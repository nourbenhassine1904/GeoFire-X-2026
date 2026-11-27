"""CPU-only saved-score crossfit; bounded ZIP streaming and archived exact AP."""
from pathlib import Path
import csv
import ctypes
import json
import os
import shutil
import time
import zipfile
import numpy as np
from geofirex import newpos4_posthoc as n
from geofirex import c8_experiments as c
from geofirex.c8_comparison import rows_from_result,bootstrap_events
from geofirex.c8_thresholds import event_counts,summarize,serial_rows

REPORT=Path('reports/newpos4_thresholds_20261121_v1')


def memory_rss():
    """Measured Windows peak working set, or Linux /proc current RSS."""
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in (
                'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        result=Counters();result.cb=ctypes.sizeof(result)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(result),result.cb):raise OSError('RSS measurement unavailable')
        return int(result.WorkingSetSize),int(result.PeakWorkingSetSize)
    fields=Path('/proc/self/status').read_text().splitlines()
    values={r.split(':')[0]:int(r.split()[1])*1024 for r in fields if r.startswith(('VmRSS:','VmHWM:'))}
    return values['VmRSS'],values['VmHWM']


class ResourceGuard:
    """Separate streaming executor qualification; legacy A budgets stay frozen."""
    def __init__(self,destination,minimum_start=768*1024**2):
        self.destination=Path(destination);self.start_available=c.old.host_memory()[1]
        self.start_rss=memory_rss()[0];self.max_rss=self.start_rss;self.minimum_seen=self.start_available
        if self.start_available<minimum_start:raise RuntimeError('Less than 768 MiB available: use Kaggle CPU notebook')
        if shutil.disk_usage(self.destination).free<1024**3:raise RuntimeError('1 GiB output headroom required')

    def __call__(self):
        available=c.old.host_memory()[1];rss,peak=memory_rss()
        self.minimum_seen=min(self.minimum_seen,available);self.max_rss=max(self.max_rss,rss)
        if available<512*1024**2:raise RuntimeError('CPU reserve below 512 MiB; preserve partial results')
        if rss-self.start_rss>128*1024**2:raise RuntimeError('Streaming RSS increase exceeds 128 MiB budget')
        if shutil.disk_usage(self.destination).free<256*1024**2:raise RuntimeError('Output disk reserve exhausted')

    def receipt(self):
        rss,peak=memory_rss()
        return dict(start_available_bytes=self.start_available,minimum_available_sample_bytes=self.minimum_seen,
            initial_rss_bytes=self.start_rss,maximum_rss_sample_bytes=max(rss,self.max_rss),process_peak_rss_bytes=peak,
            maximum_increment_allowed_bytes=128*1024**2,initial_available_required_bytes=768*1024**2,
            reserve_available_bytes=512*1024**2,block_pixels=65536,
            reason='Compressed score streaming; no tensor inference, AP sorter, extraction or full-array load',
            legacy_A_2GiB_guard_unchanged=True)


def to_rows(serial):
    return {e:dict(pooled=np.array(r['pooled'],np.int64),new_fire=np.array(r['new_fire'],np.int64),squared_error=r['squared_error']) for e,r in serial.items()}


def analysis(root,archive,out):
    root=Path(root);out=Path(out)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only CUDA mask required')
    # Verify only frozen code and small pinned inputs; never instantiate a model.
    c.identity(root)
    pins=c.read(root/REPORT/'analysis_inputs.json')
    c.old.verify_manifest(root,pins['files'])
    if out.exists():raise FileExistsError('New output required; no overwrite')
    if not out.parent.is_dir():raise ValueError('Existing output parent required')
    guard=ResourceGuard(out.parent);start=time.monotonic()
    if c.sha256(archive)!=pins['source_archive_sha256']:raise ValueError('Different NEWPOS4 COMPLETE archive')
    # Full global/member SHA verification precedes every read of score arrays.
    c.old.verify_zip(archive,pins['source_archive_sha256'])
    store=n.ArchivedPredictions(archive,expected_identity=dict(run_id=n.RUN,checkpoint_sha256=n.CHECKPOINT))
    expected=c.read(root/REPORT/'prediction_verification.json')
    if len(store.events)!=80 or store.manifest['tiles']!=30882 or expected['status']!='PASS':raise ValueError('Coverage qualification missing')
    store.guard=guard;store.deadline=start+10800
    policy=c.read(root/REPORT/'threshold_policy.json');reference=c.read(root/REPORT/'baseline_reference.json')
    official=c.read(root/REPORT/'newpos4_evaluation.json')
    out.mkdir();c.save(out/'policy_before_analysis.json',policy)
    c.save(out/'analysis_identity.json',dict(source_archive_sha256=pins['source_archive_sha256'],
        checkpoint_sha256=n.CHECKPOINT,predictions_manifest_sha256=store.manifest_sha256,
        policy_sha256=c.sha256(root/REPORT/'threshold_policy.json'),analysis_inputs_sha256=c.sha256(root/REPORT/'analysis_inputs.json')))
    complete=False
    try:
        result=n.crossfit_costs(store,policy,out/'crossfit',reference)
        guard()
        calibrated=n.reliability(store);c.save(out/'calibration_diagnostics.json',calibrated)
        if abs(calibrated['pooled']['brier']-official['pooled']['brier'])>1e-10:raise ValueError('Brier reproduction differs')
        fixed=summarize(event_counts(store,store.events,.78))
        for domain in ('pooled','new_fire'):
            if any(fixed[domain][k]!=official[domain][k] for k in ('tp','fp','fn','tn')):raise ValueError('Fixed threshold counts changed')
        c.save(out/'fixed_078_reproduction.json',fixed)
        base_rows=rows_from_result(reference);new_rows=rows_from_result(official)
        costrows=[]
        for scenario in result['scenarios']:
            guard();candidate=to_rows(scenario['per_event'])
            c.save(out/('bootstrap_'+scenario['scenario']+'.json'),bootstrap_events(base_rows,candidate))
            c.save(out/('bootstrap_vs_newpos4_078_'+scenario['scenario']+'.json'),bootstrap_events(new_rows,candidate))
            scenario['AP_new']=official['propagation_pr']['average_precision'];scenario['AP_pooled']=official['pooled']['average_precision']
            for e,r in candidate.items():
                costrows.append(dict(scenario=scenario['scenario'],event_id=e,
                    **{f'{domain}_{k}':int(r[domain][i]) for domain in ('pooled','new_fire') for i,k in enumerate(('tp','fp','fn','tn'))}))
        with (out/'per_event_all_scenarios.csv').open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(costrows[0]));w.writeheader();w.writerows(costrows)
        c.save(out/'results.json',dict(**result,probability_metrics_unchanged_by_threshold=True,
            newpos4_078=official,baseline_C8_078=reference,C8_primary_crossfit=c.read(root/REPORT/'c8_crossfit.json'),
            model_promotion_gates_can_all_pass=False,
            permanent_model_gate_failures=['AP_new improvement <0.005','pooled AP below required baseline+0.005','Brier above baseline'],
            statement='Saved validation development only; no claim of test generalization'))
        comparison_plots(reference,official,c.read(root/REPORT/'c8_crossfit.json'),out/'figures',result,calibrated)
        if time.monotonic()>store.deadline:raise TimeoutError('Analysis wall budget exhausted')
        guard();c.save(out/'resources.json',dict(**guard.receipt(),seconds=time.monotonic()-start,GPU_used=False,
            training_executed=False,inference_executed=False,test_pixels_opened=0))
        c.save(out/'complete.json',dict(status='COMPLETE',thresholds_sha256=result['thresholds_sha256'],promotion_authorized=False))
        complete=True
    finally:
        if not complete:c.save(out/'partial.json',dict(status='PARTIAL_NO_SELECTION',resources=guard.receipt()))
        c.old.package(out,'CPU-ANALYSIS-COMPLETE' if complete else 'PARTIAL')
    return out


def comparison_plots(base,new,c8cross,out,exploratory=None,calibration=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=False)
    styles=[('C8',base,'#2464b5'),('NEWPOS4',new,'#bf4729')]
    for name,x,y in [('precision_recall','recall','precision'),('fpr_recall','fpr','recall')]:
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for label,r,color in styles:
            curve=r['propagation_pr']['curve'];points=[p for p in curve if p[x] is not None and p[y] is not None]
            ax.plot([p[x] for p in points],[p[y] for p in points],color=color,label=label+' : courbe archivée, affichage réduit')
            nf=r['new_fire'];xx=nf['false_positive_rate'] if x=='fpr' else nf[x]
            ax.scatter([xx],[nf[y]],color=color,marker='o',label=label+' seuil 0,78')
        if exploratory:
            for s in exploratory['scenarios']:
                r=s['summary']['new_fire'];xx=r['false_positive_rate'] if x=='fpr' else r[x]
                if xx is not None and r[y] is not None:ax.scatter([xx],[r[y]],marker='x',s=70,label='NEWPOS4 hors événement '+s['scenario'])
        ax.set(xlabel='FPR nouveaux feux' if x=='fpr' else 'Rappel nouveaux feux',ylabel='Précision' if y=='precision' else 'Rappel',
            title='Validation V2 : nouveaux pixels actifs à J+1')
        if x=='fpr':ax.set_xscale('symlog',linthresh=.0001);ax.axvline(.00028,color='gray',ls=':',label='Budget de calibration primaire')
        ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));labels=['C8 0,78','NEWPOS4 0,78'];rows=[base['new_fire'],new['new_fire']]
    if exploratory:
        for s in exploratory['scenarios']:labels.append(s['scenario']);rows.append(s['summary']['new_fire'])
    axes[0].bar(labels,[r['fp'] for r in rows],color='#d06454');axes[0].axhline(29134,color='black',ls=':',label='Garde FP 29 134')
    axes[0].set_ylabel('Faux positifs nouveaux feux');axes[0].legend(fontsize=8)
    for ratio in (1,5,20):axes[1].plot(labels,[r['fp']+ratio*r['fn'] for r in rows],marker='o',label=f'Coût FN / FP = {ratio}')
    axes[1].set_ylabel('FP + coût relatif × FN');axes[1].legend(fontsize=8)
    for ax in axes:ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Compromis de coût exploratoires ; aucun choix post-hoc de scénario');fig.tight_layout()
    fig.savefig(out/'false_alarms_and_costs.png',dpi=150);plt.close(fig)
    if calibration:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,domain in zip(axes,('pooled','new_fire')):
            bins=[b for b in calibration[domain]['bins'] if b['n']]
            ax.plot([b['mean_probability'] for b in bins],[b['observed_frequency'] for b in bins],marker='o',label='NEWPOS4, 20 bins fixes')
            ax.plot([0,1],[0,1],ls='--',color='gray');ax.set(xlabel='Score sigmoid moyen',ylabel='Fréquence observée',title=domain)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Calibration descriptive ; aucune transformation ajustée');fig.tight_layout()
        fig.savefig(out/'reliability.png',dpi=150);plt.close(fig)
