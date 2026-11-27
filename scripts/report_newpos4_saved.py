"""Small reports/figures from verified archived metrics, no score calibration."""
import csv
import ctypes
import json
import os
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/newpos4_thresholds_20261121_v1'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def save(p,value):
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def available_ram():
    if os.name!='nt':
        return int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:')))*1024
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(k,ctypes.c_uint64) for k in (
            'total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('Memory query failed')
    return m.available


def main():
    initial=available_ram()
    if initial<256*1024**2:raise RuntimeError('Insufficient RAM for small metadata-only figures')
    base=read(OUT/'baseline_reference.json');new=read(OUT/'newpos4_evaluation.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if available_ram()<128*1024**2:raise RuntimeError('Figure reserve exhausted')
    figures=OUT/'figures_archived';figures.mkdir(exist_ok=False)
    for name,x,y in [('precision_recall','recall','precision'),('fpr_recall','fpr','recall')]:
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for label,r,color in [('C8',base,'#2464b5'),('NEWPOS4',new,'#bf4729')]:
            points=[p for p in r['propagation_pr']['curve'] if p[x] is not None and p[y] is not None]
            ax.plot([p[x] for p in points],[p[y] for p in points],color=color,label=label+' courbe archivée, affichage réduit')
            nf=r['new_fire'];xx=nf['false_positive_rate'] if x=='fpr' else nf[x]
            ax.scatter([xx],[nf[y]],color=color,label=label+' seuil 0,78')
        ax.set(xlabel='FPR nouveaux feux' if x=='fpr' else 'Rappel nouveaux feux',
            ylabel='Précision' if y=='precision' else 'Rappel',title='Validation V2 : nouveaux pixels actifs à J+1')
        if x=='fpr':ax.set_xscale('symlog',linthresh=.0001);ax.axvline(.00028,color='gray',ls=':',label='Budget primaire à étudier hors événement')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));labels=['C8 0,78','NEWPOS4 0,78'];rows=[base['new_fire'],new['new_fire']]
    axes[0].bar(labels,[r['fp'] for r in rows],color=['#2464b5','#bf4729']);axes[0].axhline(29134,color='black',ls=':',label='Garde FP 29 134')
    axes[0].set_ylabel('Faux positifs nouveaux feux');axes[0].legend(fontsize=8)
    for ratio in (1,5,20):axes[1].plot(labels,[r['fp']+ratio*r['fn'] for r in rows],marker='o',label=f'Coût FN / FP = {ratio}')
    axes[1].set_ylabel('FP + coût relatif × FN');axes[1].legend(fontsize=8)
    fig.suptitle('Coûts descriptifs à seuil fixe, pas de choix post-hoc');fig.tight_layout()
    fig.savefig(figures/'false_alarms_and_costs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,label,values in [(axes[0],'Dice global',[base['pooled']['dice'],new['pooled']['dice']]),
        (axes[1],'Macro Dice événementiel',[base['macro_event_dice'],new['macro_event_dice']]),
        (axes[2],'Brier global (plus petit = meilleur)',[base['pooled']['brier'],new['pooled']['brier']])]:
        ax.bar(['C8','NEWPOS4'],values,color=['#2464b5','#bf4729']);ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation V2 au seuil 0,78 ; Brier indépendant du seuil');fig.tight_layout()
    fig.savefig(OUT/'figures_archived/calibration_and_segmentation.png',dpi=150);plt.close(fig)
    ratios=[1,5,20];costs=[]
    for name,result in [('C8_078',base),('NEWPOS4_078',new)]:
        for ratio in ratios:costs.append(dict(model=name,miss_to_false_alarm_cost_ratio=ratio,
            cost=result['new_fire']['fp']+ratio*result['new_fire']['fn']))
    delta_fp=new['new_fire']['fp']-base['new_fire']['fp'];delta_tp=new['new_fire']['tp']-base['new_fire']['tp']
    save(OUT/'costs_fixed078.json',dict(costs=costs,break_even_miss_cost_per_false_alarm=delta_fp/delta_tp,
        interpretation='Engineering scenarios, no operational cost validated; no choice based on heldout scores'))
    with (OUT/'archived_descriptive_envelopes.csv').open('x',encoding='utf8',newline='') as f:
        fields=['model','fpr_budget','fp','tp','precision','recall','scope'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for label,result in [('C8',base),('NEWPOS4',new)]:
            for r in result['propagation_pr']['budgets']:
                w.writerow(dict(model=label,**{k:r[k] for k in fields[1:-1]},scope='archived same-panel oracle; not held-out or a deployable threshold'))
    baseline_rows={r['event_id']:r for r in base['per_event']};events=[]
    for r in new['per_event']:
        b=baseline_rows[r['event_id']];e=dict(event_id=r['event_id'])
        for metric in ('tp','fp','fn','tn','recall','precision','dice','iou'):
            x,y=b['new_fire'][metric],r['new_fire'][metric]
            e[metric+'_C8']=x;e[metric+'_NEWPOS4']=y;e[metric+'_delta']=y-x if x is not None and y is not None else None
        events.append(e)
    with (OUT/'paired_events_fixed078.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
    def dist(values):
        v=[x for x in values if x is not None];return dict(minimum=float(min(v)),median=float(median(v)),maximum=float(max(v)))
    diagnosis={}
    for name,result in [('C8',base),('NEWPOS4',new)]:
        rows=[r['new_fire'] for r in result['per_event']];fp=result['new_fire']['fp'];fn=result['new_fire']['fn']
        diagnosis[name]=dict(events_with_new_positives=sum(r['tp']+r['fn']>0 for r in rows),
            events_with_new_positives_but_zero_TP=sum(r['tp']==0 and r['fn']>0 for r in rows),
            event_new_recall=dist([r['recall'] for r in rows]),event_new_dice=dist([r['dice'] for r in rows]),
            top10_event_share_FP=sum(sorted([r['fp'] for r in rows],reverse=True)[:10])/fp,
            top10_event_share_FN=sum(sorted([r['fn'] for r in rows],reverse=True)[:10])/fn)
    save(OUT/'event_diagnosis.json',diagnosis)
    save(OUT/'metadata_figure_resources.json',dict(initial_available_bytes=initial,final_available_bytes=available_ram(),
        model_framework_imported=False,uses_saved_metrics_only=True,threshold_assessment_executed=False))
    print('Metadata figures, event table and cost diagnostics written; no threshold assessment',flush=True)


if __name__=='__main__':main()
