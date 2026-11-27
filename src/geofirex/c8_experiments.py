"""Independent C8 A/B execution. No action runs without an exact external grant.

Official C8 code/data/results and C36 work remain immutable dependencies.
"""
from pathlib import Path
from contextlib import contextmanager
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
import numpy as np

from geofirex import v2_execution as old
from geofirex.protocol_v2 import sha256,save,V2CompactTiles,sample_keys,make_v2_loaders
from geofirex.c8_predictions import PredictionWriter,PredictionStore

REPORT=Path('reports/c8_propagation_20261121_v1')
A_ID='C8_V2_THRESHOLD_CROSSFIT_seed42_v1'
B_ID='B2_C8_V2_NEWPOS4_seed42'
BASE_CKPT='c2273fea252d709f4d316eba76515d669be97eac8bfa9302ff4831da2c9aedc1'
BASE_ZIP='37a8d69647ddebaa902d88e21fddf4df3f56eaa57e7094695b478d04116c1415'
DOMAIN='efc995be596d7d520bac472ceb1b5122fe2111fe9d5a53c5edbec02851c1accc'


def read(path):return old.read(path)


def identity(root):
    root=Path(root);base,_,u=old.identity(root)
    lock=read(root/REPORT/'execution_manifest.json');old.verify_manifest(root,lock['files'])
    b=read(root/'configs/c8_newpos4_v1.json');a=read(root/'configs/c8_thresholds_v1.json')
    expected=dict(base);expected.update(run_id=B_ID,loss=dict(base['loss'],name='NewPositiveLoss',new_positive_weight=4.),
        status='PREPARED_DISABLED',base_config_sha256=sha256(root/'configs/protocol_v2/baseline_c8.json'))
    if b!=expected:raise ValueError('NEWPOS4 may change only its registered loss and identity')
    if a['analysis_id']!=A_ID or a['baseline_checkpoint_sha256']!=BASE_CKPT or a['fpr_budget']!=.00028:
        raise ValueError('Threshold analysis contract differs')
    return base,b,a,lock,u


def authorize(root,path,action,device,out_parent):
    _,_,_,_,_=identity(root)
    a=read(path);policy=read(Path(root)/REPORT/'budget.json')
    experiment=a.get('experiment_id') if action=='gpu-preflight' else A_ID if action.startswith('threshold-') else B_ID
    if experiment not in (A_ID,B_ID):raise ValueError('Unknown experiment grant')
    expected=dict(authorized=True,action=action,experiment_id=experiment,device=device,runs=1,
        execution_manifest_sha256=sha256(Path(root)/REPORT/'execution_manifest.json'))
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError('Explicit action/experiment/device/manifest grant required')
    if (not a.get('authorized_by') or not a.get('authorized_at')
            or Path(a.get('output_parent','__absent__')).resolve()!=Path(out_parent).resolve()
            or not a.get('local_backup_destination') or not a.get('external_backup_destination')):
        raise ValueError('Grant requires operator, time, exact destination and two backup roots')
    required=policy['A_session_minutes'] if experiment==A_ID else policy['B_session_minutes']
    if action=='threshold-infer' and device=='cpu':required=policy['A_cpu_inference_session_minutes']
    if action=='threshold-analyze':required=policy['A_cpu_analysis_session_minutes']
    def number(name):
        value=a.get(name)
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
            raise ValueError('Finite numeric authorization field required: '+name)
        return value
    if number('session_remaining_minutes')<required:raise ValueError('Insufficient authorized session time')
    if device=='cuda':
        cards=policy['A_card_hours'] if experiment==A_ID else policy['B_card_hours']
        if number('quota_remaining_card_hours')<cards or number('card_hours_reserved')<cards or not a.get('quota_checked_at'):
            raise ValueError('Current quota and conservative card reserve required')
    return a


def preflight(root,maps=None,work=None,device='cpu',gpu=False):
    import torch
    base,b,a,_,u=identity(root);policy=read(Path(root)/REPORT/'budget.json')
    result=dict(status='CPU_IDENTITY_PASS',**u,validation_tiles=30882,validation_events=80,
        gpu_checked=False,inference_executed=False,training_executed=False,test_pixels_opened=0,
        host_ram_total_available=old.host_memory(),torch=str(torch.__version__),numpy=np.__version__)
    if maps:
        mounts,receipt=old.verify_mounts(root,maps)
        train,val=make_v2_loaders('',Path(root)/base['index'],mounts=mounts,workers=0)
        try:
            if len(train.dataset)!=115900 or len(val.dataset)!=30882 or len({k[0] for k in sample_keys(val.dataset)})!=80:
                raise ValueError('Frozen coverage differs')
            result.update(mounts=receipt,loader_lengths=[len(train.dataset),len(val.dataset)])
        finally:train.dataset.close();val.dataset.close()
    if work is not None:
        total,available=old.host_memory();minimum=8*old.GIB if gpu else 2*old.GIB
        if available<minimum or (gpu and total<16*old.GIB):raise RuntimeError('Insufficient total/available host RAM')
        result['disk_free_bytes']=old.disk_check(work,policy['required_free_bytes'])
    if gpu:
        # The caller must have validated its grant before requesting this branch.
        qualified=old.preflight(root,maps,gpu=True)
        if torch.backends.cudnn.version()!=92700:raise ValueError('cuDNN differs from archived C8 environment')
        result.update(gpu_checked=True,status='GPU_RUNTIME_PASS',gpu=qualified['gpu'],gpu_memory_bytes=qualified['gpu_memory_bytes'])
    return result


def new_output(path,name):
    p=Path(path).resolve()
    if p.exists() or p.name!=name or 'input' in p.parts or '2021' in p.parts:
        raise ValueError('New, correctly named output outside input/test paths required')
    p.parent.mkdir(parents=True,exist_ok=True)
    save(p.parent/(name+'_SINGLE_RUN.lock.json'),dict(experiment_id=name))
    p.mkdir();return p


def preserve(root,out,lock):
    names=set(lock['files'])|{(old.REPORT/'execution_manifest.json').as_posix(),(REPORT/'execution_manifest.json').as_posix()}
    for name in sorted(names):
        src=Path(root)/name;dest=out/'inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as target,src.open('rb') as source:shutil.copyfileobj(source,target)


def baseline_json(root):return read(Path(root)/REPORT/'baseline_reference.json')


def load_official(root,archive):
    """Only our exact known archive and exact final checkpoint may be unpickled."""
    import torch
    old.verify_zip(archive,BASE_ZIP)
    with zipfile.ZipFile(archive) as z:
        data=z.read('checkpoint_u0028976.pt')
    import hashlib
    if hashlib.sha256(data).hexdigest()!=BASE_CKPT:raise ValueError('Official final checkpoint differs')
    state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    base,lock,_=old.identity(root)
    if (state['config']!=base or state['execution_manifest']!=lock or state['epoch']!=4
            or state['updates']!=28976 or state['cursor']!=0):raise ValueError('Official checkpoint state differs')
    return state


def make_model(config,device):
    from geofirex.benchmark_models import StackedUNet
    return StackedUNet(8,3,base_channels=config['base_channels'],depth=config['depth']).to(device)


def configure_runtime(seed,device):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def predict_validation(root,maps,model,out,source_identity,device,seconds):
    """Shared single pass, validation only. No optimizer or threshold calibration."""
    import torch
    from torch.utils.data import DataLoader
    base,_,_,_,_=identity(root);mounts,_=old.verify_mounts(root,maps)
    ds=V2CompactTiles('',Path(root)/base['index'],'val',mounts=mounts)
    writer=None;start=time.monotonic();offset=0
    try:
        keys=sample_keys(ds);writer=PredictionWriter(out,keys,source_identity)
        model.eval()
        with torch.inference_mode():
            for x,y,v in DataLoader(ds,batch_size=16,num_workers=2 if device=='cuda' else 0,shuffle=False):
                if time.monotonic()-start>seconds:raise TimeoutError('Prediction budget exhausted; partial store cannot be analyzed')
                p=model(x.to(device)).sigmoid().cpu().numpy().astype(np.float32)
                for i in range(len(x)):
                    writer.add(keys[offset+i],p[i,0],y[i,0].numpy(),v[i,0].numpy(),x[i,-1,7].numpy())
                offset+=len(x)
                if offset%1024==0:old.disk_check(Path(out).parent,2*old.GIB)
        writer.finish(DOMAIN)
        save(Path(out)/'resources.json',dict(seconds=time.monotonic()-start,tiles=offset,device=device,
            host_rss_sample_bytes=old.host_rss(),training_executed=False,test_pixels_opened=0))
    finally:
        if writer is not None:writer.close()
        ds.close()


def infer_thresholds(root,maps,archive,out,authorization,device='cuda'):
    authorize(root,authorization,'threshold-infer',device,Path(out).resolve().parent)
    base,_,a,lock,_=identity(root);policy=read(Path(root)/REPORT/'budget.json')
    pre=preflight(root,maps,Path(out).parent,device,gpu=device=='cuda')
    state=load_official(root,archive)  # grant and capacity checked before deserialization
    out=new_output(out,A_ID);preserve(root,out,lock);save(out/'preflight.json',pre)
    shutil.copyfile(authorization,out/'authorization.json')
    try:
        configure_runtime(42,device);model=make_model(base,device);model.load_state_dict(state['model'])
        predict_validation(root,maps,model,out/'predictions',dict(analysis_id=A_ID,checkpoint_sha256=BASE_CKPT),
            device,policy['A_inference_seconds_cuda' if device=='cuda' else 'A_inference_seconds_cpu'])
        save(out/'inference_complete.json',dict(status='COMPLETE',checkpoint_sha256=BASE_CKPT))
    finally:old.package(out,'PREDICTIONS-SAFE' if (out/'inference_complete.json').exists() else 'PARTIAL')
    return out


def exact_metrics(store,destination,policy):
    from geofirex.v2_propagation import PropagationMetrics
    keys=[tuple(k) for row in store.files.values() for k in row['keys']]
    acc=PropagationMetrics(Path(destination)/'_scratch',keys,policy)
    for key,a in store.tiles():acc.update(key,a['p'].copy(),a['y'].copy(),a['v'].copy(),a['o'].copy())
    result=acc.result();save(Path(destination)/'evaluation.json',result);return result


@contextmanager
def analysis_deadline(seconds):
    """Linux alarm; Windows cooperative chunks plus completion-time guard."""
    import signal
    if getattr(signal,'SIGALRM',None) is not None:
        with old.evaluation_deadline(seconds):yield
    else:
        end=time.monotonic()+seconds
        yield
        if time.monotonic()>end:raise TimeoutError('Analysis exceeded cooperative Windows deadline')


def analyze_thresholds(root,predictions,out,authorization):
    from geofirex.c8_thresholds import crossfit,plots
    from geofirex.c8_comparison import bootstrap_events
    authorize(root,authorization,'threshold-analyze','cpu',Path(out).resolve().parent)
    receipt=preflight(root,work=Path(out).parent)
    store=PredictionStore(predictions,expected_identity=dict(analysis_id=A_ID,checkpoint_sha256=BASE_CKPT),expected_domain=DOMAIN)
    if len(store.events)!=80 or store.manifest['tiles']!=30882:raise ValueError('All 80 validation events required')
    seconds=read(Path(root)/REPORT/'budget.json')['A_analysis_seconds']
    store.deadline=time.monotonic()+seconds
    out=new_output(out,A_ID+'_ANALYSIS')
    preserve(root,out,identity(root)[3]);save(out/'preflight.json',receipt)
    shutil.copyfile(authorization,out/'authorization.json')
    try:
        with analysis_deadline(seconds):
            result=crossfit(store,out/'crossfit')  # locked before assessment, before global score calculation
            exact=exact_metrics(store,out,read(Path(root)/old.REPORT/'propagation_policy.json'))
        original=baseline_json(root)
        reproduction=dict(domain_match=exact['evaluation_domain_sha256']==original['evaluation_domain_sha256'],
            pooled_counts_match=all(exact['pooled'][k]==original['pooled'][k] for k in ('tp','fp','fn','tn')),
            new_counts_match=all(exact['new_fire'][k]==original['new_fire'][k] for k in ('tp','fp','fn','tn')),
            AP_delta=exact['pooled']['average_precision']-original['pooled']['average_precision'],
            new_AP_delta=exact['propagation_pr']['average_precision']-original['propagation_pr']['average_precision'],
            Brier_delta=exact['pooled']['brier']-original['pooled']['brier'])
        reproduction['pass']=(reproduction['domain_match'] and reproduction['pooled_counts_match']
            and reproduction['new_counts_match'] and abs(reproduction['AP_delta'])<=1e-6
            and abs(reproduction['new_AP_delta'])<=1e-6 and abs(reproduction['Brier_delta'])<=1e-8)
        save(out/'baseline_reproduction.json',reproduction)
        def convert(rows):return {e:dict(pooled=np.array(r['pooled'],np.int64),new_fire=np.array(r['new_fire'],np.int64),squared_error=r['squared_error']) for e,r in rows.items()}
        save(out/'bootstrap_events.json',bootstrap_events(convert(result['per_event_baseline']),convert(result['per_event_crossfit'])))
        plots(exact,result,out/'figures')
        if time.monotonic()>store.deadline:raise TimeoutError('Analysis budget exceeded; results remain partial')
        save(out/'REPORT.json',dict(exploratory=True,threshold_acceptance=result['acceptance'],
            baseline_reproduction_pass=reproduction['pass'],AP_new=exact['propagation_pr']['average_precision'],
            AP_and_Brier_improvement_from_threshold=False,
            selection_eligible=result['acceptance_all'] and reproduction['pass'],automatic_selection=False))
        (out/'RAPPORT.md').write_text('# Analyse exploratoire des seuils C8\n\n'+
            'Les seuils ont été figés hors événement avant estimation. Les probabilités sont inchangées : aucune amélioration AP/Brier par le seuil.\n\n'+
            '```json\n'+json.dumps(dict(baseline=result['baseline_078_same_probabilities'],crossfit=result['crossfit'],
                acceptance=result['acceptance'],baseline_reproduction=reproduction),indent=2)+'\n```\n',encoding='utf8')
        save(out/'complete.json',dict(status='COMPLETE',thresholds_sha256=result['thresholds_sha256']))
    finally:old.package(out,'ANALYSIS-COMPLETE' if (out/'complete.json').exists() else 'PARTIAL')
    return out


def final_candidate(root,source):
    import torch
    _,c,_,lock,_=identity(root);source=Path(source);receipt=read(source/'training_complete.json')
    p=source/receipt['checkpoint']
    if (p.parent.resolve()!=source.resolve() or receipt['updates']!=28976 or receipt['checkpoint_sha256']==BASE_CKPT
            or sha256(p)!=receipt['checkpoint_sha256'] or sha256(p)!=read(p.with_suffix('.json'))['sha256']):
        raise ValueError('Candidate final checkpoint differs')
    state=torch.load(p,map_location='cpu',weights_only=False)
    if state['config']!=c or state['execution_manifest']!=lock or (state['epoch'],state['updates'],state['cursor'])!=(4,28976,0):
        raise ValueError('Not the fixed final NEWPOS4 checkpoint')
    return state,p


def restore_candidate(root,archive,sidecar,destination):
    """Restore only an own, verified NEWPOS4 TRAINING-SAFE into a fresh folder."""
    _,config,_,_,_=identity(root)
    expected=Path(sidecar).read_text(encoding='ascii').split()[0]
    old.verify_zip(archive,expected)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('artifact_manifest.json'))
        cfg=json.loads(z.read('inputs/configs/c8_newpos4_v1.json'))
        lock=json.loads(z.read('inputs/'+REPORT.as_posix()+'/execution_manifest.json'))
        if (manifest['stage']!='TRAINING-SAFE' or cfg!=config
                or lock!=read(Path(root)/REPORT/'execution_manifest.json')):
            raise ValueError('Not this NEWPOS4 training-safe archive')
    return old.restore(archive,sidecar,destination)


def safeguarded_candidate_evaluation(root,out,callback):
    """CPU-testable handoff: checkpoint archive precedes all inference/AP."""
    complete=False
    try:
        final_candidate(root,out)
        old.package(out,'TRAINING-SAFE')
        result=callback()
        if not (Path(out)/'evaluation/complete.json').exists():raise ValueError('Evaluation coverage not complete')
        complete=True;return result
    finally:old.package(out,'COMPLETE' if complete else 'PARTIAL')


def candidate_evaluation(root,maps,source,destination):
    from geofirex.c8_comparison import compare
    _,c,_,_,_=identity(root);destination=Path(destination);destination.mkdir(exist_ok=False)
    state,p=final_candidate(root,source);model=make_model(c,'cuda');model.load_state_dict(state['model'])
    policy=read(Path(root)/REPORT/'budget.json')
    predict_validation(root,maps,model,destination/'predictions',dict(run_id=B_ID,checkpoint_sha256=sha256(p)),
        'cuda',policy['B_evaluation_seconds'])
    store=PredictionStore(destination/'predictions',expected_domain=DOMAIN)
    result=exact_metrics(store,destination,read(Path(root)/old.REPORT/'propagation_policy.json'))
    comparison=compare(baseline_json(root),result,destination/'paired_comparison',store.events)
    save(destination/'complete.json',dict(status='COMPLETE',checkpoint_sha256=sha256(p),tiles=result['tiles']))
    return comparison


def evaluate_candidate(root,maps,source,out,authorization):
    authorize(root,authorization,'newpos4-evaluate','cuda',Path(out).resolve().parent)
    preflight(root,maps,Path(out).parent,'cuda',gpu=True)
    if (Path(source)/'evaluation/complete.json').exists():raise ValueError('Already completely evaluated; preserve it')
    _,_,_,lock,_=identity(root);_,p=final_candidate(root,source)
    out=new_output(out,B_ID+'_EVALUATION_RECOVERY');preserve(root,out,lock)
    shutil.copyfile(authorization,out/'authorization.json')
    for src in (p,p.with_suffix('.json'),Path(source)/'training_complete.json'):
        with (out/src.name).open('xb') as dst,src.open('rb') as stream:shutil.copyfileobj(stream,dst)
    try:
        with old.evaluation_deadline(read(Path(root)/REPORT/'budget.json')['B_evaluation_seconds']):
            candidate_evaluation(root,maps,out,out/'evaluation')
    finally:old.package(out,'EVALUATION-RECOVERY')
    return out


def fit(root,maps,out,authorization):
    """Only NEWPOS4, fresh seed42. No resume or C8 checkpoint initialization."""
    import torch
    from geofirex.c8_newpos_loss import NewPositiveLoss
    authorize(root,authorization,'newpos4-fit','cuda',Path(out).resolve().parent)
    base,c,_,lock,u=identity(root);policy=read(Path(root)/REPORT/'budget.json')
    receipt=preflight(root,maps,Path(out).parent,'cuda',gpu=True)
    if Path(out).resolve().parent!=Path('/kaggle/working').resolve():raise ValueError('New direct Kaggle working output required')
    out=new_output(out,B_ID);preserve(root,out,lock);save(out/'preflight.json',receipt)
    shutil.copyfile(authorization,out/'authorization.json')
    train=val=None;last=None;start=time.monotonic();io_seconds=0.;peak_rss=0;handed_off=False
    try:
        configure_runtime(42,'cuda');model=make_model(c,'cuda')
        if sum(p.numel() for p in model.parameters())!=123977:raise ValueError('Architecture changed')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        criterion=NewPositiveLoss(4.,.5,10.).to('cuda')
        mounts,_=old.verify_mounts(root,maps)
        train,val=make_v2_loaders('',Path(root)/base['index'],mounts=mounts,batch_size=16,workers=2)
        state=dict(epoch=0,cursor=0,updates=0,history=[])
        last=old.checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
        torch.cuda.reset_peak_memory_stats()
        with (out/'steps.jsonl').open('x',encoding='utf8') as log:
            for epoch in range(4):
                train.sampler.epoch=epoch;model.train();total_loss=0.;valid_pixels=0;before=time.monotonic()
                for x,y,v in train:
                    io_seconds+=time.monotonic()-before
                    if time.monotonic()-start>policy['B_fit_seconds']:
                        if read(last.with_suffix('.json'))['updates']!=state['updates']:
                            last=old.checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
                        save(out/'paused.json',dict(status='PARTIAL_NO_AUTOMATIC_RESUME',updates=state['updates']));return out
                    observed=x[:,-1,7:8].to('cuda')
                    optimizer.zero_grad(set_to_none=True)
                    loss=criterion(model(x.to('cuda')),y.to('cuda'),v.to('cuda'),observed)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();n=int(v.sum());total_loss+=float(loss.detach())*n;valid_pixels+=n
                    state['updates']+=1;state['cursor']+=len(x)
                    log.write(json.dumps(dict(epoch=epoch+1,updates=state['updates'],cursor=state['cursor'],loss=float(loss.detach())))+'\n')
                    if state['updates']%1000==0:
                        log.flush();old.disk_check(out,2*old.GIB)
                        last=old.checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
                    peak_rss=max(peak_rss,old.host_rss());before=time.monotonic()
                if state['cursor']!=115900:raise ValueError('Nonexhaustive training epoch')
                state['history'].append(dict(epoch=epoch+1,loss=total_loss/valid_pixels))
                state['epoch']=epoch+1;state['cursor']=0
                last=old.checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
        if state['updates']!=28976:raise ValueError('Update budget differs')
        torch.cuda.synchronize();fit_seconds=time.monotonic()-start
        save(out/'resources_fit.json',dict(fit_seconds=fit_seconds,loader_wait_seconds=io_seconds,
            peak_host_rss_bytes=peak_rss,peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),visible_gpu_count=1))
        save(out/'training_complete.json',dict(status='COMPLETE',epochs=4,updates=28976,checkpoint=last.name,checkpoint_sha256=sha256(last)))
        def evaluate_final():
            with old.evaluation_deadline(policy['B_evaluation_seconds']):
                torch.cuda.reset_peak_memory_stats();candidate_evaluation(root,maps,out,out/'evaluation')
            save(out/'runtime_budget.json',dict(fit_seconds=fit_seconds,total_seconds=time.monotonic()-start,
                fit_within_budget=fit_seconds<=policy['B_fit_seconds'],
                session_within_budget=time.monotonic()-start<=60*policy['B_session_minutes'],
                quota_card_hours='Record actual Kaggle deduction separately; visibility does not prove billing'))
        handed_off=True;safeguarded_candidate_evaluation(root,out,evaluate_final)
        return out
    finally:
        if train is not None:train.dataset.close()
        if val is not None:val.dataset.close()
        if not handed_off:old.package(out,'PARTIAL')
