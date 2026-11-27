"""FIRE13 input-only execution adapted from C8/NEWPOS2. Requires an exact grant.

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
from geofirex.fire13_features import Fire13Tiles,make_fire13_loaders,NAMES
from geofirex.c8_predictions import PredictionWriter,PredictionStore

REPORT=Path('reports/fire13_20261122_v1')
B_ID='B2_C13_FIRE_V2_seed42'
BASE_CKPT='c2273fea252d709f4d316eba76515d669be97eac8bfa9302ff4831da2c9aedc1'
BASE_ZIP='37a8d69647ddebaa902d88e21fddf4df3f56eaa57e7094695b478d04116c1415'
DOMAIN='efc995be596d7d520bac472ceb1b5122fe2111fe9d5a53c5edbec02851c1accc'


def read(path):return old.read(path)


def identity(root):
    root=Path(root)
    base,original,u=old.identity(root)
    lock=read(root/REPORT/'execution_manifest.json')
    old.verify_manifest(root,lock['files'])
    candidate=read(root/'configs/c8_fire13_v1.json')
    expected=dict(base,run_id=B_ID,status='PREPARED_DISABLED',channels_per_day=13,model_input_channels=39,
        channel_names=base['channel_names']+NAMES,feature_policy='C8_plus_FIRE13',
        derived_normalization_sha256=sha256(root/REPORT/'derived_normalization.json'),
        scene_geometry_sha256=sha256(root/REPORT/'scene_geometry.json'),
        base_config_sha256=sha256(root/'configs/protocol_v2/baseline_c8.json'))
    if candidate!=expected:raise ValueError('Only registered FIRE13 input enrichment may differ')
    parity=read(root/REPORT/'parity_full.json')
    if parity['status']!='PASS' or parity['roles']['train']['tiles']!=115900 or parity['roles']['val']['tiles']!=30882:
        raise ValueError('Exhaustive real FIRE13 parity required')
    if len(original['files'])!=77 or lock['run_id']!=B_ID:
        raise ValueError('Original scientific identities or new identity differ')
    return base,candidate,None,lock,u


def make_criterion():
    from geofirex.unet import SegmentationLoss
    return SegmentationLoss(.5,10.)


def authorize(root,path,action,device,out_parent):
    identity(root)
    if action not in ('gpu-preflight','fire13-fit','fire13-evaluate') or device!='cuda':
        raise ValueError('Unknown action/device')
    a=read(path);policy=read(Path(root)/REPORT/'budget.json')
    expected=dict(authorized=True,action=action,experiment_id=B_ID,device='cuda',runs=1,
        execution_manifest_sha256=sha256(Path(root)/REPORT/'execution_manifest.json'))
    if any(a.get(k)!=v for k,v in expected.items()):
        raise ValueError('Explicit FIRE13 action/device/manifest authorization required')
    if (not a.get('authorized_by') or not a.get('authorized_at') or not a.get('quota_checked_at')
            or Path(a.get('output_parent','__absent__')).resolve()!=Path(out_parent).resolve()
            or not a.get('local_backup_destination') or not a.get('external_backup_destination')
            or a['local_backup_destination']==a['external_backup_destination']):
        raise ValueError('Operator/date/current quota/exact output/two backup roots required')
    for field,minimum in [('session_remaining_minutes',policy['B_session_minutes']),
                          ('quota_remaining_card_hours',policy['B_card_hours']),
                          ('card_hours_reserved',policy['B_card_hours'])]:
        value=a.get(field)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<minimum:
            raise ValueError('Insufficient finite authorized budget: '+field)
    return a


def loaders(root,mounts,workers=0):
    base=read(Path(root)/'configs/protocol_v2/baseline_c8.json')
    return make_fire13_loaders('',Path(root)/base['index'],mounts=mounts,workers=workers,
        geometry=read(Path(root)/REPORT/'scene_geometry.json'),stats=read(Path(root)/REPORT/'derived_normalization.json'))


def preflight(root,maps=None,work=None,device='cpu',gpu=False):
    import torch
    base,b,a,_,u=identity(root);policy=read(Path(root)/REPORT/'budget.json')
    from geofirex.fire13_features import numerical_selftest
    import scipy
    numerics=numerical_selftest()
    result=dict(status='CPU_IDENTITY_PASS',**u,validation_tiles=30882,validation_events=80,
        gpu_checked=False,inference_executed=False,training_executed=False,test_pixels_opened=0,
        scipy=scipy.__version__,derived_numerics=numerics,
        host_ram_total_available=old.host_memory(),torch=str(torch.__version__),numpy=np.__version__)
    if maps is not None:
        mounts,receipt=old.verify_mounts(root,maps)
        train,val=loaders(root,mounts,workers=0)
        try:
            if len(train.dataset)!=115900 or len(val.dataset)!=30882 or len({k[0] for k in sample_keys(val.dataset)})!=80:
                raise ValueError('Frozen coverage differs')
            result.update(mounts=receipt,loader_lengths=[len(train.dataset),len(val.dataset)])
        finally:train.dataset.close();val.dataset.close()
    if work is not None:
        total,available=old.host_memory();minimum=8*old.GIB if gpu else 2*old.GIB
        if available<minimum or (gpu and total<16*old.GIB):raise RuntimeError('Insufficient total/available host RAM')
        result['disk_free_bytes']=old.disk_check(work,policy['required_free_bytes'])
    if gpu and not maps:raise ValueError('All eleven maps required before GPU qualification')
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


def make_model(config,device):
    from geofirex.benchmark_models import StackedUNet
    return StackedUNet(13,3,base_channels=config['base_channels'],depth=config['depth']).to(device)


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
    ds=Fire13Tiles(V2CompactTiles('',Path(root)/base['index'],'val',mounts=mounts),
        read(Path(root)/REPORT/'scene_geometry.json'),read(Path(root)/REPORT/'derived_normalization.json'))
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


def exact_metrics(store,destination,policy):
    from geofirex.v2_propagation import PropagationMetrics
    keys=[tuple(k) for row in store.files.values() for k in row['keys']]
    acc=PropagationMetrics(Path(destination)/'_scratch',keys,policy)
    for key,a in store.tiles():acc.update(key,a['p'].copy(),a['y'].copy(),a['v'].copy(),a['o'].copy())
    result=acc.result();save(Path(destination)/'evaluation.json',result);return result


def final_candidate(root,source):
    import torch
    _,c,_,lock,_=identity(root);source=Path(source);receipt=read(source/'training_complete.json')
    p=source/receipt['checkpoint']
    if (p.parent.resolve()!=source.resolve() or receipt['updates']!=28976 or receipt['checkpoint_sha256']==BASE_CKPT
            or sha256(p)!=receipt['checkpoint_sha256'] or sha256(p)!=read(p.with_suffix('.json'))['sha256']):
        raise ValueError('Candidate final checkpoint differs')
    state=torch.load(p,map_location='cpu',weights_only=False)
    if state['config']!=c or state['execution_manifest']!=lock or (state['epoch'],state['updates'],state['cursor'])!=(4,28976,0):
        raise ValueError('Not the fixed final FIRE13 checkpoint')
    return state,p


def restore_candidate(root,archive,sidecar,destination):
    """Restore only an own, verified FIRE13 TRAINING-SAFE into a fresh folder."""
    _,config,_,_,_=identity(root)
    expected=Path(sidecar).read_text(encoding='ascii').split()[0]
    old.verify_zip(archive,expected)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('artifact_manifest.json'))
        cfg=json.loads(z.read('inputs/configs/c8_fire13_v1.json'))
        lock=json.loads(z.read('inputs/'+REPORT.as_posix()+'/execution_manifest.json'))
        if (manifest['stage']!='TRAINING-SAFE' or cfg!=config
                or lock!=read(Path(root)/REPORT/'execution_manifest.json')):
            raise ValueError('Not this FIRE13 training-safe archive')
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
    from geofirex.fire13_comparison import compare
    _,c,_,_,_=identity(root);destination=Path(destination);destination.mkdir(exist_ok=False)
    state,p=final_candidate(root,source);model=make_model(c,'cuda');model.load_state_dict(state['model'])
    policy=read(Path(root)/REPORT/'budget.json')
    predict_validation(root,maps,model,destination/'predictions',dict(run_id=B_ID,checkpoint_sha256=sha256(p)),
        'cuda',policy['B_evaluation_seconds'])
    store=PredictionStore(destination/'predictions',expected_domain=DOMAIN)
    result=exact_metrics(store,destination,read(Path(root)/old.REPORT/'propagation_policy.json'))
    comparison=compare(root,result,destination/'paired_comparison',store.events)
    from geofirex.fire13_comparison import comparable_fp
    comparable_fp(root,store,destination/'comparable_fp')
    save(destination/'complete.json',dict(status='COMPLETE',checkpoint_sha256=sha256(p),tiles=result['tiles']))
    return comparison


def evaluate_candidate(root,maps,source,out,authorization):
    authorize(root,authorization,'fire13-evaluate','cuda',Path(out).resolve().parent)
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
    """Only FIRE13, fresh seed42. No resume or C8 checkpoint initialization."""
    import torch
    from geofirex.unet import SegmentationLoss
    authorize(root,authorization,'fire13-fit','cuda',Path(out).resolve().parent)
    base,c,_,lock,u=identity(root);policy=read(Path(root)/REPORT/'budget.json')
    receipt=preflight(root,maps,Path(out).parent,'cuda',gpu=True)
    if Path(out).resolve().parent!=Path('/kaggle/working').resolve():raise ValueError('New direct Kaggle working output required')
    out=new_output(out,B_ID);preserve(root,out,lock);save(out/'preflight.json',receipt)
    shutil.copyfile(authorization,out/'authorization.json')
    train=val=None;last=None;start=time.monotonic();io_seconds=0.;peak_rss=0;handed_off=False
    try:
        configure_runtime(42,'cuda');model=make_model(c,'cuda')
        if sum(p.numel() for p in model.parameters())!=125057:raise ValueError('Architecture changed')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        criterion=make_criterion().to('cuda')
        mounts,_=old.verify_mounts(root,maps)
        train,val=loaders(root,mounts,workers=2)
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
                    optimizer.zero_grad(set_to_none=True)
                    loss=criterion(model(x.to('cuda')),y.to('cuda'),v.to('cuda'))
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
            elapsed=time.monotonic()-start
            save(out/'runtime_budget.json',dict(fit_seconds=fit_seconds,total_seconds=elapsed,
                fit_within_budget=fit_seconds<=policy['B_fit_seconds'],
                session_within_budget=elapsed<=60*policy['B_session_minutes'],
                quota_card_hours='Record actual Kaggle deduction separately; visibility does not prove billing'))
            if elapsed>60*policy['B_session_minutes']:raise TimeoutError('Authorized session budget exceeded')
        handed_off=True;safeguarded_candidate_evaluation(root,out,evaluate_final)
        return out
    finally:
        if train is not None:train.dataset.close()
        if val is not None:val.dataset.close()
        if not handed_off:old.package(out,'PARTIAL')
