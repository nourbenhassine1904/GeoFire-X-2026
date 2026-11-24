"""Disabled-by-default, single controlled V2 experiment. No FIRE13 path."""
from pathlib import Path
import math
import shutil
import time
import numpy as np
from geofirex import v2_execution as old
from geofirex.protocol_v2 import save,sha256,make_v2_loaders,sample_keys
from geofirex.auxnew_model import AuxiliaryUNet,AuxiliaryLoss
from geofirex.c8_predictions import PredictionWriter,PredictionStore
from geofirex.v2_propagation import PropagationMetrics
from geofirex.c8_comparison import compare

REPORT=Path('reports/auxnew_20261123_v1')
RUN='B2_C8_AUXNEW_V2_seed42'
DOMAIN='efc995be596d7d520bac472ceb1b5122fe2111fe9d5a53c5edbec02851c1accc'


def identity(root):
    root=Path(root);base,original,_=old.identity(root)
    lock=old.read(root/REPORT/'execution_manifest.json');old.verify_manifest(root,lock['files'])
    protocol=old.read(root/REPORT/'protocol.json')
    expected=dict(base,run_id=RUN,status='EXPERIMENTAL_DISABLED',
        auxiliary_task=dict(coefficient=.25,loss='BCEWithLogits',pos_weight=10.,domain='valid & observed_J==0',
            inference='primary_head_only',additional_parameters=9))
    if protocol['config']!=expected or len(original['files'])!=77:
        raise ValueError('Only preregistered C8 auxiliary supervision may differ')
    return expected,lock,protocol


def authorize(root,path,action,parent):
    _,_,protocol=identity(root);a=old.read(path)
    if action not in ('auxnew-gpu-preflight','auxnew-fit','auxnew-evaluate'):
        raise ValueError('Unknown AUXNEW action')
    expected=dict(authorized=True,action=action,experiment_id=RUN,device='cuda',runs=1,
        execution_manifest_sha256=sha256(Path(root)/REPORT/'execution_manifest.json'))
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError('New explicit AUXNEW grant required')
    if (Path(a.get('output_parent','__absent__')).resolve()!=Path(parent).resolve() or
        not all(a.get(k) for k in ['authorized_by','authorized_at','quota_checked_at','authorization_id',
            'local_backup_destination','external_backup_destination']) or
        a['local_backup_destination']==a['external_backup_destination']):raise ValueError('Operator/quota/output/backups required')
    for key,minimum in [('session_remaining_minutes',300),('quota_remaining_card_hours',10),('card_hours_reserved',10)]:
        value=a.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<minimum:
            raise ValueError('Insufficient finite authorization budget: '+key)
    return a


def preflight(root,maps,work,gpu=False):
    identity(root)
    total,available=old.host_memory()
    if available<8*old.GIB or total<16*old.GIB:raise RuntimeError('16 GiB total / 8 GiB available RAM required')
    free=old.disk_check(work,20*old.GIB)
    result=old.preflight(root,maps,gpu=gpu)
    if gpu:
        import torch
        if torch.backends.cudnn.version()!=92700:raise ValueError('Archived cuDNN version required')
        free_gpu,_=torch.cuda.mem_get_info()
        if free_gpu<2*old.GIB:raise RuntimeError('At least 2 GiB available GPU memory required')
        result['gpu_free_bytes']=free_gpu
    return dict(**result,free_bytes=free,ram_total=total,ram_available=available,
        training_executed=False,evaluation_deadline_seconds=8100,session_reserve_minutes=300,
        card_hours_reserved=10,GPU_memory_estimate_only=not gpu)


def final_state(root,out):
    import torch
    config,lock,_=identity(root);out=Path(out);receipt=old.read(out/'training_complete.json')
    p=out/receipt['checkpoint']
    if (p.parent.resolve()!=out.resolve() or receipt['status']!='COMPLETE' or receipt['epochs']!=4 or
        receipt['updates']!=28976 or sha256(p)!=receipt['checkpoint_sha256'] or
        old.read(p.with_suffix('.json'))['sha256']!=sha256(p)):raise ValueError('Wrong final AUXNEW checkpoint')
    state=torch.load(p,map_location='cpu',weights_only=False)
    if state['config']!=config or state['execution_manifest']!=lock or (state['epoch'],state['cursor'],state['updates'])!=(4,0,28976):
        raise ValueError('Checkpoint scientific identity differs')
    return state,p


def verify_initial(model):
    from geofirex.propagation_experiment import state_hash,EXPECTED_MAIN_INITIAL_SHA
    if sum(p.numel() for p in model.parameters())!=123986 or state_hash(model,True)!=EXPECTED_MAIN_INITIAL_SHA:
        raise ValueError('Fresh main initialization differs from C8; no optimizer permitted')


def evaluate(root,maps,out,authorization):
    import torch
    authorize(root,authorization,'auxnew-evaluate',Path(out).resolve().parent)
    root=Path(root);out=Path(out);state,p=final_state(root,out)
    save(out.parent/(RUN+'_EVALUATION_SINGLE_RUN.lock.json'),dict(checkpoint_sha256=sha256(p),consumed=True))
    dest=out/'evaluation';dest.mkdir(exist_ok=False)
    model=AuxiliaryUNet().to('cuda');model.load_state_dict(state['model'],strict=True);model.eval()
    base=old.identity(root)[0];mounts,_=old.verify_mounts(root,maps)
    train,val=make_v2_loaders('',root/base['index'],mounts=mounts,workers=2)
    writer=None
    try:
        keys=sample_keys(val.dataset);writer=PredictionWriter(dest/'predictions',keys,dict(run_id=RUN,checkpoint_sha256=sha256(p)))
        offset=0
        with torch.inference_mode():
            for x,y,v in val:
                probabilities=model(x.to('cuda')).sigmoid().cpu().numpy().astype(np.float32)
                for i in range(len(x)):
                    writer.add(keys[offset+i],probabilities[i,0],y[i,0].numpy(),v[i,0].numpy(),x[i,-1,7].numpy())
                offset+=len(x)
                if offset%1024==0:old.disk_check(out,2*old.GIB)
        writer.finish(DOMAIN)
        store=PredictionStore(dest/'predictions',expected_keys=keys,expected_domain=DOMAIN)
        accumulator=PropagationMetrics(dest/'_scratch',keys,old.read(root/old.REPORT/'propagation_policy.json'))
        for key,a in store.tiles():accumulator.update(key,*(a[k].copy() for k in ('p','y','v','o')))
        result=accumulator.result();save(dest/'evaluation.json',result)
        compare(old.read(root/REPORT/'baseline_reference.json'),result,dest/'paired_comparison',store.events)
        policy=old.read(root/REPORT/'acceptance.json')
        fields={'AP_new':(result['propagation_pr']['average_precision'],'AP_new_min',True),
            'AP_pooled':(result['pooled']['average_precision'],'AP_pooled_min',True),
            'Dice_pooled':(result['pooled']['dice'],'Dice_pooled_min',True),
            'recall_new':(result['new_fire']['recall'],'recall_new_min',True),
            'Dice_new':(result['new_fire']['dice'],'Dice_new_min',True),
            'FP_new':(result['new_fire']['fp'],'FP_new_max',False),
            'macro_Dice':(result['macro_event_dice'],'macro_Dice_min',True),
            'Brier':(result['pooled']['brier'],'Brier_max',False)}
        gates={name:value>=policy[key] if higher else value<=policy[key] for name,(value,key,higher) in fields.items()}
        save(dest/'acceptance.json',dict(scientific_gates=gates,all_scientific_gates_pass=all(gates.values()),
            promotion_authorized=False,exploratory_validation_already_observed=True,test_pixels_opened=0))
        save(dest/'complete.json',dict(status='COMPLETE',tiles=30882,events=80,checkpoint_sha256=sha256(p)))
    finally:
        if writer is not None:writer.close()
        train.dataset.close();val.dataset.close()


def fit(root,maps,out,fit_authorization,evaluation_authorization,gpu_authorization):
    """Only a separately authorized manual Kaggle run can enter this function."""
    import torch
    # All grants validated before CUDA, output, initialization or optimizer.
    root=Path(root);out=Path(out).resolve()
    for path,action in [(fit_authorization,'auxnew-fit'),(evaluation_authorization,'auxnew-evaluate'),(gpu_authorization,'auxnew-gpu-preflight')]:
        authorize(root,path,action,out.parent)
    if out.parent!=Path('/kaggle/working').resolve() or out.name!=RUN or out.exists():raise ValueError('Fresh fixed Kaggle output required')
    config,lock,protocol=identity(root)
    receipt=preflight(root,maps,out.parent,gpu=True)
    save(out.parent/(RUN+'_SINGLE_RUN.lock.json'),dict(experiment_id=RUN,consumed=True))
    out.mkdir()
    train=val=None;started=time.monotonic()
    try:
        for name in list(lock['files'])+[(REPORT/'execution_manifest.json').as_posix()]:
            dest=out/'inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as target,(root/name).open('rb') as source:shutil.copyfileobj(source,target)
        for path,name in [(fit_authorization,'fit_authorization.json'),(evaluation_authorization,'evaluation_authorization.json'),(gpu_authorization,'gpu_authorization.json')]:
            shutil.copyfile(path,out/name)
        save(out/'preflight.json',receipt)
        import random
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=AuxiliaryUNet().to('cuda');criterion=AuxiliaryLoss().to('cuda')
        verify_initial(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        mounts,_=old.verify_mounts(root,maps)
        train,val=make_v2_loaders('',root/config['index'],mounts=mounts,workers=2)
        state=dict(epoch=0,cursor=0,updates=0,history=[])
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(4):
            model.train();train.sampler.epoch=epoch;total=0.;pixels=0
            for x,y,v in train:
                if time.monotonic()-started>5400:raise TimeoutError('Fit budget exhausted; no automatic continuation')
                x=x.to('cuda');y=y.to('cuda');v=v.to('cuda');optimizer.zero_grad(set_to_none=True)
                main,aux=model.heads(x);loss,_,_=criterion(main,aux,y,v,x[:,-1,7:8])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                n=int(v.sum());total+=float(loss.detach())*n;pixels+=n
                state['updates']+=1;state['cursor']+=len(x)
                if state['updates']%1000==0:old.disk_check(out,2*old.GIB)
            if state['cursor']!=115900:raise ValueError('Incomplete epoch')
            state['epoch']=epoch+1;state['cursor']=0;state['history'].append(dict(epoch=epoch+1,loss=total/max(pixels,1)))
            checkpoint=old.checkpoint(out,model,optimizer,state,config,lock,train.dataset.normalization)
        if state['updates']!=28976:raise ValueError('Update budget differs')
        save(out/'training_complete.json',dict(status='COMPLETE',epochs=4,updates=28976,
            checkpoint=checkpoint.name,checkpoint_sha256=sha256(checkpoint)))
        final_state(root,out);old.package(out,'TRAINING-SAFE')
        with old.evaluation_deadline(8100):evaluate(root,maps,out,evaluation_authorization)
        elapsed=time.monotonic()-started
        save(out/'runtime_budget.json',dict(total_seconds=elapsed,session_within_budget=elapsed<=18000,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),promotion_authorized=False,
            actual_Kaggle_quota_receipt_required=True,test_pixels_opened=0))
    finally:
        if train is not None:train.dataset.close()
        if val is not None:val.dataset.close()
        old.package(out,'COMPLETE' if (out/'evaluation/complete.json').exists() else 'PARTIAL')
    return out
