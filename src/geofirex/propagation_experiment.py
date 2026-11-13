"""ONE preregistered experiment: U-Net with a new-detection specialist head.

Metadata-only by default. Training requires --mode train --execute-training.
No historical checkpoint is initialized, no 169/2021 inference is exposed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from geofirex.benchmark_models import ForecastModel
from geofirex.calibration_io import save, sha
from geofirex.compact_calibration import evaluate, event_ids, render_maps
from geofirex.fast_compact import FastCompactTiles, LocalitySampler
from geofirex.unet import LightUNet, SegmentationLoss

EXPERIMENT = 'UNET_NEW_FIRE_HEAD_V1'
GPU_WALL_SECONDS = 7200
EXPECTED_MAIN_INITIAL_SHA = 'a835383534d061787c2449bbb4a896c32c3e990d92b80fe2b3105f9c91ac6f5f'
THRESHOLDS = np.unique(np.round(np.r_[np.arange(.1,.951,.005),.96,.97,.98,.99,.995,.999],3))


class PropagationUNet(ForecastModel):
    """Unchanged U-Net decoder features, main head + full-resolution new head.

    At deployment the observed J mask chooses a head, never the J+1 label.
    The main head is supervised on the entire valid domain as before.
    """
    def __init__(self):
        super().__init__(8,3)
        self.network = LightUNet(24,8,3)
        self.new_head = nn.Conv2d(8,1,1)
        nn.init.constant_(self.new_head.bias,-3.)

    def heads(self,x):
        self.validate(x)
        if min(x.shape[-2:])<8:raise ValueError('Minimum spatial size is 8')
        z=x.flatten(1,2);skips=[]
        for encoder in self.network.encoders:
            z=encoder(z);skips.append(z);z=F.max_pool2d(z,2)
        z=self.network.bottleneck(z)
        for decoder,skip in zip(self.network.decoders,reversed(skips)):
            z=F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            z=decoder(torch.cat((skip,z),dim=1))
        return self.network.head(z),self.new_head(z)

    def forward(self,x):
        main,new=self.heads(x)
        return torch.where(x[:,-1,7:8]==0,new,main)


class PropagationLoss(nn.Module):
    """L = historical main loss(w10) + 0.5 * new-domain loss(w10).

    Aux Dice pools the batch to avoid averaging almost-constant losses on
    empty foreground tiles. Aux BCE includes ALL valid nonburning negatives.
    No adjacency/dilation rule removes isolated new detections.
    """
    def __init__(self):
        super().__init__()
        self.main=SegmentationLoss(.5,10)

    def forward(self,main,new,target,valid,observed):
        if not (main.shape==new.shape==target.shape==valid.shape==observed.shape):
            raise ValueError('Head/label/mask shape mismatch')
        mask=valid*(observed==0)
        bce=F.binary_cross_entropy_with_logits(new,target,pos_weight=self.main.pos_weight,reduction='none')
        bce=(bce*mask).sum()/mask.sum().clamp_min(1)
        p=new.sigmoid()*mask;t=target*mask
        # Empty-new-target batches retain their BCE negatives; avoid an almost
        # constant Dice penalty with negligible useful gradient on such batches.
        dice=1-(2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6) if t.sum()>0 else new.sum()*0
        auxiliary=.5*bce+.5*dice if mask.any() else new.sum()*0
        primary=self.main(main,target,valid)
        return primary+.5*auxiliary,primary,auxiliary


def state_hash(model,core_only=False):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if core_only and name.startswith('new_head.'):continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def resume_budget(checkpoint,state):
    """An uncleanly killed GPU session spends its whole reserved remainder."""
    path=Path(checkpoint).parent/'gpu_budget.json'
    if not path.is_file():raise ValueError('Resume budget accounting missing; no automatic GPU extension')
    account=json.loads(path.read_text(encoding='utf-8'))
    if account['status']!='PAUSED' or account['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('GPU budget is reserved/spent or checkpoint differs; new explicit budget review required')
    used=max(state['gpu_wall_seconds'],account['spent_seconds'])
    if used>=GPU_WALL_SECONDS-30:raise ValueError('Cumulative GPU budget exhausted')
    return used


def preflight(args):
    protocol=json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    train=FastCompactTiles(args.corpus,args.index,'train');val=FastCompactTiles(args.corpus,args.index,'val')
    try:
        selected=protocol['calibration_events'];allval=set(event_ids(val));alltrain=set(event_ids(train))
        if len(train)!=93603 or len(alltrain)!=250 or any(e[:4] not in ('2018','2019') for e in alltrain):
            raise ValueError('Expected complete 250-event training corpus')
        if len(val)!=78771 or len(allval)!=201 or any(not e.startswith('2020/') for e in allval):
            raise ValueError('Expected 201-event validation metadata')
        if len(selected)!=32 or len(set(selected))!=32 or set(selected)&set(protocol['holdout_events']):
            raise ValueError('32-event scope/overlap differs')
        if set(selected)|set(protocol['holdout_events'])!=allval or len(protocol['holdout_events'])!=169:
            raise ValueError('Validation partition differs')
        if int(np.isin(event_ids(val),selected).sum())!=13759:raise ValueError('32-event tile scope differs')
        for name,key in [('index.json','index_sha256'),('normalization.json','normalization_sha256')]:
            if sha(Path(args.index)/name)!=protocol[key]:raise ValueError('Pinned metadata differs: '+name)
        norm=train.normalization
        if norm['continuous']!=[12,14] or norm['include_landcover'] or not norm['fit_paths']:
            raise ValueError('Expected historical eight-channel train-only normalizer')
        for p in norm['fit_paths']:
            parts=Path(p.replace('\\','/')).parts
            if not set(parts)&{'2018','2019'} or set(parts)&{'2020','2021'}:raise ValueError('Normalizer leakage')
        source_names=('propagation_experiment.py','unet.py','fast_compact.py','compact_calibration.py','benchmark_metrics.py')
        plan=dict(experiment=EXPERIMENT,runs=1,seed=42,epochs=4,updates=23404,train_tiles=93603,train_events=250,
            batch_size=16,model_context=3,target='J+1',optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
            gradient_clip=1.,main_pos_weight=10,auxiliary_pos_weight=10,auxiliary_coefficient=.5,
            auxiliary_dice='batch pooled on valid & observed_J==0; zero on empty-new-target batches, BCE negatives retained',main_dice='historical per-image',
            initialization='fresh seed42; no historical trained weights; require historical initial main U-Net digest before training',
            expected_initial_main_state_sha256=EXPECTED_MAIN_INITIAL_SHA,
            calibration_events=selected,calibration_tiles=13759,checkpoint_selection='epoch4 only; no early stopping',
            threshold_candidates=THRESHOLDS.tolist(),threshold_rule='max global micro Dice; ties precision then higher threshold, 32 only',
            acceptance_rule='on same 32: global Dice >= saved w10; new-fire Dice >= saved w10; new-fire recall > saved w10',
            inference='if fire_J==0 use specialist logit; otherwise use main logit; one global threshold',
            normalizer='reuse already-normalized compact values; no fitting or second normalization',
            index_sha256=protocol['index_sha256'],normalization_sha256=protocol['normalization_sha256'],
            protocol_sha256=sha(args.protocol),baseline32_sha256=sha(args.baseline32),
            source_sha256={n:sha(Path(__file__).parent/n) for n in source_names},
            gpu_wall_limit_seconds=GPU_WALL_SECONDS,gpu_count=1,device_target='single T4; no DDP; float32',
            resource_log='wall/data/step seconds, CPU RSS, GPU allocated/reserved peaks and nvidia-smi snapshots',
            holdout169_policy='already observed panel; inaccessible to this runner; no hyperparameter choices',
            test_2021_pixels_opened=0,training_executed=False)
        baseline=json.loads(Path(args.baseline32).read_text())
        if baseline['events']!=selected or baseline['tiles']!=13759:raise ValueError('Baseline scope differs')
        if not any(r['threshold']==.78 for r in baseline['curves']):raise ValueError('Saved baseline .78 missing')
        return plan,train,val
    except BaseException:
        train.close();val.close();raise


def train_run(args,plan,train):
    started=time.monotonic()
    if not args.execute_training:raise ValueError('Explicit --execute-training authorization flag required')
    resume_state=None
    if args.resume:
        resume_state=torch.load(args.resume,map_location='cpu',weights_only=False)
        resume_budget(args.resume,resume_state)  # Reject spent sessions before initializing CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    if not torch.cuda.is_available():raise ValueError('CUDA required; no silent CPU training')
    if torch.cuda.device_count()!=1:raise ValueError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0')
    import psutil
    out=Path(args.out).resolve()
    for protected in (Path(args.corpus).resolve(),Path(args.index).resolve()):
        if out==protected or out.is_relative_to(protected):raise ValueError('Output overlaps immutable input')
    out.mkdir(parents=True,exist_ok=False)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    network=PropagationUNet()
    if state_hash(network,True)!=EXPECTED_MAIN_INITIAL_SHA:
        raise ValueError('Initial main U-Net differs from saved ablation; use pinned Kaggle torch2.11.0/numpy2.1.3. No optimizer update performed.')
    network=network.cuda();optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.0001)
    criterion=PropagationLoss().cuda()
    config=dict(plan,training_executed=True,initial_state_sha256=state_hash(network),
        initial_main_state_sha256=state_hash(network,True),torch_version=torch.__version__,numpy_version=np.__version__,
        gpu_name=torch.cuda.get_device_name(0),cuda_version=torch.version.cuda,workers=args.workers)
    epoch=1;cursor=updates=0;history=[];used_seconds=0.;sums=[0.,0.,0.];samples=0
    generator=torch.Generator().manual_seed(43)
    if args.resume:
        # Only checkpoints produced by THIS new experiment are acceptable.
        state=resume_state
        if state['config']!=config:raise ValueError('Resume model/code/data/protocol differs')
        network.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if isinstance(v,torch.Tensor) and k!='step':values[k]=v.cuda()
        epoch,cursor,updates=state['epoch'],state['cursor'],state['updates']
        used_seconds=resume_budget(args.resume,state);history=state['history'];sums=state['loss_sums'];samples=state['loss_samples']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);generator.set_state(state['loader_rng'])
    save(out/'config.json',config)
    # Reserve all remaining allocation. If killed before graceful accounting,
    # continuation is rejected rather than resetting the wall-time counter.
    atomic_json(out/'gpu_budget.json',dict(status='RESERVED',limit_seconds=GPU_WALL_SECONDS,
        previously_spent_seconds=used_seconds,reserved_seconds=GPU_WALL_SECONDS-used_seconds))
    sampler=LocalitySampler(train,seed=42)
    loader=DataLoader(train,batch_size=16,sampler=sampler,num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0,prefetch_factor=2 if args.workers else None,generator=generator)
    process=psutil.Process();torch.cuda.reset_peak_memory_stats()
    def elapsed():return used_seconds+time.monotonic()-started
    def checkpoint():
        torch.cuda.synchronize();tmp=out/'checkpoint_latest.tmp';path=out/'checkpoint_latest.pt'
        torch.save(dict(config=config,model=network.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,cursor=cursor,
            updates=updates,history=history,loss_sums=sums,loss_samples=samples,gpu_wall_seconds=elapsed(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state()),tmp)
        os.replace(tmp,path)
        return path
    def log(value):
        with (out/'resources.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
    checkpoint()
    # CUDA timing uses synchronize, no asynchronous wall-time underestimation.
    while epoch<=4:
        sampler.epoch=epoch-1;sampler.start=cursor;network.train();wait_started=time.monotonic()
        for x,y,v in loader:
            data_seconds=time.monotonic()-wait_started
            if elapsed()>=GPU_WALL_SECONDS-30:
                path=checkpoint();save(out/'paused.json',dict(status='BUDGET_PAUSED',updates=updates,
                    checkpoint=str(path),gpu_wall_seconds=elapsed(),remaining_seconds=max(0,GPU_WALL_SECONDS-elapsed())))
                atomic_json(out/'gpu_budget.json',dict(status='PAUSED',spent_seconds=elapsed(),
                    limit_seconds=GPU_WALL_SECONDS,checkpoint_sha256=sha(path)))
                return
            step_started=time.monotonic()
            x,y,v=[t.cuda(non_blocking=True) for t in (x,y,v)]
            optimizer.zero_grad(set_to_none=True);main,new=network.heads(x)
            loss,primary,auxiliary=criterion(main,new,y,v,x[:,-1,7:8])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step();torch.cuda.synchronize()
            cursor+=len(x);updates+=1;samples+=len(x)
            for i,item in enumerate((loss,primary,auxiliary)):sums[i]+=float(item.detach())*len(x)
            log(dict(epoch=epoch,updates=updates,tiles=cursor,wall_seconds=elapsed(),data_wait_seconds=data_seconds,
                step_seconds=time.monotonic()-step_started,loss=float(loss.detach()),main_loss=float(primary.detach()),
                auxiliary_loss=float(auxiliary.detach()),cpu_rss_bytes=process.memory_info().rss,
                worker_rss_bytes=sum(p.memory_info().rss for p in process.children() if p.is_running()),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            if updates%100==0:checkpoint();print(f'{EXPERIMENT} epoch={epoch} updates={updates} wall={elapsed():.1f}s',flush=True)
            if updates%500==0:
                import subprocess
                try:
                    snapshot=subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw','--format=csv'],capture_output=True,text=True,timeout=5)
                    log(dict(updates=updates,nvidia_smi=snapshot.stdout.strip(),returncode=snapshot.returncode))
                except (OSError,subprocess.TimeoutExpired) as exc:log(dict(updates=updates,nvidia_smi_error=str(exc)))
            wait_started=time.monotonic()
        history.append(dict(epoch=epoch,updates=updates,loss=sums[0]/samples,main_loss=sums[1]/samples,auxiliary_loss=sums[2]/samples))
        epoch+=1;cursor=0;sums=[0.,0.,0.];samples=0;checkpoint()
        # Preserve each new epoch checkpoint; never touch historical checkpoints.
        import shutil
        shutil.copyfile(out/'checkpoint_latest.pt',out/f'checkpoint_epoch_{epoch-1:02d}.pt')
    if updates!=23404:raise ValueError('Training update budget differs')
    path=checkpoint();save(out/'training_complete.json',dict(status='COMPLETE',updates=updates,epochs=4,
        checkpoint_sha256=sha(path),gpu_wall_seconds=elapsed(),selection32_executed=False,test_2021_pixels_opened=0))
    atomic_json(out/'gpu_budget.json',dict(status='COMPLETE',spent_seconds=elapsed(),
        limit_seconds=GPU_WALL_SECONDS,checkpoint_sha256=sha(path)))


def select32(args,plan,val):
    if not args.checkpoint:raise ValueError('--checkpoint is required')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=state['config']
    if any(config.get(k)!=v for k,v in plan.items() if k!='training_executed'):
        raise ValueError('New experiment checkpoint protocol/code differs')
    if (state['epoch'],state['cursor'],state['updates'])!=(5,0,23404):raise ValueError('Only completed fixed epoch4 can be selected')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    # Freeze 32 IDs and scope before opening a single validation tile.
    checkpoint_sha=sha(args.checkpoint)
    save(out/'selection_scope_locked.json',dict(experiment=EXPERIMENT,checkpoint_sha256=checkpoint_sha,
        events=plan['calibration_events'],thresholds=plan['threshold_candidates'],holdout169_pixels_opened=0,test_2021_pixels_opened=0))
    torch.set_num_threads(2);network=PropagationUNet();network.load_state_dict(state['model'],strict=True);network.eval()
    args.device='cpu';args.batch_size=32
    events=plan['calibration_events'];map_events={events[i] for i in (0,len(events)//2,len(events)-1)}
    started=time.monotonic()
    result=evaluate(network,val,events,THRESHOLDS,args,out/'calibration32','new-fire-head calibration32',map_events)
    selected=max(result['curves'],key=lambda r:(r['dice'],r['precision'] if r['precision'] is not None else -1,r['threshold']))
    new=next(r for r in result['new_fire_candidates'] if r['threshold']==selected['threshold'])
    baseline=json.loads(Path(args.baseline32).read_text())
    oldfull=next(r for r in baseline['curves'] if r['threshold']==.78)
    oldnew=next(r for r in baseline['new_fire_candidates'] if r['threshold']==.78)
    accepted=selected['dice']>=oldfull['dice'] and new['dice']>=oldnew['dice'] and new['recall']>oldnew['recall']
    render_maps(result,out/'calibration32',EXPERIMENT,selected['threshold'],checkpoint_sha)
    save(out/'selection32.json',dict(status='COMPLETE',experiment=EXPERIMENT,checkpoint_sha256=checkpoint_sha,
        threshold=selected['threshold'],full=selected,new_fire=new,baseline32=dict(full=oldfull,new_fire=oldnew),
        accepted_on_development32=accepted,rule=plan['acceptance_rule'],boundary_warning=selected['threshold'] in (float(THRESHOLDS[0]),float(THRESHOLDS[-1])),
        cpu_selection_seconds=time.monotonic()-started,holdout169_pixels_opened=0,test_2021_pixels_opened=0,
        limitation='32 repeatedly used for development; 169 previously seen, neither is a fresh blind test. No automatic promotion.'))


def run(args):
    plan,train,val=preflight(args)
    try:
        if args.mode=='preflight':print(json.dumps(plan,indent=2));return plan
        if args.mode=='train':train_run(args,plan,train)
        elif args.mode=='select32':select32(args,plan,val)
    finally:train.close();val.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','index','protocol','baseline32','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=['preflight','train','select32'],default='preflight')
    p.add_argument('--execute-training',action='store_true');p.add_argument('--resume');p.add_argument('--checkpoint')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not 0<=args.workers<=4:p.error('workers must be 0..4')
    if args.execute_training and args.mode!='train':p.error('Training flag only valid with --mode train')
    run(args)


if __name__=='__main__':main()
