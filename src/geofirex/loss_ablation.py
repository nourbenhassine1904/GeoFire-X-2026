"""Independent, paired pos_weight runs. Default is metadata-only preflight.

No existing checkpoint is loaded for initialization. Only the 32 event IDs from
the completed protocol can reach inference. The 169 and 2021 stay inaccessible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from geofirex.benchmark_models import make_model
from geofirex.calibration_io import save,sha
from geofirex.compact_calibration import evaluate, event_ids, write_csv
from geofirex.fast_compact import FastCompactTiles, LocalitySampler
from geofirex.unet import SegmentationLoss

WEIGHTS=(100,30,10)
THRESHOLDS=np.unique(np.round(np.r_[np.arange(.1,.951,.005),.96,.97,.98,.99,.995,.999],3))


def model_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def preflight(corpus,index,protocol):
    protocol=json.loads(Path(protocol).read_text(encoding='utf-8'))
    train=FastCompactTiles(corpus,index,'train');val=FastCompactTiles(corpus,index,'val')
    events=protocol['calibration_events']
    ids=event_ids(val);train_events=set(event_ids(train))
    if len(train)!=93603 or len(train_events)!=250 or any(e[:4] not in ('2018','2019') for e in train_events):
        raise ValueError('Training corpus differs')
    if len(events)!=32 or len(set(events))!=32 or any(not e.startswith('2020/') for e in events):
        raise ValueError('Expected exact 32 calibration IDs')
    if set(events)&set(protocol['holdout_events']):raise ValueError('Split overlap')
    if int(np.isin(ids,events).sum())!=13759:raise ValueError('Expected 13759 calibration tiles')
    for name,key in [('index.json','index_sha256'),('normalization.json','normalization_sha256')]:
        if sha(Path(index)/name)!=protocol[key]:raise ValueError('Protocol digest mismatch: '+name)
    plan=dict(schema='geofirex-independent-loss-ablation-v1',weights=list(WEIGHTS),seed=42,
        epochs=4,batch_size=16,updates_per_run=23404,train_tiles=93603,
        calibration_events=events,calibration_tiles=13759,model_context=3,target='J+1',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),gradient_clip=1.,dice_weight=.5,
        sampler='same LocalitySampler seed42, epoch order and every tile exactly once',
        index_sha256=protocol['index_sha256'],normalization_sha256=protocol['normalization_sha256'],
        initialization='fresh make_model seed42 per independent run; never load historical trained weights',
        threshold_candidates=THRESHOLDS.tolist(),checkpoint_selection='fixed epoch 4; equal training budget',
        selection_scope='32 only; 169 confirmation panel not opened; no 2021 data',
        total_planned_runs=6,total_planned_updates=140424,training_executed=False)
    return plan,train,val


def select_threshold(curves):
    if not curves or not curves[0]['positive_pixels']:raise ValueError('No calibration positives')
    return max(curves,key=lambda r:(r['dice'],r['precision'] if r['precision'] is not None else -1.,r['threshold']))


def run(args):
    plan,train,val=preflight(args.corpus,args.index,args.protocol)
    if not args.execute_training:
        print(json.dumps(plan,indent=2));train.close();val.close();return plan
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    out=Path(args.out).resolve()
    if out==Path(args.corpus).resolve() or out.is_relative_to(Path(args.corpus).resolve()):raise ValueError('Output must be outside immutable corpus')
    out.mkdir(parents=True,exist_ok=False)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    network=make_model(args.model,8,3,**({'architecture':'spatial_cnn_v2'} if args.model=='B1' else {})).to(args.device)
    initial_sha=model_hash(network)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.0001)
    criterion=SegmentationLoss(.5,args.pos_weight).to(args.device)
    config=dict(**plan,model=args.model,pos_weight=args.pos_weight,initial_state_sha256=initial_sha,
                protocol_sha256=sha(args.protocol),qualification='controlled-independent-training',
                device=args.device,torch_version=torch.__version__,numpy_version=np.__version__)
    config['training_executed']=True
    save(out/'config.json',config)
    epoch=1;cursor=updates=0;history=[];loss_sum=0.;loss_pixels=0
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Resume config differs: no cross-weight/model/protocol resume')
        network.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        # Optimizer state tensors follow model parameters on CUDA after load.
        for values in optimizer.state.values():
            for k,v in values.items():
                if isinstance(v,torch.Tensor) and k!='step':values[k]=v.to(args.device)
        epoch,cursor,updates=state['epoch'],state['cursor'],state['updates']
        history,loss_sum,loss_pixels=state['history'],state['loss_sum'],state['loss_pixels']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if args.device=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    sampler=LocalitySampler(train,seed=42)
    loader=DataLoader(train,batch_size=16,sampler=sampler,num_workers=args.workers,
        pin_memory=args.device=='cuda',persistent_workers=args.workers>0,
        prefetch_factor=2 if args.workers else None,generator=torch.Generator().manual_seed(43))
    deadline=time.monotonic()+args.seconds
    def checkpoint():
        path=out/'checkpoint_latest.pt';temporary=out/'checkpoint_latest.tmp'
        torch.save(dict(config=config,model=network.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,cursor=cursor,updates=updates,history=history,loss_sum=loss_sum,loss_pixels=loss_pixels,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),temporary)
        temporary.replace(path)
        return path
    while epoch<=4:
        sampler.epoch=epoch-1;sampler.start=cursor;network.train()
        for x,y,v in loader:
            if time.monotonic()>=deadline:
                path=checkpoint();save(out/'paused.json',dict(status='WALL_LIMIT',checkpoint=str(path),epoch=epoch,cursor=cursor,updates=updates))
                print('Paused; resume this independent variant into a NEW output directory:',path,flush=True)
                train.close();val.close();return
            x,y,v=[t.to(args.device,non_blocking=True) for t in (x,y,v)]
            optimizer.zero_grad(set_to_none=True);loss=criterion(network(x),y,v)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step()
            cursor+=len(x);updates+=1;pixels=int(v.sum());loss_sum+=float(loss.detach())*pixels;loss_pixels+=pixels
            if updates%100==0:checkpoint();print(f'{args.model} w={args.pos_weight} epoch={epoch} updates={updates}',flush=True)
        history.append(dict(epoch=epoch,updates=updates,train_loss=loss_sum/loss_pixels))
        epoch+=1;cursor=0;loss_sum=0.;loss_pixels=0;checkpoint()
    if updates!=23404:raise ValueError('Training budget differs')
    checkpoint()
    network.eval()
    # No 169 or 2021 event ID is ever passed to evaluation; map output disabled.
    result=evaluate(network,val,plan['calibration_events'],THRESHOLDS,args,out/'calibration32',
                    f'{args.model} w={args.pos_weight} calibration32',set())
    selected=select_threshold(result['curves'])
    save(out/'selection32.json',dict(status='COMPLETE',model=args.model,pos_weight=args.pos_weight,
        initial_state_sha256=initial_sha,updates=updates,threshold=selected['threshold'],metrics=selected,
        boundary_warning=selected['threshold']==float(THRESHOLDS[-1]),selection_scope='32 only',
        checkpoint_sha256=sha(out/'checkpoint_latest.pt'),index_sha256=plan['index_sha256'],
        normalization_sha256=plan['normalization_sha256'],protocol_sha256=sha(args.protocol),
        holdout_pixels_opened=0,test_2021_pixels_opened=0))
    train.close();val.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True);p.add_argument('--index',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--model',choices=['B1','B2'],required=True);p.add_argument('--pos-weight',type=int,choices=WEIGHTS,required=True)
    p.add_argument('--out',required=True);p.add_argument('--execute-training',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,choices=[32],default=32,help='Validation batch; training is fixed at 16')
    p.add_argument('--seconds',type=int,default=5400);p.add_argument('--resume')
    args=p.parse_args()
    if args.seconds<1 or args.workers<0:p.error('Invalid budget/workers')
    run(args)


if __name__=='__main__':main()
