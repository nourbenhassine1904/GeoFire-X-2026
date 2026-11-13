"""B1/B2 on immutable compact corpus, resumable mid-epoch, no publication.

Default one epoch; execution on CUDA is a separately authorized campaign.
The small --max-updates option is for a discarded-model verification only.
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from geofirex.fast_compact import FastCompactTiles,LocalitySampler
from geofirex.benchmark_models import make_model
from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.runner_evaluation import Counts
from geofirex.unet import SegmentationLoss
from geofirex.mission_streaming import save,sha,binary_metrics,peak_ram

def evaluate(model,ds,workers,device,out,deadline):
    counts=Counts([.5]);ap=ExternalAveragePrecision(out,retain_runs=False);seen=0;loss_sum=pixels=0;criterion=SegmentationLoss(.5,100).to(device);complete=True
    loader=DataLoader(ds,batch_size=32,num_workers=workers,pin_memory=device=='cuda',prefetch_factor=2 if workers else None,generator=torch.Generator().manual_seed(42))
    model.eval()
    with torch.inference_mode():
        for x,y,v in loader:
            if time.perf_counter()>deadline:complete=False;break
            x,y,v=[a.to(device,non_blocking=True) for a in (x,y,v)];logits=model(x);n=int(v.sum());loss_sum+=float(criterion(logits,y,v))*n;pixels+=n;mask=v.bool();p=logits.sigmoid()[mask].cpu().numpy();target=y[mask].cpu().numpy();counts.update(p,target);ap.update(p,target);seen+=len(x)
    if complete:average_precision=ap.compute()
    else:ap._release(list(ap.owned));average_precision=None
    result=dict(**counts.at(0),average_precision=average_precision,loss=loss_sum/pixels if pixels else None,samples_seen=seen,status='COMPLETE' if complete else 'PARTIAL_WALL_LIMIT')
    if not result['positive_pixels']:
        result.update(dice=None,iou=None,precision=None,recall=None,average_precision=None,informative=False)
    return result

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA required')
    train=FastCompactTiles(a.corpus,a.index,'train');val=FastCompactTiles(a.corpus,a.index,'val')
    events=sorted(set(val.meta['events'].values()) & {val.event(i) for i in range(len(val))})
    selected=events if a.validation_events==201 else [events[i] for i in np.linspace(0,len(events)-1,min(a.validation_events,len(events)),dtype=int)]
    val_ids=[i for i in range(len(val)) if val.event(i) in selected];validation=Subset(val,val_ids)
    config=dict(model=a.model,seed=42,context=3,target='J+1',threshold=.5,epochs=a.epochs,batch_size=a.batch_size,workers=a.workers,validation_events=selected,validation_tiles=len(validation),train_tiles=len(train),sampling='random block order; temporal/spatial locality within blocks, every tile once',index_sha256=sha(Path(a.index)/'index.json'),normalization_sha256=sha(Path(a.index)/'normalization.json'),feature_policy='existing 8-channel train-only normalized float32',test_pixels_opened=0,qualification='technical_profile' if a.max_updates else 'annual-fold0-compact')
    save(out/'config.json',config);save(out/'normalization.json',train.normalization)
    model=make_model(a.model,8,3,**({'architecture':'spatial_cnn_v2'} if a.model=='B1' else {})).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=SegmentationLoss(.5,100).to(a.device)
    epoch=1;cursor=updates=seen=0;history=[];loss_sum=loss_pixels=0.;sampler=LocalitySampler(train)
    if a.resume:
        state=torch.load(a.resume,map_location=a.device,weights_only=False)
        for key in ('model','seed','context','target','batch_size','validation_events','index_sha256','normalization_sha256'):
            if state['config'][key]!=config[key]:raise ValueError(f'Resume contract mismatch: {key}')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);epoch=state['epoch'];cursor=state['cursor'];updates=state['updates'];seen=state['seen'];history=state['history'];loss_sum=state['loss_sum'];loss_pixels=state['loss_pixels'];torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if a.device=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    started=time.perf_counter();deadline=started+a.seconds;last_checkpoint=None
    def checkpoint(reason):
        nonlocal last_checkpoint
        name=f'checkpoint_e{epoch:03d}_u{updates:07d}_{reason}.pt';path=out/name
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,cursor=cursor,updates=updates,seen=seen,history=history,config=config,normalization=train.normalization,loss_sum=loss_sum,loss_pixels=loss_pixels,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else None)
        with path.open('xb') as f:torch.save(state,f)
        last_checkpoint=str(path)
    status='FINISHED'
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=a.device=='cuda',persistent_workers=a.workers>0,prefetch_factor=2 if a.workers else None,generator=torch.Generator().manual_seed(43))
    with (out/'steps.jsonl').open('x') as log:
        while epoch<=a.epochs:
            sampler.epoch=epoch-1;sampler.start=cursor
            model.train()
            for x,y,v in loader:
                if time.perf_counter()>deadline:status='WALL_LIMIT';break
                x,y,v=[t.to(a.device,non_blocking=True) for t in (x,y,v)];optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y,v)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1;cursor+=len(x);seen+=len(x);n=int(v.sum());loss_sum+=float(loss.detach())*n;loss_pixels+=n
                log.write(json.dumps(dict(epoch=epoch,updates=updates,samples_seen=seen,epoch_cursor=cursor,loss=float(loss.detach()),seconds=time.perf_counter()-started))+'\n');log.flush()
                if updates%100==0:checkpoint('periodic')
                if a.max_updates and updates>=a.max_updates:status='SHORT_TEST_LIMIT';break
            if status!='FINISHED':checkpoint('paused');break
            # Save before validation: if wall limit hits during exhaustive AP,
            # resume with cursor==len(train), so no training sample is repeated.
            checkpoint('prevalidation')
            result=evaluate(model,validation,a.workers,a.device,out/f'ap_epoch_{epoch}',deadline)
            if result['status']!='COMPLETE':status='WALL_LIMIT_VALIDATION';checkpoint('validation_interrupted');break
            if a.b0:
                reference=json.loads(Path(a.b0).read_text());rows=[r for r in reference['per_event'] if r['event_id'] in selected];baseline=binary_metrics([sum(r[k] for r in rows) for k in ('tp','fp','fn','tn')])
                if baseline['valid_pixels']!=result['pixels'] or baseline['positive_pixels']!=result['positive_pixels']:raise ValueError('B0 comparison domain differs')
                save(out/f'b0_same_domain_epoch_{epoch}.json',dict(source_sha256=sha(a.b0),events=selected,metrics=baseline,recomputed_predictions=False))
            history.append(dict(epoch=epoch,train_loss=loss_sum/loss_pixels,validation=result,updates=updates,samples_seen=seen));save(out/f'epoch_{epoch:03d}.json',history[-1])
            # A real prediction map from a positive held-out tile; illustrative.
            positive=[i for i in val_ids if val.records[i]['category']==1]
            if positive:
                x,y,v=val[positive[0]]
                with torch.no_grad():p=model.predict_proba(x[None].to(a.device))[0,0].cpu().numpy()
                np.savez_compressed(out/f'map_epoch_{epoch}.npz',observed=x[-1,7].numpy(),target=y[0].numpy(),probability=p,valid=v[0].numpy(),val_record=val.records[positive[0]])
                import matplotlib;matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(1,4,figsize=(12,3))
                for ax,values,title in zip(axes,[x[-1,7].numpy(),y[0].numpy(),p,p>=.5],['Observed J','Ground truth J+1',a.model+' probability','Prediction >= 0.5']):
                    ax.imshow(np.ma.array(values,mask=~v[0].numpy().astype(bool)),vmin=0,vmax=1,cmap='inferno');ax.set_title(title);ax.axis('off')
                fig.tight_layout();fig.savefig(out/f'map_epoch_{epoch}.png',dpi=140);plt.close(fig)
            epoch+=1;cursor=0;loss_sum=loss_pixels=0.;checkpoint('epoch_complete')
            if time.perf_counter()>deadline and epoch<=a.epochs:status='WALL_LIMIT';break
    summary=dict(status=status,model=a.model,complete_epochs=len(history),optimizer_updates=updates,samples_seen=seen,next_epoch=epoch,cursor=cursor,elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=peak_ram(),checkpoint=last_checkpoint,history=history)
    save(out/'summary.json',summary)
    if history:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();ax.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],label='train');ax.plot([h['epoch'] for h in history],[h['validation']['loss'] for h in history],label='validation');ax.legend();ax.set(xlabel='epoch',ylabel='BCE + Dice');fig.savefig(out/'loss.png',dpi=140);plt.close(fig)
    print(json.dumps(summary));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--index',required=True);p.add_argument('--out',required=True);p.add_argument('--model',choices=['B1','B2'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=2);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--epochs',type=int,default=1);p.add_argument('--validation-events',type=int,default=32);p.add_argument('--seconds',type=int,default=3600);p.add_argument('--resume');p.add_argument('--max-updates',type=int,default=0);p.add_argument('--b0',default='reports/mission_20261111/b0_v2/results.json');run(p.parse_args())
