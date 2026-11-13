"""Real B1 optimization on compact shards or already attached GeoTIFF lots.

Full epochs visit every tile once. Profiling is a separate model/run. No upload,
submission, dependency installation or test-2021 reading is performed here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset,DataLoader,Subset

from geofirex.benchmark_models import make_model
from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.mission_streaming import CompactTiles,load,save,sha,peak_ram
from geofirex.runner_data import TemporalStore
from geofirex.runner_evaluation import Counts
from geofirex.unet import SegmentationLoss
from geofirex.unet_data import tile_coordinates


class SceneTiles(Dataset):
    def __init__(self,store):
        self.store=store
        self.samples=[dict(event_id=w.event_id,target_date=w.target_date,row=r,col=c,index=i) for i,w in enumerate(store.windows) for r,c in tile_coordinates(*store.shapes[i],64)]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]; x,y,v,_=self.store.tile(s['index'],s['row'],s['col'],64)
        return tuple(torch.from_numpy(a.copy()) for a in (x,y,v))


def attached_datasets(prepared,roots):
    """Resolve complete events only; published training lots need real val mount.

    No missing event is silently represented by zero tensors. All roles and
    dates come from pinned metadata. SHA is checked for every resolved TIFF.
    """
    import pyarrow.parquet as pq
    rows,groups,domains,n=load(prepared,roots[0])
    resolved={}; available=[]
    for event in sorted({r['event_id'] for r in rows}):
        members=[r for r in rows if r['event_id']==event]; paths={}
        for r in members:
            found={p.resolve() for root in roots if (p:=Path(root)/r['relative_path']).is_file()}
            if found:
                for p in found:
                    if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']: raise ValueError(f'Uncertified mounted raster {p}')
                p=sorted(found)[0]
                paths[r['date']]=str(p)
        if len(paths)==len(members):
            resolved[event]=paths;available.append(event)
    stores={}
    from geofirex.windows import Window
    for role in ('train','val'):
        ws=[]; ds=[]
        for w in groups[role]:
            if w.event_id not in resolved: continue
            lookup=resolved[w.event_id]
            ws.append(Window(event_id=w.event_id,context_dates=w.context_dates,target_date=w.target_date,context_paths=tuple(lookup[d] for d in w.context_dates),target_path=lookup[w.target_date]))
            dd=[Path(p).stem for p in domains[(w.event_id,w.target_date)]]
            ds.append(Window(event_id=w.event_id,context_dates=tuple(dd),target_date=w.target_date,context_paths=tuple(lookup[d] for d in dd),target_path=lookup[w.target_date]))
        if not ws: raise ValueError(f'No complete {role} event mounted; attach an existing 2020 sample or approved compact validation. No train-as-validation fallback.')
        stores[role]=SceneTiles(TemporalStore(ws,n,domain_windows=ds,cache_bytes=64*1024**2))
    return stores['train'],stores['val'],dict(available_events=available,normalization_sha256=sha(Path(prepared)/'normalization.json'),source='attached certified original TIFFs')


def evaluate(model,loader,criterion,device,out,deadline=None):
    ap=ExternalAveragePrecision(out,retain_runs=False)
    counts=Counts([.5]);loss_sum=pixels=seen=0
    model.eval();complete=True
    with torch.inference_mode():
        for x,y,v in loader:
            if deadline is not None and time.perf_counter()>deadline:complete=False;break
            x,y,v=[a.to(device) for a in (x,y,v)]; logits=model(x);loss=criterion(logits,y,v)
            n=int(v.sum());loss_sum+=float(loss)*n;pixels+=n;seen+=len(x)
            mask=v.bool(); p=logits.sigmoid()[mask].cpu().numpy();t=y[mask].cpu().numpy()
            if len(t): counts.update(p,t);ap.update(p,t)
    if complete:average_precision=ap.compute()
    else:
        ap._release(list(ap.owned));average_precision=None
    return dict(**counts.at(0),average_precision=average_precision,loss=loss_sum/pixels if pixels else None,samples_seen=seen,status='COMPLETE' if complete else 'WALL_LIMIT')


def train(train_ds,val_ds,out,*,device='cpu',epochs=2,profile_steps=0,batch_size=16,provenance=None,wall_seconds=600):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA required; preserve installed Kaggle torch')
    events=lambda ds: sorted({s['event_id'] for s in ds.samples})
    assert all(e[:4] in ('2018','2019') for e in events(train_ds))
    assert all(e.startswith('2020/') for e in events(val_ds))
    assert not set(events(train_ds)) & set(events(val_ds))
    model=make_model('B1',channels=8,context=3,architecture='spatial_cnn_v2').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    # Existing loss, fixed weighting; no label statistics estimated on validation.
    criterion=SegmentationLoss(pos_weight=100.,dice_weight=.5).to(device)
    gen=torch.Generator().manual_seed(42)
    loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,generator=gen,num_workers=0,pin_memory=device=='cuda')
    evaluation_ds=Subset(val_ds,range(min(64,len(val_ds)))) if profile_steps else val_ds
    val_loader=DataLoader(evaluation_ds,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=device=='cuda')
    normalization=train_ds.normalization if hasattr(train_ds,'normalization') else train_ds.store.normalizer.to_dict()
    save(out/'normalization.json',normalization)
    config=dict(model='B1',architecture='spatial_cnn_v2',context=3,channels=8,device=device,epochs=epochs,profile_steps=profile_steps,train_tiles=len(train_ds),val_tiles=len(val_ds),train_events=events(train_ds),val_events=events(val_ds),sampling='each grid tile once per complete epoch, shuffle train only',threshold=.5,pos_weight=100.,dice_weight=.5,seed=42,wall_seconds=wall_seconds,provenance=provenance,normalization='existing train-only; already applied once in compact shards',normalization_sha256=sha(out/'normalization.json'),test_pixels_opened=0)
    config['validation_scope']='first 64 metadata-ordered tiles, profiling only; no benchmark claim' if profile_steps else 'all available validation events/windows/tiles'
    config['validation_tiles_scheduled']=len(evaluation_ds)
    save(out/'config.json',config);history=[];updates=seen=0;start=time.perf_counter();best=-1.;status='FINISHED'
    with (out/'steps.jsonl').open('x') as log:
        for epoch in range(1,epochs+1):
            epoch_start=time.perf_counter()
            model.train();loss_sum=valid_pixels=epoch_seen=0;completed=True
            for x,y,v in loader:
                if time.perf_counter()-start>wall_seconds:
                    completed=False;status='WALL_LIMIT';break
                x,y,v=[a.to(device) for a in (x,y,v)];opt.zero_grad(set_to_none=True)
                loss=criterion(model(x),y,v)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                n=int(v.sum());loss_sum+=float(loss.detach())*n;valid_pixels+=n;seen+=len(x);epoch_seen+=len(x);updates+=1
                row=dict(epoch=epoch,step=updates,loss=float(loss.detach()),samples_seen=seen,elapsed_seconds=time.perf_counter()-start)
                log.write(json.dumps(row)+'\n');log.flush()
                if updates%50==0: print(row,flush=True)
                if profile_steps and updates>=profile_steps: completed=False;break
            if not valid_pixels: break
            train_seconds=time.perf_counter()-epoch_start;validation_start=time.perf_counter()
            val=evaluate(model,val_loader,criterion,device,out/f'ap_epoch_{epoch}',deadline=start+wall_seconds)
            if val['status']=='WALL_LIMIT':status='WALL_LIMIT'
            row=dict(epoch=epoch,complete_epoch=completed and epoch_seen==len(train_ds),train_samples_seen=epoch_seen,total_train_samples_seen=seen,optimizer_updates=updates,train_loss=loss_sum/valid_pixels,validation=val,train_seconds=train_seconds,validation_seconds=time.perf_counter()-validation_start,elapsed_seconds=time.perf_counter()-start)
            history.append(row);save(out/f'epoch_{epoch:02d}.json',row)
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,complete_epoch=row['complete_epoch'],optimizer_updates=updates,samples_seen=seen,config=config,normalization=normalization,history=history,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=gen.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None)
            with (out/f'checkpoint_epoch_{epoch:02d}.pt').open('xb') as f: torch.save(state,f)
            # Immutable checkpoint name per epoch; best.json points to best AP.
            if val['average_precision'] is not None and val['average_precision']>best: best=val['average_precision'];best_epoch=epoch
            print(row,flush=True)
            if profile_steps or not completed or status=='WALL_LIMIT': break
    if device=='cuda': torch.cuda.synchronize()
    summary=dict(status=status,device=device,epochs_completed=sum(r['complete_epoch'] for r in history),train_samples_seen=seen,optimizer_updates=updates,history=history,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_ram(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,best_validation_ap=best if best>=0 else None,best_epoch=best_epoch if best>=0 else None)
    save(out/'summary.json',summary)
    if history:
        checkpoint=torch.load(out/f'checkpoint_epoch_{history[-1]["epoch"]:02d}.pt',map_location=device,weights_only=False)
        restored=make_model('B1',channels=8,context=3,architecture='spatial_cnn_v2').to(device);restored.load_state_dict(checkpoint['model']);restored.eval();model.eval()
        x=val_ds[0][0].unsqueeze(0).to(device)
        with torch.no_grad(): assert torch.equal(model(x),restored(x))
        save(out/'checkpoint-verification.json',dict(status='PASS',same_predictions=True,optimizer_updates=checkpoint['optimizer_updates'],contains_optimizer_and_rng=True))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Train');ax.plot([r['epoch'] for r in history],[r['validation']['loss'] for r in history],label='Validation');ax.set(xlabel='Epoch',ylabel='Masked weighted BCE + Dice',title=f'B1 real optimization ({device})');ax.legend();fig.tight_layout();fig.savefig(out/'loss.png',dpi=140);plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compact',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=2);p.add_argument('--profile-steps',type=int,default=0);p.add_argument('--wall-seconds',type=int,default=600);a=p.parse_args()
    print(train(CompactTiles(a.compact,'train'),CompactTiles(a.compact,'val'),a.out,device=a.device,epochs=a.epochs,profile_steps=a.profile_steps,wall_seconds=a.wall_seconds,provenance=dict(compact_report_sha256=sha(Path(a.compact)/'report.json'))))
