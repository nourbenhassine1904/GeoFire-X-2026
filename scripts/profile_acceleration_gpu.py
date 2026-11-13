"""At most a short discarded-model GPU profile; no full epochs/corpus upload."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,Dataset
from geofirex.fast_raw import scoped_stores,FastRawTiles
from geofirex.mission_training import SceneTiles
from geofirex.fast_compact import diagnostic_indices
from geofirex.benchmark_models import make_model
from geofirex.unet import SegmentationLoss
from geofirex.mission_streaming import save,peak_ram

class Timed(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        t=time.perf_counter();sample=self.base[i];return (*sample,time.perf_counter()-t)

def run(prepared,scope,roots,out,max_seconds=420,steps=8,device='cuda'):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(42)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    stores,provenance=scoped_stores(prepared,roots,scope['train'],scope['val']);fast=FastRawTiles(stores['train']);old=SceneTiles(stores['train']);rec=fast.records;setup_seconds=time.perf_counter()-start
    save(out/'setup.json',dict(seconds=setup_seconds,provenance=provenance));print('SETUP_SECONDS',setup_seconds,flush=True)
    ordered=np.lexsort((rec['day'],rec['col'],rec['row'],rec['archive']));count=16*(steps+2)
    anchors=np.linspace(0,len(ordered)-16,16,dtype=int);local=np.concatenate([ordered[j:j+16] for j in anchors])[:count];scattered=np.random.default_rng(42).permutation(local)
    for i in local[:1]:assert all(torch.equal(x,y) for x,y in zip(old[int(i)],fast[int(i)]))
    rows=[];lossfn=SegmentationLoss(.5,100).to(device)
    configs=[('old',old,0,False,False),('cached',fast,0,False,True),('cached',fast,2,True,True),('cached',fast,4,True,True)]
    for name,ds,workers,pin,locality in configs:
        if time.perf_counter()-start>max_seconds:break
        fast.close();selected=local if locality else scattered
        loader=DataLoader(Timed(Subset(ds,selected.tolist())),batch_size=16,num_workers=workers,pin_memory=pin and device=='cuda',persistent_workers=workers>0,prefetch_factor=2 if workers else None)
        for model_name in ('B1','B2') if name=='cached' and workers==2 else ('B1',):
            torch.manual_seed(42);model=make_model(model_name,8,3,**({'architecture':'spatial_cnn_v2'} if model_name=='B1' else {})).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001);iterator=iter(loader);times=[]
            for step in range(steps+2):
                if time.perf_counter()-start>max_seconds:break
                tick=time.perf_counter();x,y,v,loading=next(iterator);wait=time.perf_counter()-tick
                tick=time.perf_counter();x,y,v=[a.to(device,non_blocking=True) for a in (x,y,v)]
                if device=='cuda':torch.cuda.synchronize()
                h2d=time.perf_counter()-tick;tick=time.perf_counter();opt.zero_grad(set_to_none=True);loss=lossfn(model(x),y,v);loss.backward();opt.step()
                if device=='cuda':torch.cuda.synchronize()
                compute=time.perf_counter()-tick
                if step>=2:times.append(dict(wait=wait,h2d=h2d,compute=compute,worker_loading_sum=float(loading.sum()),tiles=len(x),loss=float(loss.detach())))
            if times:
                total=sum(t['wait']+t['h2d']+t['compute'] for t in times);seen=sum(t['tiles'] for t in times);rate=seen/total
                row=dict(loader=name,workers=workers,pin_memory=pin,persistent_workers=workers>0,model=model_name,measured_updates=len(times),tiles=seen,seconds=total,tiles_per_second=rate,estimated_119_train_epoch_seconds=len(fast)/rate,scaled_250_train_seconds=93603/rate,scope='raw119 measured; compact throughput not measured on this GPU',stages=times,peak_parent_rss=peak_ram(),gpu_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else None)
                rows.append(row);save(out/f'result_{len(rows):02d}.json',row);print(json.dumps(row),flush=True)
        del iterator,loader
    # Positive diagnostic: first declared historical positive val target tile;
    # selection supplied from existing compact metadata, not official metrics.
    val=FastRawTiles(stores['val']);positive=0;diagnostic=[]
    if time.perf_counter()-start<max_seconds:
        for item in scope['diagnostic'][:12]:
            if time.perf_counter()-start>max_seconds:break
            event,day,r,c=item
            from datetime import date
            matches=np.flatnonzero((val.records['archive']==val.meta['archives'].index(event))&(val.records['day']==date.fromisoformat(day).toordinal())&(val.records['row']==r)&(val.records['col']==c))
            if len(matches):
                _,y,v=val[int(matches[0])];positive+=int((y*v).sum());diagnostic.append(item)
    report=dict(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,rows=rows,setup_seconds=setup_seconds,provenance=provenance,train_tiles=len(fast),diagnostic_cases=diagnostic,diagnostic_positive_pixels=positive,diagnostic_scope='label-informed loader diagnostic, not representative validation',elapsed_seconds=time.perf_counter()-start,complete_epochs=0,test_2021=False)
    save(out/'summary.json',report);return report
