"""Short measured old/new loading and real forward/backward profile."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from geofirex.fast_compact import FastCompactTiles,LocalitySampler,diagnostic_indices
from geofirex.mission_streaming import CompactTiles,save,peak_ram
from geofirex.benchmark_models import make_model
from geofirex.unet import SegmentationLoss

class TimedDataset(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        t=time.perf_counter();sample=self.base[i]
        return (*sample,time.perf_counter()-t)

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu';rows=[]
    rng=np.random.default_rng(42)
    fast=FastCompactTiles(a.corpus,a.index,'train')
    records=fast.records;ordered=np.lexsort((records['day'],records['col'],records['row'],records['archive']))
    anchors=np.linspace(0,len(fast)-max(1,a.tiles//16),16,dtype=int)
    local_ids=np.concatenate([ordered[j:j+max(1,a.tiles//16)] for j in anchors])[:a.tiles]
    ids=rng.permutation(local_ids).tolist()
    old=CompactTiles(a.corpus,'train')
    for i in ids[:12]:
        assert all(torch.equal(x,y) for x,y in zip(old[i],fast[i]))
    for name,ds,workers,pin,persistent,locality in [('old',old,0,False,False,False),('fast-random',fast,0,False,False,False),('fast-local',fast,0,False,False,True),('fast-local',fast,2,True,True,True),('fast-local',fast,4,True,True,True)]:
        fast.close()
        selected=local_ids.tolist() if locality else ids
        loader=DataLoader(TimedDataset(Subset(ds,selected)),batch_size=16,num_workers=workers,pin_memory=pin and device=='cuda',persistent_workers=persistent if workers else False,prefetch_factor=2 if workers else None)
        # Two passes distinguish worker startup from persistent-worker throughput.
        for repeat in range(2):
            start=time.perf_counter();iterator=iter(loader);startup=time.perf_counter()-start;wait=0.;seen=0;item_seconds=0.
            for _ in range(len(loader)):
                t=time.perf_counter();batch=next(iterator);wait+=time.perf_counter()-t;seen+=len(batch[0]);item_seconds+=float(batch[3].sum())
            seconds=time.perf_counter()-start
            row=dict(kind='loader',name=name,workers=workers,pin_requested=pin,pin_active=pin and device=='cuda',persistent=persistent,pass_number=repeat+1,tiles=seen,seconds=seconds,tiles_per_second=seen/seconds,iterator_start_seconds=startup,dataloader_wait_seconds=wait,worker_loading_seconds_summed=item_seconds,peak_parent_rss=peak_ram(),cache_hits=fast.hits if not workers else None,cache_misses=fast.misses if not workers else None,decode_seconds=fast.decode_seconds if not workers else None,scope='LOCAL_'+device.upper())
            rows.append(row);print(json.dumps(row),flush=True)
        del iterator,loader
    del old
    # Compute and copies are isolated from dataset reads using one real batch.
    batch=next(iter(DataLoader(Subset(fast,ids[:16]),batch_size=16)))
    for name in ('B1','B2'):
        model=make_model(name,8,3,**({'architecture':'spatial_cnn_v2'} if name=='B1' else {})).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001);lossfn=SegmentationLoss(.5,100).to(device)
        h2d=compute=0.;steps=5
        for step in range(steps+2):
            t=time.perf_counter();x,y,v=[b.to(device,non_blocking=True) for b in batch]
            if device=='cuda':torch.cuda.synchronize()
            copied=time.perf_counter()-t;t=time.perf_counter();opt.zero_grad();loss=lossfn(model(x),y,v);loss.backward();opt.step()
            if device=='cuda':torch.cuda.synchronize()
            calculated=time.perf_counter()-t
            if step>=2:h2d+=copied;compute+=calculated
        rows.append(dict(kind='compute',model=name,device=device,updates=steps,tiles=steps*len(x),h2d_seconds=h2d if device=='cuda' else None,compute_seconds=compute,compute_tiles_per_second=steps*len(x)/compute,gpu_memory=torch.cuda.max_memory_allocated() if device=='cuda' else None))
    val=FastCompactTiles(a.corpus,a.index,'val');diagnostic=diagnostic_indices(val)
    positive=sum(int((val[i][1]*val[i][2]).sum()) for i in diagnostic)
    save(out/'results.json',dict(rows=rows,equality_samples=12,diagnostic_tiles=len(diagnostic),diagnostic_positive_pixels=positive,diagnostic_scope='label-informed diagnostic, not official validation',cuda_available=device=='cuda',timing_warning='synchronized stages for attribution; throughput is loader-only or compute-only, not combined GPU training',official_validation_tiles=len(val)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',default='reports/mission_20261111/compact_full');p.add_argument('--index',required=True);p.add_argument('--out',required=True);p.add_argument('--tiles',type=int,default=256);a=p.parse_args();run(a)
