"""Bounded real-data CUDA diagnostic; no scientific score or saved model."""
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from geofirex.fast_compact import FastCompactTiles, diagnostic_indices
from geofirex.benchmark_models import make_model
from geofirex.unet import SegmentationLoss
from geofirex.mission_streaming import peak_ram


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


class ProfileDeadline(Exception):pass


class TimedTiles(torch.utils.data.Dataset):
    def __init__(self,base,ids):self.base,self.ids=base,list(map(int,ids))
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        start=time.perf_counter();x,y,v=self.base[self.ids[i]]
        return x,y,v,time.perf_counter()-start


def run(corpus,index,out,max_seconds=180):
    assert max_seconds==180 and torch.cuda.is_available()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    start=time.perf_counter();deadline=start+max_seconds
    result=dict(status='STARTED',budget_seconds=180,seed=42,T=3,target='J+1',rows=[],
                gpu=torch.cuda.get_device_name(0),scientific_metrics=False,
                warning='Short diagnostic subsets and synchronized stages; epoch estimates provisional.')
    def alarm(*_):raise ProfileDeadline()
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    loader=None;iterator=None
    try:
        ds=FastCompactTiles(corpus,index,'train',cache_mb=64,zip_handles=8)
        assert len(ds)==93603
        order=np.lexsort((ds.records['day'],ds.records['col'],ds.records['row'],ds.records['archive']))
        anchors=np.linspace(0,len(order)-16,16,dtype=int)
        ids=np.concatenate([order[a:a+16] for a in anchors])
        random_ids=np.random.default_rng(42).permutation(ids)
        val=FastCompactTiles(corpus,index,'val',cache_mb=16,zip_handles=2)
        positive=sum(int((val[i][1]*val[i][2]).sum()) for i in diagnostic_indices(val,12))
        assert positive>0
        result.update(train_tiles=len(ds),val_tiles=len(val),diagnostic_positive_pixels=positive,
                      train_events=250,val_events=201,normalization_sha256=ds.meta['normalization_sha256'])
        val.close()
        # Include cache/locality, worker count, pinning and worker persistence.
        configs=[('B1',0,False,False,False),('B1',0,False,False,True),
                 ('B1',2,True,False,True),('B1',2,True,True,True),
                 ('B1',4,False,True,True),('B1',4,True,True,True),
                 ('B2',4,True,True,True)]
        lossfn=SegmentationLoss(.5,100).cuda()
        for model_name,workers,pin,persistent,locality in configs:
            if time.perf_counter()>deadline-15:break
            ds.close()
            model=make_model(model_name,8,3,**({'architecture':'spatial_cnn_v2'} if model_name=='B1' else {})).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            subset=TimedTiles(ds,ids if locality else random_ids)
            loader=DataLoader(subset,batch_size=16,num_workers=workers,pin_memory=pin,
                persistent_workers=persistent,prefetch_factor=2 if workers else None)
            for pass_number in (1,2):
                if time.perf_counter()>deadline-8:break
                t=time.perf_counter();iterator=iter(loader);startup=time.perf_counter()-t
                begin=time.perf_counter();wait=copy=compute=loading=0.;seen=updates=0
                # A single pass across 256 actual tiles, limited to 8 s/config pass.
                for _ in range(len(loader)):
                    if time.perf_counter()>min(deadline-3,begin+8):break
                    t=time.perf_counter();x,y,v,load=next(iterator);wait+=time.perf_counter()-t
                    loading+=float(load.sum())
                    t=time.perf_counter();x,y,v=[b.cuda(non_blocking=pin) for b in (x,y,v)]
                    torch.cuda.synchronize();copy+=time.perf_counter()-t
                    t=time.perf_counter();opt.zero_grad(set_to_none=True)
                    loss=lossfn(model(x),y,v);loss.backward();opt.step()
                    torch.cuda.synchronize();compute+=time.perf_counter()-t
                    seen+=len(x);updates+=1
                seconds=time.perf_counter()-begin+startup
                row=dict(model=model_name,workers=workers,pin_memory=pin,persistent_workers=persistent,
                    locality=locality,pass_number=pass_number,tiles=seen,updates=updates,
                    seconds=seconds,tiles_per_second=seen/seconds if seen else None,
                    loader_start_seconds=startup,dataloader_wait_seconds=wait,
                    worker_load_seconds_summed=loading,cpu_to_gpu_seconds=copy,gpu_compute_seconds=compute,
                    peak_parent_rss=peak_ram(),gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                    estimated_train_epoch_seconds=93603*seconds/seen if seen else None,
                    scope='diagnostic across 16 corpus locations; not a full-epoch measurement')
                result['rows'].append(row);save(out/'profile.json',result)
                print(json.dumps(row),flush=True)
            if iterator is not None and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
            iterator=None;loader=None
            del model,opt
        result['status']='COMPLETE_SHORT_DIAGNOSTIC'
    except ProfileDeadline:
        result['status']='WALL_LIMIT_180_SECONDS'
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if iterator is not None and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        result['elapsed_seconds']=time.perf_counter()-start
        result['models_saved']=False
        save(out/'profile.json',result)
    return result
