"""C8 + five causal fire descriptors, using existing compact shards only.

No target or target-validity mask enters a descriptor or its normalization.
Scene-wide fire maps prevent artificial 64-pixel tile boundaries.
"""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, uniform_filter, minimum_filter
from torch.utils.data import Dataset,DataLoader
from geofirex.fast_compact import LocalitySampler
from geofirex.protocol_v2 import V2CompactTiles,sha256

NAMES=['distance_fire_cap8_z','density_fire_7x7_z','appeared_since_oldest',
       'change_since_oldest','derived_support']


def daily_maps(fire,valid):
    f=np.asarray(fire);v=np.asarray(valid,dtype=bool)
    if f.shape!=v.shape or f.ndim!=2 or not np.isin(f,[0,1]).all():
        raise ValueError('Aligned binary scene fire/footprint required')
    f=f.astype(bool)&v
    # Unknown outside the scene/footprint is never asserted to be no fire.
    support=minimum_filter(v.astype(np.uint8),size=17,mode='constant',cval=0).astype(bool)
    distance=np.minimum(distance_transform_edt(~f),8).astype(np.float32) if f.any() else np.full(f.shape,8,np.float32)
    density=uniform_filter(f.astype(np.float32),size=7,mode='constant',cval=0)
    return f,distance,density,support


def descriptors(maps):
    if len(maps)!=3:raise ValueError('Exactly the three observed input days required')
    fires=np.stack([m[0] for m in maps]);support=np.logical_and.reduce([m[3] for m in maps])
    shape=fires.shape[1:];out=np.zeros((3,5,*shape),np.float32)
    for t,m in enumerate(maps):
        out[t,0]=m[1];out[t,1]=m[2]
        out[t,2]=fires[t]&~fires[0]
        out[t,3]=fires[t].astype(np.float32)-fires[0].astype(np.float32)
        out[t,:4]*=support;out[t,4]=support
    return out


def normalize(raw,stats):
    if raw.shape[:2]!=(3,5) or not np.isfinite(raw).all():raise ValueError('Finite T3 five-feature tensor required')
    mean=np.asarray(stats['mean'],np.float32);std=np.asarray(stats['std'],np.float32)
    if mean.shape!=(2,) or std.shape!=(2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Frozen train-only moments required')
    out=raw.copy()
    out[:,:2]=np.clip((out[:,:2]-mean[None,:,None,None])/std[None,:,None,None],-8,8)
    out[:,:4]*=out[:,4:5]
    return out


class Fire13Tiles(Dataset):
    def __init__(self,base,geometry,stats=None,cache_mb=64):
        if base.role not in ('train','val'):raise ValueError('Only frozen development roles')
        self.base=base;self.role=base.role;self.records=base.records;self.meta=base.meta
        self.geometry=geometry;self.stats=stats;self.budget=int(cache_mb*1024**2)
        self.normalization=dict(base=base.normalization,derived=stats)
        self.cache=OrderedDict();self.cache_bytes=0
        if self.budget<0:raise ValueError('Nonnegative cache budget required')
    def __len__(self):return len(self.base)
    def event(self,i):return self.base.event(i)
    def __getstate__(self):return dict(self.__dict__,cache=OrderedDict(),cache_bytes=0)

    def _scene(self,a,day,allowed):
        if day not in allowed or day>=737791: # 2021-01-01 ordinal
            raise ValueError('Descriptor may read only its three dates <=J')
        if a<0 or a>=len(self.meta['archives']) or self.base.paths[a] is None:
            raise ValueError('Forbidden archive')
        key=(a,day)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        event=self.meta['events'][self.meta['archives'][a]]
        if self.meta['event_roles'][event]!=self.role or event.startswith('2021/'):
            raise ValueError('Wrong role or locked event')
        h,w=self.geometry[event]['shape'];f=np.zeros((h,w),bool);v=np.zeros((h,w),bool)
        for row in range(0,h,64):
            for col in range(0,w,64):
                x,valid=self.base._day(a,day,row,col)
                hh,ww=min(64,h-row),min(64,w-col)
                if x.shape!=(8,64,64) or valid.shape!=(1,64,64):raise ValueError('Certified C8 shard shape differs')
                f[row:row+hh,col:col+ww]=x[7,:hh,:ww]
                v[row:row+hh,col:col+ww]=valid[0,:hh,:ww]
        maps=daily_maps(f,v);size=sum(m.nbytes for m in maps)
        if size<=self.budget:
            while self.cache and self.cache_bytes+size>self.budget:
                self.cache_bytes-=sum(m.nbytes for m in self.cache.popitem(last=False)[1])
            self.cache[key]=maps;self.cache_bytes+=size
        return maps

    def context_maps(self,a,target_day):
        allowed=tuple(target_day+d for d in (-3,-2,-1))
        return [self._scene(int(a),day,allowed) for day in allowed]

    def raw_tile(self,i):
        r=self.records[i];a,day,row,col=[int(r[k]) for k in ('archive','day','row','col')]
        maps=self.context_maps(a,day);tiles=[]
        for scene in maps:
            parts=[]
            for grid in scene:
                patch=np.zeros((64,64),grid.dtype);source=grid[row:row+64,col:col+64]
                patch[:source.shape[0],:source.shape[1]]=source;parts.append(patch)
            tiles.append(parts)
        return descriptors(tiles)

    def __getitem__(self,i):
        if self.stats is None:raise ValueError('Fitted train-only normalization required for model input')
        raw=self.raw_tile(i) # This path has not read target or its mask.
        x,y,v=self.base[i]
        extra=torch.from_numpy(normalize(raw,self.stats))
        return torch.cat((x,extra),dim=1),y,v

    def close(self):self.cache.clear();self.cache_bytes=0;self.base.close()


def fit_moments(dataset):
    if dataset.role!='train' or dataset.base.role!='train':raise ValueError('Statistics may fit train only')
    windows=np.unique(dataset.records[['archive','day']])
    count=0;total=np.zeros(2,np.float64);squares=np.zeros(2,np.float64);seen=[]
    import time
    started=time.monotonic()
    for j,r in enumerate(windows):
        if time.monotonic()-started>1200:raise TimeoutError('CPU moments budget exhausted')
        raw=descriptors(dataset.context_maps(int(r['archive']),int(r['day'])))
        mask=raw[:,4].astype(bool);values=raw[:,:2].transpose(1,0,2,3)[:,mask].astype(np.float64)
        count+=values.shape[1];total+=values.sum(axis=1);squares+=np.square(values).sum(axis=1)
        seen.append([int(r['archive']),int(r['day'])])
        if (j+1)%256==0:print('TRAIN-ONLY MOMENTS',j+1,'/',len(windows),flush=True)
    if count==0:raise ValueError('No supported training pixels')
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,0));std[std<1e-6]=1
    return dict(schema='fire13-train-moments-1',mean=mean.tolist(),std=std.tolist(),count=count,
        fit_role='train',windows=seen,window_count=len(seen),clip=8,
        policy='population over unique train windows, three input days, supported unpadded pixels; target mask unused',
        fixed_channels=['appeared binary','change -1/0/1','support binary'],seconds=time.monotonic()-started)


def make_fire13_loaders(corpus,index,*,mounts=None,batch_size=16,workers=0,geometry=None,stats=None):
    if geometry is None or stats is None:raise ValueError('Frozen geometry and train moments required')
    if stats.get('fit_role')!='train':raise ValueError('Normalization must be fitted exclusively on train')
    train=Fire13Tiles(V2CompactTiles(corpus,index,'train',mounts=mounts),geometry,stats)
    val=Fire13Tiles(V2CompactTiles(corpus,index,'val',mounts=mounts),geometry,stats)
    options=dict(batch_size=batch_size,num_workers=workers,drop_last=False,pin_memory=False,
                 prefetch_factor=2 if workers else None)
    return DataLoader(train,sampler=LocalitySampler(train,seed=42),**options),DataLoader(val,shuffle=False,**options)


def numerical_selftest():
    f=np.zeros((64,64),np.uint8);f[32,32]=1
    a=daily_maps(f,np.ones_like(f));empty=daily_maps(f*0,np.ones_like(f))
    if not (a[1][32,32]==0 and a[1][32,35]==3 and a[1][32,48]==8
            and abs(float(a[2][32,32])-1/49)<1e-7 and empty[1][32,32]==8
            and not a[3][0,0] and a[3][8,8]):
        raise ValueError('Unqualified SciPy descriptor numerics')
    return dict(status='PASS',distance='Euclidean grid pixels, cap8',density='7x7 mean',support='8-pixel complete-footprint rim')
