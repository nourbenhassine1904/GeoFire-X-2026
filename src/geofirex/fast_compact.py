"""Small mmap indices and byte-bounded, process-local caches for existing NPZs.

Does not convert or alter any raster/archive. Each record still addresses the
original day tiles, T=3, J+1 and the common T=5 validity domain.
"""
from __future__ import annotations
from collections import OrderedDict
from datetime import date
import io
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from geofirex.mission_streaming import save,sha

DTYPE=np.dtype([('archive','<u2'),('day','<i4'),('row','<u2'),('col','<u2'),('category','u1')])

def build_index(corpus,out):
    """Metadata-only derivative; preserve source sample order for comparisons."""
    corpus,out=Path(corpus),Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();samples=json.loads((corpus/'samples.json').read_text())
    archives=sorted({s['target'].split('!')[0] for s in samples});codes={s:i for i,s in enumerate(archives)}
    roles={};events={};normalizer=json.loads((corpus/'normalization.json').read_text())
    for role in ('train','val'):
        selected=[s for s in samples if s['split']==role];records=np.empty(len(selected),DTYPE)
        for i,s in enumerate(selected):
            archive,member=s['target'].split('!');day=date.fromisoformat(s['target_date']).toordinal();r,c=s['row'],s['col']
            expected=lambda delta:f'{archive}!{date.fromordinal(day+delta).isoformat()}_{r}_{c}.npz'
            if s['context']!=[expected(d) for d in (-3,-2,-1)] or s['domain']!=[expected(d) for d in (-5,-4,-3,-2,-1)] or s['target']!=expected(0):raise ValueError('Noncanonical temporal paths')
            if (role=='train' and s['event_id'][:4] not in ('2018','2019')) or (role=='val' and not s['event_id'].startswith('2020/')):raise ValueError('Split leakage')
            if not member.endswith(f'_{r}_{c}.npz'):raise ValueError('Coordinate mismatch')
            records[i]=(codes[archive],day,r,c,{'negative':0,'positive':1,'hard_negative':2}[s['category']]);events[archive]=s['event_id']
        with (out/f'{role}.npy').open('xb') as f:np.save(f,records,allow_pickle=False)
        roles[role]=dict(tiles=len(records),events=len({s['event_id'] for s in selected}))
    save(out/'normalization.json',normalizer)
    save(out/'index.json',dict(schema='fast-compact-index-v1',archives=archives,events=events,roles=roles,source_samples_sha256=sha(corpus/'samples.json'),normalization_sha256=sha(corpus/'normalization.json'),index_sha256={r:sha(out/f'{r}.npy') for r in roles},seconds=time.perf_counter()-started,raw_arrays_rebuilt=False))
    return roles


class FastCompactTiles(Dataset):
    def __init__(self,corpus,index,role,*,mounts=None,cache_mb=64,zip_handles=8):
        if role not in ('train','val'):raise ValueError('2021 locked')
        self.root=Path(corpus);self.index=Path(index);self.role=role
        self.meta=json.loads((self.index/'index.json').read_text());self.normalization=json.loads((self.index/'normalization.json').read_text())
        if sha(self.index/'normalization.json')!=self.meta['normalization_sha256']:raise ValueError('Normalizer differs')
        if sha(self.index/f'{role}.npy')!=self.meta['index_sha256'][role]:raise ValueError('Compact index differs')
        self.records=np.load(self.index/f'{role}.npy',mmap_mode='r',allow_pickle=False)
        roots=[Path(p) for p in mounts] if mounts else [self.root]
        self.paths=[]
        for logical in self.meta['archives']:
            candidates={p.resolve() for root in roots if (p:=root/logical).is_file()}
            # A remote dataset may add a single wrapper directory, explicitly
            # supplied as a mount root; never guess/extract/rewrite archives.
            if len(candidates)!=1:raise ValueError(f'Missing/ambiguous archive: {logical}')
            self.paths.append(str(candidates.pop()))
        self.budget=int(cache_mb*1024**2);self.handle_limit=zip_handles;self._reset()
    def _reset(self):
        self.pid=os.getpid();self.tiles=OrderedDict();self.handles=OrderedDict();self.cache_bytes=0;self.hits=self.misses=0;self.decode_seconds=0.
    def __getstate__(self):
        return {**self.__dict__,'records':None,'tiles':None,'handles':None}
    def __setstate__(self,value):
        self.__dict__.update(value);self.records=np.load(self.index/f'{self.role}.npy',mmap_mode='r');self._reset()
    def __len__(self):return len(self.records)
    def event(self,i):return self.meta['events'][self.meta['archives'][int(self.records[i]['archive'])]]
    def _day(self,a,day,r,c):
        if self.pid!=os.getpid():self._reset()
        key=(a,day,r,c)
        if key in self.tiles:
            self.hits+=1;self.tiles.move_to_end(key);return self.tiles[key]
        self.misses+=1;started=time.perf_counter()
        if a not in self.handles:
            if len(self.handles)>=self.handle_limit:self.handles.popitem(last=False)[1].close()
            self.handles[a]=zipfile.ZipFile(self.paths[a])
        self.handles.move_to_end(a)
        member=f'{date.fromordinal(day).isoformat()}_{r}_{c}.npz'
        with np.load(io.BytesIO(self.handles[a].read(member)),allow_pickle=False) as z:x=z['x'];valid=z['valid']
        self.decode_seconds+=time.perf_counter()-started;n=x.nbytes+valid.nbytes
        if n<=self.budget:
            while self.tiles and self.cache_bytes+n>self.budget:
                old=self.tiles.popitem(last=False)[1];self.cache_bytes-=old[0].nbytes+old[1].nbytes
            self.tiles[key]=(x,valid);self.cache_bytes+=n
        return x,valid
    def __getitem__(self,i):
        record=self.records[i];a,day,r,c=[int(record[k]) for k in ('archive','day','row','col')]
        context=[self._day(a,day+d,r,c) for d in (-3,-2,-1)]
        target,valid=self._day(a,day,r,c);valid=valid.copy()
        for _,mask in context:valid &= mask
        for d in (-5,-4):valid &= self._day(a,day+d,r,c)[1]
        return torch.from_numpy(np.stack([x for x,_ in context])),torch.from_numpy(target[7:8].copy()),torch.from_numpy(valid.astype(np.float32))
    def close(self):
        for handle in self.handles.values():handle.close()
        self._reset()


class LocalitySampler(Sampler):
    """Shuffle bounded blocks; every tile once, with resumable batch offsets.

    Block order is randomized per epoch. Within blocks group a spatial tile's
    temporal windows together, so six-day fragments are reused by the LRU.
    No label-based sampling or omission, including negative tiles.
    """
    def __init__(self,dataset,seed=42,block_size=256):
        self.dataset=dataset;self.seed=seed;self.block_size=block_size;self.epoch=0;self.start=0
    def order(self):
        r=self.dataset.records
        order=np.lexsort((r['day'],r['col'],r['row'],r['archive']))
        blocks=[order[i:i+self.block_size] for i in range(0,len(order),self.block_size)]
        rng=np.random.default_rng(self.seed+self.epoch)
        return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    def __iter__(self):return iter(self.order()[self.start:].tolist())
    def __len__(self):return len(self.dataset)-self.start


def diagnostic_indices(dataset,count=64):
    """Diagnostic only: deterministic equal-ish positive, hard-negative, negative.

    This label-informed subset must NEVER be called representative validation.
    Official scores retain all tiles of metadata-selected complete events.
    """
    pools=[np.flatnonzero(dataset.records['category']==k) for k in (1,2,0)]
    selected=[]
    for pool in pools:
        n=min(len(pool),count//3)
        if n:selected.extend(pool[np.linspace(0,len(pool)-1,n,dtype=int)].tolist())
    return selected
