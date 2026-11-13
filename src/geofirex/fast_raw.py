"""Explicit train/val scopes and bounded normalized tile cache for raw lots."""
from pathlib import Path
import numpy as np
import torch
from geofirex.fast_compact import FastCompactTiles,DTYPE
from geofirex.mission_streaming import load
from geofirex.runner_data import TemporalStore,RasterReader
from geofirex.unet_data import tile_coordinates,padded_patch
from geofirex.windows import Window

def scoped_stores(prepared,roots,train_events,val_events):
    allowed={'train':set(train_events),'val':set(val_events)}
    if any(not e.startswith(('2018/','2019/')) for e in train_events) or any(not e.startswith('2020/') for e in val_events):raise ValueError('Annual scope invalid')
    rows,groups,domains,n=load(prepared,roots[0]);paths={};duplicates=[];event_roots={};shapes={r['event_id']:(r['height'],r['width']) for r in rows}
    for role,events in allowed.items():
        for event in events:
            preferred=[Path(root) for root in roots if ('geofirex-wsts-fold0-tv-00' in str(root))==(role=='train')]
            candidates=[root for root in (preferred or list(map(Path,roots))) if (root/event).is_dir()]
            if not candidates:raise ValueError(f'Missing scoped event {event}')
            event_roots[event]=sorted(candidates)[0]
            if len(candidates)>1:duplicates.append(event)
    for r in rows:
        if r['event_id'] not in allowed[r['split']]:continue
        paths[(r['event_id'],r['date'])]=str(event_roots[r['event_id']]/r['relative_path'])
    stores={}
    for role in ('train','val'):
        windows=[];maximum=[]
        for w in groups[role]:
            if w.event_id not in allowed[role]:continue
            window=lambda dates:Window(event_id=w.event_id,context_dates=tuple(dates),target_date=w.target_date,context_paths=tuple(paths[(w.event_id,d)] for d in dates),target_path=paths[(w.event_id,w.target_date)])
            windows.append(window(w.context_dates));maximum.append(window([Path(p).stem for p in domains[(w.event_id,w.target_date)]]))
        if {w.event_id for w in windows}!=allowed[role]:raise ValueError('Requested scope incomplete')
        store=TemporalStore.__new__(TemporalStore);store.windows=windows;store.normalizer=n;store.domain_windows=maximum;store.reader=RasterReader(64*1024**2);store.shapes=[shapes[w.event_id] for w in windows]
        stores[role]=store
    return stores,dict(train_events=sorted(allowed['train']),val_events=sorted(allowed['val']),duplicate_events_resolved=len(duplicates),source_verification='pinned certified mount scope and event directory presence; file/header failures propagate at reads; no repeated per-file audit',tiff_headers_opened_during_setup=0)

class FastRawTiles(FastCompactTiles):
    def __init__(self,store,cache_mb=64):
        self.normalizer=store.normalizer;self.normalization=self.normalizer.to_dict()
        events=sorted({w.event_id for w in store.windows});codes={e:i for i,e in enumerate(events)}
        self.meta=dict(archives=events,events={e:e for e in events});self.paths=[str(Path(next(w.target_path for w in store.windows if w.event_id==e)).parent) for e in events]
        from datetime import date
        self.records=np.array([(codes[w.event_id],date.fromisoformat(w.target_date).toordinal(),r,c,0) for i,w in enumerate(store.windows) for r,c in tile_coordinates(*store.shapes[i],64)],dtype=DTYPE)
        self.budget=int(cache_mb*1024**2);self.handle_limit=0;self._reset();self.reader=RasterReader(64*1024**2)
    def __getstate__(self):return {**self.__dict__,'tiles':None,'handles':None,'reader':None}
    def __setstate__(self,value):self.__dict__.update(value);self._reset();self.reader=RasterReader(64*1024**2)
    def _day(self,a,day,r,c):
        from datetime import date
        import time
        key=(a,day,r,c)
        if key in self.tiles:self.hits+=1;self.tiles.move_to_end(key);return self.tiles[key]
        self.misses+=1;t=time.perf_counter();raw,valid=self.reader.tile(str(Path(self.paths[a])/f'{date.fromordinal(day).isoformat()}.tif'),r,c,64)
        x=padded_patch(self.normalizer.transform(raw),0,0,64);valid=padded_patch(valid,0,0,64).astype(bool)
        self.decode_seconds+=time.perf_counter()-t;n=x.nbytes+valid.nbytes
        while self.tiles and self.cache_bytes+n>self.budget:
            old=self.tiles.popitem(last=False)[1];self.cache_bytes-=old[0].nbytes+old[1].nbytes
        if n<=self.budget:self.tiles[key]=(x,valid);self.cache_bytes+=n
        return x,valid
    def close(self):super().close();self.reader=RasterReader(64*1024**2)
