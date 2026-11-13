"""Executed mission: bounded scene reads, binary exact AP, lossless day shards.

Uses the existing prepared annual protocol and Normalizer. Never opens 2021.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq
import rasterio
import torch
from torch.utils.data import Dataset

from geofirex.runner_data import RasterReader, read_tile
from geofirex.unet_data import Normalizer, padded_patch, tile_coordinates, validate_layout
from geofirex.windows import Window


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def peak_ram():
    if __import__('os').name == 'nt':
        from ctypes import wintypes
        class PM(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(k, ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        v = PM(); v.cb = ctypes.sizeof(v)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(v), v.cb)
        if not ok: raise ctypes.WinError()
        return v.PeakWorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def load(prepared, root):
    prepared, root = Path(prepared), Path(root)
    rows = [r for r in pq.read_table(prepared/'observations.parquet').to_pylist() if r['split'] in ('train','val')]
    ws = [w for w in pq.read_table(prepared/'windows.parquet').to_pylist() if w['split'] in ('train','val')]
    assert {r['event_id'][:4] for r in rows} <= {'2018','2019','2020'}
    assert all(r['status'] == 'ok' for r in rows)
    n = Normalizer(**json.loads((prepared/'normalization.json').read_text()))
    assert len(n.channel_names) == 8 and all('/2020/' not in p.replace('\\','/') and '/2021/' not in p.replace('\\','/') for p in n.fit_paths)
    if any(r['split'] != ('val' if r['event_id'].startswith('2020/') else 'train') for r in rows):
        raise ValueError('Annual assignment mismatch')
    tokens=[p.replace('\\','/').split('/rasters/')[-1] for p in n.fit_paths]
    expected={p for w in ws if w['split']=='train' and w['context']==5 for p in w['context_paths']}
    if len(tokens)!=len(set(tokens)) or set(tokens)!=expected:
        raise ValueError('Normalizer fit paths differ from unique train-only maximum-context inputs')
    groups = {role: [Window(event_id=w['event_id'], context_dates=tuple(w['context_dates']), target_date=w['target_date'], context_paths=tuple(str(root/p) for p in w['context_paths']), target_path=str(root/w['target_path'])) for w in ws if w['split']==role and w['context']==3] for role in ('train','val')}
    domains = {(w['event_id'],w['target_date']): [str(root/p) for p in w['context_paths']] for w in ws if w['context']==5}
    return rows, groups, domains, n


def binary_metrics(c):
    tp, fp, fn, tn = map(int,c)
    pos, pred, pixels = tp+fn, tp+fp, sum(c)
    # Exact sklearn-style noninterpolated AP with TWO global tied scores.
    ap = (tp/pos)*(tp/pred) + (fn/pos)*(pos/pixels) if pos and pred else (pos/pixels if pos else None)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,valid_pixels=int(pixels),positive_pixels=pos,
        dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
        precision=tp/pred if pred else None,recall=tp/pos if pos else None,average_precision=ap)


def counts(p,y,v):
    p,y = p[v].astype(bool),y[v].astype(bool)
    return np.array([(p&y).sum(),(p&~y).sum(),(~p&y).sum(),(~p&~y).sum()],np.int64)


def b0(prepared, root, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); rows,groups,domains,n=load(prepared,root)
    reader=RasterReader(64*1024**2); per_window=[]; by_event=defaultdict(lambda:np.zeros(4,np.int64)); all_c=np.zeros(4,np.int64)
    selected=groups['val']; invalid=0; figures=0; ignition=np.zeros(4,np.int64)
    with (out/'progress.jsonl').open('x') as log:
        for i,w in enumerate(selected):
            with rasterio.open(w.target_path) as ds:
                validate_layout(ds); shape=ds.shape; geometry=dict(crs=str(ds.crs),transform=list(ds.transform))
            total=np.zeros(4,np.int64); bad=0
            for r,c in tile_coordinates(*shape,256):
                raw,valid=reader.tile(w.target_path,r,c,256); y=np.isfinite(raw[22]); v=valid[0].astype(bool)
                for path in domains[(w.event_id,w.target_date)]:
                    _,mask=reader.tile(path,r,c,256); v &= mask[0].astype(bool)
                observed,_=reader.tile(w.context_paths[-1],r,c,256); p=np.isfinite(observed[22])
                cc=counts(p,y,v); total+=cc; ignition+=counts(p,y,v&~p); bad+=int((~v).sum())
                if figures<4 and (cc[1]+cc[2])>0:
                    import matplotlib; matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig,axs=plt.subplots(1,4,figsize=(14,4))
                    error=np.where(y&~p,1,np.where(p&~y,-1,0))
                    for ax,a,title,cmap,lo,hi in zip(axs,[p,y,p,error],['Observation J','Verite terrain J+1','B0 Persistence','Erreur: rouge FN, bleu FP'],['hot','hot','hot','coolwarm'],[0,0,0,-1],[1,1,1,1]):
                        ax.imshow(np.ma.array(a,mask=~v),cmap=cmap,vmin=lo,vmax=hi); ax.set_title(title,fontsize=9); ax.axis('off')
                    fig.suptitle(f'{w.event_id} | {w.target_date} | tile row={r}, col={c}'); fig.tight_layout(); fig.savefig(out/f'map_{figures:02d}.png',dpi=140); plt.close(fig)
                    np.savez_compressed(out/f'map_{figures:02d}.npz',observed=p,target=y,prediction=p,valid=v,error=error)
                    save(out/f'map_{figures:02d}.json',dict(event_id=w.event_id,target_date=w.target_date,row=r,col=c,**geometry))
                    figures+=1
            all_c+=total; by_event[w.event_id]+=total; invalid+=bad
            row=dict(event_id=w.event_id,target_date=w.target_date,invalid_pixels=bad,**binary_metrics(total)); per_window.append(row)
            if i==len(selected)-1 or selected[i+1].event_id!=w.event_id:
                progress=dict(events=len(by_event),windows=i+1,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_ram())
                log.write(json.dumps(progress)+'\n'); log.flush(); print(progress,flush=True)
    events=[dict(event_id=e,**binary_metrics(c)) for e,c in sorted(by_event.items())]
    def macro(rr):
        return {k:dict(mean=float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None,defined=sum(r[k] is not None for r in rr)) for k in ('dice','iou','precision','recall','average_precision')}
    report=dict(status='EXECUTED',model='B0',protocol='annual aligned T=1/3/5 targets; T=3 context; joint T=5+target footprint',split='val',year=2020,events=len(events),windows=len(per_window),micro=binary_metrics(all_c),macro_event=macro(events),macro_window=macro(per_window),new_ignitions=binary_metrics(ignition),invalid_pixels=invalid,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_ram(),aggregation='micro pools confusion counts across all valid pixels; macro event pools within event then equal event mean; macro window equal window mean',empty_policy='Dice/IoU empty-empty=1; precision undefined without predicted positives; recall/AP undefined without target positives; null excluded from macro with defined count',average_precision='exact two-score tied AP; not trapezoidal PR area',validity='NaN fire=no detection; explicit GDAL/alpha masks and finite nodata invalid; intersect target and all 5 context footprints',selection='all official 2020 aligned windows, no label sampling',test_pixels_opened=0,inputs={k:sha(Path(prepared)/k) for k in ('observations.parquet','windows.parquet','normalization.json')},per_event=events,per_window=per_window)
    save(out/'results.json',report); print(json.dumps({k:v for k,v in report.items() if k not in ('per_event','per_window')}),flush=True)


_zip_cache=OrderedDict()


@contextmanager
def open_npz(root,name):
    if '!' not in name:
        with np.load(Path(root)/name,allow_pickle=False) as z: yield z
    else:
        archive,member=name.split('!',1); path=str(Path(root)/archive)
        if path not in _zip_cache:
            if len(_zip_cache)>=4: _zip_cache.popitem(last=False)[1].close()
            _zip_cache[path]=zipfile.ZipFile(path)
        _zip_cache.move_to_end(path)
        with np.load(io.BytesIO(_zip_cache[path].read(member)),allow_pickle=False) as z: yield z


def compact(prepared,root,out,train_events=10,val_events=6,all_events=False,archives=False,partition=0,partitions=1):
    out=Path(out); out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    rows,groups,domains,n=load(prepared,root)
    chosen=[]
    for role,limit in [('train',train_events),('val',val_events)]:
        events=sorted({w.event_id for w in groups[role]})
        chosen+=events if all_events else [events[j] for j in np.linspace(0,len(events)-1,min(limit,len(events)),dtype=int)]
    if partitions<1 or not 0<=partition<partitions: raise ValueError('Invalid partition')
    chosen=chosen[partition::partitions]
    save(out/'normalization.json',n.to_dict())
    shards=[]; samples=[]; raw_bytes=0; logical_bytes=0; verified=0
    reader=RasterReader(64*1024**2)
    with (out/'progress.jsonl').open('x') as log:
        for event in chosen:
            event_start=time.perf_counter(); role='val' if event.startswith('2020/') else 'train'; event_rows=[r for r in rows if r['event_id']==event]; ws=[w for w in groups[role] if w.event_id==event]
            eventdir=out/role/event; eventdir.mkdir(parents=True)
            archive_path=f'{role}/{event}.zip'
            archive=zipfile.ZipFile(out/archive_path,'x',compression=zipfile.ZIP_STORED) if archives else None
            tile_info={}
            for row in event_rows:
                path=Path(root)/row['relative_path']; raw_bytes+=path.stat().st_size
                with rasterio.open(path) as ds:
                    validate_layout(ds); shape=ds.shape
                    if shape!=(row['height'],row['width']) or str(ds.crs)!=row['crs'] or list(ds.transform)!=row['transform']:
                        raise ValueError('Geometry changed since prepared manifest')
                for r,c in tile_coordinates(*shape,64):
                    raw,v=reader.tile(str(path),r,c,64); x=n.transform(raw)
                    tile_info[(row['date'],r,c)]=(bool(v.all()),int((x[7:8]*v).sum()))
                    x=padded_patch(x,0,0,64); v=padded_patch(v,0,0,64).astype(bool)
                    member=f"{row['date']}_{r}_{c}.npz"
                    file=eventdir/member
                    if archives:
                        buffer=io.BytesIO();np.savez_compressed(buffer,x=x,valid=v);payload=buffer.getvalue()
                        archive.writestr(member,payload)
                        name=archive_path+'!'+member; num_bytes=len(payload); checksum=hashlib.sha256(payload).hexdigest()
                    else:
                        with file.open('xb') as f: np.savez_compressed(f,x=x,valid=v)
                        name=file.relative_to(out).as_posix();num_bytes=file.stat().st_size;checksum=sha(file)
                    logical_bytes+=x.nbytes+v.nbytes
                    shards.append(dict(path=name,event_id=event,date=row['date'],row=r,col=c,bytes=num_bytes,sha256=checksum,source_sha256=row['sha256'],crs=row['crs'],transform=row['transform'],shape=list(shape)))
                    # Every day: first tile roundtrip, float32 bit-preserving including masks.
                    if r==c==0:
                        with np.load(io.BytesIO(payload) if archives else file,allow_pickle=False) as z:
                            assert np.array_equal(z['x'],x) and np.array_equal(z['valid'],v)
                        verified+=1
            if archive: archive.close()
            shape=(event_rows[0]['height'],event_rows[0]['width'])
            for w in ws:
                for r,c in tile_coordinates(*shape,64):
                    names=lambda dates:[f'{archive_path}!{d}_{r}_{c}.npz' if archives else f'{role}/{event}/{d}_{r}_{c}.npz' for d in dates]
                    context=names(w.context_dates); target=names([w.target_date])[0]
                    domain_dates=[Path(p).stem for p in domains[(event,w.target_date)]]
                    if all(tile_info[(d,r,c)][0] for d in [w.target_date,*domain_dates]):
                        pos=tile_info[(w.target_date,r,c)][1]; obs=tile_info[(w.context_dates[-1],r,c)][1]
                    else:
                        with open_npz(out,target) as z: y=z['x'][7:8];valid=z['valid'].copy()
                        for fn in names(domain_dates):
                            with open_npz(out,fn) as z: valid &= z['valid']
                        with open_npz(out,context[-1]) as z: observed=z['x'][7:8]
                        pos=(y*valid).sum();obs=(observed*valid).sum()
                    category='positive' if pos else 'hard_negative' if obs else 'negative'
                    samples.append(dict(event_id=event,split=role,target_date=w.target_date,context_dates=list(w.context_dates),context=context,target=target,domain=names(domain_dates),row=r,col=c,category=category,crs=event_rows[0]['crs'],scene_transform=event_rows[0]['transform']))
            progress=dict(event_id=event,events=chosen.index(event)+1,shards=len(shards),samples=len(samples),event_seconds=time.perf_counter()-event_start,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_ram())
            log.write(json.dumps(progress)+'\n');log.flush();print(progress,flush=True)
    save(out/'shards.json',shards);save(out/'samples.json',samples)
    data_bytes=sum(p.stat().st_size for p in out.rglob('*.zip')) if archives else sum(s['bytes'] for s in shards)
    meta_bytes=sum((out/f).stat().st_size for f in ('shards.json','samples.json','normalization.json','progress.jsonl'))
    from collections import Counter
    report=dict(status='EXECUTED',format='lossless normalized float32 NPZ day/tile shards; temporal references avoid duplicated T=3 arrays',events=len(chosen),train_events=sum(not e.startswith('2020/') for e in chosen),val_events=sum(e.startswith('2020/') for e in chosen),event_ids=chosen,observations=sum(r['event_id'] in chosen for r in rows),windows=sum(w.event_id in chosen for ws in groups.values() for w in ws),samples=len(samples),shards=len(shards),raw_geotiff_bytes=raw_bytes,logical_array_bytes=logical_bytes,shard_bytes=data_bytes,metadata_bytes=meta_bytes,transfer_payload_bytes=data_bytes+meta_bytes,ratio_to_raw=(data_bytes+meta_bytes)/raw_bytes,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_ram(),roundtrip_days=verified,categories={role:dict(Counter(s['category'] for s in samples if s['split']==role)) for role in ('train','val')},selection='metadata-only evenly spaced sorted event ids; all days/all tiles; validation exhaustive without balancing',train_sampling='all positive/hard-negative/negative tiles retained; optional sampler is train-only',test_pixels_opened=0,normalization_sha256=sha(out/'normalization.json'))
    save(out/'report.json',report)
    # Verify actual torch DataLoader against original TemporalStore for sampled windows.
    dataset=CompactTiles(out,'train'); from torch.utils.data import DataLoader
    batch=next(iter(DataLoader(dataset,batch_size=4,num_workers=0)))
    assert batch[0].shape[1:]==(3,8,64,64)
    for index in sorted(set([0,len(dataset)//2,len(dataset)-1])):
        s=dataset.samples[index]; xx,yy,vv=dataset[index]
        expected=[]
        for d in s['context_dates']:
            raw,_=read_tile(Path(root)/s['event_id']/f'{d}.tif',s['row'],s['col'],64); expected.append(padded_patch(n.transform(raw),0,0,64))
        assert np.array_equal(xx.numpy(),np.stack(expected))
        raw,valid=read_tile(Path(root)/s['event_id']/f"{s['target_date']}.tif",s['row'],s['col'],64)
        y=padded_patch(np.isfinite(raw[22:23]).astype(np.float32),0,0,64)
        for fn in s['domain']:
            raw_d,mask=read_tile(Path(root)/s['event_id']/f'{Path(fn.rsplit("!",1)[-1]).name[:10]}.tif',s['row'],s['col'],64); valid*=mask
        assert np.array_equal(yy.numpy(),y) and np.array_equal(vv.numpy(),padded_patch(valid,0,0,64))
    save(out/'pytorch-verification.json',dict(status='PASS',batch_shape=list(batch[0].shape),original_comparisons=3,all_day_first_tile_roundtrips=verified,torch_version=torch.__version__))
    print(json.dumps(report),flush=True)


class CompactTiles(Dataset):
    def __init__(self,root,split):
        if split not in ('train','val'): raise ValueError('Test locked')
        self.root=Path(root); self.samples=[s for s in json.loads((self.root/'samples.json').read_text()) if s['split']==split]
        self.normalization=json.loads((self.root/'normalization.json').read_text())
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]; x=[]; context_valid={}
        for fn in s['context']:
            with open_npz(self.root,fn) as z: x.append(z['x']);context_valid[fn]=z['valid']
        with open_npz(self.root,s['target']) as z: y=z['x'][7:8].copy(); valid=z['valid'].copy()
        for fn in s['domain']:
            if fn in context_valid: valid &= context_valid[fn]
            else:
                with open_npz(self.root,fn) as z: valid &= z['valid']
        return torch.from_numpy(np.stack(x)),torch.from_numpy(y),torch.from_numpy(valid.astype(np.float32))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['b0','compact']);p.add_argument('--prepared',default='reports/full_corpus_v1/prepared_spark_v1');p.add_argument('--root',default='D:/GeoFire-X/WildfireSpreadTS/rasters');p.add_argument('--out',required=True);p.add_argument('--all-events',action='store_true');p.add_argument('--archives',action='store_true');p.add_argument('--partition',type=int,default=0);p.add_argument('--partitions',type=int,default=1);a=p.parse_args()
    with rasterio.Env(GDAL_CACHEMAX=32*1024**2):
        if a.mode=='b0': b0(a.prepared,a.root,a.out)
        else: compact(a.prepared,a.root,a.out,all_events=a.all_events,archives=a.archives,partition=a.partition,partitions=a.partitions)
