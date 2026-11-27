"""Immutable, event-sharded validation predictions. No model or test reader."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from geofirex.protocol_v2 import sha256, save

PIXEL = np.dtype([('p','<f4'),('y','u1'),('v','u1'),('o','u1')])


def validate_keys(keys):
    if not keys or len(keys)!=len(set(keys)):
        raise ValueError('Nonempty unique validation tile keys required')
    for e,d,r,c in keys:
        if not re.fullmatch('2020/fire_[0-9]+',e) or not '2020-08-23'<=d<'2021-01-01':
            raise ValueError('Only frozen V2 validation; 2021 forbidden')
        if type(r) is not int or type(c) is not int or r<0 or c<0 or r%64 or c%64:
            raise ValueError('Invalid tile grid')


def domain_tile(key,y,v,o):
    h=hashlib.sha256(json.dumps([key,list(np.asarray(y).shape)]).encode())
    for a in (y,v,o):h.update(np.asarray(a).astype(bool).tobytes())
    return h.hexdigest()


class PredictionWriter:
    def __init__(self,directory,keys,identity,shape=(64,64)):
        keys=list(keys);validate_keys(keys)
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=False)
        self.keys=keys;self.identity=identity;self.shape=tuple(shape);self.seen=set();self.domains={}
        self.events={};self.positions={};self.open=OrderedDict();self.closed=False
        for key in keys:self.events.setdefault(key[0],[]).append(key)
        for e,ek in self.events.items():
            for i,k in enumerate(ek):self.positions[k]=i
        self.counts={e:dict(new_positive=0,new_negative=0,valid_pixels=0) for e in self.events}

    def array(self,event):
        if event not in self.open:
            if len(self.open)>=4:
                _,a=self.open.popitem(last=False);a.flush();a._mmap.close()
            p=self.root/(event.replace('/','__')+'.npy')
            mode='r+' if p.exists() else 'w+'
            self.open[event]=np.lib.format.open_memmap(p,mode=mode,dtype=PIXEL,
                shape=(len(self.events[event]),*self.shape))
        self.open.move_to_end(event);return self.open[event]

    def add(self,key,p,y,v,o):
        key=tuple(key)
        if self.closed or key not in self.positions or key in self.seen:
            raise ValueError('Unexpected, duplicate or finalized tile')
        p,y,v,o=map(np.asarray,(p,y,v,o))
        if (any(a.shape!=self.shape for a in (p,y,v,o)) or p.dtype!=np.float32
                or not np.isfinite(p).all() or np.any((p<0)|(p>1))
                or not all(np.isin(a,[0,1]).all() for a in (y,v,o))):
            raise ValueError('Finite float32 probabilities and exact binary masks required')
        a=self.array(key[0]);i=self.positions[key]
        for field,value in zip(('p','y','v','o'),(p,y,v,o)):a[field][i]=value
        mask=v.astype(bool)&~o.astype(bool);c=self.counts[key[0]]
        c['new_positive']+=int((mask&y.astype(bool)).sum())
        c['new_negative']+=int((mask&~y.astype(bool)).sum());c['valid_pixels']+=int(v.sum())
        self.domains[key]=domain_tile(key,y,v,o);self.seen.add(key)

    def close(self):
        for a in self.open.values():a.flush();a._mmap.close()
        self.open.clear()

    def finish(self,expected_domain=None):
        self.close()
        if self.closed or self.seen!=set(self.keys):raise ValueError('Incomplete prediction coverage')
        domain=hashlib.sha256(json.dumps(sorted(self.domains.items())).encode()).hexdigest()
        if expected_domain is not None and domain!=expected_domain:raise ValueError('Validation domain differs from C8')
        files=[]
        for e,keys in sorted(self.events.items()):
            p=self.root/(e.replace('/','__')+'.npy')
            files.append(dict(event=e,path=p.name,keys=keys,bytes=p.stat().st_size,sha256=sha256(p),**self.counts[e]))
        save(self.root/'predictions_manifest.json',dict(schema='c8-validation-predictions-1',
            identity=self.identity,files=files,shape=list(self.shape),tiles=len(self.keys),
            evaluation_domain_sha256=domain,test_pixels_opened=0))
        save(self.root/'complete.json',dict(manifest_sha256=sha256(self.root/'predictions_manifest.json')))
        self.closed=True;return domain


class PredictionStore:
    def __init__(self,directory,expected_keys=None,expected_identity=None,expected_domain=None):
        self.root=Path(directory).resolve();p=self.root/'predictions_manifest.json'
        complete=json.loads((self.root/'complete.json').read_text())
        if sha256(p)!=complete['manifest_sha256']:raise ValueError('Prediction manifest changed')
        self.manifest=json.loads(p.read_text());self.manifest_sha256=sha256(p)
        self.deadline=None
        self.files={};keys=[]
        if self.manifest['schema']!='c8-validation-predictions-1':raise ValueError('Prediction schema differs')
        for row in self.manifest['files']:
            e=row['event'];path=(self.root/row['path']).resolve()
            if (e in self.files or not path.is_relative_to(self.root) or path.parent!=self.root
                    or path.suffix!='.npy' or '2021' in path.parts):raise ValueError('Unsafe prediction member')
            ks=[tuple(k) for k in row['keys']];validate_keys(ks)
            if any(k[0]!=e for k in ks) or sha256(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
                raise ValueError('Prediction member differs')
            self.files[e]=row;keys.extend(ks)
        validate_keys(keys)
        if len(keys)!=self.manifest['tiles']:raise ValueError('Prediction coverage count differs')
        if expected_keys is not None and set(keys)!=set(expected_keys):raise ValueError('Prediction key coverage differs')
        if expected_identity is not None and self.manifest['identity']!=expected_identity:raise ValueError('Checkpoint/analysis identity differs')
        if expected_domain is not None and self.manifest['evaluation_domain_sha256']!=expected_domain:raise ValueError('Evaluation domain differs')

    @property
    def events(self):return sorted(self.files)

    def chunks(self,events,chunk_pixels=250000):
        for e in events:
            row=self.files[e];a=np.load(self.root/row['path'],mmap_mode='r',allow_pickle=False)
            try:
                if a.dtype!=PIXEL or a.shape!=(len(row['keys']),*self.manifest['shape']):
                    raise ValueError('Prediction array schema differs')
                flat=a.reshape(-1)
                for i in range(0,len(flat),chunk_pixels):
                    if self.deadline is not None and time.monotonic()>self.deadline:raise TimeoutError('Analysis deadline exhausted')
                    yield e,flat[i:i+chunk_pixels]
            finally:a._mmap.close()

    def tiles(self,events=None):
        for e in self.events if events is None else events:
            row=self.files[e];a=np.load(self.root/row['path'],mmap_mode='r',allow_pickle=False)
            try:
                for i,key in enumerate(row['keys']):
                    if self.deadline is not None and time.monotonic()>self.deadline:raise TimeoutError('Analysis deadline exhausted')
                    yield tuple(key),a[i]
            finally:a._mmap.close()
