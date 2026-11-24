"""Read-only historical observations and archived V2 scenes. No model execution here."""
from datetime import date,timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import zipfile
import numpy as np
from geofirex.calibration_io import sha
from geofirex.c8_predictions import PIXEL
from geofirex.jury_app import evidence,ROOT


def scene_metrics(p,y,valid,observed):
    """Exact AP with tied scores grouped; counts on the joint valid domain."""
    out={}
    for name,mask in [('global',valid),('new_fire',valid & (observed==0))]:
        pp=p[mask];yy=y[mask].astype(bool);pred=pp>=np.float32(.78)
        tp=int((pred&yy).sum());fp=int((pred&~yy).sum());fn=int((~pred&yy).sum());tn=int((~pred&~yy).sum())
        ratio=lambda a,b:float(a/b) if b else None
        ap=None
        if len(pp):
            order=np.argsort(-pp,kind='stable');scores=pp[order];labels=yy[order]
            ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
            hits=np.cumsum(labels,dtype=np.int64)[ends]
            ap=float(np.sum(np.diff(np.r_[0,hits])*hits/(ends+1))/yy.sum()) if yy.any() else 0.
        out[name]=dict(tp=tp,fp=fp,fn=fn,tn=tn,valid_pixels=len(pp),recall=ratio(tp,tp+fn),
            precision=ratio(tp,tp+fp),dice=ratio(2*tp,2*tp+fp+fn),AP=ap,
            Brier=float(np.mean((pp.astype(np.float64)-yy)**2)) if len(pp) else None)
    return out


class SavedScenes:
    """Only the whitelisted 80-event domain. Bounded, sequential ZIP reads."""
    def __init__(self):
        self.sources={r['model']:r for r in json.loads(evidence('comparison.json'))['sources']}
        self.manifests={};self.stamps={};self.lock=threading.Lock()

    def manifest(self,model):
        source=self.sources[model];path=Path(source['archive']);st=path.stat();stamp=(st.st_size,st.st_mtime_ns)
        if self.stamps.get(model)!=stamp:
            if sha(path)!=source['archive_sha256']:raise ValueError('Saved archive SHA-256 differs: '+model)
            with zipfile.ZipFile(path) as z:
                name='evaluation/predictions/predictions_manifest.json'
                raw=z.read(name);artifact=json.loads(z.read('artifact_manifest.json'))
                row=next(r for r in artifact['files'] if r['path']==name)
                if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Prediction manifest differs')
                m=json.loads(raw)
                if m['evaluation_domain_sha256']!=json.loads(evidence('comparison.json'))['evaluation_domain_sha256']:
                    raise ValueError('Saved domain differs')
                if len(m['files'])!=80 or any(not r['event'].startswith('2020/') for r in m['files']):
                    raise ValueError('Only validation2020 permitted')
                self.manifests[model]=m
            self.stamps[model]=stamp
        return self.manifests[model]

    def catalogue(self):
        with self.lock:
            manifests=[self.manifest(m) for m in ('NEWPOS2','NEWPOS4','FIRE13')]
            keys=[{tuple(k) for r in m['files'] for k in r['keys']} for m in manifests]
            if any(k!=keys[0] for k in keys[1:]):raise ValueError('Saved spatial keys differ')
            groups={}
            for e,d,r,c in sorted(keys[0]):groups.setdefault(e,set()).add(d)
            return dict(events=[dict(event_id=e,target_dates=sorted(ds)) for e,ds in sorted(groups.items())],
                models=['NEWPOS2','NEWPOS4','FIRE13'],threshold=.78,scope='validation2020_V2_80',
                source='saved predictions; no inference',C8_spatial_unavailable='C8 COMPLETE contains no full saved prediction arrays',
                test_2021_locked=True)

    def scene(self,event_id,target_date,max_side=384):
        # Refuse 2021 before opening any archive, including prediction metadata.
        if not event_id.startswith('2020/') or not target_date.startswith('2020-'):
            raise PermissionError('Only validation2020 saved scenes authorized; test2021 locked')
        date.fromisoformat(target_date)
        if not self.lock.acquire(blocking=False):raise BlockingIOError('Lecture V2 en cours ; réessayer')
        try:
            cases=[];reference=None;refkeys=None
            for model in ('NEWPOS2','NEWPOS4','FIRE13'):
                manifest=self.manifest(model)
                rows=[r for r in manifest['files'] if r['event']==event_id]
                if not rows:raise KeyError('Unknown V2 event')
                row=rows[0];keys=[tuple(k) for k in row['keys']];selected=[i for i,k in enumerate(keys) if k[1]==target_date]
                if not selected:raise KeyError('Unknown saved target date')
                scene_keys=[keys[i] for i in selected]
                if refkeys is not None and scene_keys!=refkeys:raise ValueError('Scene tile keys differ')
                refkeys=scene_keys
                h=max(k[2]+64 for k in scene_keys);w=max(k[3]+64 for k in scene_keys)
                if h*w>20_000_000 or row['bytes']>256_000_000:raise ValueError('Scene exceeds local memory limit')
                with zipfile.ZipFile(self.sources[model]['archive']) as z:
                    raw=z.read('evaluation/predictions/'+row['path'])
                if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                    raise ValueError('Saved event predictions integrity differs')
                tiles=np.load(io.BytesIO(raw),allow_pickle=False)
                if tiles.dtype!=PIXEL or tiles.shape!=(len(keys),64,64):raise ValueError('Saved prediction schema differs')
                values=np.zeros((h,w),PIXEL);occupied=np.zeros((h,w),bool)
                for i,k in zip(selected,scene_keys):
                    r,c=k[2:];region=np.s_[r:r+64,c:c+64]
                    if occupied[region].any():raise ValueError('Overlapping tiles')
                    values[region]=tiles[i];occupied[region]=True
                del tiles,raw
                if not np.isfinite(values['p']).all() or not np.isin(values['v'],[0,1]).all():raise ValueError('Invalid saved values')
                if reference is None:reference={k:values[k].copy() for k in ('y','v','o')}
                elif any(not np.array_equal(values[k],reference[k]) for k in reference):raise ValueError('Targets/observations/masks differ')
                valid=values['v']==1;stride=max(1,math.ceil(max(h,w)/max_side))
                nullable=lambda a:np.where(valid[::stride,::stride],a[::stride,::stride].astype(object),None).tolist()
                source=self.sources[model];pred=values['p']>=np.float32(.78)
                metrics=scene_metrics(values['p'],values['y'],valid,values['o'])
                result=dict(event_id=event_id,target_date=target_date,as_of=(date.fromisoformat(target_date)-timedelta(days=1)).isoformat(),
                    model=model+' V2',threshold=.78,full_shape=[h,w],preview_stride=stride,scope='validation2020_V2_80_saved_scene',
                    probability=nullable(values['p']),observed_fire=nullable(values['o']),predicted_fire=nullable(pred.astype(np.uint8)),
                    new_predicted_fire=nullable((pred&(values['o']==0)).astype(np.uint8)),
                    checkpoint_sha256=source['checkpoint_sha256'],source_archive_sha256=source['archive_sha256'],
                    saved_member_sha256=row['sha256'],optimizer_updates=28976,crs=None,transform=None,
                    provenance='Saved COMPLETE predictions; no new inference',scene_metrics=metrics,metrics_scope='full saved scene; joint valid mask',
                    input_dates=[(date.fromisoformat(target_date)-timedelta(days=i)).isoformat() for i in (3,2,1)])
                cases.append(dict(model=model,result=result,truth=nullable(values['y'])))
            return dict(cases=cases,same_pixels_verified=True,source='saved predictions; no inference',
                full_scene_metrics=True,test_2021_locked=True,new_inference=False)
        finally:self.lock.release()


def attach_spatial(app):
    from fastapi import APIRouter,HTTPException,Query
    from starlette.routing import Mount
    router=APIRouter();saved=SavedScenes();history_lock=threading.Lock()
    def call(function,*args):
        try:return function(*args)
        except PermissionError as e:raise HTTPException(403,str(e)) from e
        except KeyError as e:raise HTTPException(404,str(e)) from e
        except BlockingIOError as e:raise HTTPException(409,str(e)) from e
        except (ValueError,FileNotFoundError) as e:raise HTTPException(422,str(e)) from e
    def overview_data():
        folder=ROOT/'reports/studio_propagation_20261123_v1'
        seal=json.loads((folder/'visualization_manifest.json').read_text())
        path=folder/'visualization_sources.json'
        if sha(path)!=seal[path.name]:raise ValueError('Visualization provenance differs')
        data=json.loads(path.read_text());hero=data['hero'];source=ROOT/hero['path']
        if sha(source)!=hero['sha256']:raise ValueError('Historical hero output SHA differs')
        split=data['v2_split']
        if sha(ROOT/split['path'])!=split['sha256']:raise ValueError('Split metadata differs')
        with np.load(source,allow_pickle=False) as a:
            valid=a['valid'];nullable=lambda v:np.where(valid,v.astype(object),None).tolist()
            return dict(**data,observed=nullable(a['observed']),predicted=nullable((a['probability']>=np.float32(.78)).astype(np.uint8)),
                full_shape=list(valid.shape),threshold=.78,source='saved historical output; not V2; no new inference')
    @router.get('/studio/overview')
    def overview():return call(overview_data)
    @router.get('/science/v2/scenes')
    def catalogue():return call(saved.catalogue)
    @router.get('/science/v2/scene')
    def scene(event_id:str,target_date:str,max_side:int=Query(384,ge=16,le=512)):
        return call(saved.scene,event_id,target_date,max_side)
    def past(event_id,as_of):
        if not event_id.startswith('2020/'):raise PermissionError('Test2021 locked; validation2020 only')
        with history_lock:
            holder=app.state.prediction_lab
            if 'model' not in holder:
                from geofirex.prediction_lab import CompactPredictor
                holder['model']=CompactPredictor()
            predictor=holder['model'];day,indices=predictor.locate(event_id,as_of)
            if not predictor.lock.acquire(timeout=5):raise BlockingIOError('Inférence en cours')
            try:
                predictor.verify_sources()
                ds=predictor.dataset;a=int(ds.records[indices[0]]['archive']);predictor.verify_archive(a)
                h=max(int(ds.records[i]['row'])+64 for i in indices);w=max(int(ds.records[i]['col'])+64 for i in indices)
                if h*w>20_000_000:raise ValueError('Scene exceeds local limit')
                stride=max(1,math.ceil(max(h,w)/512));observations=[]
                for delta in (-3,-2,-1):
                    fire=np.zeros((h,w),np.uint8);mask=np.zeros((h,w),bool)
                    for i in indices:
                        r,c=int(ds.records[i]['row']),int(ds.records[i]['col']);x,v=predictor.day(a,day+delta,r,c)
                        fire[r:r+64,c:c+64]=x[7];mask[r:r+64,c:c+64]=v[0]
                    observations.append(dict(date=date.fromordinal(day+delta).isoformat(),fire=predictor.nullable(fire,mask,stride)))
                return dict(event_id=event_id,as_of=as_of,observations=observations,preview_stride=stride,full_shape=[h,w],
                    source_archive_sha256=predictor.files[ds.meta['archives'][a]]['sha256'],target_pixels_read=False,crs=None,transform=None)
            finally:predictor.lock.release()
    @router.get('/lab/observations')
    def observations(event_id:str,as_of:str):return call(past,event_id,as_of)
    idx=next(i for i,r in enumerate(app.router.routes) if isinstance(r,Mount) and r.path in ('','/'))
    app.router.routes[idx:idx]=router.routes
    app.state.saved_scenes=saved
