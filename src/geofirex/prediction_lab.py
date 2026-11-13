"""CPU inference of the retained compact U-Net. Never read J+1 in predict().

Independent compact-serving contract; the original TIFF Predictor is untouched.
Only server-pinned checkpoints, training sources, normalizer and archives load.
"""
from datetime import date,timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time
import uuid
import zipfile

import numpy as np

from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/prediction_lab_v1.json'


class CompactPredictor:
    def __init__(self,config=CONFIG):
        self.contract=json.loads(Path(config).read_text(encoding='utf-8'))
        self.lock=threading.Lock();self.dataset=None;self.model=None;self.archive_stamps={}
        self.verify_sources()
        import torch
        from geofirex.benchmark_models import make_model
        from geofirex.fast_compact import FastCompactTiles
        c=self.contract
        with zipfile.ZipFile(ROOT/c['checkpoint_archive']) as z:
            raw=z.read(c['checkpoint_member'])
        if hashlib.sha256(raw).hexdigest()!=c['checkpoint_sha256']:raise ValueError('Retained checkpoint SHA differs')
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        cfg=state['config']
        if (cfg['model'],cfg['pos_weight'],cfg['seed'],cfg['model_context'],cfg['train_tiles'])!=('B2',10,42,3,93603):
            raise ValueError('Retained model protocol differs')
        if (state['epoch'],state['cursor'],state['updates'])!=(5,0,23404) or len(state['history'])!=4:
            raise ValueError('Retained checkpoint is not completed epoch4')
        for key in ('index_sha256','normalization_sha256'):
            if cfg[key]!=c[key]:raise ValueError('Checkpoint/data identity differs: '+key)
        torch.set_num_threads(2)
        model=make_model('B2',8,3)
        model.load_state_dict(state['model'],strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
        self.model=model.cpu().eval();self.torch_version=torch.__version__
        self.dataset=FastCompactTiles(ROOT/c['corpus'],ROOT/c['index'],'val')
        ds=self.dataset
        if ds.normalization['continuous']!=[12,14] or ds.normalization['include_landcover']:
            raise ValueError('Expected normalized compact eight-channel features')
        for p in ds.normalization['fit_paths']:
            parts=Path(p.replace('\\','/')).parts
            if not set(parts)&{'2018','2019'} or set(parts)&{'2020','2021'}:raise ValueError('Normalizer is not train-only')
        self.groups={}
        for i,r in enumerate(ds.records):
            event=ds.event(i)
            if not event.startswith('2020/'):raise ValueError('Only historical validation2020 allowed')
            self.groups.setdefault(event,{}).setdefault(int(r['day']),[]).append(i)
        if len(self.groups)!=201 or len(ds)!=78771:raise ValueError('Validation inventory differs')
        with zipfile.ZipFile(ROOT/c['checkpoint_archive']) as z:
            selection=json.loads(z.read('B2_posweight_10/selection32.json'))
        if selection['threshold']!=.78 or selection['checkpoint_sha256']!=c['checkpoint_sha256']:
            raise ValueError('Retained threshold lock differs')
        self.calibration=set(cfg['calibration_events'])
        plan=json.loads((ROOT/c['publication_plan']).read_text())
        self.files={f['path']:f for part in plan['parts'] for f in part['files']}

    def verify_sources(self):
        c=self.contract
        for path,digest in c['pinned_files'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Prediction Lab integrity differs: '+path)
        with zipfile.ZipFile(ROOT/c['source_bundle']) as z:
            for name in ('benchmark_models.py','unet.py','fast_compact.py'):
                current=(ROOT/'src/geofirex'/name).read_text(encoding='utf-8')
                if name=='fast_compact.py':
                    current=current.replace('from geofirex.mission_streaming import save,sha','from geofirex.calibration_io import save,sha')
                if current!=z.read('geofirex/'+name).decode():raise ValueError('Training source differs: '+name)

    def status(self):
        return dict(ready=True,device='cpu',model='B2 U-Net poids 10',threshold=.78,
            checkpoint_sha256=self.contract['checkpoint_sha256'],updates=23404,events=len(self.groups),
            torch_version=self.torch_version,training_torch_version='2.11.0+cu130',test_2021_locked=True,
            preprocessing='already normalized compact T3; train-only normalizer; no re-normalization',
            spatial_reference='pixel grid; CRS/affine unavailable in compact',historical_predictor_bypassed=False)

    def events(self):
        return [dict(event_id=e,split='val',scope='calibration32' if e in self.calibration else 'observed_confirmation169',
            forecast_dates=[date.fromordinal(d-1).isoformat() for d in sorted(days)],observations=len(days))
            for e,days in sorted(self.groups.items())]

    def locate(self,event,as_of):
        if event.startswith('2021/'):raise PermissionError('Test 2021 is locked')
        if event not in self.groups:raise KeyError('Unknown historical event')
        target=date.fromisoformat(as_of).toordinal()+1
        if target not in self.groups[event]:raise ValueError('Date unavailable in compact forecast catalogue')
        return target,sorted(self.groups[event][target],key=lambda i:(int(self.dataset.records[i]['row']),int(self.dataset.records[i]['col'])))

    def verify_archive(self,a):
        ds=self.dataset;logical=ds.meta['archives'][a];path=Path(ds.paths[a]);stat=path.stat()
        stamp=(stat.st_size,stat.st_mtime_ns)
        if self.archive_stamps.get(a)!=stamp:
            expected=self.files[logical]
            if stat.st_size!=expected['bytes'] or sha(path)!=expected['sha256']:raise ValueError('Compact archive identity differs')
            # Drop old handles/cache after an archive has changed, even if bytes restored.
            ds.close();self.archive_stamps[a]=stamp

    def day(self,a,d,r,c):
        x,v=self.dataset._day(a,d,r,c)
        if x.shape!=(8,64,64) or v.shape!=(1,64,64) or x.dtype!=np.float32:
            raise ValueError('Compact tile shape/dtype differs')
        if not np.isfinite(x).all() or not np.isin(v,[0,1]).all() or not np.isin(x[7],[0,1]).all():
            raise ValueError('Nonfinite features or invalid fire/valid mask')
        return x,v.astype(bool)

    @staticmethod
    def nullable(array,mask,stride):
        return np.where(mask[::stride,::stride],array[::stride,::stride].astype(object),None).tolist()

    def predict(self,event_id,as_of,max_side=256):
        if not 16<=max_side<=512:raise ValueError('Preview size16..512 required')
        day,indices=self.locate(event_id,as_of)
        if not self.lock.acquire(blocking=False):raise BlockingIOError('Une prédiction est déjà en cours ; réessayer après sa fin')
        try:
            self.verify_sources();start=time.perf_counter();ds=self.dataset
            a=int(ds.records[indices[0]]['archive']);self.verify_archive(a)
            h=max(int(ds.records[i]['row'])+64 for i in indices);w=max(int(ds.records[i]['col'])+64 for i in indices)
            if h*w>20_000_000 or len(indices)>5000:raise ValueError('Event exceeds local inference limit')
            p=np.zeros((h,w),np.float32);observed=np.zeros((h,w),np.uint8);valid=np.zeros((h,w),bool)
            occupied=valid.copy();tensors=[];tiles=[]
            import torch
            def flush():
                if not tensors:return
                with torch.inference_mode():probs=self.model(torch.from_numpy(np.stack(tensors))).sigmoid().numpy()[:,0]
                if not np.isfinite(probs).all():raise ValueError('Nonfinite inference')
                for pp,(r,c,mask,obs) in zip(probs,tiles):
                    region=np.s_[r:r+64,c:c+64]
                    if occupied[region].any():raise ValueError('Overlapping compact tiles')
                    p[region]=pp;valid[region]=mask;observed[region]=obs;occupied[region]=True
                tensors.clear();tiles.clear()
            for i in indices:
                rec=ds.records[i];r,c=int(rec['row']),int(rec['col'])
                # day is J+1. Read ONLY J-4..J: never day itself, even for masks.
                history=[self.day(a,day+delta,r,c) for delta in (-5,-4,-3,-2,-1)]
                mask=np.logical_and.reduce([v[0] for x,v in history])
                tensors.append(np.stack([x for x,v in history[-3:]]));tiles.append((r,c,mask,history[-1][0][7]))
                if len(tensors)==32:flush()
            flush();stride=max(1,math.ceil(max(h,w)/max_side))
            return dict(request_id=uuid.uuid4().hex,event_id=event_id,as_of=as_of,target_date=date.fromordinal(day).isoformat(),
                model='B2 U-Net poids 10',threshold=.78,checkpoint_sha256=self.contract['checkpoint_sha256'],
                optimizer_updates=23404,device='cpu',elapsed_seconds=time.perf_counter()-start,tiles=len(indices),
                input_dates=[date.fromordinal(day+d).isoformat() for d in (-3,-2,-1)],
                validity_dates=[date.fromordinal(day+d).isoformat() for d in (-5,-4,-3,-2,-1)],
                target_pixels_read=False,validity_scope='history_only; J+1 excluded until separate ground-truth request',
                full_shape=[h,w],preview_stride=stride,crs=None,transform=None,spatial_reference='compact pixel grid',
                probability=self.nullable(p,valid,stride),observed_fire=self.nullable(observed,valid,stride),
                predicted_fire=self.nullable((p>=np.float32(.78)).astype(np.uint8),valid,stride),
                new_predicted_fire=self.nullable(((p>=np.float32(.78)) & (observed==0)).astype(np.uint8),valid,stride),
                normalizer_sha256=self.contract['normalization_sha256'],source_archive_sha256=self.files[ds.meta['archives'][a]]['sha256'],
                scope='calibration32' if event_id in self.calibration else 'observed_confirmation169',test_2021_locked=True)
        finally:self.lock.release()

    def ground_truth(self,event_id,as_of,max_side=256):
        if not 16<=max_side<=512:raise ValueError('Preview size16..512 required')
        day,indices=self.locate(event_id,as_of)
        if not self.lock.acquire(timeout=5):raise BlockingIOError('Prédiction en cours ; réessayer après sa fin')
        try:
            a=int(self.dataset.records[indices[0]]['archive']);self.verify_archive(a)
            h=max(int(self.dataset.records[i]['row'])+64 for i in indices);w=max(int(self.dataset.records[i]['col'])+64 for i in indices)
            if h*w>20_000_000:raise ValueError('Event exceeds local limit')
            y=np.zeros((h,w),np.uint8);valid=np.zeros((h,w),bool)
            for i in indices:
                r,c=int(self.dataset.records[i]['row']),int(self.dataset.records[i]['col'])
                x,v=self.day(a,day,r,c);y[r:r+64,c:c+64]=x[7];valid[r:r+64,c:c+64]=v[0]
            stride=max(1,math.ceil(max(h,w)/max_side))
            return dict(event_id=event_id,target_date=date.fromordinal(day).isoformat(),full_shape=[h,w],preview_stride=stride,
                crs=None,transform=None,ground_truth=self.nullable(y,valid,stride),purpose='historical comparison only; never an inference input')
        finally:self.lock.release()


def attach_lab(app,config=CONFIG):
    """Lazy model loading; results-only start still imports no Torch."""
    from fastapi import APIRouter,HTTPException,Query
    from starlette.routing import Mount
    router=APIRouter(prefix='/lab',tags=['prediction-lab']);holder={};loading=threading.Lock()
    def predictor():
        with loading:
            if 'model' not in holder:holder['model']=CompactPredictor(config)
        return holder['model']
    def call(method,*args):
        try:return getattr(predictor(),method)(*args)
        except PermissionError as e:raise HTTPException(403,str(e)) from e
        except KeyError as e:raise HTTPException(404,str(e)) from e
        except BlockingIOError as e:raise HTTPException(409,str(e)) from e
        except (ValueError,FileNotFoundError) as e:raise HTTPException(422,str(e)) from e
    @router.get('/status')
    def status():return call('status')
    @router.get('/lineage')
    def lineage():
        c=json.loads(Path(config).read_text(encoding='utf-8'))
        for name,digest in c['pinned_files'].items():
            if sha(ROOT/name)!=digest:raise HTTPException(422,'Lineage integrity differs')
        plan=json.loads((ROOT/c['publication_plan']).read_text());index=json.loads((ROOT/c['index']/'index.json').read_text())
        return dict(partitions=len(plan['parts']),archive_count=len(index['archives']),source_bytes=plan['original_file_bytes'],
            roles=index['roles'],context=3,target='J+1',normalization_sha256=c['normalization_sha256'],index_sha256=c['index_sha256'],
            format='lossless float32 NPZ day/tile archives; temporal references, no duplicated T3 arrays',
            live_inference='only J-4..J read; T3 model inputs J-2,J-1,J; target opened in separate comparison request',
            features=['slope_z','elevation_z','slope_missing','elevation_missing','aspect_sin','aspect_cos','aspect_missing','active_fire'],
            corpus_rebuilt=False,spark_restarted=False,test_2021_pixels_opened=0)
    @router.get('/events')
    def events():return call('events')
    @router.get('/predict')
    def predict(event_id:str,as_of:str,max_side:int=Query(256,ge=16,le=512)):
        return call('predict',event_id,as_of,max_side)
    @router.get('/ground-truth')
    def ground_truth(event_id:str,as_of:str,max_side:int=Query(256,ge=16,le=512)):
        return call('ground_truth',event_id,as_of,max_side)
    index=next((i for i,r in enumerate(app.router.routes) if isinstance(r,Mount) and r.path in ('','/')),len(app.router.routes))
    app.router.routes[index:index]=router.routes
    app.state.prediction_lab=holder
