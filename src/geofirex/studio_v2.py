"""Inference-only Studio adapter. The only data reader is validation-2020 V2.

No final-test runner, optimizer, training or raw TIFF reader is imported here.
The historical serving contract remains a separate, unchanged option.
"""
from datetime import date
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
import zipfile

import numpy as np
from geofirex.calibration_io import sha
from geofirex.prediction_lab import CompactPredictor, ROOT

INDEX = ROOT/'reports/protocol_v2_20261118/index'
QUALIFICATION = ROOT/'artifacts/studio-v2-deployment/qualification.json'
CHANNELS = ['slope_z', 'elevation_z', 'slope_missing', 'elevation_missing',
            'aspect_sin', 'aspect_cos', 'aspect_missing', 'fire_J']
LABELS = {'C8': 'C8 V2 · référence évaluée indépendamment sur 2021',
          'NEWPOS2': 'NEWPOS2 · variante expérimentale',
          'NEWPOS4': 'NEWPOS4 · variante expérimentale haute sensibilité',
          'FIRE13': 'FIRE13 · descripteurs causaux du feu · expérimental',
          'legacy': 'U-Net historique · ancien protocole'}
APPLICATION_SOURCES = ('src/geofirex/studio_v2.py', 'src/geofirex/prediction_lab.py',
                       'src/geofirex/fire13_features.py')


def implementation_hashes():
    return {p: sha(ROOT/p) for p in APPLICATION_SOURCES}


def sources():
    from geofirex.jury_app import evidence
    return {r['model']: r for r in json.loads(evidence('comparison.json'))['sources']}


def archive_path(source):
    override = os.environ.get('GEOFIREX_V2_ROOT')
    return Path(override)/Path(source['archive']).name if override else Path(source['archive'])


def require_validation(event_id, as_of):
    # Check BEFORE model loading or any archive resolution, including metadata.
    import re
    if not re.fullmatch(r'2020/fire_[0-9]+', event_id) or not as_of.startswith('2020-'):
        raise PermissionError('Studio limité à la validation 2020 ; test 2021 terminé, aucune réexécution autorisée.')
    date.fromisoformat(as_of)


def qualification(model):
    from importlib.metadata import version
    q = json.loads(QUALIFICATION.read_text(encoding='utf-8'))
    row = q['models'][model]
    if (q['implementation_sha256'] != implementation_hashes()
            or row['checkpoint_sha256'] != sources()[model]['checkpoint_sha256']
            or q.get('runtime') != {p:version(p) for p in ('torch','numpy','scipy')}):
        raise ValueError('Qualification CPU absente ou périmée ; exécuter scripts/qualify_studio_v2.py sur 2020.')
    if row['status']!='PASS':
        error=max(c['max_abs_error'] for c in row['checks'])
        raise ValueError(f'{model} reste archivé : parité CPU hors tolérance (écart maximal {error:.3g}, atol=rtol=1e-6). Aucun changement de seuil ou de normaliseur autorisé.')
    return row


class V2Predictor(CompactPredictor):
    def __init__(self, model_id='C8', *, require_qualification=True):
        if model_id not in ('C8', 'NEWPOS2', 'NEWPOS4','FIRE13'): raise KeyError('Modèle V2 inconnu')
        self.model_id = model_id
        self.source = sources()[model_id]
        self.path = archive_path(self.source)
        self.lock = threading.Lock(); self.archive_stamps = {}; self.source_stamp = None
        self.dataset = None; self.model = None
        if require_qualification: qualification(model_id)
        baseline = json.loads((ROOT/'configs/protocol_v2/baseline_c8.json').read_text())
        self.pins = {**baseline['source_sha256'],
                     'reports/protocol_v2_20261118/index/index.json': baseline['index_sha256'],
                     'reports/protocol_v2_20261118/index/normalization.json': baseline['normalization_sha256'],
                     baseline['event_manifest']: baseline['event_manifest_sha256']}
        self.verify_sources()
        import torch
        from geofirex.benchmark_models import StackedUNet
        from geofirex.protocol_v2 import V2CompactTiles
        with zipfile.ZipFile(self.path) as z:
            complete = json.loads(z.read('training_complete.json'))
            raw = z.read('checkpoint_u0028976.pt')
        if hashlib.sha256(raw).hexdigest() != self.source['checkpoint_sha256']:
            raise ValueError('SHA-256 checkpoint V2 incorrect')
        if (complete['status'], complete['epochs'], complete['updates'], complete['checkpoint_sha256']) != (
                'COMPLETE', 4, 28976, self.source['checkpoint_sha256']):
            raise ValueError('Checkpoint V2 incomplet')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        cfg = state['config']
        config_name={'C8':'protocol_v2/baseline_c8','NEWPOS2':'c8_newpos2_v1','NEWPOS4':'c8_newpos4_v1','FIRE13':'c8_fire13_v1'}[model_id]
        if cfg!=json.loads((ROOT/('configs/'+config_name+'.json')).read_text()):
            raise ValueError('Configuration du checkpoint choisi différente du protocole enregistré')
        self.channels=cfg['channel_names'];self.fire13=None
        channels=13 if model_id=='FIRE13' else 8
        expected = dict(model='B2/StackedUNet/LightUNet', channels_per_day=channels, context_days=3,
                        base_channels=8, depth=3, channel_names=self.channels, fire_channel=7,
                        threshold=.78, index_sha256=baseline['index_sha256'],
                        normalization_sha256=baseline['normalization_sha256'])
        if any(cfg.get(k) != v for k,v in expected.items()): raise ValueError('Architecture/prétraitement V2 incorrect')
        if (state['epoch'], state['cursor'], state['updates'], len(state['history'])) != (4,0,28976,4):
            raise ValueError('Époque 4 V2 requise')
        norm = json.loads((INDEX/'normalization.json').read_text())
        checkpoint_norm=state['normalizer']['base'] if model_id=='FIRE13' else state['normalizer']
        if checkpoint_norm != norm or norm['continuous'] != [12,14] or norm['include_landcover']:
            raise ValueError('Normaliseur train-only incorrect')
        for p in norm['fit_paths']:
            parts = Path(p.replace('\\','/')).parts
            if not set(parts)&{'2018','2019'} or set(parts)&{'2020','2021'}:
                raise ValueError('Fuite temporelle du normaliseur')
        torch.set_num_threads(2)
        net = StackedUNet(channels,3,base_channels=8,depth=3)
        net.load_state_dict(state['model'], strict=True)
        self.parameters=sum(p.numel() for p in net.parameters())
        if channels==8 and self.parameters != 123977: raise ValueError('Architecture V2 différente')
        if any(not torch.isfinite(t).all() for t in net.state_dict().values()): raise ValueError('Poids non finis')
        self.model = net.cpu().eval().requires_grad_(False); self.torch_version = torch.__version__
        corpus = Path(os.environ.get('GEOFIREX_COMPACT_ROOT', str(ROOT/'reports/mission_20261111/compact_full')))
        self.dataset = V2CompactTiles(corpus, INDEX, 'val')
        self.contract = dict(checkpoint_sha256=self.source['checkpoint_sha256'],
                             normalization_sha256=baseline['normalization_sha256'])
        self.calibration = set(); self.groups = {}
        ds = self.dataset
        for i,r in enumerate(ds.records):
            event = ds.event(i)
            if not event.startswith('2020/'): raise PermissionError('Validation 2020 requise')
            self.groups.setdefault(event,{}).setdefault(int(r['day']),[]).append(i)
        if len(self.groups) != 80 or len(ds) != 30882: raise ValueError('Domaine V2 incorrect')
        self.files = ds.meta['routes']
        if model_id=='FIRE13':
            from geofirex.fire13_features import Fire13Tiles,NAMES,numerical_selftest
            if self.channels!=CHANNELS+NAMES:raise ValueError('Ordre des 13 canaux différent')
            folder=ROOT/'reports/fire13_20261122_v1'
            for name,key in [('derived_normalization.json','derived_normalization_sha256'),('scene_geometry.json','scene_geometry_sha256')]:
                if sha(folder/name)!=cfg[key]:raise ValueError('Prétraitement FIRE13 différent : '+name)
                self.pins[(folder/name).relative_to(ROOT).as_posix()]=cfg[key]
            manifest=json.loads((folder/'execution_manifest.json').read_text())
            self.pins['src/geofirex/fire13_features.py']=manifest['files']['src/geofirex/fire13_features.py']
            self.verify_sources();numerical_selftest()
            stats=json.loads((folder/'derived_normalization.json').read_text())
            if stats!=state['normalizer']['derived'] or stats['fit_role']!='train':raise ValueError('Normaliseur dérivé FIRE13 différent')
            if any(ds.meta['event_roles'][ds.meta['events'][ds.meta['archives'][a]]]!='train' for a,d in stats['windows']):
                raise ValueError('Moments FIRE13 hors train')
            geometry=json.loads((folder/'scene_geometry.json').read_text())
            self.fire13=Fire13Tiles(ds,geometry,stats)
            self._context=threading.local()
            owner=self
            class Fire13Input(torch.nn.Module):
                def __init__(self,network):super().__init__();self.network=network
                def forward(self,inputs):
                    batch=np.stack([owner.context_tensor(next(owner._context.indices),x.numpy()) for x in inputs])
                    return self.network(torch.from_numpy(batch))
            self.model=Fire13Input(self.model).eval().requires_grad_(False)

    def context_tensor(self,index,base):
        # raw_tile reads exactly J-2/J-1/J scene-wide; __getitem__ would also read J+1.
        from geofirex.fire13_features import normalize
        extra=normalize(self.fire13.raw_tile(index),self.fire13.stats)
        result=np.concatenate((base,extra),axis=1)
        if result.shape!=(3,13,64,64) or not np.isfinite(result).all():raise ValueError('Tenseur FIRE13 incorrect')
        return result

    def verify_sources(self):
        for path,digest in self.pins.items():
            if sha(ROOT/path) != digest: raise ValueError('Intégrité source V2 incorrecte : '+path)
        stat = self.path.stat(); stamp = (stat.st_size,stat.st_mtime_ns)
        if self.source_stamp != stamp:
            if sha(self.path) != self.source['archive_sha256']: raise ValueError('SHA-256 archive V2 incorrect')
            self.source_stamp = stamp

    def verify_archive(self, a):
        self.verify_sources()
        super().verify_archive(a)

    def locate(self,event,as_of):
        require_validation(event,as_of)
        return super().locate(event,as_of)

    def events(self):
        return [dict(r,scope='validation2020_V2_80') for r in super().events()]

    def status(self):
        return dict(ready=True, model=LABELS[self.model_id], model_id=self.model_id, device='cpu',
                    threshold=.78, checkpoint_sha256=self.source['checkpoint_sha256'],
                    normalizer_sha256=self.contract['normalization_sha256'], channels=self.channels,parameters=self.parameters,
                    observations=3, events=80, tiles=30882, epoch=4, updates=28976,
                    scope='validation2020_V2_80', test_2021_status='COMPLETE_READ_ONLY',
                    preprocessing='compact float32 déjà normalisé ; train-only ; aucune renormalisation',
                    qualification=qualification(self.model_id))

    def predict(self,event_id,as_of,max_side=256):
        if self.fire13 is not None:self._context.indices=iter(self.locate(event_id,as_of)[1])
        try:result = super().predict(event_id,as_of,max_side)
        finally:
            if self.fire13 is not None:del self._context.indices
        result.update(model=LABELS[self.model_id],model_id=self.model_id,optimizer_updates=28976,
                      scope='validation2020_V2_80',new_inference=True,
                      provenance='Real CPU inference · frozen epoch 4 · validation 2020 V2',
                      channels=self.channels,epoch=4,test_2021_status='COMPLETE_READ_ONLY')
        if self.fire13 is not None:
            result['derived_normalizer_sha256']=self.pins['reports/fire13_20261122_v1/derived_normalization.json']
        return result

    def observations(self,event_id,as_of):
        day,indices=self.locate(event_id,as_of)
        if not self.lock.acquire(blocking=False): raise BlockingIOError('Inférence CPU en cours ; réessayer.')
        try:
            ds=self.dataset; a=int(ds.records[indices[0]]['archive']); self.verify_archive(a)
            h=max(int(ds.records[i]['row'])+64 for i in indices)
            w=max(int(ds.records[i]['col'])+64 for i in indices)
            if h*w>20_000_000: raise ValueError('Scène au-delà de la limite mémoire locale')
            stride=max(1,math.ceil(max(h,w)/512)); observations=[]
            for delta in (-3,-2,-1):
                fire=np.zeros((h,w),np.uint8); mask=np.zeros((h,w),bool)
                for i in indices:
                    r,c=int(ds.records[i]['row']),int(ds.records[i]['col'])
                    x,v=self.day(a,day+delta,r,c); fire[r:r+64,c:c+64]=x[7]; mask[r:r+64,c:c+64]=v[0]
                observations.append(dict(date=date.fromordinal(day+delta).isoformat(),fire=self.nullable(fire,mask,stride)))
            return dict(event_id=event_id,as_of=as_of,observations=observations,preview_stride=stride,
                        full_shape=[h,w],source_archive_sha256=self.files[ds.meta['archives'][a]]['sha256'],
                        target_pixels_read=False,crs=None,transform=None,scope='validation2020_V2_80')
        finally: self.lock.release()


def attach_studio_v2(app):
    from fastapi import APIRouter, HTTPException, Query
    from starlette.routing import Mount
    router=APIRouter(prefix='/studio',tags=['CPU-validation-2020'])
    models={}; load_lock=threading.Lock(); cpu_lock=threading.Lock()

    def get(model_id):
        if model_id not in LABELS or model_id=='legacy': raise KeyError('Modèle V2 inconnu')
        with load_lock:
            if model_id not in models: models[model_id]=V2Predictor(model_id)
        return models[model_id]

    def call(model_id,method,*args):
        try:
            if method in ('predict','observations','ground_truth'): require_validation(args[0],args[1])
            # A single CPU operation across all V2 models bounds peak RAM.
            if not cpu_lock.acquire(blocking=False): raise BlockingIOError('Studio CPU occupé ; réessayer après la fin du calcul.')
            try: return getattr(get(model_id),method)(*args)
            finally: cpu_lock.release()
        except PermissionError as e: raise HTTPException(403,str(e)) from e
        except KeyError as e: raise HTTPException(404,str(e)) from e
        except BlockingIOError as e: raise HTTPException(409,str(e)) from e
        except (OSError,ValueError,zipfile.BadZipFile) as e:
            raise HTTPException(503,'Modèle indisponible : vérifier ADATA, les chemins configurés et les SHA-256. '+str(e)) from e

    @router.get('/models')
    def catalogue():
        rows=[]
        for model_id,source in sources().items():
            reason=None
            try:
                qualification(model_id)
                if not archive_path(source).is_file(): raise FileNotFoundError('Archive absente : reconnecter ADATA ou configurer GEOFIREX_V2_ROOT.')
                corpus=Path(os.environ.get('GEOFIREX_COMPACT_ROOT',str(ROOT/'reports/mission_20261111/compact_full')))
                meta=json.loads((INDEX/'index.json').read_text())
                for logical,route in meta['routes'].items():
                    if meta['event_roles'][meta['events'][logical]]=='val' and not any((corpus/p).is_file() for p in (logical,route['alias'])):
                        raise FileNotFoundError('Compact validation 2020 absent : vérifier GEOFIREX_COMPACT_ROOT et ADATA.')
            except (OSError,ValueError,KeyError) as e: reason=str(e)
            rows.append(dict(id=model_id,label=LABELS[model_id],enabled=reason is None,reason=reason,
                             checkpoint_sha256=source['checkpoint_sha256'],scope='validation2020_V2_80'))
        legacy=json.loads((ROOT/'configs/prediction_lab_v1.json').read_text())
        legacy_missing=not all((ROOT/p).is_file() for p in legacy['pinned_files'])
        rows.append(dict(id='legacy',label=LABELS['legacy'],enabled=not legacy_missing,
                         reason='Artefacts du U-Net historique absents ; restaurer le contrat historique.' if legacy_missing else None,scope='legacy201'))
        return dict(models=rows,default='C8',test_2021_status='COMPLETE_READ_ONLY')

    @router.get('/v2/{model_id}/status')
    def status(model_id:str): return call(model_id,'status')
    @router.get('/v2/{model_id}/events')
    def events(model_id:str): return call(model_id,'events')
    @router.get('/v2/{model_id}/predict')
    def predict(model_id:str,event_id:str,as_of:str,max_side:int=Query(256,ge=16,le=512)):
        return call(model_id,'predict',event_id,as_of,max_side)
    @router.get('/v2/{model_id}/ground-truth')
    def truth(model_id:str,event_id:str,as_of:str,max_side:int=Query(256,ge=16,le=512)):
        return call(model_id,'ground_truth',event_id,as_of,max_side)
    @router.get('/v2/{model_id}/observations')
    def observations(model_id:str,event_id:str,as_of:str): return call(model_id,'observations',event_id,as_of)
    idx=next(i for i,r in enumerate(app.router.routes) if isinstance(r,Mount) and r.path in ('','/'))
    app.router.routes[idx:idx]=router.routes
    app.state.studio_v2=models
