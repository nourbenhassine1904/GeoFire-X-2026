"""Read-only integration of completed Kaggle and Spark artifacts.

The ZIP stays intact. No model is loaded, no calibration or Spark job is run.
"""
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import zipfile

from geofirex.calibration_io import sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = ROOT/'reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip'
DEFAULT_SPARK = ROOT/'reports/mission_20261111/spark'
PHASES = ('calibration32', 'holdout169')
MODELS = ('B1', 'B2')


class ScientificResults:
    def __init__(self, archive=DEFAULT_ZIP, spark=DEFAULT_SPARK):
        self.archive, self.spark = Path(archive).resolve(), Path(spark).resolve()
        self.zip_sha256 = sha(self.archive)
        with zipfile.ZipFile(self.archive) as z:
            self.members={n for n in z.namelist() if not n.endswith('/')}
        # Support a single wrapper folder without extracting arbitrary ZIP paths.
        prefixes=[n.removesuffix('app_manifest.json') for n in self.members if n.endswith('app_manifest.json')]
        if len(prefixes)!=1:raise ValueError('Ambiguous calibration ZIP')
        self.prefix=prefixes[0]
        self.manifest=self.read('app_manifest.json')
        self.protocol=self.read('protocol.json')
        self.lock=self.read('thresholds_locked.json')
        self.comparison=self.read('comparison.json')
        self.train=self.read('train_audit.json')
        self.diagnosis=self.read('diagnosis.json')
        self.complete=self.read('status_complete.json')
        import hashlib
        self.lock_sha=hashlib.sha256(self.bytes('thresholds_locked.json')).hexdigest()
        if (self.complete['status']!='COMPLETE' or self.lock_sha!=self.manifest['threshold_lock_sha256']
                or self.lock_sha!=self.complete['threshold_lock_sha256']
                or self.lock['thresholds']!=self.manifest['thresholds']):
            raise ValueError('Incomplete or inconsistent threshold lock')
        self.rows=self.comparison['rows']
        if self.rows!=self.manifest['comparison']:raise ValueError('Comparison manifest mismatch')
        self.per_event={}
        self.probabilities={}
        self.curves={}
        for phase in PHASES:
            domain=None
            for model in MODELS:
                data=self.read(f'{model}_{phase}/evaluation.json')
                if data['status']!='COMPLETE' or any(not e.startswith('2020/') for e in data['events']):
                    raise ValueError('Only complete 2020 validation is served')
                if len(data['events'])!=(32 if phase=='calibration32' else 169):raise ValueError('Event count mismatch')
                if domain is not None and domain!=data['events']:raise ValueError('Model domains differ')
                domain=data['events']
                thresholds={.5,self.lock['thresholds'][model]}
                per=[r for r in data['per_event'] if r['threshold'] in thresholds]
                for threshold in thresholds:
                    grouped=[r for r in per if r['threshold']==threshold]
                    row=next(r for r in self.rows if r['phase']==phase and r['model']==model and r['threshold']==threshold)
                    if len(grouped)!=len(domain):raise ValueError('Incomplete per-event table')
                    for key in ('tp','fp','fn','tn','valid_pixels','positive_pixels'):
                        if sum(r[key] for r in grouped)!=row[key]:raise ValueError('Per-event/global counts differ')
                pair=[r for r in self.rows if r['phase']==phase and r['model']==model]
                if len(pair)!=2 or any(pair[0][key]!=pair[1][key] for key in ('average_precision','brier_score')):
                    raise ValueError('Probability metrics changed with threshold')
                self.per_event[phase,model]=per
                self.probabilities[phase,model]=data['probabilities']
                # Holdout never exposes an exploratory threshold-search curve.
                self.curves[phase,model]=data['curves'] if phase=='calibration32' else []
        if set(self.protocol['calibration_events']) & set(self.protocol['holdout_events']):raise ValueError('Split overlap')
        self.assets={'calibration_curves.png','calibration32_diagnostics.png','holdout169_diagnostics.png'}
        for case in self.manifest['maps']:
            if not case['event_id'].startswith('2020/'):raise ValueError('Forbidden map year')
            for name in [case['preview'],case['npz'],*case['layers'].values()]:
                path=PurePosixPath(case['directory'])/name
                if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe map path')
                self.assets.add(path.as_posix())
        if any(self.prefix+p not in self.members for p in self.assets):raise ValueError('Missing calibration asset')

    def bytes(self,name):
        with zipfile.ZipFile(self.archive) as z:return z.read(self.prefix+name)

    def read(self,name):return json.loads(self.bytes(name))

    def summary(self):
        return dict(schema='geofirex-science-results-v1',status='COMPLETE',comparison=self.rows,
            distributions=self.comparison['distributions'],thresholds=self.lock['thresholds'],
            boundaries={m:self.lock['selection'][m]['boundary_warning'] for m in MODELS},
            provenance=dict(archive=str(self.archive.relative_to(ROOT)) if self.archive.is_relative_to(ROOT) else self.archive.name,
                archive_sha256=self.zip_sha256,threshold_lock_sha256=self.lock_sha,
                checkpoints={m:dict(sha256=self.protocol['checkpoints'][m]['sha256'],
                    updates=self.protocol['checkpoints'][m]['updates'],
                    next_epoch=self.protocol['checkpoints'][m]['epoch'],
                    completed_epochs=len(self.protocol['checkpoints'][m]['history'])) for m in MODELS},
                index_sha256=self.protocol['index_sha256'],normalization_sha256=self.protocol['normalization_sha256']),
            train=self.train,diagnosis={k:self.diagnosis[k] for k in ('loss','train_prevalence','rows')},
            experiment=dict(weights=[100,30,10],independent_training=True,seed=42,epochs=4,
                updates_per_run=23404,training_executed=False,selection_scope='32 only',
                historical_checkpoint_initialization=False,test_2021_locked=True),
            history={m:self.protocol['checkpoints'][m]['history'] for m in MODELS},
            limitations=[
                '0,95 est le meilleur seuil testé dans [0,10 ; 0,95], pas un optimum global démontré.',
                'Le gain de Dice échange des faux positifs contre des faux négatifs ; lire aussi le rappel.',
                'AP et Brier utilisent les probabilités originales et ne changent pas avec le seuil.',
                '32 événements de réglage ; 169 événements de confirmation désormais observés. Test 2021 verrouillé.',
                'Cartes réelles en grille de pixels ; aucun CRS ou géoréférencement inventé.'])

    def events(self,phase):
        return dict(phase=phase,models={m:self.per_event[phase,m] for m in MODELS},
                    probability_statistics={m:self.probabilities[phase,m] for m in MODELS},
                    calibration_curves={m:self.curves[phase,m] for m in MODELS})

    def maps(self,phase):
        return [dict(**case,phase=phase) for case in self.manifest['maps'] if case['directory'].endswith(phase)]

    @lru_cache(maxsize=1)
    def big_data(self):
        import pyarrow.parquet as pq
        report_path=self.spark/'report.json'
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if report['status']!='SPARK_EXECUTED':raise ValueError('Spark result not executed')
        tables={};files=[]
        for name in ('annual','windows','quality','annual_leakage','window_join'):
            path=self.spark/(name+'.parquet')
            rows=pq.read_table(path).to_pylist()
            canonical=lambda rs:sorted(json.dumps(r,sort_keys=True) for r in rs)
            if canonical(rows)!=canonical(report['results'][name]):raise ValueError('Parquet/report mismatch: '+name)
            tables[name]=rows
            files.append(dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
                              sha256=sha(path),bytes=path.stat().st_size,rows=len(rows)))
        return dict(status=report['status'],spark_version=report['spark_version'],master=report['master'],
            distributed_cluster=report['distributed_cluster'],total_seconds=report['total_seconds'],
            processing_seconds=report['processing_seconds'],jvm_start_seconds=report['jvm_start_seconds'],
            query_seconds=report['query_seconds'],tables=tables,queries=report['queries'],
            duplicate_keys=report['results']['duplicate_keys'],scope=report['scope'],
            source_hashes=report['source_hashes'],parquet_files=files,report_sha256=sha(report_path),
            provenance='reports/mission_20261111/spark/report.json + existing aggregate Parquet; read only, no Spark restart',
            caveats=['local[2] : exécution locale à deux threads, pas un cluster distribué.',
                     '2021 : inventaire de métadonnées existantes seulement ; aucun pixel ni score de test.',
                     'Les comptes qualité concernent les observations ; la calibration utilise des fenêtres et un masque joint.'])


def attach_science(app,archive=DEFAULT_ZIP,spark=DEFAULT_SPARK):
    from fastapi import APIRouter,HTTPException,Query,Response
    from starlette.routing import Mount
    store=ScientificResults(archive,spark)
    router=APIRouter(prefix='/science',tags=['science-results'])
    @router.get('/summary')
    def summary():return store.summary()
    @router.get('/events')
    def events(phase:str=Query('holdout169',pattern='^(calibration32|holdout169)$')):return store.events(phase)
    @router.get('/maps')
    def maps(phase:str=Query('holdout169',pattern='^(calibration32|holdout169)$')):return store.maps(phase)
    @router.get('/big-data')
    def big_data():return store.big_data()
    @router.get('/assets/{name:path}')
    def asset(name:str):
        if name not in store.assets:raise HTTPException(404,'Asset scientifique indisponible')
        return Response(store.bytes(name),media_type='image/png' if name.endswith('.png') else 'application/octet-stream',
                        headers={'Cache-Control':'private, max-age=3600','ETag':'"'+store.zip_sha256+'"'})
    from geofirex.unet_results import attach_unet_routes
    app.state.unet_results=attach_unet_routes(router,store)
    from geofirex.propagation_results import attach_propagation
    attach_propagation(router)
    index=next((i for i,r in enumerate(app.router.routes) if isinstance(r,Mount) and r.path in ('','/')),len(app.router.routes))
    app.router.routes[index:index]=router.routes
    app.state.scientific_results=store
    from geofirex.prediction_lab import attach_lab
    attach_lab(app)
    return app


def create_results_app(archive=DEFAULT_ZIP,spark=DEFAULT_SPARK):
    """Same React app without requiring the optional historical predictor snapshot."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    app=FastAPI(title='GeoFire-X — résultats scientifiques')
    attach_science(app,archive,spark)
    @app.get('/health')
    def health():return dict(ready=True,mode='scientific-results',live_inference=False,prediction_lab_available=True,test_2021_locked=True)
    @app.get('/events')
    def events():return []
    app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
    return app
