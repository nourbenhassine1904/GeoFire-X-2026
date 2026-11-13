"""Real-artifact integration and experiment guardrails; no campaign is run."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from geofirex.calibration_io import sha
from geofirex.science_results import ScientificResults,attach_science,create_results_app,ROOT,DEFAULT_ZIP
from geofirex import loss_ablation


async def request(app,path,query=''):
    messages=[]
    delivered=False
    async def receive():
        nonlocal delivered
        if delivered:await asyncio.Event().wait()
        delivered=True
        return dict(type='http.request',body=b'',more_body=False)
    async def send(m):messages.append(m)
    await app(dict(type='http',asgi=dict(version='3.0',spec_version='2.4'),http_version='1.1',method='GET',scheme='http',
        path=path,raw_path=path.encode(),query_string=query.encode(),headers=[],client=('127.0.0.1',100),server=('localhost',80),root_path=''),receive,send)
    return next(m['status'] for m in messages if m['type']=='http.response.start'),b''.join(m.get('body',b'') for m in messages)


@pytest.fixture(scope='module')
def science():return ScientificResults()


def test_real_results_and_unchanged_probability_metrics(science):
    summary=science.summary()
    assert summary['thresholds']=={'B1':.95,'B2':.95}
    assert summary['experiment']['independent_training']
    assert 'future_experiments' not in summary['diagnosis']
    assert summary['provenance']['checkpoints']['B1']['completed_epochs']==4
    actual={(r['model'],r['mode']):r for r in summary['comparison'] if r['phase']=='holdout169'}
    assert actual['B1','calibrated']['dice']==pytest.approx(.440878681922247)
    assert actual['B2','calibrated']['dice']==pytest.approx(.4066946488933403)
    for model in ('B1','B2'):
        a,b=actual[model,'baseline_05'],actual[model,'calibrated']
        assert a['average_precision']==b['average_precision']
        assert a['brier_score']==b['brier_score']
        assert b['fp']<a['fp'] and b['fn']>a['fn']
        assert len(science.events('holdout169')['models'][model])==338
        assert science.events('holdout169')['calibration_curves'][model]==[]


def test_routes_before_static_root_and_restricted_assets(science):
    original=sha(DEFAULT_ZIP)
    app=FastAPI();app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True));attach_science(app)
    for path in ('/science/summary','/science/events','/science/maps','/science/big-data'):
        status,body=asyncio.run(request(app,path))
        assert status==200 and json.loads(body) is not None
    status,_=asyncio.run(request(app,'/science/events','phase=test2021'));assert status==422
    for path in ('/science/assets/protocol.json','/science/assets/../protocol.json','/science/assets/unknown.png'):
        status,_=asyncio.run(request(app,path));assert status==404
    case=science.maps('holdout169')[0]
    status,body=asyncio.run(request(app,'/science/assets/'+case['directory']+'/'+case['layers']['errors']))
    assert status==200 and body.startswith(b'\x89PNG')
    assert sha(DEFAULT_ZIP)==original
    status,body=asyncio.run(request(app,'/'));assert status==200 and b'app.js' in body


def test_big_data_actual_parquet_values_and_provenance(science):
    report=science.big_data()
    assert report['master']=='local[2]' and not report['distributed_cluster']
    assert sum(r['events'] for r in report['tables']['annual'])==607
    assert sum(r['observations'] for r in report['tables']['annual'])==13607
    assert report['tables']['quality'][0]['positive_pixels']==429682
    assert report['tables']['windows'][1]['windows']==3948
    assert report['tables']['annual_leakage']==[{'violations':0}]
    assert len(report['parquet_files'])==5
    assert all(len(f['sha256'])==64 for f in report['parquet_files'])


def test_science_extension_preserves_predictor_routes(monkeypatch):
    from geofirex import serving
    class Fake:
        def __init__(self,*a,**kw):self.config={'model':'B1'};self.inventory_identity='existing'
        def events(self):return [{'event_id':'2020/existing'}]
        def predict(self,*a,**kw):return {'existing_prediction':True}
        def ground_truth(self,*a,**kw):return {'existing_truth':True}
    monkeypatch.setattr(serving,'Predictor',Fake)
    app=serving.create_app('unused',[],'existing')
    assert json.loads(asyncio.run(request(app,'/events'))[1])==[{'event_id':'2020/existing'}]
    assert json.loads(asyncio.run(request(app,'/predict','event_id=2020%2Fexisting&as_of=2020-01-05'))[1])=={'existing_prediction':True}
    assert asyncio.run(request(app,'/science/summary'))[0]==200


def test_experiment_preflight_reads_no_pixels_and_never_initializes_model(monkeypatch,tmp_path):
    import zipfile
    with zipfile.ZipFile(DEFAULT_ZIP) as z:(tmp_path/'protocol.json').write_bytes(z.read('protocol.json'))
    from geofirex.fast_compact import FastCompactTiles
    def forbidden(*a,**kw):raise AssertionError('No pixels or training allowed in preflight')
    monkeypatch.setattr(FastCompactTiles,'__getitem__',forbidden)
    monkeypatch.setattr(loss_ablation,'make_model',forbidden)
    args=SimpleNamespace(corpus=ROOT/'reports/mission_20261111/compact_full',index=ROOT/'reports/gpu_acceleration_20261111_v1/index',
        protocol=tmp_path/'protocol.json',execute_training=False,out=tmp_path/'must-not-exist')
    plan=loss_ablation.run(args)
    assert plan['weights']==[100,30,10] and plan['updates_per_run']==23404
    assert len(plan['calibration_events'])==32
    assert not args.out.exists()
    assert max(plan['threshold_candidates'])==.999


def test_paired_fresh_initialization_and_loss_weight_direction():
    from geofirex.benchmark_models import make_model
    from geofirex.unet import SegmentationLoss
    hashes=[]
    for weight in (100,30,10):
        torch.manual_seed(42)
        model=make_model('B1',8,3,architecture='spatial_cnn_v2')
        hashes.append(loss_ablation.model_hash(model))
    assert len(set(hashes))==1
    logits=torch.zeros((1,1,8,8));target=torch.zeros_like(logits);target[0,0,2,2]=1
    losses=[float(SegmentationLoss(.5,w)(logits,target)) for w in (100,30,10)]
    assert losses[0]>losses[1]>losses[2]
    # Threshold selection uses only supplied 32-event curves, with explicit tie order.
    curves=[dict(threshold=t,positive_pixels=2,dice=.5,precision=.4) for t in (.95,.99)]
    assert loss_ablation.select_threshold(curves)['threshold']==.99


def test_final_selection_requires_six_paired_runs_and_refuses_holdout_use(tmp_path):
    spec=importlib.util.spec_from_file_location('selector',ROOT/'scripts/select_loss_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dirs=[]
    for m in ('B1','B2'):
        for w in (100,30,10):
            p=tmp_path/f'{m}-{w}';p.mkdir();dirs.append(p)
            (p/'checkpoint_latest.pt').write_bytes(b'synthetic checkpoint, not a model')
            initial='paired-'+m
            config=dict(seed=42,epochs=4,batch_size=16,index_sha256='index',normalization_sha256='norm',
                protocol_sha256='protocol',torch_version='same',device='cpu',initial_state_sha256=initial,
                model=m,pos_weight=w,
                initialization='fresh make_model seed42 per independent run; never load historical trained weights')
            row=dict(status='COMPLETE',model=m,pos_weight=w,updates=23404,holdout_pixels_opened=0,
                test_2021_pixels_opened=0,initial_state_sha256=initial,checkpoint_sha256=sha(p/'checkpoint_latest.pt'),
                metrics=dict(dice=.6 if (m,w)==('B1',30) else .4,average_precision=.3,brier_score=.01))
            (p/'config.json').write_text(json.dumps(config));(p/'selection32.json').write_text(json.dumps(row))
    result=module.select(dirs,tmp_path/'selection.json')
    assert result['winner']['model']=='B1' and result['winner']['pos_weight']==30
    row=json.loads((dirs[0]/'selection32.json').read_text());row['holdout_pixels_opened']=1
    (dirs[0]/'selection32.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='contaminated'):module.select(dirs,tmp_path/'rejected.json')


def test_ablation_notebook_bundle_is_compilable_and_disabled_by_default():
    import zipfile
    directory=ROOT/'reports/loss_ablation_20261112_v1'
    with zipfile.ZipFile(directory/'loss-ablation-code.zip') as z:
        for name in z.namelist():
            if name.endswith('.py'):compile(z.read(name),name,'exec')
        assert 'geofirex/loss_ablation.py' in z.namelist()
        assert json.loads(z.read('protocol.json'))['calibration_events']==ScientificResults().protocol['calibration_events']
    notebook=json.loads((directory/'GEOFIREX_POS_WEIGHT_INDEPENDENT_20261112.ipynb').read_text(encoding='utf-8'))
    code=[c['source'] for c in notebook['cells'] if c['cell_type']=='code']
    for source in code:compile(source,'notebook','exec')
    assert any('RUN_TRAINING=False' in source for source in code)
    assert not any('torch.load(' in source for source in code)
