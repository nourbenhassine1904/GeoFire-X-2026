"""Serving safety and actual CPU parity, confined to validation 2020."""
import asyncio
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from geofirex.studio_v2 import (ROOT, V2Predictor, require_validation, qualification,
                               implementation_hashes)
from geofirex.studio_final_results import CompletedResults, RESULTS_SHA
from geofirex.jury_app import create_jury_app
from test_unet_confirmation import request


@pytest.mark.parametrize('event,day',[
    ('2021/fire_24935867','2021-08-01'),('2019/fire_1','2019-08-01'),
    ('2020/../2021/fire_1','2020-08-30'),('2020/fire_1','2021-01-01')])
def test_reject_outside_2020_before_io(event,day):
    with pytest.raises(PermissionError):require_validation(event,day)


@pytest.mark.parametrize('model',['C8','NEWPOS2','NEWPOS4'])
def test_cpu_parity_receipts_current_and_classes_identical(model):
    q=qualification(model)
    assert q['status']=='PASS' and q['device']=='cpu'
    assert all(c['passed'] and c['threshold_classification_equal'] for c in q['checks'])
    assert max(c['max_abs_error'] for c in q['checks'])<1e-6


@pytest.fixture(scope='module')
def c8():
    p=V2Predictor('C8')
    yield p
    p.dataset.close()


def test_predict_never_opens_target_then_truth_is_separate(c8,monkeypatch):
    seen=[]; original=c8.day
    def tracked(a,day,r,c):
        seen.append(day)
        return original(a,day,r,c)
    monkeypatch.setattr(c8,'day',tracked)
    r=c8.predict('2020/fire_24332704','2020-09-09',512)
    from datetime import date
    target=date.fromisoformat(r['target_date']).toordinal()
    assert set(seen)==set(range(target-5,target))
    assert not r['target_pixels_read'] and r['new_inference']
    assert r['model_id']=='C8' and r['optimizer_updates']==28976
    assert 'ground_truth' not in r and 'scene_metrics' not in r
    assert len(r['input_dates'])==3 and r['threshold']==.78
    p=np.array(r['probability'],float); pred=np.array(r['predicted_fire'],float)
    observed=np.array(r['observed_fire'],float); new=np.array(r['new_predicted_fire'],float)
    valid=np.isfinite(p)
    np.testing.assert_array_equal(pred[valid],p[valid]>=np.float32(.78))
    np.testing.assert_array_equal(new[valid],(pred[valid]==1)&(observed[valid]==0))
    seen.clear(); t=c8.ground_truth(r['event_id'],r['as_of'],512)
    assert set(seen)=={target}
    assert t['full_shape']==r['full_shape'] and t['preview_stride']==r['preview_stride']


def test_observation_dates_are_only_T3(c8,monkeypatch):
    seen=[]; original=c8.day
    def tracked(a,day,r,c):seen.append(day);return original(a,day,r,c)
    monkeypatch.setattr(c8,'day',tracked)
    r=c8.observations('2020/fire_24332704','2020-09-09')
    assert [o['date'] for o in r['observations']]==['2020-09-07','2020-09-08','2020-09-09']
    assert len(set(seen))==3 and not r['target_pixels_read']


def test_2021_api_guard_precedes_predictor_loading(monkeypatch):
    app=create_jury_app()
    def forbidden(*args,**kwargs):raise AssertionError('Must not load a model for forbidden data')
    monkeypatch.setattr(V2Predictor,'__init__',forbidden)
    for endpoint in ('predict','ground-truth','observations'):
        for model in ('C8','NEWPOS2','NEWPOS4'):
            status,body=asyncio.run(request(app,f'/studio/v2/{model}/{endpoint}',
                'event_id=2021%2Ffire_24935867&as_of=2021-08-01'))
            assert status==403 and '2021' in body.decode()
    assert not app.state.studio_v2


def test_models_disabled_with_disconnected_ADATA(monkeypatch):
    monkeypatch.setenv('GEOFIREX_V2_ROOT',str(ROOT/'artifacts/no-such-drive'))
    status,body=asyncio.run(request(create_jury_app(),'/studio/models'))
    assert status==200
    models=json.loads(body)['models']
    assert all(not r['enabled'] and 'ADATA' in r['reason'] for r in models[:3])


def test_missing_and_corrupt_results_fail_closed(tmp_path,monkeypatch):
    p=tmp_path/'fake.zip';p.write_bytes(b'not official results')
    with pytest.raises(ValueError,match='SHA-256'):CompletedResults(p).read()
    monkeypatch.setenv('GEOFIREX_RESULTS_ARCHIVE',str(tmp_path/'missing.zip'))
    status,body=asyncio.run(request(create_jury_app(),'/science/final-2021'))
    assert status==503 and 'ADATA' in body.decode()


def test_official_results_and_exports_match_archive_without_new_inference():
    app=create_jury_app()
    status,body=asyncio.run(request(app,'/science/final-2021'));r=json.loads(body)
    assert status==200 and r['status']=='COMPLETE' and r['read_only']
    assert r['events']==156 and r['tiles']==75540 and not r['new_inference']
    assert r['provenance']['archive_sha256']==RESULTS_SHA
    assert r['pooled']['average_precision']==pytest.approx(.45500139551181146)
    assert r['pooled']['dice']==pytest.approx(.5348317337214559)
    assert r['new_fire']['recall']==pytest.approx(.10403844483686998)
    status,body=asyncio.run(request(app,'/science/final-2021/export.csv'))
    rows=list(csv.DictReader(io.StringIO(body.decode())))
    assert status==200 and len(rows)==2 and rows[0]['model']=='C8 V2'
    status,body=asyncio.run(request(app,'/science/final-2021/export.json'))
    assert status==200 and json.loads(body)==r


def test_invalid_model_and_missing_mount_return_explanations(monkeypatch):
    app=create_jury_app()
    assert asyncio.run(request(app,'/studio/v2/not-a-model/events'))[0]==404
    monkeypatch.setenv('GEOFIREX_V2_ROOT',str(ROOT/'artifacts/no-such-drive'))
    status,body=asyncio.run(request(app,'/studio/v2/C8/events'))
    assert status==503 and 'ADATA' in body.decode()
