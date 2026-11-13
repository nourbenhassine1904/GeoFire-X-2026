"""Point inference tests, not aggregate evaluations or training."""
import asyncio
from datetime import date
import json
from pathlib import Path

import numpy as np
import pytest

from geofirex.prediction_lab import CompactPredictor,ROOT,CONFIG
from geofirex.science_results import create_results_app
from test_unet_confirmation import request


@pytest.fixture(scope='module')
def predictor():return CompactPredictor()


def test_real_day_inference_never_opens_target_and_matches_saved_probabilities(predictor,monkeypatch):
    target=date(2020,9,4).toordinal();seen=[];original=predictor.day
    def guarded(a,d,r,c):
        assert d<target,'Future pixels must never reach inference, even as a mask'
        seen.append(d);return original(a,d,r,c)
    monkeypatch.setattr(predictor,'day',guarded)
    result=predictor.predict('2020/fire_24332880','2020-09-03',512)
    assert set(seen)==set(range(target-5,target))
    assert not result['target_pixels_read'] and result['device']=='cpu' and result['threshold']==.78
    assert result['tiles']==20 and result['optimizer_updates']==23404
    raw=ROOT/'reports/unet_confirmation_20261112_v1/confirmation_cpu/B2_w10_holdout169/2020__fire_24332880_2020-09-04_raw.npz'
    with np.load(raw) as z:
        actual=np.asarray(result['probability'],dtype=float);mask=z['valid']
        assert result['preview_stride']==1
        np.testing.assert_allclose(actual[mask],z['probability'][mask],rtol=0,atol=1e-6)
        predicted=np.asarray(result['predicted_fire'],dtype=float)
        assert np.array_equal(predicted[mask],(z['probability'][mask]>=np.float32(.78)).astype(float))
    monkeypatch.setattr(predictor,'day',original)
    truth=predictor.ground_truth('2020/fire_24332880','2020-09-03',512)
    assert truth['full_shape']==result['full_shape'] and truth['preview_stride']==1


def test_catalogue_2021_and_invalid_dates_rejected_before_pixels(predictor,monkeypatch):
    events=predictor.events();assert len(events)==201 and len({e['event_id'] for e in events})==201
    assert sum(e['scope']=='calibration32' for e in events)==32
    def forbidden(*args):raise AssertionError('No pixel read permitted')
    monkeypatch.setattr(predictor,'day',forbidden)
    with pytest.raises(PermissionError):predictor.predict('2021/fire_locked','2021-01-01')
    with pytest.raises(PermissionError):predictor.ground_truth('2021/fire_locked','2021-01-01')
    with pytest.raises(KeyError):predictor.predict('2020/missing','2020-09-03')
    with pytest.raises(ValueError):predictor.predict('2020/fire_24332880','2020-01-01')


def test_integrity_failure_closes_lab_instead_of_repinning(tmp_path):
    config=json.loads(CONFIG.read_text());name=next(iter(config['pinned_files']));config['pinned_files'][name]='0'*64
    path=tmp_path/'tampered.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='integrity'):CompactPredictor(path)


def test_api_lab_and_comparison_keep_main_model_and_historical_routes(predictor):
    app=create_results_app();app.state.prediction_lab['model']=predictor
    status,body=asyncio.run(request(app,'/lab/status'));data=json.loads(body)
    assert status==200 and data['threshold']==.78 and data['device']=='cpu'
    assert asyncio.run(request(app,'/lab/predict','event_id=2021%2Flocked&as_of=2021-01-01'))[0]==403
    assert asyncio.run(request(app,'/lab/predict','event_id=2020%2Funknown&as_of=2020-01-01'))[0]==404
    status,body=asyncio.run(request(app,'/science/propagation-comparison'));data=json.loads(body)
    assert status==200 and not data['promoted'] and data['selection']['accepted_on_development32'] is False
    assert data['delta']['new_fire']['fp']==13055
    assert asyncio.run(request(app,'/science/big-data'))[0]==200
    assert asyncio.run(request(app,'/science/unet-ablation'))[0]==200
    status,body=asyncio.run(request(app,'/lab/lineage'));lineage=json.loads(body)
    assert status==200 and lineage['partitions']==11 and lineage['archive_count']==451
    assert lineage['roles']['train']['tiles']==93603 and lineage['test_2021_pixels_opened']==0


def test_changed_event_archive_is_rejected_before_decode(predictor,monkeypatch,tmp_path):
    _,indices=predictor.locate('2020/fire_24332880','2020-09-03')
    a=int(predictor.dataset.records[indices[0]]['archive'])
    bad=tmp_path/'changed.zip';bad.write_bytes(b'corrupt archive')
    paths=predictor.dataset.paths.copy();paths[a]=str(bad);monkeypatch.setattr(predictor.dataset,'paths',paths)
    with pytest.raises(ValueError,match='archive identity'):predictor.verify_archive(a)


@pytest.mark.parametrize('event',['2020/fire_23756984','2020/fire_24695963'])
def test_other_events_multiple_dates_keep_temporal_and_mask_contract(predictor,monkeypatch,event):
    from datetime import timedelta
    catalogue=next(e for e in predictor.events() if e['event_id']==event)
    days=catalogue['forecast_dates'];middle=len(days)//2;ids=set()
    original=predictor.day
    for as_of in days[middle-1:middle+2]:
        target=date.fromisoformat(as_of).toordinal()+1;seen=[]
        def guarded(a,d,r,c):
            assert target-5<=d<target;seen.append(d);return original(a,d,r,c)
        monkeypatch.setattr(predictor,'day',guarded)
        result=predictor.predict(event,as_of,512)
        assert set(seen)==set(range(target-5,target))
        ids.add(result['request_id'])
        assert result['checkpoint_sha256']==predictor.contract['checkpoint_sha256']
        p=np.asarray(result['probability'],dtype=float);valid=np.isfinite(p)
        pred=np.asarray(result['predicted_fire'],dtype=float);obs=np.asarray(result['observed_fire'],dtype=float)
        new=np.asarray(result['new_predicted_fire'],dtype=float)
        assert np.array_equal(np.isfinite(pred),valid) and np.array_equal(np.isfinite(obs),valid)
        assert ((p[valid]>=0)&(p[valid]<=1)).all()
        assert np.array_equal(pred[valid],(p[valid]>=np.float32(.78)).astype(float))
        assert np.array_equal(new[valid],((pred[valid]==1)&(obs[valid]==0)).astype(float))
        stride=result['preview_stride'];h,w=result['full_shape']
        assert p.shape==((h+stride-1)//stride,(w+stride-1)//stride)
        assert result['input_dates'][-1]==as_of and result['target_date']==date.fromordinal(target).isoformat()
        monkeypatch.setattr(predictor,'day',original)
        truth=predictor.ground_truth(event,as_of,512)
        assert np.asarray(truth['ground_truth']).shape==p.shape and truth['preview_stride']==stride
    assert len(ids)==3,'Each date must cause an actual new model request'


def test_busy_prediction_does_not_start_and_truth_waits_for_reader(predictor):
    import threading
    predictor.lock.acquire()
    try:
        with pytest.raises(BlockingIOError):predictor.predict('2020/fire_24332880','2020-09-03')
    finally:predictor.lock.release()
    predictor.lock.acquire();release=threading.Timer(.05,predictor.lock.release);release.start()
    try:
        truth=predictor.ground_truth('2020/fire_24332880','2020-09-03')
        assert truth['target_date']=='2020-09-04'
    finally:release.join()
