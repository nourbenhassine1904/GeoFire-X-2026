"""Real archived scenes and past-only observation integration; CPU only."""
import asyncio
from datetime import date
from pathlib import Path
import json
import os
import numpy as np
import pytest
from geofirex.studio_spatial import SavedScenes,scene_metrics
from geofirex.jury_app import create_jury_app,ROOT
from geofirex.prediction_lab import CompactPredictor
from test_unet_confirmation import request


@pytest.fixture(scope='module')
def saved():return SavedScenes()


def test_saved_catalogue_exact80_common_keys_and_C8_limitation(saved):
    catalogue=saved.catalogue()
    assert len(catalogue['events'])==80 and catalogue['models']==['NEWPOS2','NEWPOS4','FIRE13']
    assert all(e['event_id'].startswith('2020/') for e in catalogue['events'])
    assert 'no full saved' in catalogue['C8_spatial_unavailable']


@pytest.mark.parametrize('offset',[0,1])
def test_full_saved_scenes_same_targets_masks_threshold_and_exact_counts(saved,offset,tmp_path):
    catalogue=saved.catalogue();event=catalogue['events'][0]
    scene=saved.scene(event['event_id'],event['target_dates'][offset],512)
    assert scene['same_pixels_verified'] and not scene['new_inference']
    first=scene['cases'][0]
    for case in scene['cases']:
        r=case['result'];assert case['truth']==first['truth']
        assert r['observed_fire']==first['result']['observed_fire']
        assert r['crs'] is None and r['transform'] is None
        assert r['threshold']==.78 and r['scene_metrics']['new_fire']['valid_pixels']>0
        p=np.array(r['probability'],float);pred=np.array(r['predicted_fire'],float);mask=np.isfinite(p)
        np.testing.assert_array_equal(pred[mask],p[mask]>=np.float32(.78))
        assert len(r['source_archive_sha256'])==64 and len(r['saved_member_sha256'])==64
        if r['preview_stride']==1:
            y=np.nan_to_num(np.array(case['truth'],float)).astype('u1');o=np.nan_to_num(np.array(r['observed_fire'],float)).astype('u1')
            check=scene_metrics(np.nan_to_num(p).astype('f4'),y,mask,o)
            assert check==r['scene_metrics']
    out=Path(os.environ.get('GEOFIREX_SPATIAL_REPORT_DIR',str(tmp_path)))
    write_scene_receipt(out,f'saved-scene-{offset}-metrics.json',{
        'event':event['event_id'],'target_date':event['target_dates'][offset],
        'models':[{k:c['result'][k] for k in ('model','full_shape','preview_stride','scene_metrics','source_archive_sha256','saved_member_sha256')} for c in scene['cases']],
        'new_inference':False})


def write_scene_receipt(out,name,data):
    out=Path(out)
    if any((out/name).exists() for name in ('delivery_manifest.json','science_manifest.json','artifact_manifest.json')):
        raise ValueError('Refusing to overwrite sealed delivery')
    out.mkdir(parents=True,exist_ok=True)
    (out/name).write_text(json.dumps(data,indent=2),encoding='utf8')


def test_repeated_checks_never_overwrite_sealed_delivery(tmp_path):
    (tmp_path/'delivery_manifest.json').write_text('{}')
    (tmp_path/'receipt.json').write_text('preserved')
    with pytest.raises(ValueError,match='sealed'):write_scene_receipt(tmp_path,'receipt.json',{'changed':True})
    assert (tmp_path/'receipt.json').read_text()=='preserved'


def test_2021_refused_before_archive_access(monkeypatch,saved):
    def forbidden(*args):raise AssertionError('Archive must not open')
    monkeypatch.setattr(saved,'manifest',forbidden)
    with pytest.raises(PermissionError):saved.scene('2021/locked','2021-01-01')
    with pytest.raises(PermissionError):saved.scene('2020/fire_24332704','2021-01-01')


def test_exact_AP_groups_ties_and_excludes_invalid_and_old_fire():
    p=np.array([[.9,.9,.1,.8,.99]],dtype='f4');y=np.array([[1,0,1,1,1]],dtype='u1')
    valid=np.array([[1,1,1,1,0]],bool);observed=np.array([[0,0,0,1,0]],dtype='u1')
    result=scene_metrics(p,y,valid,observed)
    assert result['new_fire']['tp']==1 and result['new_fire']['fp']==1 and result['new_fire']['fn']==1
    assert result['new_fire']['AP']==pytest.approx((.5+2/3)/2)
    assert result['global']['valid_pixels']==4


def test_history_API_never_opens_Jplus1_and_exports_real_provenance(monkeypatch):
    app=create_jury_app();predictor=CompactPredictor();app.state.prediction_lab['model']=predictor
    target=date(2020,9,4).toordinal();seen=[];original=predictor.day
    def guarded(a,d,r,c):
        assert target-3<=d<target
        seen.append(d);return original(a,d,r,c)
    monkeypatch.setattr(predictor,'day',guarded)
    status,body=asyncio.run(request(app,'/lab/observations','event_id=2020%2Ffire_24332880&as_of=2020-09-03'))
    data=json.loads(body)
    assert status==200 and not data['target_pixels_read']
    assert [o['date'] for o in data['observations']]==['2020-09-01','2020-09-02','2020-09-03']
    assert set(seen)==set(range(target-3,target)) and data['full_shape']==[320,256]
    assert asyncio.run(request(app,'/lab/observations','event_id=2021%2Flocked&as_of=2021-01-01'))[0]==403
    assert asyncio.run(request(app,'/science/v2/scene','event_id=2021%2Flocked&target_date=2021-01-01'))[0]==403
    status,body=asyncio.run(request(app,'/studio/overview'));overview=json.loads(body)
    assert status==200 and overview['v2_split']['roles']=={'train':318,'purged':53,'val':80,'test':156}
    assert overview['hero']['event_id']=='2020/fire_24332880'
    assert not overview['new_inference']
    predictor.dataset.close()


def test_saved_archive_integrity_failure_is_closed(saved,monkeypatch,tmp_path):
    bad=tmp_path/'bad.zip';bad.write_bytes(b'not original bytes')
    source=dict(saved.sources['NEWPOS2'],archive=str(bad))
    monkeypatch.setitem(saved.sources,'NEWPOS2',source)
    with pytest.raises(ValueError,match='SHA-256'):saved.manifest('NEWPOS2')
