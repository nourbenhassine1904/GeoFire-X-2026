"""No real 2021 data. Temporary synthetic 2020 TIFFs and unit-only fake grants."""
import json
from pathlib import Path
from datetime import date,timedelta

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from sklearn.metrics import average_precision_score

from geofirex import final_test_v2 as ft
from geofirex.real_constants import RAW_BAND_NAMES


@pytest.fixture
def synthetic(tmp_path):
    root=tmp_path/'rasters'; folder=root/'2020/fire_1';folder.mkdir(parents=True)
    norm=dict(continuous=[12,14],mean=[10.,100.],std=[2.,20.],finite_count=[100,100],fit_paths=[],include_landcover=False,clip=8.)
    rows=[]
    for i in range(6):
        day=(date(2020,9,1)+timedelta(days=i)).isoformat(); path=folder/(day+'.tif')
        raw=np.full((23,65,66),np.nan,np.float32)
        raw[12]=10+i;raw[14]=100+i;raw[13]=90;raw[22,0,0]=1
        if i==5: raw[22,3,3]=1  # strictly future ignition
        if i==2: raw[12,2,2]=np.nan;raw[13,2,3]=-999
        with rasterio.open(path,'w',driver='GTiff',width=66,height=65,count=23,dtype='float32',
                           crs='EPSG:32610',transform=from_origin(0,1000,375,375)) as ds:
            ds.write(raw);ds.descriptions=RAW_BAND_NAMES
        rows.append(dict(event_id='2020/fire_1',date=day,relative_path=f'2020/fire_1/{day}.tif',
                         height=65,width=66,bytes=path.stat().st_size,status='ok',crs='EPSG:32610',
                         transform=list(from_origin(0,1000,375,375)),sha256=ft.sha(path),sidecars_json='{}'))
    return root,rows,norm


def test_inventory_stat_only_no_raster_open(synthetic,monkeypatch):
    root,rows,_=synthetic
    monkeypatch.setattr(rasterio,'open',lambda *a,**k:pytest.fail('metadata must not open TIFF'))
    selected,keys,stats=ft.inventory(rows,['2020/fire_1'],root)
    assert len(keys)==4 and stats['windows']==1 and stats['observations']==6
    assert stats['missing_paths']==stats['size_errors']==stats['raster_headers_opened']==0
    assert stats['upper_padded_pixels']==4*4096


def test_inventory_rejects_duplicate_dates_and_geometry(synthetic):
    _,rows,_=synthetic
    with pytest.raises(ValueError,match='duplicate'):ft.inventory(rows+[rows[0]],['2020/fire_1'])
    bad=[dict(r) for r in rows];bad[1]['height']=12
    with pytest.raises(ValueError,match='geometry'):ft.inventory(bad,['2020/fire_1'])


def test_causal_reader_and_padding(synthetic):
    root,rows,norm=synthetic; reader=ft.FinalReader(rows,root,norm)
    key=('2020/fire_1','2020-09-06',64,64)
    x,token=reader.context(key)
    assert x.shape==(3,8,64,64)
    assert ('2020/fire_1','2020-09-06') not in reader.checked
    with pytest.raises(PermissionError,match='before prediction'):reader.target_after_prediction(token)
    reader.record_prediction(token,np.full((1,64,64),.5,np.float32))
    y,v,o=reader.target_after_prediction(token)
    assert v.sum()==2 and not x[...,1:,:].any() and not x[...,:,2:].any()
    assert not y.any() and not o.any()
    with pytest.raises(PermissionError):reader.target_after_prediction(token)


def test_future_fire_not_in_inputs_and_target_after_model(synthetic):
    root,rows,norm=synthetic;reader=ft.FinalReader(rows,root,norm)
    x,token=reader.context(('2020/fire_1','2020-09-06',0,0))
    assert not x[:,7,3,3].any()
    assert np.array_equal(x[:,0,0,0],np.array([1.,1.5,2.],np.float32))
    assert ('2020/fire_1','2020-09-06') not in reader.checked
    reader.record_prediction(token,np.zeros((1,64,64),np.float32))
    y,v,o=reader.target_after_prediction(token)
    assert y[0,3,3]==1 and o[0,3,3]==0 and v[0,3,3]


def test_2021_constructor_refused_before_io(monkeypatch,tmp_path):
    monkeypatch.setattr(rasterio,'open',lambda *a,**k:pytest.fail('2021 open'))
    monkeypatch.setattr(ft,'sha',lambda *a:pytest.fail('2021 byte read'))
    with pytest.raises(PermissionError,match='2021 reader'):
        ft.FinalReader([dict(event_id='2021/fire_1')],tmp_path,{})


def test_source_tamper_refused(synthetic):
    root,rows,norm=synthetic;rows[0]['sha256']='0'*64
    reader=ft.FinalReader(rows,root,norm)
    with pytest.raises(ValueError,match='content changed'):reader.context(('2020/fire_1','2020-09-06',0,0))


def test_common_validity_and_nodata(synthetic):
    root,rows,norm=synthetic
    path=root/rows[0]['relative_path']
    # Explicit finite fire nodata invalidates domain; NaN fire is no detection.
    with rasterio.open(path,'r+') as ds:
        raw=ds.read();raw[22,2,2]=-999;ds.write(raw);ds.nodata=-999
    rows[0]['sha256']=ft.sha(path)
    rows[0]['bytes']=path.stat().st_size
    reader=ft.FinalReader(rows,root,norm);_,token=reader.context(('2020/fire_1','2020-09-06',0,0))
    reader.record_prediction(token,np.zeros((1,64,64),np.float32));y,v,o=reader.target_after_prediction(token)
    assert not v[0,2,2] and v[0,3,4] and y[0,3,4]==0


def test_exact_global_new_and_event_ap_brier(tmp_path):
    keys=[('2021/SYNTHETIC_1','2021-01-06',0,0),('2021/SYNTHETIC_2','2021-01-06',0,0)]
    acc=ft.FinalMetrics(tmp_path/'sort',keys)
    p=np.array([.9,.8,.8,.1],np.float32);y=np.array([1,0,1,0]);v=np.array([1,1,1,0]);o=np.array([1,0,0,0])
    acc.update(keys[0],p,y,v,o)
    with pytest.raises(ValueError,match='incomplete'):acc.result()
    with pytest.raises(ValueError,match='duplicate'):acc.update(keys[0],p,y,v,o)
    acc.update(keys[1],np.array([.8,.2],np.float32),np.array([0,1]),np.ones(2),np.zeros(2))
    r=acc.result()
    assert r['pooled']['average_precision']==pytest.approx(average_precision_score([1,0,1,0,1],[.9,.8,.8,.8,.2]))
    assert r['new_fire']['average_precision']==pytest.approx(average_precision_score([0,1,0,1],[.8,.8,.8,.2]))
    assert r['pooled']['dice']==pytest.approx(4/7) and r['new_fire']['fp']==2
    assert r['pooled']['brier']==pytest.approx((.01+.64+.04+.64+.64)/5)
    assert r['new_fire']['brier']==pytest.approx((.64+.04+.64+.64)/4)
    assert r['macro_event_dice']==.4 and len(r['per_event'])==2
    assert r['per_event'][0]['pooled']['average_precision']==pytest.approx(average_precision_score([1,0,1],[.9,.8,.8]))


@pytest.mark.parametrize('valid,expected_dice',[(0,None),(1,1.)])
def test_empty_conventions(tmp_path,valid,expected_dice):
    key=('SYNTHETIC','2020-01-01',0,0);acc=ft.FinalMetrics(tmp_path/'sort',[key])
    acc.update(key,np.array([.1],np.float32),np.array([0]),np.array([valid]),np.array([0]))
    r=acc.result();assert r['pooled']['average_precision'] is None and r['pooled']['dice']==expected_dice
    assert r['new_fire']['recall'] is None


@pytest.mark.parametrize('bad',[np.array([np.nan],np.float32),np.array([1.1],np.float32),np.array([.5],np.float64)])
def test_metric_bad_probability(tmp_path,bad):
    k=('SYNTHETIC','2020-01-01',0,0);acc=ft.FinalMetrics(tmp_path/'sort',[k])
    with pytest.raises(ValueError):acc.update(k,bad,np.array([0]),np.array([1]),np.array([0]))


def test_buffered_ap_global_ties_across_chunks(tmp_path):
    ap=ft.BufferedAP(tmp_path/'ap'); ap.acc.chunk_pixels=2;ap.acc.fan_in=2
    p=np.array([.5,.5,.8,.1,.5,.2,.8],np.float32);y=np.array([1,0,1,1,1,0,0])
    ap.update(p[:3],y[:3]);ap.flush();ap.update(p[3:],y[3:]);
    assert ap.compute()==pytest.approx(average_precision_score(y,p))


def test_exclusive_writes_and_safe_paths(tmp_path):
    p=tmp_path/'f.json';ft.save(p,{'old':1})
    with pytest.raises(FileExistsError):ft.save(p,{'new':2})
    assert ft.load(p)=={'old':1}
    for token in ('../2021/a.tif','C:/x.tif','/tmp/x.tif','a\\b'):
        with pytest.raises(ValueError):ft.safe_source(tmp_path,token)


def test_disabled_grant_before_output_or_pixels(tmp_path,monkeypatch):
    frozen=tmp_path/'freeze';frozen.mkdir();ft.save(frozen/'selection.json',{'test_identity':'SYNTHETIC'})
    auth=tmp_path/'disabled.json';ft.save(auth,{'authorized':False})
    with pytest.raises(PermissionError,match='disabled'):
        ft.consume_authorization(auth,frozen,None,None,tmp_path/'output',tmp_path,tmp_path,tmp_path,'cpu')
    monkeypatch.setattr(ft,'verify_freeze',lambda *a:({},[],[]))
    monkeypatch.setattr(rasterio,'open',lambda *a,**k:pytest.fail('pixel I/O forbidden'))
    with pytest.raises(PermissionError,match='NOT authorized'):
        ft.evaluate(tmp_path,frozen,tmp_path,tmp_path,tmp_path/'output',tmp_path,auth,None,None,'cpu')
    assert not (tmp_path/'output').exists()


def test_synthetic_single_use_durable_lock(tmp_path,monkeypatch):
    # This is not a real grant: synthetic identity, no TIFF paths, temp files only.
    frozen=tmp_path/'freeze';frozen.mkdir();ft.save(frozen/'selection.json',{'test_identity':'SYNTHETIC-NOT-2021-DATA'})
    from datetime import datetime,timezone
    fs=ft.sha(frozen/'selection.json')
    q=tmp_path/'q.json';ft.save(q,{'device':'cpu','status':'PASS','freeze_sha256':fs,'runtime':{}})
    pre=tmp_path/'p.json';ft.save(pre,{'checks':{k:'PASS' for k in ('frozen_identities','tiff_paths_sizes','c8_archive','ram','disk','runtime_validation_parity')},
                                    'freeze_sha256':fs,'runtime':{},'created_utc':datetime.now(timezone.utc).isoformat()})
    ledger=tmp_path/'ledger';ledger.mkdir();out=tmp_path/'output'
    a=dict(schema='geofirex-final-test-v2-authorization-1',authorized=True,
           freeze_sha256=ft.sha(frozen/'selection.json'),test_identity='SYNTHETIC-NOT-2021-DATA',
           preflight_sha256=ft.sha(pre),qualification_sha256=ft.sha(q),output=str(out.resolve()),
           scratch_root=str(tmp_path.resolve()),raster_root=str(tmp_path.resolve()),archive=str(tmp_path.resolve()),device='cpu',
           authorization_id='SYNTHETIC-UNIT-TEST',authorized_by='synthetic-fixture',authorized_at_utc='2026-11-25T00:00:00Z',training_authorized=False,
           quota_evidence='synthetic only',durable_shared_ledger_confirmed=True,runtime_mounts_verified=True,
           session_seconds_remaining=14400,quota_minutes_remaining=240,max_seconds=14400,ledger_root=str(ledger))
    grant=tmp_path/'SYNTHETIC_ONLY.json';ft.save(grant,a)
    assert ft.ephemeral_ledger(ledger)
    with pytest.raises(PermissionError,match='persistent ledger'):
        ft.consume_authorization(grant,frozen,pre,q,out,tmp_path,tmp_path,tmp_path,'cpu')
    # Only this unit fixture substitutes persistence; no production bypass flag.
    monkeypatch.setattr(ft,'ephemeral_ledger',lambda _:False)
    permit=ft.consume_authorization(grant,frozen,pre,q,out,tmp_path,tmp_path,tmp_path,'cpu')
    assert Path(permit.ledger_file).is_file()
    with pytest.raises(FileExistsError):ft.consume_authorization(grant,frozen,pre,q,out,tmp_path,tmp_path,tmp_path,'cpu')


def test_prediction_roundtrip_and_verified_archive(tmp_path):
    out=tmp_path/'result';out.mkdir();w=ft.PredictionArchive(out/'predictions')
    k=('SYNTHETIC','2020-09-01',0,0);p=np.array([.78,.12345678],np.float32);y=np.array([0,1]);v=np.ones(2);o=np.zeros(2)
    w.add(k,p,y,v,o);w.finish()
    a=np.fromfile(out/'predictions/event_000.bin',dtype=ft.PIXEL)
    assert np.array_equal(a['p'],p) and np.array_equal(a['y'],y)
    package=Path(ft.result_package(out));assert package.is_file()
    assert ft.load(package.with_suffix('.sha256.json'))['sha256']==ft.sha(package)
    with pytest.raises(FileExistsError):ft.result_package(out)


def test_budget_accounts_for_two_domains_event_ap_and_archive(synthetic):
    _,rows,_=synthetic;_,_,stats=ft.inventory(rows,['2020/fire_1'])
    plan=ft.resource_plan(stats)
    assert stats['exact_sort_scratch_upper_bytes']==4*24*(4*4096+4*4096)
    assert plan['same_filesystem_required_bytes']==plan['scratch_required_bytes']+plan['output_required_bytes']
    assert plan['output_required_bytes']>2*stats['prediction_upper_bytes']


def test_legacy_2021_guards_remain(tmp_path):
    from geofirex.protocol_v2 import V2CompactTiles
    from geofirex.v2_evaluation import V2Metrics
    from geofirex.c8_predictions import validate_keys
    with pytest.raises(ValueError):V2CompactTiles(tmp_path,tmp_path,'test')
    with pytest.raises(ValueError):V2Metrics(tmp_path/'ap',[('2021/fire_1','2021-01-06',0,0)])
    with pytest.raises(ValueError):validate_keys([('2021/fire_1','2021-01-06',0,0)])
    assert not (tmp_path/'ap').exists()


def test_2021_inventory_without_any_byte_open(tmp_path,monkeypatch):
    folder=tmp_path/'2021/fire_1';folder.mkdir(parents=True)
    rows=[]
    for i in range(6):
        day=(date(2021,1,1)+timedelta(days=i)).isoformat();p=folder/(day+'.tif');p.touch()
        rows.append(dict(event_id='2021/fire_1',date=day,relative_path=f'2021/fire_1/{day}.tif',
          height=2,width=2,bytes=0,status='ok',crs='SYNTHETIC',transform=[1],sha256='0'*64,sidecars_json='{}'))
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('metadata byte read'))
    _,keys,stats=ft.inventory(rows,['2021/fire_1'],tmp_path)
    assert len(keys)==1 and stats['raster_pixels_opened']==0


def test_metadata_column_whitelist(monkeypatch,tmp_path):
    import pyarrow.parquet as pq
    calls=[]
    class Table:
        def to_pylist(self):return []
    def read(path,columns):calls.extend(columns);return Table()
    monkeypatch.setattr(pq,'read_table',read)
    assert ft.metadata_rows(tmp_path)==[]
    assert not {'positive_pixels','valid_pixels','invalid_pixels','no_detection_pixels'}&set(calls)


def test_reader_encoder_bit_parity_to_historical(synthetic):
    from geofirex.runner_data import read_tile
    from geofirex.unet_data import Normalizer,padded_patch
    root,rows,norm=synthetic;r=ft.FinalReader(rows,root,norm)
    for row,col in [(0,0),(64,64)]:
        x,v=r._day('2020/fire_1','2020-09-03',row,col)
        raw,mask=read_tile(root/rows[2]['relative_path'],row,col,64)
        assert np.array_equal(x,padded_patch(Normalizer(**norm).transform(raw),0,0,64))
        assert np.array_equal(v,padded_patch(mask,0,0,64).astype(bool))


def test_prediction_rejects_bad_payload(tmp_path):
    w=ft.PredictionArchive(tmp_path/'pred')
    with pytest.raises(ValueError):w.add(('SYNTHETIC','2020-01-01',0,0),np.array([.5]),np.array([2]),np.array([1]),np.array([0]))
    assert not list((tmp_path/'pred').iterdir())


def test_recovery_refuses_without_original_complete_inference(tmp_path,monkeypatch):
    monkeypatch.setattr(ft,'verify_freeze',lambda *a:({'test_identity':'SYNTHETIC'},[],[]))
    with pytest.raises(FileNotFoundError):ft.recover_metrics(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_copy_preserves_exact_authorization_bytes(tmp_path):
    src=tmp_path/'source.json';src.write_bytes(b'{ "authorized" : false }\r\n')
    dst=tmp_path/'copy.json';ft.copy_exclusive(src,dst)
    assert ft.sha(src)==ft.sha(dst)
    with pytest.raises(FileExistsError):ft.copy_exclusive(src,dst)


def test_reader_rejects_future_path_disguised_as_validation(synthetic):
    root,rows,norm=synthetic;rows[0]['relative_path']='2021/fire_1/2021-01-01.tif'
    with pytest.raises(PermissionError,match='path/date/event'):ft.FinalReader(rows,root,norm)


def test_target_does_not_call_normalizer(synthetic,monkeypatch):
    root,rows,norm=synthetic;reader=ft.FinalReader(rows,root,norm)
    _,token=reader.context(('2020/fire_1','2020-09-06',0,0))
    monkeypatch.setattr(reader.normalizer,'transform',lambda *a:pytest.fail('future encoding forbidden'))
    reader.record_prediction(token,np.zeros((1,64,64),np.float32))
    y,_,_=reader.target_after_prediction(token);assert y[0,3,3]==1


def test_end_to_end_synthetic_only_no_optimizer_no_normalizer_fit(tmp_path,synthetic,monkeypatch):
    import torch
    from geofirex.benchmark_models import StackedUNet
    from geofirex import unet_data
    root,rows,norm=synthetic
    _,keys,stats=ft.inventory(rows,['2020/fire_1'])
    f=dict(test_identity='SYNTHETIC-UNIT-ONLY',keys_sha256=ft.digest(keys),selected='SYNTHETIC',inventory=stats)
    frozen=tmp_path/'freeze';frozen.mkdir();ft.save(frozen/'selection.json',f)
    (tmp_path/ft.INDEX).mkdir(parents=True);ft.save(tmp_path/ft.INDEX/'normalization.json',norm)
    auth=tmp_path/'SYNTHETIC_ONLY.json';ft.save(auth,{'authorized':True})
    q=tmp_path/'q.json';ft.save(q,{'synthetic':True})
    ledger=tmp_path/'SYNTHETIC.lock';ft.save(ledger,{'synthetic':True})
    model=StackedUNet(8,3,base_channels=8,depth=3)
    monkeypatch.setattr(ft,'verify_freeze',lambda *a:(f,rows,keys))
    monkeypatch.setattr(ft,'preflight',lambda *a:dict(checks={k:'PASS' for k in ('frozen_identities','tiff_paths_sizes','c8_archive','ram','disk','runtime_validation_parity')}))
    monkeypatch.setattr(ft,'selected_checkpoint',lambda *a:dict(model=model.state_dict()))
    monkeypatch.setattr(ft,'consume_authorization',lambda *a:ft.PixelPermit('SYNTHETIC-UNIT-ONLY',str(ledger),'SYNTHETIC-ONLY'))
    monkeypatch.setattr(torch.optim.Optimizer,'step',lambda *a,**k:pytest.fail('optimizer forbidden'))
    monkeypatch.setattr(unet_data,'fit_normalizer',lambda *a,**k:pytest.fail('refit forbidden'))
    out=tmp_path/'output'
    package=ft.evaluate(tmp_path,frozen,tmp_path,root,out,tmp_path,auth,q,q,'cpu')
    r=ft.load(out/'evaluation.json')
    assert r['status']=='COMPLETE' and r['tiles']==4 and Path(package).is_file()
    assert (out/'INFERENCE_COMPLETE.json').exists() and (out/'per_event.csv').exists()
    assert ft.load(out/'INFERENCE_COMPLETE.json')['test_observations_accessed']==0
