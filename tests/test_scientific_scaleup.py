import zlib
from datetime import date, timedelta

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.scientific_scaleup import (acquisition_stages, audit_observations,
                                       load_window_manifest, summarize_audit, write_metadata_tables, feature_contracts,
                                       stage_download_entries)
from geofirex.runner_data import TemporalStore
from geofirex.unet_data import Normalizer
from geofirex.benchmark import event_assignments


def fixture(tmp_path, days=7, invalid=False, fire=False):
    rows, entries = [], []
    for i in range(days):
        day = (date(2018,1,1)+timedelta(days=i)).isoformat()
        path = tmp_path/f'{day}.tif'
        raw = np.full((23,4,5),np.nan,np.float32)
        raw[12:15] = 1
        if fire: raw[22,0,0] = 0  # finite zero is a detection code, not background
        with rasterio.open(path,'w',driver='GTiff',height=4,width=5,count=23,
                           dtype='float32',crs='EPSG:32610',transform=from_origin(10,10,1,1),
                           nodata=np.nan) as ds:
            ds.write(raw)
            ds.descriptions = RAW_BAND_NAMES
            if invalid: ds.write_mask(np.zeros((4,5),np.uint8))
        rows.append(dict(event_id='2018/fire_1',date=day,relative_path=str(path),height=4,width=5))
        body=path.read_bytes()
        entries.append(dict(event_id='2018/fire_1',date=day,year=2018,
                            path=f'2018/fire_1/{day}.tif',compressed_bytes=len(body),
                            geotiff_bytes=len(body),crc32=f'{zlib.crc32(body):08x}'))
    return rows,dict(entries=entries)


@pytest.mark.parametrize('invalid,fire,positive,valid',[(False,False,0,20),(True,True,0,0),(False,True,1,20)])
def test_zero_invalid_and_finite_detection(tmp_path,invalid,fire,positive,valid):
    rows,index=fixture(tmp_path,invalid=invalid,fire=fire)
    records=audit_observations(rows,index,tmp_path,block_size=2)
    assert all(r['positive_pixels']==positive and r['valid_pixels']==valid for r in records)
    summary=summarize_audit(records,index)
    assert summary['window_counts']['train']['1']['windows']==6
    assert summary['window_counts']['train']['5']['windows']==2
    assert summary['events'][0]['complete']
    assert summary['events'][0]['entirely_invalid_days']==(7 if invalid else 0)


def test_missing_corrupt_and_incomplete(tmp_path):
    rows,index=fixture(tmp_path)
    # Fixture mutation only; production auditing is read-only.
    (tmp_path/'2018-01-02.tif').unlink()
    (tmp_path/'2018-01-04.tif').write_bytes(b'broken')
    records=audit_observations(rows,index,tmp_path)
    assert records[1]['status']=='missing' and records[1]['positive_pixels'] is None
    assert records[3]['status']=='corrupt' and records[3]['positive_pixels'] is None
    summary=summarize_audit(records,index)
    assert not summary['events'][0]['complete']
    assert summary['events'][0]['gaps']
    assert summary['window_counts']['train']['3']['windows']==0


def test_duplicate_and_test_lock(tmp_path):
    rows,index=fixture(tmp_path)
    with pytest.raises(ValueError,match='Duplicate'):
        audit_observations(rows+[rows[0]],index,tmp_path)
    rows[0]['event_id']='2021/fire_1'
    index['entries'][0]['event_id']='2021/fire_1'
    with pytest.raises(ValueError,match='locked'):
        audit_observations(rows,index,tmp_path)


def test_parquet_to_temporal_store_and_tampering(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows,index=fixture(tmp_path,fire=True)
    records=audit_observations(rows,index,tmp_path)
    summary=summarize_audit(records,index)
    pq.write_table(pa.Table.from_pylist(summary['aligned_windows']),tmp_path/'windows.parquet')
    loaded=pq.read_table(tmp_path/'windows.parquet').to_pylist()
    assignment=event_assignments(['2018/fire_1'],0)
    windows=load_window_manifest(loaded,records,assignment,'train',3)
    norm=Normalizer([12,14],[0,0],[1,1],[20,20],[],False)
    store=TemporalStore(windows,norm)
    x,y,valid,_=store.tile(0,0,0,4)
    assert x.shape==(3,8,4,4) and y.sum()==1 and valid.sum()==16
    loaded[0]['target_date']='2018-12-31'
    with pytest.raises(ValueError,match='canonical'):
        load_window_manifest(loaded,records,assignment,'train',1)


def test_nested_plans_exact_and_no_test():
    entries=[]
    for year,n in [(2018,6),(2019,3),(2020,5),(2021,2)]:
        for i in range(n):
            event=f'{year}/fire_{i}'
            entries.append(dict(path=f'{event}/{year}-01-01.tif',event_id=event,date=f'{year}-01-01',
                                year=year,compressed_bytes=10,geotiff_bytes=20,crc32='00000000'))
    plans=acquisition_stages(dict(entries=entries),[entries[0]['path']],targets=(4,9,14))
    assert [p['target_events'] for p in plans]==[4,9,14]
    assert set(plans[0]['events']) < set(plans[1]['events']) < set(plans[2]['events'])
    for p in plans:
        assert p['compressed_payload_bytes']==10*p['target_events']
        assert p['additional_payload_bytes']==10*(p['target_events']-1)
        assert not any(e.startswith('2021') for e in p['events'])
        assert p['verified_reused_files']==1 and not p['approved']
    assert plans==acquisition_stages(dict(entries=entries),[entries[0]['path']],targets=(4,9,14))


def test_empty_parquet_schema_and_no_overwrite(tmp_path):
    import pyarrow.parquet as pq
    write_metadata_tables([],[],tmp_path)
    assert pq.read_table(tmp_path/'observations.parquet').num_rows==0
    assert pq.read_table(tmp_path/'windows.parquet').schema.field('context_paths').type.value_type
    with pytest.raises(FileExistsError): write_metadata_tables([],[],tmp_path)


def test_observation_parquet_with_empty_sidecars_and_corrupt_raster(tmp_path):
    rows,index=fixture(tmp_path,days=1)
    records=audit_observations(rows,index,tmp_path)
    write_metadata_tables(records,[],tmp_path)
    path=tmp_path/'2018-01-01.tif'
    path.write_bytes(b'not a raster but correct index checksum')
    index['entries'][0].update(geotiff_bytes=path.stat().st_size,crc32=f'{zlib.crc32(path.read_bytes()):08x}')
    assert audit_observations(rows,index,tmp_path)[0]['status']=='corrupt'


def test_feature_contracts_are_ordered_and_do_not_authorize_modalities():
    contracts=feature_contracts()
    assert {k:v['channels'] for k,v in contracts.items()} == {
        'conservative_fire_terrain':8,'terrain_vegetation_v1':36,'multimodal_observed_j_v1':50}
    assert contracts['conservative_fire_terrain']['channel_names'][-1]=='fire_J'
    for name,profile in contracts.items():
        assert profile == feature_contracts()[name]
        assert len(set(profile['channel_names']))==profile['channels']
        if name != 'conservative_fire_terrain': assert profile['availability'].startswith('BLOCKED')


def test_reuse_across_roots_and_plan_tampering(tmp_path):
    rows,index=fixture(tmp_path,days=1)
    entry=index['entries'][0]
    target=tmp_path/entry['path']
    target.parent.mkdir(parents=True)
    target.write_bytes((tmp_path/'2018-01-01.tif').read_bytes())
    plan=acquisition_stages(index,[entry['path']],targets=(1,))[0]
    missing,reused=stage_download_entries(plan,index,[tmp_path/'absent',tmp_path])
    assert not missing and len(reused)==1
    plan['target_events']=2
    with pytest.raises(ValueError,match='identity'): stage_download_entries(plan,index,[tmp_path])
    plan=acquisition_stages(index,[entry['path']],targets=(1,))[0]
    target.write_bytes(b'bad')
    with pytest.raises(ValueError,match='preserved'): stage_download_entries(plan,index,[tmp_path])


def test_window_loader_requires_qc_and_annual_split(tmp_path):
    rows,index=fixture(tmp_path)
    records=audit_observations(rows,index,tmp_path)
    windows=summarize_audit(records,index)['aligned_windows']
    with pytest.raises(ValueError,match='annual'):
        load_window_manifest(windows,records,{'2018/fire_1':'val'},'val',3)
    records[0]['status']='corrupt'
    with pytest.raises(ValueError,match='QC'):
        load_window_manifest(windows,records,{'2018/fire_1':'train'},'train',3)
