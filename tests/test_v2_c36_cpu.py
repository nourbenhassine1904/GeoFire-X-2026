"""C36 CPU fixtures only. No optimizer, GPU, real raster pixels or checkpoints."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_protocol_v2 import compact  # reuse existing synthetic compact fixture
from geofirex.protocol_v2 import sha256
from geofirex.v2_c36_readiness import V2Contract, PRODUCTS, validate_causal_evidence
from geofirex.v2_c36_overlay import (geometry,require_exact_geometry,require_coverage,
    TrainMoments,assemble_c36,parity_stream,COMMON_C36,fit_extra_statistics)


@pytest.fixture
def contract(compact):
    _,index,meta=compact
    observations=[]
    for role in ('train','val'):
        records=np.load(index/(role+'.npy'))
        for rec in records:
            event=meta['events'][meta['archives'][int(rec['archive'])]]
            target=date.fromordinal(int(rec['day']))
            for delta in range(-5,1):
                day=(target+timedelta(days=delta)).isoformat()
                observations.append(dict(event_id=event,date=day,relative_path=f'{event}/{day}.tif',
                    status='ok',height=64,width=64,crs='EPSG:4326',transform=[1,0,0,0,-1,64,0,0,1],
                    sha256=hashlib.sha256((event+day).encode()).hexdigest(),bytes=1))
    return V2Contract(index,observations)


def dossier(contract,tmp_path):
    rows=[]
    for i,(event,j,day,band) in enumerate(sorted(contract.required_keys())):
        start=day+'T00:00:00Z';end=day+'T08:00:00Z';available=day+'T09:00:00Z'
        kind='observed'
        if band in (3,4):
            start=(date.fromisoformat(day)-timedelta(days=16)).isoformat()+'T00:00:00Z';kind='composite'
        if band==16:
            start='2016-01-01T00:00:00Z';end='2017-01-01T00:00:00Z'
            available='2017-02-01T00:00:00Z';kind='annual'
        row=dict(event_id=event,as_of_J=j,input_date=day,band=band,
            decision_at=(date.fromisoformat(j)+timedelta(days=1)).isoformat()+'T00:00:00Z',
            observation_start=start,observation_end=end,available_at=available,kind=kind,
            source_product=PRODUCTS[band],product_version='synthetic-only-v1',
            asset_id=f'{event}/{day}/band{band}',asset_sha256=hashlib.sha256(f'{event}{day}{band}'.encode()).hexdigest(),
            raster_sha256=contract.observations[event,day]['sha256'],
            reviewer='synthetic test fixture, not a real asset reviewer',
            justification='Synthetic fixture exclusively; never qualifies real C36 data',
            reviewed=True,reviewed_at='2026-11-21T00:00:00Z',
            source_uri='https://example.invalid/synthetic-primary-record')
        if band==16:row['product_year']=2016
        primary=tmp_path/(event.replace('/','_')+'_'+day+'_'+str(band)+'_primary.txt')
        primary.write_text('SYNTHETIC TEST RECORD ONLY\n'+json.dumps({k:row[k] for k in
            ('asset_id','available_at','source_product','product_version')}),encoding='utf8')
        row.update(primary_record_path=primary.name,primary_record_sha256=sha256(primary),
                   primary_record_locator='synthetic text record, lines 1-2; not a real scientific source')
        # Canonical proof name is independent of J, as is the asset itself.
        p=tmp_path/(event.replace('/','_')+'_'+day+'_'+str(band)+'.json')
        proof={k:row[k] for k in ('asset_id','asset_sha256','source_product','product_version',
            'raster_sha256','observation_start','observation_end','available_at','source_uri',
            'primary_record_sha256','primary_record_locator')}
        p.write_text(json.dumps(proof),encoding='utf8')
        row.update(evidence_path=p.name,evidence_sha256=sha256(p));rows.append(row)
    return dict(schema='geofirex-c36-causal-evidence-1',index_sha256=contract.index_sha256,
        events_sha256=contract.events_sha256,records=rows)


def test_v2_2017_observation_in_2018_event_is_admitted_by_membership(contract,tmp_path):
    import shutil
    observations=list(contract.observations.values())
    # Windows keeps the original fixture mmap locked; mutate a fresh fixture copy.
    isolated=tmp_path/'cross_year_fixture'
    shutil.copytree(contract.index.parent,isolated)
    index=isolated/'index'
    arr=np.array(contract.records['train'],copy=True)
    arr['day']=date(2018,1,3).toordinal();np.save(index/'train.npy',arr)
    meta=deepcopy(contract.meta);meta['index_sha256']['train']=sha256(index/'train.npy')
    (index/'index.json').write_text(json.dumps(meta))
    observations=[r for r in observations if not r['event_id'].startswith('2018/')]
    for delta in range(-5,1):
        day=(date(2018,1,3)+timedelta(days=delta)).isoformat()
        observations.append(dict(event_id='2018/fire_1',date=day,relative_path=f'2018/fire_1/{day}.tif',
            status='ok',height=64,width=64,sha256='1'*64))
    c=V2Contract(index,observations)
    assert ('2018/fire_1','2018-01-02','2017-12-31',0) in c.required_keys()
    assert validate_causal_evidence(c,dossier(c,tmp_path),tmp_path)['records']==36


@pytest.mark.parametrize('path',['2021/fire_1/2021-01-01.tif','../2021/a.tif',
    '2018/fire_1/unknown.tif','C:/2021/a.tif','2018\\fire_1\\x.tif'])
def test_reserved_and_unlisted_paths_fail_before_access(contract,tmp_path,path):
    with pytest.raises(ValueError):contract.input_path(tmp_path,path)


def test_train_path_guard_rejects_validation(contract,tmp_path):
    with pytest.raises(ValueError):
        contract.input_path(tmp_path,next(iter(contract.inputs['val'])),'train')


def test_proofs_exhaustive_and_linked_to_actual_documents(contract,tmp_path):
    d=dossier(contract,tmp_path)
    r=validate_causal_evidence(contract,d,tmp_path)
    assert r['status']=='REVIEWED_EVIDENCE_CONSISTENCY_PASS' and not r['training_authorized']
    assert r['factual_truth_requires_primary_source_review']
    d['records'].pop()
    with pytest.raises(ValueError,match='incomplete'):validate_causal_evidence(contract,d,tmp_path)


@pytest.mark.parametrize('field,value',[
    ('available_at','2022-01-01T00:00:00Z'),('observation_end','2022-01-01T00:00:00Z'),
    ('available_at','2018-01-04T09:00:00'),('source_product','GRIDMET'),
    ('product_version',''),('asset_sha256','0'*64),('raster_sha256','2'*64),
    ('reviewed',False),('reviewer',''),('justification',''),('kind','forecast'),
    ('evidence_sha256','3'*64),('decision_at','2020-09-08T00:00:00Z'),
    ('primary_record_sha256','4'*64),('primary_record_locator','')])
def test_causal_refusals(contract,tmp_path,field,value):
    d=dossier(contract,tmp_path);d['records'][0][field]=value
    with pytest.raises(ValueError):validate_causal_evidence(contract,d,tmp_path)


def test_lc_same_year_unfinished_and_duplicate_key_are_rejected(contract,tmp_path):
    d=dossier(contract,tmp_path);r=next(r for r in d['records'] if r['band']==16)
    r.update(product_year=2018,observation_start='2018-01-01T00:00:00Z',
        observation_end='2019-01-01T00:00:00Z',available_at='2019-03-01T00:00:00Z')
    with pytest.raises(ValueError):validate_causal_evidence(contract,d,tmp_path)
    d=dossier(contract,tmp_path);d['records'].append(d['records'][0])
    with pytest.raises(ValueError):validate_causal_evidence(contract,d,tmp_path)


def test_asset_vintage_may_not_change_between_referring_J(contract,tmp_path):
    event,day=next(iter(contract.windows))
    contract.windows.add((event,(date.fromisoformat(day)+timedelta(days=1)).isoformat()))
    # Add the newly referenced day as certified metadata for this fixture only.
    new_day=day
    contract.observations.setdefault((event,new_day),dict(sha256='1'*64))
    d=dossier(contract,tmp_path)
    grouped={}
    for row in d['records']:
        k=(row['event_id'],row['input_date'],row['band'])
        if k in grouped:
            row['asset_id']='different-vintage';break
        grouped[k]=row
    with pytest.raises(ValueError,match='vintage'):validate_causal_evidence(contract,d,tmp_path)


def test_real_fit_is_blocked_before_any_raster_open(contract,tmp_path,monkeypatch):
    import rasterio
    def forbidden(*a,**k):pytest.fail('No real raster may open before causal evidence passes')
    monkeypatch.setattr(rasterio,'open',forbidden)
    with pytest.raises(ValueError):fit_extra_statistics(contract,tmp_path,{},tmp_path,'1'*64)


def test_real_statistics_also_require_overlay_qualification(contract,tmp_path,monkeypatch):
    import rasterio
    monkeypatch.setattr(rasterio,'open',lambda *a,**k:pytest.fail('Overlay gate must precede reads'))
    with pytest.raises(ValueError,match='overlay'):
        fit_extra_statistics(contract,tmp_path,dossier(contract,tmp_path),tmp_path,'1'*64)


def test_fitting_io_is_only_train_even_when_validation_exists(contract,tmp_path,monkeypatch):
    import rasterio
    import geofirex.protocol_v2 as protocol
    import geofirex.v2_c36_readiness as ready
    opened=[]
    monkeypatch.setattr(ready,'validate_causal_evidence',lambda *a:receipt())
    by_path={str((tmp_path/p).resolve()):obs for p,obs in contract.inputs['train'].items()}
    def fake_sha(p):
        if Path(p).name=='normalization.json':return '9'*64
        return by_path[str(p)]['sha256']
    monkeypatch.setattr(protocol,'sha256',fake_sha)
    class Source:
        height=64;width=64;crs='EPSG:4326';transform=[1,0,0,0,-1,64,0,0,1];count=23
        scales=(1.,)*23;offsets=(0.,)*23
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,bands,**kwargs):
            assert bands==[1,2,3,4,5]
            return np.ma.array(np.ones((5,64,64),np.float32))
    def source(path):
        assert str(path) in by_path,'Validation or test was opened by statistical fitting'
        opened.append(str(path));return Source()
    monkeypatch.setattr(rasterio,'open',source)
    overlay=dict(status='EXACT_CAUSAL_OVERLAY_PASS',index_sha256=contract.index_sha256,
        causal_evidence_sha256='1'*64,train_input_paths_sha256=ready.json_digest(sorted(contract.inputs['train'])))
    r=fit_extra_statistics(contract,tmp_path,{},tmp_path,'9'*64,overlay)
    assert len(opened)==3 and r['finite_count']==[3*64*64]*5
    assert r['mean']==[1.]*5 and r['std']==[1.]*5
    assert set(r['fit_paths'])==set(contract.inputs['train'])


def grid():return dict(height=2,width=3,crs='EPSG:4326',transform=[1,0,0,0,-1,2,0,0,1])


@pytest.mark.parametrize('mutation',['shape','half_pixel','resolution','crs','axis','rotation'])
def test_exact_geometry_refuses_misalignment(mutation):
    a=grid();b=deepcopy(a)
    if mutation=='shape':b['width']=4
    elif mutation=='half_pixel':b['transform'][2]=.5
    elif mutation=='resolution':b['transform'][0]=2
    elif mutation=='crs':b['crs']='EPSG:3857'
    elif mutation=='axis':b['transform'][4]=1
    else:b['transform'][1]=.01
    with pytest.raises(ValueError):require_exact_geometry(a,b)


def test_exact_geometry_accepts_equivalent_crs_and_coverage_is_ordered():
    a=grid();b=deepcopy(a);b['crs']='urn:ogc:def:crs:EPSG::4326'
    assert require_exact_geometry(a,b)['resolution']==[1.,1.]
    require_coverage([1,2],[1,2])
    for values in ([2,1],[1,1],[1],[]):
        with pytest.raises(ValueError):require_coverage([1,2],values)


def receipt():return dict(status='REVIEWED_EVIDENCE_CONSISTENCY_PASS',evidence_sha256='1'*64)


def test_train_moments_population_chunking_missing_constant_and_provenance():
    a=np.arange(30,dtype=np.float32).reshape(5,2,3);a[0,0,0]=np.nan;a[1]=3
    x=TrainMoments(['train/a']);x.add_chunks('train/a',[a[:,:1],a[:,1:]])
    r=x.result(causal_receipt=receipt(),base_normalization_sha256='2'*64,source_sha256={'train/a':'3'*64})
    assert r['fit_paths']==['train/a'] and r['finite_count']==[5,6,6,6,6]
    assert r['std'][1]==1 and r['continuous']==[0,1,2,3,4]
    np.testing.assert_allclose(r['mean'],np.nanmean(a,axis=(1,2)),rtol=1e-6)
    for i in (0,2,3,4):assert r['std'][i]==pytest.approx(float(np.nanstd(a[i])),rel=1e-6)
    assert r['base_normalization_sha256']=='2'*64 and not r['padding_included']
    with pytest.raises(ValueError):x.add('train/a',a)
    for path in ('val/a','purged/a','2021/a'):
        with pytest.raises(ValueError):x.add(path,a*100)


def test_all_missing_and_incomplete_training_are_refused():
    x=TrainMoments(['a','b']);x.add('a',np.ones((5,1,1)))
    with pytest.raises(ValueError):x.result(causal_receipt=receipt(),base_normalization_sha256='2'*64,source_sha256={})
    x=TrainMoments(['a']);x.add('a',np.full((5,1,1),np.nan))
    with pytest.raises(ValueError):x.result(causal_receipt=receipt(),base_normalization_sha256='2'*64,source_sha256={'a':'3'*64})


def tiles():
    b=np.arange(3*8*2*3,dtype=np.float32).reshape(3,8,2,3);b[:,7]=0;b[-1,7,0,1]=1
    y=np.zeros((1,2,3),np.float32);y[0,1,1]=1;v=np.ones_like(y);v[0,0,0]=0
    extra=np.ones((3,5,2,3),np.float32);extra[:,0,0,0]=np.nan;extra[:,0,1,1]=np.nan
    lc=np.ones((3,2,3),np.float32);lc[:,0,1]=2.5;lc[:,1,0]=np.nan
    c=assemble_c36(b,extra,lc,dict(mean=[1]*5,std=[1]*5,clip=8.))
    return (b,y,v,b[-1,7:8].copy()),(c,y.copy(),v.copy(),b[-1,7:8].copy())


def test_assembly_preserves_base_unknown_classes_and_missing_flags():
    b,c=tiles()
    assert np.array_equal(c[0][:,COMMON_C36],b[0])
    assert c[0][:,7,0,0].all() and c[0][:,35,0,1].all() and c[0][:,35,1,0].all()
    assert (c[0][:,0,0,0]==0).all()
    assert c[0][:,7,1,1].all() and c[2][0,1,1]==1  # missing extra never removes a valid target


def test_full_parity_stream_matches_original_domain_algorithm(tmp_path):
    from geofirex.v2_evaluation import V2Metrics
    keys=[('2020/fire_2','2020-09-07',0,0),('2020/fire_2','2020-09-08',0,0)]
    b,c=tiles();r=parity_stream(keys,[b,b],[c,c])
    acc=V2Metrics(tmp_path/'sort',keys)
    for k in keys:acc.update(k,np.zeros((2,3),np.float32),b[1][0],b[2][0],b[3][0])
    assert r['evaluation_domain_sha256']==acc.result()['evaluation_domain_sha256']
    assert r['tiles']==2


@pytest.mark.parametrize('changed',[0,1,2,3])
def test_parity_refuses_changed_shared_features_targets_masks_or_J(changed):
    b,c=tiles();altered=list(c);altered[changed]=altered[changed].copy()
    if changed==0:altered[0][0,5,0,0]+=1
    else:altered[changed][0,0,0]=1-altered[changed][0,0,0]
    with pytest.raises(ValueError):parity_stream([('2020/fire_2','2020-09-07',0,0)],[b],[altered])


def test_parity_refuses_incomplete_or_test_coverage():
    b,c=tiles();key=('2020/fire_2','2020-09-07',0,0)
    with pytest.raises(ValueError):parity_stream([key,key],[b],[c])
    with pytest.raises(ValueError):parity_stream([('2021/fire_2','2021-01-01',0,0)],[b],[c])


def test_real_contract_metadata_only_and_causal_population():
    c=V2Contract.from_repository(Path('.'))
    assert [len(c.keys[r]) for r in ('train','val')]==[115900,30882]
    assert [len(c.inputs[r]) for r in ('train','val')]==[5439,1474]
    assert len(c.required_keys())==110106
    assert len(c.required_keys((0,1,2,3,4,12,13,14,16,22)))==183510


def test_fallback_proposal_preserves_frozen_identity_and_is_disabled():
    root=Path('.')
    p=json.loads((root/'configs/c8_propagation_proposal_v1.json').read_text())
    b=json.loads((root/'configs/protocol_v2/baseline_c8.json').read_text())
    assert not p['training_authorized'] and not p['launch_enabled'] and p['test_2021_reserved']
    assert p['events_sha256']==b['event_manifest_sha256']
    assert p['index_sha256']==b['index_sha256'] and p['normalization_sha256']==b['normalization_sha256']
    for k in ('train_tiles','validation_tiles','context_days','tile_size','seed','epochs','batch_size','total_updates','threshold'):
        assert p['unchanged_protocol'][k]==b[k]
    assert p['small_variant']['source_C8_channels']==list(range(7))
    assert not p['priority_no_fit_analysis']['executed'] and not p['priority_single_future_fit']['executed']
