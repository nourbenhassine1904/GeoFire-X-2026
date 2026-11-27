"""Small CPU fixtures only. No inference, optimizer, GPU or reserved pixels."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from geofirex import newpos4_posthoc as n
from geofirex import newpos4_analysis as a
from geofirex.c8_predictions import PredictionWriter
from geofirex.c8_thresholds import choose_threshold
from geofirex.newpos4_studio import error_map
from geofirex import v2_execution as old


def fixture(tmp_path,seed=42):
    source=tmp_path/'fixture';source.mkdir();rng=np.random.default_rng(seed)
    keys=[(f'2020/fire_{i}','2020-09-05',0,0) for i in range(5)]
    writer=PredictionWriter(source/'evaluation/predictions',keys,dict(run_id=n.RUN,checkpoint_sha256=n.CHECKPOINT))
    arrays=[]
    for key in keys:
        p=rng.integers(0,5001,(64,64)).astype(np.float32)/np.float32(5000)
        y=(rng.random((64,64))<.1).astype(np.uint8);v=np.ones_like(y);o=np.zeros_like(y)
        writer.add(key,p,y,v,o);arrays.append((p,y))
    writer.finish();archive=old.package(source,'COMPLETE')
    return n.ArchivedPredictions(archive,keys,expected_domain=None),archive,arrays


def test_archive_sha_complete_keys_headers_and_bounded_streaming(tmp_path):
    store,path,arrays=fixture(tmp_path)
    receipt=n.verify_archive(path,path.with_suffix('.sha256'));assert receipt['all_members_verified']
    chunks=list(store.chunks(store.events));assert max(len(v) for _,v in chunks)<=65536
    assert sum(len(v) for _,v in chunks)==5*4096
    for (_,tile),(p,y) in zip(store.tiles(),arrays):
        np.testing.assert_array_equal(tile['p'],p);np.testing.assert_array_equal(tile['y'],y)
    with pytest.raises(ValueError):list(store.chunks(store.events,65537))
    with pytest.raises(ValueError):n.ArchivedPredictions(path,expected_domain='bad')
    store.deadline=0
    with pytest.raises(TimeoutError):next(store.chunks(store.events))
    wrong=tmp_path/'bad.sha256';wrong.write_text('0'*64)
    with pytest.raises(ValueError):n.verify_archive(path,wrong)


@pytest.mark.parametrize('rate',[(7,50000),(7,25000),(7,12500)])
@pytest.mark.parametrize('seed',[42,987])
def test_sensitivity_thresholds_exact_global_ties_against_independent_sort(tmp_path,rate,seed):
    store,_,_=fixture(tmp_path,seed);events=store.events[:4]
    result=n.select_threshold(store,events,rate);p=[];y=[]
    for _,x in store.chunks(events):p.extend(x['p']);y.extend(x['y'])
    p=np.array(p);y=np.array(y);score,inv,ct=np.unique(p,return_inverse=True,return_counts=True)
    positives=np.bincount(inv,weights=y,minlength=len(score))[::-1]
    tp=np.cumsum(positives);fp=np.cumsum(ct[::-1])-tp
    allowed=fp<=result['fp_budget'];best=int(tp[allowed].max()) if allowed.any() else 0
    expected=float(score[::-1][np.flatnonzero(allowed&(tp==best))[0]]) if best else None
    assert result['threshold']==expected and result['calibration_tp']==best
    if rate==(7,25000):assert result['threshold']==choose_threshold(store,events)['threshold']


def test_all_fifteen_decisions_locked_before_any_assessment(tmp_path,monkeypatch):
    store,_,_=fixture(tmp_path);original=n.event_counts;seen=[];out=tmp_path/'costs'
    def assess(*args,**kwargs):
        p=out/'thresholds_locked.json';lock=json.loads(p.read_text())
        assert len(lock['selections'])==15
        assert all(set(r['calibration_events']).isdisjoint(r['assessment_events']) for r in lock['selections'])
        seen.append(hashlib.sha256(p.read_bytes()).hexdigest());return original(*args,**kwargs)
    monkeypatch.setattr(n,'event_counts',assess)
    reference={'new_fire':{'recall':.1,'dice':.1},'pooled':{'dice':.4}}
    result=n.crossfit_costs(store,n.cost_policy(),out,reference)
    assert len(set(seen))==1 and len(result['scenarios'])==3
    assert result['selected_scenario'] is None and not result['promotion_authorized']


def test_fixed_bin_calibration_uses_saved_scores_and_brier(tmp_path):
    store,_,arrays=fixture(tmp_path);result=n.reliability(store)
    p=np.concatenate([x.ravel() for x,y in arrays]).astype(np.float64);y=np.concatenate([y.ravel() for x,y in arrays])
    assert result['new_fire']['brier']==pytest.approx(float(np.square(p-y).mean()),abs=1e-14)
    assert sum(r['n'] for r in result['pooled']['bins'])==5*4096
    assert len(result['pooled']['bins'])==20


def test_studio_new_categories_exclude_observed_and_invalid():
    from geofirex.c8_predictions import PIXEL
    x=np.zeros(6,dtype=PIXEL);x['p']=[.9,.9,.1,.1,.9,.9];x['y']=[1,0,1,0,1,1];x['v']=1;x['v'][4]=0;x['o'][5]=1
    error,mask,pred,truth=error_map(x,.78)
    assert error.tolist()==[1,2,3,0,255,255]
    assert not mask[4:].any() and pred.sum()==2 and truth.sum()==2


def test_resource_guard_qualifies_bounded_reader_and_refuses_low_reserve(tmp_path,monkeypatch):
    monkeypatch.setattr(a.c.old,'host_memory',lambda:(16*1024**3,700*1024**2))
    monkeypatch.setattr(a,'memory_rss',lambda:(200*1024**2,200*1024**2))
    with pytest.raises(RuntimeError):a.ResourceGuard(tmp_path)
    monkeypatch.setattr(a.c.old,'host_memory',lambda:(16*1024**3,1024**3))
    guard=a.ResourceGuard(tmp_path);guard()
    monkeypatch.setattr(a.c.old,'host_memory',lambda:(16*1024**3,500*1024**2))
    with pytest.raises(RuntimeError):guard()
    monkeypatch.setattr(a.c.old,'host_memory',lambda:(16*1024**3,1024**3))
    monkeypatch.setattr(a,'memory_rss',lambda:(400*1024**2,400*1024**2))
    with pytest.raises(RuntimeError):guard()


def test_cpu_notebook_offline_disabled_and_payload_verified():
    import ast,base64,io
    path=Path('notebooks/GEOFIREX_NEWPOS4_THRESHOLDS_CPU_DISABLED.ipynb')
    nb=json.loads(path.read_text(encoding='utf8'));found=False
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        code=''.join(cell['source']);compile(code,'CPU notebook','exec')
        assert 'newpos4-fit' not in code and 'infer_thresholds(' not in code
        if 'RUN_ANALYSIS = False' in code:found=True;assert "CUDA_VISIBLE_DEVICES']=''" in code
        if 'DATA=base64.b64decode(' in code:
            tree=ast.parse(code)
            literal=next(s.value.args[0].value for s in tree.body if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATA' for t in s.targets))
            data=base64.b64decode(literal)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                pins=json.loads(z.read('reports/newpos4_thresholds_20261121_v1/analysis_inputs.json'))
                for name,digest in pins['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
                assert not any(name.endswith(('.pt','.tif')) for name in z.namelist())
    assert found
