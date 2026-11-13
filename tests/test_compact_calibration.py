"""Numerical and isolation checks; synthetic outputs are never scientific scores."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import zipfile

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, brier_score_loss

from geofirex import compact_calibration as calibration
from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.benchmark_models import make_model
from geofirex.calibration_io import save, sha
from geofirex.fast_compact import FastCompactTiles, DTYPE


def test_counts_match_brute_force_including_float32_ties_and_empty():
    rng=np.random.default_rng(8)
    p=np.r_[rng.random(211).astype(np.float32),calibration.DENSE.astype(np.float32),[0,1]].astype(np.float32)
    y=rng.integers(0,2,len(p),dtype=np.uint8)
    accumulator=calibration.ThresholdCounts(calibration.DENSE)
    accumulator.update(p[:123],y[:123]);accumulator.update(p[123:],y[123:])
    accumulator.update(np.array([],np.float32),np.array([],np.uint8))
    for row in accumulator.rows():
        prediction=p>=np.float32(row['threshold'])
        expected=[int((prediction & (y==1)).sum()),int((prediction & (y==0)).sum()),
                  int((~prediction & (y==1)).sum()),int((~prediction & (y==0)).sum())]
        assert [row[k] for k in ('tp','fp','fn','tn')]==expected


def test_probability_metrics_match_sklearn_and_preserve_global_ties(tmp_path):
    p=np.array([.1,.1,.5,.5,.5,.9,.9,1],np.float32)
    y=np.array([0,1,0,1,1,0,1,1])
    ap=ExternalAveragePrecision(tmp_path/'ap',chunk_pixels=2,fan_in=2,retain_runs=False)
    stats=calibration.ProbabilityStats()
    for start in range(0,len(p),3):
        ap.update(p[start:start+3],y[start:start+3]);stats.update(p[start:start+3],y[start:start+3])
    assert ap.compute()==pytest.approx(average_precision_score(y,p),abs=1e-12)
    result=stats.result()
    assert result['brier_score']==pytest.approx(brier_score_loss(y,p.astype(np.float64)),abs=1e-12)
    assert sum(result['histogram_fire'])==5
    assert sum(r['count'] for r in result['reliability'])==len(p)
    assert not list((tmp_path/'ap').glob('*.bin'))


def test_selection_coarse_then_local_refinement():
    # Fine optimum .735 is inside +/- .05 of coarse best .75.
    rows=[dict(threshold=float(t),**calibration.metrics([int(1000-1000*abs(t-.735)),100,100,1000]))
          for t in calibration.DENSE]
    assert calibration.choose_threshold(rows)['threshold']==.735
    empty=[dict(threshold=float(t),**calibration.metrics([0,0,0,100])) for t in calibration.DENSE]
    with pytest.raises(ValueError,match='positive'):calibration.choose_threshold(empty)


def mini_corpus(root):
    from datetime import date,timedelta
    root.mkdir();index=root/'index';index.mkdir()
    norm=dict(fit_paths=['trusted/rasters/2018/train/2018-06-01.tif'])
    save(root/'normalization.json',norm);save(index/'normalization.json',norm)
    archives=[];events={};records={'train':[],'val':[]}
    for event,role in [('2018/train','train'),('2020/cal','val'),('2020/holdA','val'),('2020/holdB','val')]:
        archive=event+'.zip';archives.append(archive);events[archive]=event
        (root/archive).parent.mkdir(parents=True,exist_ok=True)
        day=date(int(event[:4]),6,6)
        with zipfile.ZipFile(root/archive,'w',zipfile.ZIP_DEFLATED) as z:
            import io
            for delta in range(-5,1):
                x=np.zeros((8,8,8),np.float32)
                x[:7]=.1
                x[7,2,2]=1;x[7,3,3]=1
                if delta==0:
                    x[7,3,3]=0;x[7,4,4]=1;x[7,0,0]=1
                valid=np.ones((1,8,8),bool)
                if delta==0:valid[0,0,0]=False
                if delta==-4:valid[0,1,1]=False
                buffer=io.BytesIO();np.savez_compressed(buffer,x=x,valid=valid)
                z.writestr(f'{day+timedelta(days=delta)}_0_0.npz',buffer.getvalue())
        records[role].append((len(archives)-1,day.toordinal(),0,0,1))
    for role in records:np.save(index/(role+'.npy'),np.array(records[role],dtype=DTYPE),allow_pickle=False)
    save(index/'index.json',dict(archives=archives,events=events,normalization_sha256=sha(index/'normalization.json'),
         index_sha256={role:sha(index/(role+'.npy')) for role in records}))
    return root,index


def test_pipeline_masks_lock_holdout_and_maps(tmp_path,monkeypatch):
    corpus,index=mini_corpus(tmp_path/'corpus')
    ds=FastCompactTiles(corpus,index,'val')
    x,y,v=ds[0]
    assert int(v.sum())==62  # target invalid and -4 day invalid are both excluded
    assert float(y[v.bool()].sum())==2
    args=argparse.Namespace(corpus=str(corpus),index=str(index),out=str(tmp_path/'synthetic-only'),
        device='cpu',workers=0,batch_size=2,cache_mb=1)
    torch.set_num_threads(1)
    for name in ('B1','B2'):
        torch.manual_seed(42)
        network=make_model(name,8,3,**({'architecture':'spatial_cnn_v2'} if name=='B1' else {})).eval()
        with torch.inference_mode():p=network(x[None]).sigmoid()[0].numpy()[v.numpy().astype(bool)]
        target=y.numpy()[v.numpy().astype(bool)]
        counts=calibration.ThresholdCounts([.5]);counts.update(p,target)
        validation=dict(**counts.rows()[0],pixels=62,samples_seen=1,
                        average_precision=float(average_precision_score(target,p)))
        state=dict(model=network.state_dict(),normalization=ds.normalization,updates=23404,epoch=5,cursor=0,
            history=[dict(epoch=i,validation=validation) for i in range(1,5)],
            config=dict(model=name,seed=42,context=3,target='J+1',validation_events=['2020/cal'],
                index_sha256=sha(index/'index.json'),normalization_sha256=sha(index/'normalization.json'),
                feature_policy='existing 8-channel train-only normalized float32'))
        path=tmp_path/(name+'.pt');torch.save(state,path);setattr(args,name.lower(),str(path))
    original_contract=calibration.validate_contract
    expected=dict(train_events=1,train_tiles=1,val_events=3,val_tiles=3,calibration_events=1,calibration_tiles=1)
    monkeypatch.setattr(calibration,'validate_contract',lambda states,ds,train:original_contract(states,ds,train,expected))
    evaluate=calibration.evaluate
    seen=[]
    def guarded_evaluate(model,ds,events,thresholds,args,out,label,map_events):
        lock=Path(args.out)/'thresholds_locked.json'
        if 'holdout169' in label:
            assert lock.is_file()
            thresholds_locked=json.loads(lock.read_text())['thresholds']
            name=label.split()[0]
            assert list(thresholds)==sorted(set([.5,thresholds_locked[name]]))
            assert not set(events)&{'2020/cal'}
        else:assert not lock.exists()
        seen.append(label)
        return evaluate(model,ds,events,thresholds,args,out,label,map_events)
    monkeypatch.setattr(calibration,'evaluate',guarded_evaluate)
    assets=calibration.run(args)
    assert seen==['B1 calibration32','B2 calibration32','B1 holdout169','B2 holdout169']
    assert len(assets['comparison'])==8
    for row in assets['comparison']:
        assert row['valid_pixels']==62*row['events']
    for phase in ('calibration32','holdout169'):
        for name in ('B1','B2'):
            rows=[r for r in assets['comparison'] if r['phase']==phase and r['model']==name]
            assert rows[0]['average_precision']==rows[1]['average_precision']
            assert rows[0]['brier_score']==rows[1]['brier_score']
    for case in assets['maps']:
        out=Path(args.out)/case['directory']
        with np.load(out/case['npz']) as z:
            assert np.all(z['error'][~z['valid']]==255)
            assert not z['segmentation'][~z['valid']].any()
            assert z['probability'].dtype==np.float32
        assert (out/case['layers']['errors']).is_file()
    assert (Path(args.out)/'status_complete.json').is_file()
    assert not list(Path(args.out).rglob('*.bin'))
    ds.close()


def test_bundle_compiles_and_loader_only_changes_utility_import(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('builder',root/'scripts/prepare_calibration_notebook.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    out=builder.build(tmp_path/'delivery')
    with zipfile.ZipFile(out/'calibration-code.zip') as z:
        loader_source=z.read('geofirex/fast_compact.py').decode()
        original=(root/'src/geofirex/fast_compact.py').read_text(encoding='utf-8')
        assert loader_source==original.replace('from geofirex.mission_streaming import save,sha','from geofirex.calibration_io import save,sha')
        for name in z.namelist():compile(z.read(name),name,'exec')
        isolated=tmp_path/'isolated';z.extractall(isolated)
    # The standalone bundle must import without geospatial/training dependencies.
    source = '''import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'rasterio','pyarrow','sklearn','mlflow'}:
        raise AssertionError('Unexpected dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from geofirex.compact_calibration import main
from geofirex.fast_compact import FastCompactTiles
print('standalone imports PASS')
'''
    completed=subprocess.run([sys.executable,'-I','-c',source,str(isolated)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert 'PASS' in completed.stdout
    notebook=json.loads((out/'GEOFIREX_B1_B2_CALIBRATION_20261112.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'notebook','exec')
    assert len(json.loads((out/'dataset-sources.json').read_text()))==11
    assert json.loads((out/'kernel-metadata.json').read_text())['enable_internet'] is False


def test_contract_rejects_different_checkpoint_split_and_year_leakage(tmp_path):
    corpus,index=mini_corpus(tmp_path/'corpus')
    ds=FastCompactTiles(corpus,index,'val');train=FastCompactTiles(corpus,index,'train')
    config=dict(seed=42,context=3,target='J+1',validation_events=['2020/cal'],index_sha256=sha(index/'index.json'),
                normalization_sha256=sha(index/'normalization.json'),feature_policy='normalized')
    states={name:dict(config=dict(**config,model=name),normalization=ds.normalization,updates=23404,
                     history=[{}, {}, {}, {}],cursor=0,epoch=5) for name in ('B1','B2')}
    expected=dict(train_events=1,train_tiles=1,val_events=3,val_tiles=3,calibration_events=1,calibration_tiles=1)
    assert calibration.validate_contract(states,ds,train,expected)==(['2020/cal'],['2020/holdA','2020/holdB'])
    states['B2']['config']['validation_events']=['2020/holdA']
    with pytest.raises(ValueError,match='mismatch'):calibration.validate_contract(states,ds,train,expected)
    states['B2']['config']['validation_events']=['2020/cal']
    ds.meta['events']['2020/holdA.zip']='2021/forbidden'
    with pytest.raises(ValueError,match='leakage'):calibration.validate_contract(states,ds,train,expected)
    ds.close();train.close()


def test_pinned_mounts_verify_bytes_and_reject_duplicates(tmp_path):
    import hashlib
    from geofirex.calibration_mounts import restore_verified
    maps=[];expected=[]
    logicals=['normalization.json','index/index.json']+[f'object_{i}.bin' for i in range(9)]
    for i,logical in enumerate(logicals):
        part=tmp_path/f'p{i:02d}';part.mkdir()
        payload=('existing unchanged object '+str(i)).encode()
        (part/'object.bin').write_bytes(payload)
        files=[dict(path=logical,alias='object.bin',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())]
        identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        save(part/'mount-map.json',dict(identity=identity,files=files))
        maps.append(part/'mount-map.json');expected.append(identity)
    output=restore_verified(maps,expected,tmp_path/'linked')
    assert (output/'normalization.json').read_bytes()==(maps[0].parent/'object.bin').read_bytes()
    assert (output/'index/normalization.json').read_bytes()==(output/'normalization.json').read_bytes()
    with pytest.raises(ValueError,match='identity'):restore_verified(maps+[maps[0]],expected,tmp_path/'duplicate')
    (maps[5].parent/'object.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='mismatch'):restore_verified(maps,expected,tmp_path/'tampered')
