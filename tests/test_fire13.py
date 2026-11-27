"""FIRE13 targeted CPU tests: inputs, causal reads, exact parity and safeguards."""
from copy import deepcopy
from datetime import date
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
import pytest
import torch
from geofirex.fire13_features import (Fire13Tiles,daily_maps,descriptors,normalize,fit_moments,
    make_fire13_loaders,numerical_selftest,NAMES)
from geofirex.fast_compact import DTYPE,LocalitySampler
from geofirex.unet import SegmentationLoss
from geofirex import fire13_experiment as exp


class SceneFixture(torch.utils.data.Dataset):
    role='train'
    def __init__(self):
        self.day=date(2020,9,7).toordinal();self.normalization={'fixture':True}
        self.records=np.array([(0,self.day,0,c,1) for c in (0,64)],dtype=DTYPE)
        self.meta=dict(archives=['fixture.zip'],events={'fixture.zip':'2020/fire_1'},event_roles={'2020/fire_1':'train'})
        self.paths=['fixture'];self.calls=[];self.target=0.;self.target_valid=1.
    def __len__(self):return len(self.records)
    def event(self,i):return '2020/fire_1'
    def _day(self,a,day,row,col):
        assert a==0 and day in (self.day-3,self.day-2,self.day-1),'Target/future accessed by descriptor'
        self.calls.append((a,day,row,col));x=np.zeros((8,64,64),np.float32);v=np.ones((1,64,64),bool)
        t=day-(self.day-3)
        for yy,xx in [(40,65),(40,65+t)]:
            if row<=yy<row+64 and col<=xx<col+64:x[7,yy-row,xx-col]=1
        x[0]=t;x[1]=row+col
        return x,v
    def __getitem__(self,i):
        r=self.records[i];x=np.stack([self._day(0,self.day+d,int(r['row']),int(r['col']))[0] for d in (-3,-2,-1)])
        return torch.from_numpy(x),torch.full((1,64,64),self.target),torch.full((1,64,64),self.target_valid)
    def close(self):pass


@pytest.fixture
def fixture():
    base=SceneFixture();g={'2020/fire_1':dict(shape=[128,128])}
    return Fire13Tiles(base,g,dict(mean=[0.,0.],std=[1.,1.],fit_role='train'))


def test_descriptor_numerics_empty_fire_edges_and_holes():
    assert numerical_selftest()['status']=='PASS'
    f=np.zeros((64,64),np.uint8);v=np.ones_like(f);v[32,32]=0
    maps=daily_maps(f,v)
    assert np.all(maps[1]==8) and np.all(maps[2]==0)
    assert not maps[3][24:41,24:41].any() and maps[3][20,20]
    raw=descriptors([maps]*3)
    assert not raw[:,:4,:,0].any() and not raw[:,4,:,0].any()
    assert np.isfinite(normalize(raw,dict(mean=[2,.1],std=[1,.2]))).all()


def test_scene_halo_crosses_tile_boundaries_exactly(fixture):
    left=fixture.raw_tile(0);right=fixture.raw_tile(1)
    assert left[0,0,40,63]==2 and right[0,0,40,0]==1
    assert left[0,1,40,63]==pytest.approx(1/49)
    full=descriptors(fixture.context_maps(0,fixture.base.day))
    assert np.array_equal(left,full[:,:,:64,:64])
    assert np.array_equal(right,full[:,:,:64,64:128])


def test_recent_appearance_and_signed_change_are_observations_only(fixture):
    raw=fixture.raw_tile(1)
    assert raw[0,2].sum()==0 and raw[0,3].sum()==0
    assert raw[1,2,40,2]==1 and raw[2,2,40,3]==1
    assert raw[2,3,40,3]==1
    assert all(day<fixture.base.day for _,day,_,_ in fixture.base.calls)


def test_targets_and_target_masks_cannot_change_features_or_moments(fixture):
    a=fixture[0];stats=fit_moments(fixture)
    fixture.base.target=1.;fixture.base.target_valid=0.
    b=fixture[0];other=fit_moments(fixture)
    assert torch.equal(a[0],b[0]) and not torch.equal(a[1],b[1]) and not torch.equal(a[2],b[2])
    for k in ('mean','std','count','windows'):assert stats[k]==other[k]


def test_stats_do_not_call_target_reader_and_exclude_padding(fixture,monkeypatch):
    monkeypatch.setattr(SceneFixture,'__getitem__',lambda *a:pytest.fail('Statistics may not read targets'))
    stats=fit_moments(fixture)
    assert stats['count']==3*112*112 and stats['window_count']==1
    assert stats['fit_role']=='train'
    raw=descriptors(fixture.context_maps(0,fixture.base.day));mask=raw[:,4].astype(bool)
    expected=raw[:,:2].transpose(1,0,2,3)[:,mask].astype(np.float64)
    np.testing.assert_allclose(stats['mean'],expected.mean(axis=1),atol=1e-12)
    np.testing.assert_allclose(stats['std'],expected.std(axis=1),atol=1e-12)


@pytest.mark.parametrize('role',['val','test','2021','purged'])
def test_nontrain_cannot_fit_statistics_before_any_read(fixture,role):
    fixture.role=role
    with pytest.raises(ValueError,match='train only'):fit_moments(fixture)
    assert not fixture.base.calls


@pytest.mark.parametrize('day_offset',[0,1,365])
def test_future_or_target_descriptor_reads_refused(fixture,day_offset):
    with pytest.raises(ValueError):fixture._scene(0,fixture.base.day+day_offset,(fixture.base.day-3,fixture.base.day-2,fixture.base.day-1))
    assert not fixture.base.calls


def test_2021_refused_even_if_caller_forges_allowed_dates(fixture):
    day=date(2021,1,1).toordinal()
    with pytest.raises(ValueError):fixture._scene(0,day,(day,))
    assert not fixture.base.calls


def test_base_channels_targets_masks_and_sampler_are_unchanged(fixture):
    bx,by,bv=fixture.base[0];cx,cy,cv=fixture[0]
    assert cx.shape==(3,13,64,64) and torch.equal(bx,cx[:,:8])
    assert torch.equal(by,cy) and torch.equal(bv,cv)
    for epoch in range(4):
        a,b=LocalitySampler(fixture.base),LocalitySampler(fixture);a.epoch=b.epoch=epoch
        assert np.array_equal(a.order(),b.order()) and len(set(b.order()))==len(fixture)
    restored=pickle.loads(pickle.dumps(fixture))
    assert restored.cache_bytes==0 and all(torch.equal(a,b) for a,b in zip(fixture[0],restored[0]))


@pytest.mark.parametrize('empty',[False,True])
def test_model_shape_parameters_and_historical_loss_gradients(empty):
    torch.set_num_threads(2);torch.manual_seed(42)
    c={'base_channels':8,'depth':3};model=exp.make_model(c,'cpu')
    assert sum(p.numel() for p in model.parameters())==125057
    x=torch.zeros(2,3,13,32,32);y=torch.zeros(2,1,32,32);v=torch.zeros_like(y) if empty else torch.ones_like(y)
    z=model(x);a=exp.make_criterion()(z,y,v);b=SegmentationLoss(.5,10)(z,y,v)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z,retain_graph=True)[0])
    a.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_exact_integer_fp_choice_keeps_ties_indivisible(tmp_path):
    from geofirex.c8_predictions import PredictionWriter,PredictionStore
    from geofirex.fire13_comparison import exact_fp_choice
    key=('2020/fire_1','2020-09-07',0,0);w=PredictionWriter(tmp_path/'p',[key],{},(2,3))
    p=np.array([[.9,.8,.8],[.7,.6,.1]],np.float32);y=np.array([[1,0,0],[1,0,1]],np.uint8)
    w.add(key,p,y,np.ones_like(y),np.zeros_like(y));w.finish();s=PredictionStore(tmp_path/'p')
    r=exact_fp_choice(s,s.events,1)
    assert r['threshold']==pytest.approx(.9) and r['calibration_fp']==0 and r['calibration_tp']==1
    with pytest.raises(ValueError):exact_fp_choice(s,s.events,-1)


def test_comparable_fp_all_thresholds_locked_before_assessment(tmp_path,monkeypatch):
    from geofirex.c8_predictions import PredictionWriter,PredictionStore
    from geofirex import fire13_comparison as comp
    keys=[(f'2020/fire_{i}','2020-09-07',0,0) for i in range(80)]
    w=PredictionWriter(tmp_path/'predictions',keys,{'synthetic':True},(2,3))
    p=np.array([[.9,.8,.8],[.7,.6,.1]],np.float32);y=np.array([[1,0,0],[1,0,1]],np.uint8)
    for k in keys:w.add(k,p,y,np.ones_like(y),np.zeros_like(y))
    w.finish();store=PredictionStore(tmp_path/'predictions')
    (tmp_path/'exact').mkdir()
    policy=exp.read('reports/protocol_v2_execution/propagation_policy.json')
    base=exp.exact_metrics(store,tmp_path/'exact',policy)
    read=exp.read
    def fixture_read(path):
        if Path(path).name=='baseline_reference.json':return base
        if Path(path).name=='comparable_fp_policy.json':return dict(FPR_rational=[1,100])
        return read(path)
    monkeypatch.setattr(exp,'read',fixture_read)
    original=comp.event_counts;locks=[]
    def assess(*a,**kw):
        lock=tmp_path/'paired/thresholds_locked.json'
        assert lock.exists();locks.append(exp.sha256(lock))
        return original(*a,**kw)
    monkeypatch.setattr(comp,'event_counts',assess)
    comp.comparable_fp('.',store,tmp_path/'paired')
    result=read(tmp_path/'paired/result.json')
    assert len(set(locks))==1 and len(locks)==5 and len(result['per_event'])==80
    assert not result['automatic_promotion'] and result['acceptance_criteria_unchanged']
    selections=read(tmp_path/'paired/thresholds_locked.json')['selections']
    assert all(len(s['calibration_events'])==64 and len(s['assessment_events'])==16
        and not set(s['calibration_events'])&set(s['assessment_events']) for s in selections)


def test_scientific_contract_real_parity_and_train_stats():
    base,c,_,lock,u=exp.identity('.')
    assert c['loss']==base['loss'] and c['sampler']==base['sampler']
    assert c['channel_names'][:8]==base['channel_names'] and c['channel_names'][8:]==NAMES
    assert (c['channels_per_day'],c['model_input_channels'],u['total_updates'])==(13,39,28976)
    stats=exp.read(exp.REPORT/'derived_normalization.json')
    assert stats['fit_role']=='train' and stats['window_count']==4803
    parity=exp.read(exp.REPORT/'parity_full.json')
    assert parity['status']=='PASS' and parity['roles']['val']['evaluation_domain_sha256']==exp.DOMAIN
    assert exp.read(exp.REPORT/'acceptance.json')==exp.read('reports/newpos2_20261122_v1/acceptance.json')


def test_real_runner_loaders_emit_13_channels_and_fit_uses_them():
    import inspect
    torch.set_num_threads(2)
    root=Path('.').resolve()
    maps=[root/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    mounts,_=exp.old.verify_mounts(root,maps)
    train,val=exp.loaders(root,mounts,workers=0)
    try:
        x,y,v=next(iter(train))
        assert x.shape==(16,3,13,64,64) and y.shape==v.shape==(16,1,64,64)
        assert x.dtype==torch.float32 and torch.isfinite(x).all()
        idx=int(train.sampler.order()[0]);bx,by,bv=train.dataset.base[idx]
        assert torch.equal(x[0,:,:8],bx) and torch.equal(y[0],by) and torch.equal(v[0],bv)
        assert np.array_equal(train.sampler.order(),LocalitySampler(train.dataset.base).order())
        assert len(val.dataset)==30882
    finally:train.dataset.close();val.dataset.close()
    source=inspect.getsource(exp.fit)
    assert 'train,val=loaders(root,mounts,workers=2)' in source
    assert 'NewPositiveLoss' not in source
    assert "criterion(model(x.to('cuda')),y.to('cuda'),v.to('cuda'))" in source
    assert 'ds=Fire13Tiles(' in inspect.getsource(exp.predict_validation)


@pytest.mark.parametrize('field,value',[('authorized',False),('experiment_id','B2_C8_V2_PROP_HN2_seed42'),
    ('action','newpos2-fit'),('runs',2),('execution_manifest_sha256','0'*64),('device','cpu'),
    ('session_remaining_minutes',299),('quota_remaining_card_hours',9.9),('card_hours_reserved',None)])
def test_bad_grant_refused_before_cuda_or_output(tmp_path,monkeypatch,field,value):
    a=exp.read(exp.REPORT/'authorization_fire13-fit_DISABLED.json')
    a.update(authorized=True,authorized_by='SYNTHETIC',authorized_at='fixture',quota_checked_at='fixture',
        output_parent=str(tmp_path),session_remaining_minutes=300,quota_remaining_card_hours=10,
        card_hours_reserved=10,local_backup_destination='synthetic-local',external_backup_destination='synthetic-external')
    a[field]=value;p=tmp_path/'grant.json';p.write_text(json.dumps(a))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No CUDA without valid grant'))
    with pytest.raises(ValueError):exp.fit('.',[],tmp_path/exp.B_ID,p)
    assert not (tmp_path/exp.B_ID).exists()


def test_notebook_disabled_payload_pinned_compilable_and_complete(tmp_path):
    nb=exp.read('notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb');build=exp.read(exp.REPORT/'notebook_build.json');data=None
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        s=''.join(cell['source']);tree=ast.parse(s);compile(s,'notebook','exec')
        assert cell['outputs']==[] and cell['execution_count'] is None
        if 'RUN_GPU_PREFLIGHT = False' in s:
            assert all(f'{flag} = False' in s for flag in ('RUN_GPU_PREFLIGHT','RUN_FIT','RUN_RECOVERY'))
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATA' for t in node.targets):data=base64.b64decode(node.value.args[0].value)
    assert hashlib.sha256(data).hexdigest()==build['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert not any(n.endswith(('.pt','.tif','.bin')) for n in z.namelist());z.extractall(tmp_path/'payload')
    assert exp.identity(tmp_path/'payload')[1]['channels_per_day']==13
    assert exp.sha256('notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb')==build['notebook_sha256']
