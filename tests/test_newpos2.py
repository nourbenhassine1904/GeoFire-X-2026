"""Targeted NEWPOS2 qualification. Only synthetic train/val pixels and CPU work."""
from copy import deepcopy
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
import torch
from geofirex.unet import SegmentationLoss
from geofirex.c8_newpos_loss import NewPositiveLoss
from geofirex import newpos2_experiment as exp
from test_protocol_v2 import compact


def test_runner_criterion_is_factor2_and_loss_does_not_mutate_covariates_or_targets(compact):
    corpus,index,_=compact
    baseline_train,baseline_val=exp.make_v2_loaders(corpus,index)
    candidate_train,candidate_val=exp.make_v2_loaders(corpus,index)
    try:
        for a,b in zip((baseline_train,baseline_val),(candidate_train,candidate_val)):
            assert exp.sample_keys(a.dataset)==exp.sample_keys(b.dataset)
            aa,bb=next(iter(a)),next(iter(b))
            assert all(torch.equal(x,y) for x,y in zip(aa,bb))
            x,y,v=bb;before=[t.clone() for t in bb]
            logits=torch.zeros_like(y,requires_grad=True)
            criterion=exp.make_criterion()
            assert criterion.new_positive_weight==2 and criterion.pos_weight==10 and criterion.dice_weight==.5
            criterion(logits,y,v,x[:,-1,7:8]).backward()
            assert torch.isfinite(logits.grad).all()
            assert all(torch.equal(x,y) for x,y in zip(before,bb))
    finally:
        for loader in (baseline_train,baseline_val,candidate_train,candidate_val):loader.dataset.close()


def test_frozen_scientific_contract_and_all_sampler_orders_without_pixels():
    from geofirex.fast_compact import LocalitySampler
    base,c,_,lock,u=exp.identity('.')
    assert len(exp.read(exp.old.REPORT/'execution_manifest.json')['files'])==77
    differences={k for k in set(base)|set(c) if base.get(k)!=c.get(k)}
    assert differences=={'run_id','loss','base_config_sha256','status'}
    assert c['training_authorized'] is False and c['launch_enabled'] is False
    assert (c['epochs'],c['total_updates'],c['threshold'],c['model_input_channels'])==(4,28976,.78,24)
    meta=exp.read(Path(base['index'])/'index.json')
    from collections import Counter
    assert Counter(meta['event_roles'].values())==dict(train=318,val=80,purged=53,test=156)
    records=np.load(Path(base['index'])/'train.npy',mmap_mode='r',allow_pickle=False)
    class Records:
        def __init__(self):self.records=records
        def __len__(self):return len(records)
    a,b=LocalitySampler(Records(),seed=42),LocalitySampler(Records(),seed=42)
    for epoch in range(4):
        a.epoch=b.epoch=epoch
        assert np.array_equal(a.order(),b.order())
        assert np.array_equal(np.sort(a.order()),np.arange(115900))
    assert len(records)%16==12
    assert exp.make_v2_loaders is __import__('geofirex.protocol_v2',fromlist=['make_v2_loaders']).make_v2_loaders


def grant(tmp_path):
    a=exp.read(exp.REPORT/'authorization_newpos2-fit_DISABLED.json')
    a.update(authorized=True,authorized_by='SYNTHETIC ONLY',authorized_at='fixture',quota_checked_at='fixture',
             output_parent=str(tmp_path),session_remaining_minutes=225,quota_remaining_card_hours=7.5,
             card_hours_reserved=7.5,local_backup_destination='synthetic-local',external_backup_destination='synthetic-external')
    p=tmp_path/'grant.json';p.write_text(json.dumps(a));return p,a


@pytest.mark.parametrize('field,value',[('authorized',False),('experiment_id','B2_C8_V2_NEWPOS4_seed42'),
    ('action','newpos4-fit'),('runs',2),('execution_manifest_sha256','0'*64),('device','cpu'),
    ('session_remaining_minutes',224),('quota_remaining_card_hours',7.4),('card_hours_reserved',None),
    ('authorized_by',''),('authorized_at',''),('quota_checked_at',''),('output_parent','wrong'),
    ('local_backup_destination',''),('external_backup_destination','synthetic-local')])
def test_bad_grants_refused_before_gpu_or_output(tmp_path,monkeypatch,field,value):
    p,a=grant(tmp_path);a[field]=value;p.write_text(json.dumps(a))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('CUDA queried before grant'))
    with pytest.raises(ValueError):exp.fit('.',[],tmp_path/exp.B_ID,p)
    assert not (tmp_path/exp.B_ID).exists()


def test_atomic_lock_survives_deleted_run_directory(tmp_path):
    p=exp.new_output(tmp_path/exp.B_ID,exp.B_ID);p.rmdir()
    with pytest.raises(FileExistsError):exp.new_output(p,exp.B_ID)


def test_newpos2_rejects_wrong_scientific_config_before_execution(tmp_path,monkeypatch):
    original=exp.read
    def changed(p):
        r=original(p)
        if str(p).replace('\\','/').endswith('configs/c8_newpos2_v1.json'):
            r['loss']['new_positive_weight']=4
        return r
    monkeypatch.setattr(exp,'read',changed)
    with pytest.raises(ValueError,match='only registered'):exp.identity('.')


def test_notebook_disabled_self_contained_pinned_and_compilable(tmp_path):
    nb=exp.read('notebooks/GEOFIREX_C8_NEWPOS2_DISABLED.ipynb')
    build=exp.read(exp.REPORT/'notebook_build.json')
    assert exp.sha256('notebooks/GEOFIREX_C8_NEWPOS2_DISABLED.ipynb')==build['notebook_sha256']
    data=None
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        assert cell['outputs']==[] and cell['execution_count'] is None
        s=''.join(cell['source']);tree=ast.parse(s);compile(s,'notebook','exec')
        if 'RUN_GPU_PREFLIGHT = False' in s:
            assert all(f'{flag} = False' in s for flag in ('RUN_GPU_PREFLIGHT','RUN_FIT','RUN_RECOVERY'))
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATA' for t in n.targets):
                data=base64.b64decode(n.value.args[0].value)
    assert hashlib.sha256(data).hexdigest()==build['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert not any(n.endswith(('.pt','.tif','.bin')) for n in z.namelist())
        z.extractall(tmp_path/'payload')
    assert exp.identity(tmp_path/'payload')[1]['run_id']==exp.B_ID
    assert len(exp.read(tmp_path/'payload'/exp.old.REPORT/'partitions.json'))==11


def test_locked_acceptance_exact_boundaries():
    from geofirex.newpos2_comparison import gates
    p=exp.read(exp.REPORT/'acceptance.json')
    r=deepcopy(exp.baseline_json('.'))
    r['propagation_pr']['average_precision']=p['AP_new_min']
    r['pooled'].update(average_precision=p['AP_pooled_min'],dice=p['Dice_pooled_min'],brier=p['Brier_max'])
    r['new_fire'].update(recall=p['recall_new_min'],dice=p['Dice_new_min'],fp=p['FP_new_max'])
    r['macro_event_dice']=p['macro_Dice_min']
    assert all(gates(r,p).values())
    r['new_fire']['fp']+=1
    assert not gates(r,p)['FP_new']
    r['propagation_pr']['average_precision']=np.nextafter(p['AP_new_min'],-np.inf)
    assert not gates(r,p)['AP_new']


def test_three_way_paired_table_and_bootstrap_using_saved_metadata_only(tmp_path):
    from geofirex.newpos2_comparison import compare
    baseline=exp.baseline_json('.')
    events=[r['event_id'] for r in baseline['per_event']]
    r=compare('.',baseline,tmp_path/'comparison',events)
    assert not r['all_scientific_gates_pass'] and not r['promotion_authorized']
    assert len((tmp_path/'comparison/metrics.csv').read_text().splitlines())==4
    b=exp.read(tmp_path/'comparison/bootstrap_events.json')
    assert (b['repetitions'],b['seed'])==(2000,20261121)
    assert all(v['low']==v['high']==0 for v in b['intervals'].values())


@pytest.mark.parametrize('empty',[False,True])
def test_disabled_extra_weight_is_bitwise_base_value_and_gradient(empty):
    torch.manual_seed(12)
    x=torch.randn(2,1,3,4,requires_grad=True);y=torch.randint(0,2,x.shape).float()
    v=torch.zeros_like(x) if empty else torch.randint(0,2,x.shape).float();o=torch.randint(0,2,x.shape).float()
    a=SegmentationLoss(.5,10)(x,y,v);b=NewPositiveLoss(1)(x,y,v,o)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])


def test_exact_weighted_formula_and_direct_gradients():
    x=torch.tensor([[[[-2.,1.,.3,4.]]]],requires_grad=True)
    y=torch.tensor([[[[1.,1.,0.,1.]]]]);v=torch.tensor([[[[1.,1.,1.,0.]]]]);o=torch.tensor([[[[0.,1.,0.,0.]]]])
    bce=torch.nn.functional.binary_cross_entropy_with_logits(x,y,pos_weight=torch.tensor(10.),reduction='none')
    # Historical Dice isolated from baseline using dice_weight=1.
    expected=.5*(bce*v*(1+y*(o==0))).sum()/v.sum()+.5*SegmentationLoss(1,10)(x,y,v)
    actual=NewPositiveLoss(2)(x,y,v,o)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])


def test_loss_double_gradcheck_and_extreme_logits_padding():
    x=torch.tensor([[[[-.4,.2],[1.,-2.]]]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[[[1.,0.],[1.,0.]]]],dtype=torch.double);v=torch.ones_like(y);o=torch.zeros_like(y)
    assert torch.autograd.gradcheck(lambda z:NewPositiveLoss(2).double()(z,y,v,o),(x,))
    z=torch.tensor([[[[-1000.,1000.]]]],requires_grad=True);yy=torch.ones_like(z)
    for mask in (torch.ones_like(z),torch.zeros_like(z)):
        loss=NewPositiveLoss(2)(z,yy,mask,torch.zeros_like(z));assert torch.isfinite(loss)
        assert torch.isfinite(torch.autograd.grad(loss,z,retain_graph=True)[0]).all()


@pytest.mark.parametrize('kind',['shape','target','valid','observed','weight'])
def test_loss_rejects_invalid_inputs(kind):
    x=torch.zeros(1,1,2,2);y=x.clone();v=torch.ones_like(x);o=x.clone()
    if kind=='weight':
        with pytest.raises(ValueError):NewPositiveLoss(.9)
        return
    if kind=='shape':o=torch.zeros(2,2)
    if kind=='target':y[0,0,0,0]=2
    if kind=='valid':v[0,0,0,0]=float('nan')
    if kind=='observed':o[0,0,0,0]=.1
    with pytest.raises(ValueError):NewPositiveLoss(2)(x,y,v,o)


def test_cpu_identity_and_model_initialization_never_query_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No CUDA query on CPU preflight'))
    r=exp.preflight('.');assert r['total_updates']==28976 and not r['gpu_checked']
    from geofirex.benchmark_models import StackedUNet
    torch.manual_seed(42);base=StackedUNet(8,3)
    torch.manual_seed(42);candidate=exp.make_model(exp.identity('.')[1],'cpu')
    assert sum(p.numel() for p in candidate.parameters())==123977
    assert all(torch.equal(a,b) for a,b in zip(base.state_dict().values(),candidate.state_dict().values()))


def test_new_outputs_refuse_official_names_and_reuse(tmp_path):
    with pytest.raises(ValueError):exp.new_output(tmp_path/'B2_C8_V2_seed42',exp.B_ID)
    p=exp.new_output(tmp_path/exp.B_ID,exp.B_ID);assert p.is_dir()
    with pytest.raises(ValueError):exp.new_output(p,exp.B_ID)


def synthetic_final(tmp_path,monkeypatch):
    """Untrained fixture carrying a synthetic final cursor; never a result."""
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    _,config,_,lock,_=exp.identity('.')
    out=tmp_path/'UNTRAINED_FIXTURE';out.mkdir()
    model=exp.make_model(config,'cpu');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    exp.preserve('.',out,lock)
    state=dict(epoch=4,cursor=0,updates=28976,history=['SYNTHETIC UNTRAINED FIXTURE'])
    p=exp.old.checkpoint(out,model,optimizer,state,config,lock,{'fixture':True})
    exp.save(out/'training_complete.json',dict(status='COMPLETE',updates=28976,epochs=4,
        checkpoint=p.name,checkpoint_sha256=exp.sha256(p)))
    return out


@pytest.mark.parametrize('outcome',['complete','timeout','missing_coverage'])
def test_training_safe_precedes_evaluation_and_failure_stays_partial(tmp_path,monkeypatch,outcome):
    out=synthetic_final(tmp_path,monkeypatch)
    def callback():
        archives=list(tmp_path.glob('*-TRAINING-SAFE-*.zip'));assert len(archives)==1
        exp.old.verify_zip(archives[0])
        if outcome=='timeout':raise TimeoutError('Synthetic interruption')
        if outcome=='complete':
            (out/'evaluation').mkdir();exp.save(out/'evaluation/complete.json',dict(status='COMPLETE'))
        return 'fixture'
    if outcome=='complete':assert exp.safeguarded_candidate_evaluation('.',out,callback)=='fixture'
    else:
        with pytest.raises((TimeoutError,ValueError)):exp.safeguarded_candidate_evaluation('.',out,callback)
    stage='COMPLETE' if outcome=='complete' else 'PARTIAL'
    assert len(list(tmp_path.glob(f'*-{stage}-*.zip')))==1
    if outcome!='complete':assert not list(tmp_path.glob('*-COMPLETE-*.zip'))


def test_synthetic_checkpoint_restore_and_two_verified_backup_roots(tmp_path,monkeypatch):
    out=synthetic_final(tmp_path,monkeypatch)
    archive=exp.old.package(out,'TRAINING-SAFE');sidecar=archive.with_suffix('.sha256')
    restored=exp.restore_candidate('.',archive,sidecar,tmp_path/'restored')
    state,p=exp.final_candidate('.',restored)
    assert state['history']==['SYNTHETIC UNTRAINED FIXTURE']
    assert exp.sha256(p)==exp.read(out/'training_complete.json')['checkpoint_sha256']
    with pytest.raises(FileExistsError):exp.restore_candidate('.',archive,sidecar,restored)
    roots=[tmp_path/'fixture_local',tmp_path/'fixture_external']
    for r in roots:r.mkdir()
    copies=exp.old.backup(archive,sidecar,*roots)
    assert len(copies)==2 and all(exp.sha256(p)==exp.sha256(archive) for p in copies)
    with pytest.raises(ValueError):exp.old.backup(archive,sidecar,roots[0],roots[0])
    wrong=tmp_path/'wrong.sha256';wrong.write_text('0'*64)
    with pytest.raises(ValueError):exp.restore_candidate('.',archive,wrong,tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()
