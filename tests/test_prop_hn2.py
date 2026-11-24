"""PROP-HN2 qualification: train-only fixtures, frozen metadata and CPU saves."""
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
from geofirex import prop_hn2_experiment as exp
from geofirex.prop_hn2_sampling import PropHNSampler,make_prop_loaders
from geofirex.unet import SegmentationLoss
from geofirex.fast_compact import DTYPE


class TinyTrain(torch.utils.data.Dataset):
    role='train'
    def __init__(self):
        self.records=np.array([(a,737600+i,0,0,k) for a in range(3)
            for i,k in enumerate([0,0,0,0,0,0,1,1,2])],dtype=DTYPE)
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        kind=int(self.records[i]['category'])
        x=torch.zeros(3,8,8,8);y=torch.zeros(1,8,8);v=torch.ones_like(y)
        x[:,0]=i
        if kind==1:y[0,1,1]=1
        if kind==2:x[-1,7,0,0]=1
        return x,y,v


@pytest.mark.parametrize('epoch',range(4))
def test_exact_sampling_multiplicities_events_and_positive_pixel_parity(epoch):
    ds=TinyTrain();s=PropHNSampler(ds);s.epoch=epoch
    order=s.order();r=s.receipt();counts=np.bincount(order,minlength=len(ds))
    assert len(order)==len(ds)==r['draws']
    assert (counts[s.positive]==1).all() and (counts[s.hard]==2).all()
    assert r['omitted_easy_tiles']==r['repeated_draws']==len(s.hard)
    assert set(ds.records[order]['archive'])==set(ds.records['archive'])
    baseline=sum(float((ds[i][1]*ds[i][2]).sum()) for i in range(len(ds)))
    candidate=sum(float((ds[i][1]*ds[i][2]).sum()) for i in order)
    assert baseline==candidate==6
    assert np.array_equal(order,s.order())
    # Only the order/multiplicity changes, no tensor conversion or augmentation.
    for i in np.unique(order):
        assert all(torch.equal(a,b) for a,b in zip(ds[int(i)],ds[int(i)]))


def test_sampling_does_not_read_pixels_and_epochs_differ(monkeypatch):
    ds=TinyTrain();s=PropHNSampler(ds)
    monkeypatch.setattr(TinyTrain,'__getitem__',lambda *a:pytest.fail('Sampler must use frozen metadata only'))
    a=s.order();s.epoch=1;b=s.order()
    assert not np.array_equal(a,b)
    s.epoch=4
    with pytest.raises(ValueError):s.order()


@pytest.mark.parametrize('role',['val','test','purged','2021'])
def test_nontrain_roles_rejected_before_sampling(role):
    ds=TinyTrain();ds.role=role
    with pytest.raises(ValueError,match='train-only'):PropHNSampler(ds)


@pytest.mark.parametrize('mutation',['category','nohard','nopositive','noeasy'])
def test_impossible_or_unregistered_strata_rejected(mutation):
    ds=TinyTrain()
    if mutation=='category':ds.records['category'][0]=7
    if mutation=='nohard':ds.records['category'][ds.records['category']==2]=0
    if mutation=='nopositive':ds.records['category'][ds.records['category']==1]=0
    if mutation=='noeasy':ds.records['category'][ds.records['category']==0]=2
    with pytest.raises(ValueError):PropHNSampler(ds)


def test_only_train_loader_sampler_changes_validation_and_tensors_unchanged(monkeypatch):
    import geofirex.prop_hn2_sampling as sampling
    from torch.utils.data import DataLoader
    ds=TinyTrain();val=TinyTrain();val.role='val'
    baseline=DataLoader(ds,batch_size=4);validation=DataLoader(val,batch_size=4)
    monkeypatch.setattr(sampling,'make_v2_loaders',lambda *a,**k:(baseline,validation))
    train,actual_val=make_prop_loaders('unused','unused',batch_size=4)
    assert train.dataset is baseline.dataset and actual_val is validation
    assert isinstance(train.sampler,PropHNSampler)
    for x,y,v in train:
        for i in range(len(x)):
            source=ds[int(x[i,0,0,0,0])]
            assert all(torch.equal(a,b) for a,b in zip((x[i],y[i],v[i]),source))
    assert all(torch.equal(a,b) for a,b in zip(next(iter(validation)),next(iter(actual_val))))


@pytest.mark.parametrize('empty',[True,False])
def test_criterion_and_gradient_are_bitwise_historical_c8(empty):
    torch.manual_seed(42)
    x=torch.randn(2,1,8,8,requires_grad=True);y=torch.randint(0,2,x.shape).float()
    v=torch.zeros_like(x) if empty else torch.ones_like(x)
    a=exp.make_criterion()(x,y,v);b=SegmentationLoss(.5,10)(x,y,v)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])


def test_real_identity_and_fixed_cpu_plans_no_scientific_change_beyond_sampling():
    base,c,_,_,_=exp.identity('.')
    changes={k for k in set(base)|set(c) if base.get(k)!=c.get(k)}
    assert changes=={'run_id','sampler','status','base_config_sha256'}
    assert c['loss']==base['loss'] and c['loss']['name']=='SegmentationLoss'
    assert (c['epochs'],c['total_updates'],c['threshold'])==(4,28976,.78)
    q=exp.read(exp.REPORT/'train_category_qualification.json')
    assert q['checked_hard_negative_tiles']==1839 and q['validation_pixels_opened']==q['test_pixels_opened']==0
    assert len({r['order_sha256'] for r in q['epochs']})==4
    assert all(r['events']==318 and r['positive_draws']==6861 and r['draws']==115900 for r in q['epochs'])
    assert exp.read(exp.REPORT/'acceptance.json')==exp.read('reports/newpos2_20261122_v1/acceptance.json')


@pytest.mark.parametrize('field,value',[('authorized',False),('experiment_id','B2_C8_V2_NEWPOS2_seed42'),
    ('action','newpos2-fit'),('runs',2),('execution_manifest_sha256','0'*64),('device','cpu'),
    ('session_remaining_minutes',254),('quota_remaining_card_hours',8.4),('card_hours_reserved',None)])
def test_invalid_grants_block_before_cuda_or_model(tmp_path,monkeypatch,field,value):
    a=exp.read(exp.REPORT/'authorization_prop_hn2-fit_DISABLED.json')
    a.update(authorized=True,authorized_by='SYNTHETIC',authorized_at='fixture',quota_checked_at='fixture',
        output_parent=str(tmp_path),session_remaining_minutes=255,quota_remaining_card_hours=8.5,
        card_hours_reserved=8.5,local_backup_destination='synthetic-local',external_backup_destination='synthetic-external')
    a[field]=value;p=tmp_path/'grant.json';p.write_text(json.dumps(a))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No GPU query'))
    monkeypatch.setattr(exp,'make_model',lambda *a:pytest.fail('No model'))
    with pytest.raises(ValueError):exp.fit('.',[],tmp_path/exp.B_ID,p)
    assert not (tmp_path/exp.B_ID).exists()


def test_sampling_plan_matches_all_frozen_real_metadata_orders():
    from types import SimpleNamespace
    records=np.load('reports/protocol_v2_20261118/index/train.npy',allow_pickle=False)
    class RealRecords:
        role='train'
        def __init__(self):self.records=records
        def __len__(self):return len(self.records)
    s=PropHNSampler(RealRecords());q=exp.read(exp.REPORT/'train_category_qualification.json')
    for i in range(4):
        s.epoch=i
        assert s.receipt()==q['epochs'][i]


def test_four_way_comparison_with_saved_metadata_only(tmp_path):
    from geofirex.prop_hn2_comparison import compare
    base=exp.baseline_json('.')
    events=[r['event_id'] for r in base['per_event']]
    result=compare('.',base,tmp_path/'paired',events)
    assert not result['promotion_authorized'] and not result['all_scientific_gates_pass']
    assert len((tmp_path/'paired/metrics.csv').read_text().splitlines())==5
    for name in ('bootstrap_events.json','bootstrap_vs_newpos4.json','bootstrap_vs_newpos2.json'):
        b=exp.read(tmp_path/'paired'/name)
        assert (b['repetitions'],b['seed'])==(2000,20261121)


def test_no_loss_modification_hidden_in_fit():
    import inspect
    source=inspect.getsource(exp.fit)
    assert 'NewPositiveLoss' not in source and "criterion(model(x.to('cuda')),y.to('cuda'),v.to('cuda'))" in source
    assert 'sampling_plan.json' in source and 'train.sampler.receipt()!=plan[epoch]' in source


def test_atomic_lock_survives_deleted_run_directory(tmp_path):
    p=exp.new_output(tmp_path/exp.B_ID,exp.B_ID);p.rmdir()
    with pytest.raises(FileExistsError):exp.new_output(p,exp.B_ID)


def test_notebook_disabled_self_contained_pinned_and_compilable(tmp_path):
    nb=exp.read('notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb')
    build=exp.read(exp.REPORT/'notebook_build.json')
    assert exp.sha256('notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb')==build['notebook_sha256']
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
    from geofirex.prop_hn2_comparison import gates
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
