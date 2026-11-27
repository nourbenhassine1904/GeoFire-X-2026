"""Independent A/B fixtures. No real inference, GPU, optimizer step or 2021 data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from geofirex.unet import SegmentationLoss
from geofirex.c8_newpos_loss import NewPositiveLoss
from geofirex.c8_predictions import PredictionWriter,PredictionStore,domain_tile,PIXEL
from geofirex.c8_thresholds import choose_threshold,counts,crossfit,folds,plots
from geofirex import c8_experiments as exp
from geofirex.c8_comparison import bootstrap_events,compare


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
    expected=.5*(bce*v*(1+3*y*(o==0))).sum()/v.sum()+.5*SegmentationLoss(1,10)(x,y,v)
    actual=NewPositiveLoss(4)(x,y,v,o)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])


def test_loss_double_gradcheck_and_extreme_logits_padding():
    x=torch.tensor([[[[-.4,.2],[1.,-2.]]]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[[[1.,0.],[1.,0.]]]],dtype=torch.double);v=torch.ones_like(y);o=torch.zeros_like(y)
    assert torch.autograd.gradcheck(lambda z:NewPositiveLoss(4).double()(z,y,v,o),(x,))
    z=torch.tensor([[[[-1000.,1000.]]]],requires_grad=True);yy=torch.ones_like(z)
    for mask in (torch.ones_like(z),torch.zeros_like(z)):
        loss=NewPositiveLoss(4)(z,yy,mask,torch.zeros_like(z));assert torch.isfinite(loss)
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
    with pytest.raises(ValueError):NewPositiveLoss(4)(x,y,v,o)


def make_store(tmp_path,seed=3,events=5,shape=(64,64),empty=False):
    rng=np.random.default_rng(seed);keys=[(f'2020/fire_{i+1}','2020-09-07',0,0) for i in range(events)]
    writer=PredictionWriter(tmp_path/'predictions',keys,{'fixture':True},shape)
    for k in keys:
        p=(rng.integers(0,21,shape)/20).astype(np.float32);y=(rng.random(shape)<.08).astype(np.uint8)
        v=np.zeros(shape,np.uint8) if empty else (rng.random(shape)>.1).astype(np.uint8)
        o=(rng.random(shape)<.1).astype(np.uint8)
        writer.add(k,p,y,v,o)
    writer.finish();return PredictionStore(tmp_path/'predictions',expected_keys=keys)


def test_store_typed_float32_and_domain_and_corruption(tmp_path):
    store=make_store(tmp_path,shape=(2,3));assert PIXEL.itemsize==7
    first=next(iter(store.tiles()))
    assert first[1]['p'].dtype==np.float32
    p=store.root/store.files[store.events[0]]['path']
    with p.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError):PredictionStore(store.root)


def test_prediction_writer_refuses_test_duplicate_and_partial(tmp_path):
    with pytest.raises(ValueError):PredictionWriter(tmp_path/'bad',[('2021/fire_1','2021-01-01',0,0)],{})
    key=('2020/fire_1','2020-09-01',0,0);w=PredictionWriter(tmp_path/'partial',[key],{},(2,2))
    with pytest.raises(ValueError):w.finish()
    p=np.zeros((2,2),np.float32);w.add(key,p,p,p,p)
    with pytest.raises(ValueError):w.add(key,p,p,p,p)
    w.finish()


@pytest.mark.parametrize('seed',list(range(8)))
def test_exact_threshold_matches_exhaustive_global_ties(tmp_path,seed):
    store=make_store(tmp_path,seed=seed);events=store.events[:4]
    r=choose_threshold(store,events)
    p=[];y=[]
    for _,a in store.chunks(events):
        mask=(a['v']==1)&(a['o']==0);p.extend(a['p'][mask]);y.extend(a['y'][mask])
    p=np.asarray(p,np.float32);y=np.asarray(y,np.uint8)
    best=None;tp=0;fp=0
    for t in sorted(np.unique(p),reverse=True):
        cc=counts(p,y,np.ones_like(y),float(t))
        if cc[1]<=r['fp_budget'] and cc[0]>tp:best=float(t);tp=int(cc[0]);fp=int(cc[1])
    assert r['threshold']==best and r['calibration_tp']==tp and r['calibration_fp']==fp
    assert fp<=len(y[y==0])*7//25000


def test_calibration_reads_only_its_own_event_subset(tmp_path):
    store=make_store(tmp_path);allowed=store.events[:4];seen=[];original=store.chunks
    def spy(events,*args,**kwargs):
        assert set(events)<=set(allowed)
        seen.extend(events);yield from original(events,*args,**kwargs)
    store.chunks=spy
    choose_threshold(store,allowed);assert set(seen)==set(allowed)


def test_crossfit_thresholds_saved_before_assessment_and_complete_event_coverage(tmp_path,monkeypatch):
    import geofirex.c8_thresholds as th
    store=make_store(tmp_path);out=tmp_path/'analysis';original=th.event_counts;locks=[]
    def assess(*args,**kwargs):
        p=out/'thresholds_locked.json';assert p.exists();locks.append(hashlib.sha256(p.read_bytes()).hexdigest())
        return original(*args,**kwargs)
    monkeypatch.setattr(th,'event_counts',assess)
    r=crossfit(store,out)
    assert r['status']=='COMPLETE' and len(set(locks))==1
    choices=json.loads((out/'thresholds_locked.json').read_text())['selections']
    assert all(set(s['assessment_events']).isdisjoint(s['calibration_events']) for s in choices)
    assert len(r['per_event_crossfit'])==5


def test_empty_domains_reject_all_and_never_claim_acceptance(tmp_path):
    store=make_store(tmp_path,shape=(2,3),empty=True)
    r=choose_threshold(store,store.events[:4]);assert r['threshold'] is None and r['fp_budget']==0
    result=crossfit(store,tmp_path/'empty_analysis');assert not result['acceptance_all']


def test_folds_80_match_frozen_proposal_without_changing_global_roles():
    meta=json.loads(Path('reports/protocol_v2_20261118/index/index.json').read_text())
    events=[e for e,r in meta['event_roles'].items() if r=='val'];m=folds(events)
    assert all(list(m.values()).count(i)==16 for i in range(5))
    import csv
    expected={r['event_id']:int(r['proposal_fold']) for r in csv.DictReader(Path('reports/c36_cpu_20261121_v1/proposal_validation_folds.csv').open())}
    assert m==expected


def test_exact_new_and_global_AP_against_sklearn_and_plot_outputs(tmp_path):
    from sklearn.metrics import average_precision_score
    store=make_store(tmp_path,shape=(4,4));result=exp.exact_metrics(store,tmp_path,exp.read('reports/protocol_v2_execution/propagation_policy.json'))
    for new in (False,True):
        pp=[];yy=[]
        for _,a in store.chunks(store.events):
            mask=a['v']==1
            if new:mask&=a['o']==0
            pp.extend(a['p'][mask]);yy.extend(a['y'][mask])
        actual=result['propagation_pr']['average_precision'] if new else result['pooled']['average_precision']
        assert actual==pytest.approx(average_precision_score(yy,pp),abs=1e-12)
    cf=crossfit(store,tmp_path/'crossfit')
    plots(result,cf,tmp_path/'figures',title='Synthetic fixture, no GeoFire-X result')
    assert len(list((tmp_path/'figures').glob('*.png')))==3


def test_paired_bootstrap_identical_rows_zero_intervals():
    rows={f'2020/fire_{i}':dict(pooled=np.array([2,1,3,10]),new_fire=np.array([1,1,2,8]),squared_error=.3) for i in range(5)}
    a=bootstrap_events(rows,rows)
    assert a['repetitions']==2000
    assert all(r['low']==r['high']==0 for r in a['intervals'].values())
    assert a==bootstrap_events(rows,rows)


def test_comparison_rejects_changed_domain_and_reports_no_false_gain(tmp_path):
    store=make_store(tmp_path,shape=(4,4));r=exp.exact_metrics(store,tmp_path,exp.read('reports/protocol_v2_execution/propagation_policy.json'))
    c=compare(r,r,tmp_path/'paired',store.events)
    assert not c['all_scientific_gates_pass'] and c['deltas']['AP_new']==0
    changed=deepcopy(r);changed['evaluation_domain_sha256']='bad'
    with pytest.raises(ValueError):compare(r,changed,tmp_path/'bad')


def grant(tmp_path,action='newpos4-fit',device='cuda',experiment=exp.B_ID):
    a=dict(authorized=True,action=action,experiment_id=experiment,device=device,runs=1,
        execution_manifest_sha256=exp.sha256(Path(exp.REPORT)/'execution_manifest.json'),
        authorized_by='SYNTHETIC TEST ONLY',authorized_at='2026-11-21T00:00:00Z',output_parent=str(tmp_path),
        local_backup_destination='synthetic-local',external_backup_destination='synthetic-external',
        session_remaining_minutes=300,quota_remaining_card_hours=10,card_hours_reserved=8,quota_checked_at='synthetic')
    p=tmp_path/'grant.json';p.write_text(json.dumps(a));return p,a


def test_cpu_identity_and_model_initialization_never_query_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No CUDA query on CPU preflight'))
    r=exp.preflight('.');assert r['total_updates']==28976 and not r['gpu_checked']
    from geofirex.benchmark_models import StackedUNet
    torch.manual_seed(42);base=StackedUNet(8,3)
    torch.manual_seed(42);candidate=exp.make_model(exp.identity('.')[1],'cpu')
    assert sum(p.numel() for p in candidate.parameters())==123977
    assert all(torch.equal(a,b) for a,b in zip(base.state_dict().values(),candidate.state_dict().values()))


@pytest.mark.parametrize('field,value',[('authorized',False),('experiment_id','B2_C8_V2_seed42'),('action','threshold-infer'),
    ('execution_manifest_sha256','0'*64),('session_remaining_minutes',None),('quota_remaining_card_hours',0),('runs',2)])
def test_grant_scope_and_budget_refusals(tmp_path,field,value):
    path,a=grant(tmp_path);a[field]=value;path.write_text(json.dumps(a))
    with pytest.raises(ValueError):exp.authorize('.',path,'newpos4-fit','cuda',tmp_path)


def test_fit_and_inference_refuse_before_loading_model_or_cuda(tmp_path,monkeypatch):
    path,a=grant(tmp_path);a['authorized']=False;path.write_text(json.dumps(a))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No CUDA without grant'))
    monkeypatch.setattr(exp,'load_official',lambda *a:pytest.fail('No checkpoint without grant'))
    with pytest.raises(ValueError):exp.fit('.',[],tmp_path/exp.B_ID,path)
    with pytest.raises(ValueError):exp.infer_thresholds('.',[],tmp_path/'never.zip',tmp_path/exp.A_ID,path)
    assert not (tmp_path/exp.B_ID).exists() and not (tmp_path/exp.A_ID).exists()


def test_gpu_preflight_scope_A_and_B(tmp_path):
    for e in (exp.A_ID,exp.B_ID):
        p,a=grant(tmp_path,action='gpu-preflight',experiment=e)
        assert exp.authorize('.',p,'gpu-preflight','cuda',tmp_path)['experiment_id']==e


def test_new_outputs_refuse_official_names_and_reuse(tmp_path):
    with pytest.raises(ValueError):exp.new_output(tmp_path/'B2_C8_V2_seed42',exp.B_ID)
    p=exp.new_output(tmp_path/exp.B_ID,exp.B_ID);assert p.is_dir()
    with pytest.raises(ValueError):exp.new_output(p,exp.B_ID)


def test_notebooks_disabled_payload_pinned_compilable_and_no_pixels(tmp_path):
    import base64,io,zipfile
    for suffix in ('THRESHOLDS','NEWPOS4'):
        nb=json.loads(Path(f'notebooks/GEOFIREX_C8_{suffix}_DISABLED.ipynb').read_text(encoding='utf8'))
        env={}
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source']);compile(source,'notebook','exec')
                if 'RUN_GPU_PREFLIGHT = False' in source:
                    assert all(f'{flag} = False' in source for flag in ('RUN_GPU_PREFLIGHT','RUN_INFERENCE','RUN_ANALYSIS','RUN_FIT','RUN_RECOVERY'))
                if 'DATA=base64.b64decode(' in source:
                    import ast
                    tree=ast.parse(source)
                    literal=next(n.value.args[0].value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATA' for t in n.targets))
                    data=base64.b64decode(literal)
                    assert hashlib.sha256(data).hexdigest()==json.loads(Path(exp.REPORT/'notebook_build.json').read_text())['payload_sha256']
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        assert not any(n.endswith(('.pt','.tif','.bin')) for n in z.namelist())
                        assert 'src/geofirex/c8_newpos_loss.py' in z.namelist()
                        lock=json.loads(z.read('reports/c8_propagation_20261121_v1/execution_manifest.json'))
                        assert all('\\' not in n for n in lock['files'])
                        extracted=tmp_path/suffix;extracted.mkdir()
                        z.extractall(extracted)
                        assert exp.identity(extracted)[4]['total_updates']==28976
                        for n in z.namelist():
                            if n.endswith('.py'):compile(z.read(n),n,'exec')


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


def test_store_stops_on_deadline_before_reading_analysis_pixels(tmp_path):
    store=make_store(tmp_path);store.deadline=0
    with pytest.raises(TimeoutError):next(store.chunks(store.events))
    with pytest.raises(TimeoutError):next(store.tiles())


def test_cpu_inference_grant_requires_its_larger_session_budget(tmp_path):
    p,a=grant(tmp_path,action='threshold-infer',device='cpu',experiment=exp.A_ID)
    a['session_remaining_minutes']=180;p.write_text(json.dumps(a))
    with pytest.raises(ValueError):exp.authorize('.',p,'threshold-infer','cpu',tmp_path)
    a['session_remaining_minutes']=240;p.write_text(json.dumps(a))
    assert exp.authorize('.',p,'threshold-infer','cpu',tmp_path)['device']=='cpu'


@pytest.mark.parametrize('timed_out',[False,True])
def test_windows_analysis_cooperative_deadline(monkeypatch,timed_out):
    import signal
    monkeypatch.setattr(signal,'SIGALRM',None,raising=False)
    times=iter([0.,2. if timed_out else .5]);monkeypatch.setattr(exp.time,'monotonic',lambda:next(times))
    if timed_out:
        with pytest.raises(TimeoutError):
            with exp.analysis_deadline(1):pass
    else:
        with exp.analysis_deadline(1):pass
