"""CPU-only qualification: no CUDA calls, no real fit or historical checkpoint."""
from pathlib import Path
import base64
import io
import json
import math
from types import SimpleNamespace
import zipfile
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from geofirex import v2_execution as v2
from geofirex.v2_propagation import PropagationMetrics, exact_pr
from geofirex.benchmark_metrics import ExternalAveragePrecision

ROOT=Path(__file__).resolve().parents[1]
POLICY=dict(fpr_budgets=[0,.25,.5,1.],recall_curve_intervals=100)

def test_frozen_cpu_preflight_never_queries_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',Mock(side_effect=AssertionError('GPU forbidden')))
    r=v2.preflight(ROOT)
    assert r['updates_per_epoch']==math.ceil(r['train_tiles']/16)==7244
    assert r['total_updates']==28976 and r['gpu_checked'] is False
    split=v2.read(ROOT/v2.REPORT/'split_lock.json')
    v2.verify_manifest(ROOT,split['protected_sha256'])

@pytest.mark.parametrize('bad', ['missing','false','quota','manifest','destination'])
def test_launch_authorization_rejected(tmp_path,bad):
    a=v2.read(ROOT/v2.REPORT/'authorization_TEMPLATE_DISABLED.json')
    if bad=='missing':
        with pytest.raises(FileNotFoundError): v2.authorize(ROOT,tmp_path/'missing','fit')
        return
    a.update(authorized=True,quota_remaining_card_hours=7,session_remaining_minutes=210,
             quota_checked_at='2026-11-19T12:00:00+01:00',authorized_by='fixture')
    if bad=='false': a['authorized']=False
    elif bad=='quota': a['quota_remaining_card_hours']=6.9
    elif bad=='manifest': a['execution_manifest_sha256']='bad'
    else: a['external_backup_destination']=''
    path=tmp_path/'authorization.json'; v2.save(path,a)
    with pytest.raises(ValueError): v2.authorize(ROOT,path,'fit')

def test_good_authorization_and_action_scope(tmp_path):
    a=v2.read(ROOT/v2.REPORT/'authorization_TEMPLATE_DISABLED.json')
    a.update(authorized=True,quota_remaining_card_hours=7,session_remaining_minutes=210,
             quota_checked_at='2026-11-19',authorized_by='fixture')
    path=tmp_path/'auth.json'; v2.save(path,a)
    assert v2.authorize(ROOT,path,'fit')==a
    with pytest.raises(ValueError): v2.authorize(ROOT,path,'evaluate-only')

def test_budget_checks_actual_space_and_ram(tmp_path,monkeypatch):
    policy=v2.read(ROOT/v2.REPORT/'budget.json')
    assert policy['session_minutes']==210 and policy['required_free_bytes']>=16*v2.GIB
    monkeypatch.setattr(v2.shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    with pytest.raises(RuntimeError): v2.disk_check(tmp_path,100)
    monkeypatch.setattr(v2,'host_memory',lambda:(1,1))
    with pytest.raises(RuntimeError): v2.resource_check(policy,tmp_path,tmp_path)

def test_manifest_rejects_corruption_and_escape(tmp_path):
    p=tmp_path/'source'; p.write_text('original'); digest=v2.sha256(p)
    v2.verify_manifest(tmp_path,{'source':digest})
    p.write_text('changed')
    with pytest.raises(ValueError): v2.verify_manifest(tmp_path,{'source':digest})
    with pytest.raises(ValueError): v2.verify_manifest(tmp_path,{'../source':digest})

def test_pr_exact_ties_and_budget_against_bruteforce(tmp_path):
    p=np.array([.9,.8,.8,.7,.6,.5,.5,.4],dtype=np.float32)
    y=np.array([1,1,0,0,1,1,0,0],dtype=np.uint8)
    acc=ExternalAveragePrecision(tmp_path/'pr',chunk_pixels=1,fan_in=2,retain_runs=False)
    for i in range(len(p)): acc.update(p[i:i+1],y[i:i+1])
    r=exact_pr(acc,POLICY)
    from sklearn.metrics import average_precision_score
    assert r['average_precision']==pytest.approx(average_precision_score(y,p))
    for row in r['budgets']:
        possible=[(int(((p>=t)&(y==1)).sum()),int(((p>=t)&(y==0)).sum())) for t in np.unique(p)]
        expected=max([tp for tp,fp in possible if fp<=row['fp_budget']]+[0])
        assert row['tp']==expected and row['fp']<=row['fp_budget']
    assert r['budgets'][0]['tp']==1  # mixed .8 tie cannot be split
    assert r['operational_threshold_selected'] is False
    assert not list((tmp_path/'pr').glob('*.bin'))

@pytest.mark.parametrize('y', [[0,0],[1,1]])
def test_pr_empty_class_policies(tmp_path,y):
    acc=ExternalAveragePrecision(tmp_path/'pr',retain_runs=False)
    acc.update(np.array([.8,.2],np.float32),np.array(y,np.uint8))
    r=exact_pr(acc,POLICY)
    if sum(y)==0: assert r['average_precision'] is None and r['budgets'][0]['recall'] is None
    else: assert all(row['fpr'] is None for row in r['budgets'])

def test_propagation_fixed_threshold_domain_and_coverage(tmp_path):
    keys=[('2020/fire_1','2020-09-01',0,0),('2020/fire_2','2020-09-01',0,0)]
    acc=PropagationMetrics(tmp_path/'metrics',keys,POLICY)
    p=np.array([[.9,.78],[.4,.2]],np.float32); y=np.array([[1,0],[1,0]],np.uint8)
    v=np.ones_like(y); observed=np.array([[1,0],[0,0]],np.uint8)
    acc.update(keys[0],p,y,v,observed)
    with pytest.raises(ValueError): acc.result()
    with pytest.raises(ValueError): acc.update(keys[0],p,y,v,observed)
    acc.update(keys[1],p,y,v,observed); r=acc.result()
    assert r['threshold']==.78 and r['pooled']['tp']==2 and r['new_fire']['tp']==0
    assert r['new_fire']['fp']==2 and r['new_fire']['fn']==2
    assert r['propagation_pr']['positives']==2 and r['tiles']==2
    assert len(r['per_event'])==2 and all('brier' in row and 'new_fire' in row for row in r['per_event'])
    with pytest.raises(ValueError): PropagationMetrics(tmp_path/'test',[('2021/fire_1','2021-01-01',0,0)],POLICY)

def test_archive_corruption_and_both_backups(tmp_path):
    out=tmp_path/'run'; out.mkdir(); (out/'checkpoint.pt').write_bytes(b'fixture checkpoint')
    scratch=out/'evaluation/_scratch'; scratch.mkdir(parents=True); (scratch/'run.bin').write_bytes(b'scratch')
    archive=v2.package(out,'TRAINING-SAFE'); digest=v2.verify_zip(archive)
    with zipfile.ZipFile(archive) as z: assert not any('_scratch' in name for name in z.namelist())
    local=tmp_path/'local'; external=tmp_path/'external'; local.mkdir(); external.mkdir()
    paths=v2.backup(archive,archive.with_suffix('.sha256'),local,external)
    assert len(paths)==2 and all(v2.verify_zip(p,digest)==digest for p in paths)
    with pytest.raises(ValueError): v2.verify_zip(archive,'wrong')
    with archive.open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError): v2.verify_zip(archive,digest)

def test_notebook_payload_frozen_disabled_compilable(tmp_path):
    nb=v2.read(ROOT/'notebooks/GEOFIREX_V2_DISABLED.ipynb')
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert 'RUN_TRAINING = False' in cells[0] and 'RUN_GPU_PREFLIGHT = False' in cells[0]
    for source in cells: compile(source,'notebook','exec')
    import ast
    tree=ast.parse(cells[1]); values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
            values[node.targets[0].id]=node.value.value
    payload=base64.b64decode(values['PAYLOAD'])
    import hashlib
    assert hashlib.sha256(payload).hexdigest()==values['EXPECTED_SHA']
    with zipfile.ZipFile(io.BytesIO(payload)) as z: z.extractall(tmp_path)
    assert v2.preflight(tmp_path)['total_updates']==28976

def test_mount_identity_sizes_and_missing_partition(tmp_path):
    # Tiny synthetic objects, exact manifest identity; no dataset pixels.
    import hashlib
    root=tmp_path/'code'; (root/v2.REPORT).mkdir(parents=True)
    expected={}; maps=[]
    for i in range(11):
        folder=tmp_path/f'p{i:02d}'; folder.mkdir(); obj=folder/'data.bin'; obj.write_bytes(bytes([i]))
        rows=[dict(path=f'train/2018/fire_{i}.zip',alias='data.bin',bytes=1,sha256=v2.sha256(obj))]
        digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        path=folder/'mount-map.json'; v2.save(path,dict(identity=digest,files=rows))
        expected[digest]=dict(map_sha256=v2.sha256(path)); maps.append(path)
    v2.save(root/v2.REPORT/'partitions.json',expected)
    _,receipt=v2.verify_mounts(root,maps); assert receipt['objects']==11
    with pytest.raises(ValueError): v2.verify_mounts(root,maps[:-1])
    with pytest.raises(ValueError): v2.verify_mounts(root,maps[:-1]+[maps[0]])
    (maps[0].parent/'data.bin').write_bytes(b'changed size')
    with pytest.raises(ValueError): v2.verify_mounts(root,maps)

def test_final_checkpoint_cpu_roundtrip_and_incomplete_rejected(tmp_path,monkeypatch):
    # Save new untrained CPU model; no forward, backward or optimizer step.
    from geofirex.benchmark_models import StackedUNet
    c,lock,u=v2.identity(ROOT); model=StackedUNet(8,3); opt=torch.optim.AdamW(model.parameters())
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    state=dict(epoch=4,cursor=0,updates=u['total_updates'],history=[])
    p=v2.checkpoint(tmp_path,model,opt,state,c,lock,{'fixture':True})
    v2.save(tmp_path/'training_complete.json',dict(checkpoint=p.name,checkpoint_sha256=v2.sha256(p),updates=u['total_updates']))
    loaded,_=v2.final_state(tmp_path,ROOT)
    assert loaded['rng']['torch'].dtype==torch.uint8 and 'optimizer' in loaded
    assert loaded['normalizer']=={'fixture':True} and loaded['cursor']==0
    loaded['updates']-=1
    with p.open('wb') as f: torch.save(loaded,f)
    with pytest.raises(ValueError): v2.final_state(tmp_path,ROOT)

@pytest.mark.parametrize('ap_fails',[False,True])
def test_checkpoint_archive_precedes_ap_and_failure_is_recoverable(tmp_path,monkeypatch,ap_fails):
    out=tmp_path/'B2_C8_V2_seed42'; out.mkdir()
    (out/'checkpoint.pt').write_bytes(b'new fixture state')
    monkeypatch.setattr(v2,'final_state',Mock(return_value=({},out/'checkpoint.pt')))
    def fake_ap():
        archives=list(tmp_path.glob('*TRAINING-SAFE*.zip'))
        assert len(archives)==1
        v2.verify_zip(archives[0])
        if ap_fails: raise RuntimeError('simulated AP failure')
        (out/'evaluation').mkdir(); v2.save(out/'evaluation/complete.json',dict(status='COMPLETE'))
        return 'fixture_complete'
    if ap_fails:
        with pytest.raises(RuntimeError): v2.safeguarded_evaluation(ROOT,out,fake_ap)
        assert list(tmp_path.glob('*PARTIAL*.zip'))
    else:
        assert v2.safeguarded_evaluation(ROOT,out,fake_ap)=='fixture_complete'
        assert list(tmp_path.glob('*COMPLETE*.zip'))
    archive=next(tmp_path.glob('*TRAINING-SAFE*.zip'))
    restored=v2.restore(archive,archive.with_suffix('.sha256'),tmp_path/'restored')
    assert (restored/'checkpoint.pt').read_bytes()==b'new fixture state'
    with pytest.raises(FileExistsError): v2.restore(archive,archive.with_suffix('.sha256'),restored)
