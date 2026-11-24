"""Recovery-only CPU checks. No optimizer or training step is constructed."""
import ast
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch
from geofirex import fire13_experiment as c
from geofirex import fire13_recovery as r
from geofirex import fire13_comparison as comp
from geofirex.c8_thresholds import counts, event_counts, summarize, serial_rows, folds

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Training forbidden during recovery checks')
    monkeypatch.setattr(c, 'fit', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)


def payload():
    nb = c.read(ROOT/'notebooks/GEOFIREX_FIRE13_EVALUATION_RECOVERY_ONLY.ipynb')
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DATA' for t in node.targets):
                return base64.b64decode(node.value.args[0].value)
    raise AssertionError('Embedded code payload absent')


def test_historical_exact_module_and_all_35_transitive_modules():
    provenance = c.read(ROOT/r.REPORT/'module_provenance.json')
    with zipfile.ZipFile(ROOT/provenance['source']) as z:
        historical = z.read(provenance['member'])
    assert historical == (ROOT/provenance['module']).read_bytes()
    assert hashlib.sha256(historical).hexdigest() == provenance['sha256']
    audit = c.read(ROOT/r.REPORT/'dependency_audit.json')
    assert len(audit['modules']) == 35
    assert set(audit['added_modules']) == {'src/geofirex/c8_thresholds.py','src/geofirex/fire13_recovery.py'}
    assert r.identity(ROOT)['original_scientific_identities'] == 77


def test_isolated_notebook_payload_imports_every_module(tmp_path):
    data = payload()
    assert hashlib.sha256(data).hexdigest() == c.read(ROOT/r.REPORT/'notebook_build.json')['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert not any(n.endswith(('.pt','.tif','.bin','.zip')) for n in z.namelist())
        z.extractall(tmp_path)
    code = "import sys,importlib,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);root=Path(sys.argv[2]);a=json.loads((root/'reports/fire13_recovery_20261123_v1/dependency_audit.json').read_text());[importlib.import_module(n) for n in a['modules']];from geofirex import fire13_recovery as r;print(r.identity(root)['checkpoint_sha256'])"
    result = subprocess.run([sys.executable,'-I','-c',code,str(tmp_path/'src'),str(tmp_path)],
        cwd=tmp_path, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert r.identity(ROOT)['checkpoint_sha256'] in result.stdout


def test_notebook_disabled_compiles_no_training_calls():
    nb = c.read(ROOT/'notebooks/GEOFIREX_FIRE13_EVALUATION_RECOVERY_ONLY.ipynb')
    combined = ''
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source']); compile(code, 'recovery-notebook', 'exec')
        assert cell['outputs'] == [] and cell['execution_count'] is None
        combined += code
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'fit','backward','step','Adam','AdamW','SGD'}
    for flag in ['RUN_GPU_PREFLIGHT','RUN_EVALUATION','CREATE_NEW_AUTHORIZATIONS']:
        assert flag+' = False' in combined
    module = (ROOT/'src/geofirex/fire13_recovery.py').read_text()
    for node in ast.walk(ast.parse(module)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {'fit','backward','step','Adam','AdamW','SGD','make_criterion','loaders'}


def test_restore_real_training_safe_checkpoint_and_consumed_grant(tmp_path):
    archive = Path(c.read(ROOT/r.REPORT/'archive_verification.json')['archives'][0]['path'])
    source = r.restore(ROOT, archive, archive.with_suffix('.sha256'), tmp_path/'restored')
    state, checkpoint = r.verify_final(ROOT, source)
    assert (state['epoch'],state['updates'],state['cursor']) == (4,28976,0)
    assert c.sha256(checkpoint) == r.identity(ROOT)['checkpoint_sha256']
    model = c.make_model(state['config'],'cpu')
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1,3,13,64,64))
    assert output.shape == (1,1,64,64) and torch.isfinite(output).all()
    with zipfile.ZipFile(archive) as z:
        assert (source/'authorization.json').read_bytes() == z.read('authorization.json')
    with pytest.raises(FileExistsError):
        r.restore(ROOT, archive, archive.with_suffix('.sha256'), source)
    # Fail closed before extraction on wrong SHA, and before inference on repeat completion.
    wrong = tmp_path/'wrong.sha256';wrong.write_text('0'*64)
    with pytest.raises(ValueError):
        r.restore(ROOT,archive,wrong,tmp_path/'wrong-source')
    assert not (tmp_path/'wrong-source').exists()
    (source/'evaluation').mkdir()
    c.save(source/'evaluation/complete.json',dict(status='COMPLETE'))
    with pytest.raises(ValueError,match='Already evaluated'):
        r.verify_final(ROOT,source)


def test_single_use_claim_survives_output_removal_and_new_authorization(tmp_path):
    digest = r.identity(ROOT)['checkpoint_sha256']
    lock = r.claim(tmp_path,digest,'new-grant-1')
    assert c.read(lock)['consumed']
    with pytest.raises(FileExistsError):
        r.claim(tmp_path,digest,'new-grant-2')


@pytest.mark.parametrize('field,value', [('authorized',False),('action','fire13-fit'),
    ('recovery_manifest_sha256','0'*64),('checkpoint_sha256','0'*64),
    ('training_archive_sha256','0'*64),('authorization_id',''),('evaluation_only',False),
    ('session_remaining_minutes',299),('quota_remaining_card_hours',9.99)])
def test_recovery_grant_refuses_mismatch_before_cuda(tmp_path,monkeypatch,field,value):
    a = c.read(ROOT/r.REPORT/'authorization_fire13-evaluate_NEW_DISABLED.json')
    a.update(authorized=True,authorized_by='CPU TEST',authorized_at='fixture',quota_checked_at='fixture',
        authorization_id='fixture',output_parent=str(tmp_path),session_remaining_minutes=300,
        quota_remaining_card_hours=10,card_hours_reserved=10,
        local_backup_destination='local',external_backup_destination='external')
    a[field] = value
    p=tmp_path/'grant.json';p.write_text(json.dumps(a))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No CUDA on bad grant'))
    with pytest.raises(ValueError):
        r.authorize(ROOT,p,'fire13-evaluate',tmp_path)


def test_counts_masks_threshold_ties_and_reject_all():
    p=np.array([.78,.9,.1,.9],np.float32); y=np.array([1,0,1,1]);v=np.array([1,1,1,0])
    assert counts(p,y,v,.78).tolist() == [1,1,1,0]
    assert counts(p,y,v,None).tolist() == [0,0,2,1]
    with pytest.raises(ValueError):folds(['same']*80)
    mapping=folds([f'2020/fire_{i}' for i in range(80)])
    assert [list(mapping.values()).count(i) for i in range(5)] == [16]*5


def test_candidate_evaluation_cpu_chain_writes_every_output(tmp_path,monkeypatch):
    from geofirex.c8_predictions import PredictionWriter,PredictionStore
    keys=[(f'2020/fire_{i}','2020-09-07',0,0) for i in range(80)]
    p=np.array([[.9,.8,.8],[.7,.6,.1]],np.float32)
    y=np.array([[1,0,0],[1,0,1]],np.uint8)
    def write(directory):
        w=PredictionWriter(directory,keys,{'fixture':'CPU ONLY'},(2,3))
        for key in keys:w.add(key,p,y,np.ones_like(y),np.zeros_like(y))
        return w.finish()
    domain=write(tmp_path/'baseline-predictions')
    policy=c.read(ROOT/c.old.REPORT/'propagation_policy.json')
    (tmp_path/'baseline-metrics').mkdir()
    baseline=c.exact_metrics(PredictionStore(tmp_path/'baseline-predictions'),tmp_path/'baseline-metrics',policy)
    class Model:
        def load_state_dict(self,state):assert state=={'CPU':'fixture'}
    checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'NOT A SCIENTIFIC CHECKPOINT')
    monkeypatch.setattr(c,'DOMAIN',domain)
    monkeypatch.setattr(c,'final_candidate',lambda *a:({'model':{'CPU':'fixture'}},checkpoint))
    monkeypatch.setattr(c,'make_model',lambda *a:Model())
    monkeypatch.setattr(c,'predict_validation',lambda root,maps,model,out,*args:write(out))
    # Full 2000-repeat numerical bootstrap is checked against historical results above.
    monkeypatch.setattr(comp.paired,'bootstrap_events',lambda *a,**kw:dict(CPU_fixture=True))
    original=c.read
    def read(path):
        if Path(path).name in {'baseline_reference.json','newpos4_reference.json','newpos2_reference.json'}:
            return baseline
        return original(path)
    monkeypatch.setattr(c,'read',read)
    result=c.candidate_evaluation(ROOT,[],tmp_path,tmp_path/'evaluation')
    assert result['status']=='COMPLETE'
    for name in ['evaluation.json','complete.json','paired_comparison/metrics.csv',
        'paired_comparison/FIRE13_acceptance.json','comparable_fp/result.json',
        'comparable_fp/thresholds_locked.json','comparable_fp/bootstrap_events.json']:
        assert (tmp_path/'evaluation'/name).is_file()
    assert original(tmp_path/'evaluation/complete.json')['checkpoint_sha256']==c.sha256(checkpoint)
    assert len(original(tmp_path/'evaluation/comparable_fp/result.json')['per_event'])==80
    assert original(tmp_path/'evaluation/evaluation.json')['propagation_pr']['average_precision']==pytest.approx(baseline['propagation_pr']['average_precision'])


def test_historical_results_counts_and_fixed_acceptance_compatible(tmp_path):
    base=c.baseline_json(ROOT)
    candidate=c.read(ROOT/c.REPORT/'newpos4_reference.json')
    result=comp.compare(ROOT,candidate,tmp_path/'paired',[x['event_id'] for x in base['per_event']])
    assert result['status']=='COMPLETE' and not result['promotion_authorized']
    rows=comp.paired.rows_from_result(candidate)
    summary=summarize(rows)
    for domain in ('pooled','new_fire'):
        for field in ('tp','fp','fn','tn','dice','recall','precision'):
            assert summary[domain][field] == candidate[domain][field]
    assert summary['pooled']['brier'] == pytest.approx(candidate['pooled']['brier'],abs=1e-15)
    assert set(serial_rows(rows))==set(rows)
    assert len(list(tmp_path.joinpath('paired').glob('bootstrap*.json')))==3
    assert (tmp_path/'paired/metrics.csv').exists()
    assert c.read(ROOT/c.REPORT/'acceptance.json')['threshold']==.78


def test_resources_fail_closed_and_original_budget_unchanged(tmp_path,monkeypatch):
    archive=Path(c.read(ROOT/r.REPORT/'archive_verification.json')['archives'][0]['path'])
    monkeypatch.setattr(c.old,'host_memory',lambda:(16*c.old.GIB,7*c.old.GIB))
    with pytest.raises(RuntimeError,match='RAM'):
        r.resource_preflight(ROOT,archive,tmp_path)
    monkeypatch.setattr(c.old,'host_memory',lambda:(32*c.old.GIB,16*c.old.GIB))
    monkeypatch.setattr(c.old,'disk_check',lambda path,n: (_ for _ in ()).throw(RuntimeError('disk')))
    with pytest.raises(RuntimeError,match='disk'):
        r.resource_preflight(ROOT,archive,tmp_path)
    monkeypatch.setattr(c.old,'disk_check',lambda path,n: n+1)
    receipt=r.resource_preflight(ROOT,archive,tmp_path)
    assert receipt['required_free_bytes']==18*c.old.GIB+4*72925838
    assert receipt['session_minutes']==300 and receipt['card_hours_reserved']==10


def test_original_archives_and_all_frozen_identities_preserved():
    before=c.read(ROOT/r.REPORT/'preservation_before.json')['files']
    c.old.verify_manifest(ROOT,{n:h for n,h in before.items() if not Path(n).is_absolute()})
    for name,digest in before.items():
        if Path(name).is_absolute():assert c.sha256(name)==digest
    assert len(c.read(ROOT/c.old.REPORT/'execution_manifest.json')['files'])==77
