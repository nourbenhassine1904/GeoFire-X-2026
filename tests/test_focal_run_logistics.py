"""Archive, identity and disabled launch checks. No torch, pixels or GPU calls."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import sys
from types import SimpleNamespace, ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('focal_logistics', ROOT/'scripts/focal_run_logistics.py')
log = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(log)


def test_real_pass_receipt_matches_frozen_contract_without_rechecking_data():
    c = log.read(ROOT/'reports/focal_preparation_v1/focal_config.json')
    r = log.checked_pass(ROOT/'reports/focal_repro_precheck_v1/precheck_report_kaggle_PASS.json', c)
    assert r['corpus']['real_raster_or_npz_arrays_decoded'] == 0
    assert r['corpus']['datasets_downloaded'] == 0


def test_package_verifies_all_members_and_omits_ap_scratch(tmp_path):
    out = tmp_path/'synthetic-run'; out.mkdir()
    (out/'checkpoint_final.pt').write_bytes(b'NOT A REAL CHECKPOINT')
    (out/'config.json').write_text('{"synthetic":true}')
    scratch = out/'development32'/'_ap_global'; scratch.mkdir(parents=True)
    (scratch/'partial.bin').write_bytes(b'scratch')
    archive = log.package(out, 'PARTIAL')
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'checkpoint_final.pt', 'config.json', 'artifact_manifest.json'}
        assert not json.loads(z.read('artifact_manifest.json'))['complete']
    assert log.verify_zip(archive) == archive.with_suffix('.sha256').read_text().split()[0]
    again = log.package(out, 'PARTIAL')
    assert archive != again and archive.exists()


def test_verified_local_copy_and_corrupted_source_rejected(tmp_path):
    out = tmp_path/'synthetic-run'; out.mkdir()
    (out/'history.json').write_text('[]')
    archive = log.package(out, 'TEST')
    dest = tmp_path/'backup-root'; dest.mkdir()
    copies = log.backup(archive, archive.with_suffix('.sha256'), dest)
    assert len(copies) == 1 and log.sha(copies[0]) == log.sha(archive)
    assert log.read(Path(copies[0]).parent/'backup_verified.json')['archive_and_all_members_verified']
    with archive.open('ab') as f: f.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256'):
        log.backup(archive, archive.with_suffix('.sha256'), dest)


def test_manifest_member_tampering_rejected(tmp_path):
    p = tmp_path/'wrong.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('a.json','changed')
        z.writestr('artifact_manifest.json',json.dumps(dict(files=[dict(path='a.json',bytes=1,sha256='0'*64)])))
    with pytest.raises(ValueError, match='member differs'): log.verify_zip(p)


def test_disk_guard_stops_before_work(tmp_path):
    with pytest.raises(RuntimeError, match='free bytes'): log.disk_check(tmp_path,10**30)


def test_shipped_authorization_is_disabled_and_hash_bound():
    c = log.read(ROOT/'reports/focal_preparation_v1/focal_config.json')
    template = ROOT/'reports/focal_go_final_v1/authorization_TEMPLATE_DISABLED.json'
    a = log.read(template)
    assert a['training_authorized'] is False
    assert a['logistics_sha256'] == log.sha(ROOT/'scripts/focal_run_logistics.py')
    with pytest.raises(ValueError, match='authorization'): log.gate_authorization(template,c)


def test_notebook_frozen_payload_unchanged_and_launch_export_disabled():
    prep = ROOT/'reports/focal_preparation_v1'
    n = log.read(prep/'GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb')
    cells = [c for c in n['cells'] if c['cell_type']=='code']
    for c in cells:
        assert c['execution_count'] is None and c['outputs'] == []
        compile(c['source'],'notebook','exec')
    tree = ast.parse(cells[0]['source'])
    encoded = next(node.value.args[0].value for node in tree.body if isinstance(node,ast.Assign)
                   and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='payload')
    assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == log.CODE_SHA
    assert log.sha(prep/'focal_config.json') == log.CONFIG_SHA
    env = {}
    exec(cells[-2]['source'],env); exec(cells[-1]['source'],env)
    assert env['RUN_TRAINING'] is False and env['AUTHORIZATION_FILE'] is None and env['OUT'] is None
    assert cells[0]['source'].index('CUDA_VISIBLE_DEVICES') < cells[0]['source'].index('payload=')


def test_extracted_scientific_code_drift_rejected(tmp_path):
    prep = ROOT/'reports/focal_preparation_v1'
    with zipfile.ZipFile(prep/'focal-code.zip') as z: z.extractall(tmp_path)
    for source,name in ((prep/'focal-code.zip','focal-code.zip'),
                        (ROOT/'reports/focal_repro_precheck_v1/precheck_report_kaggle_PASS.json',
                         'precheck_report_kaggle_PASS.json')):
        (tmp_path/name).write_bytes(source.read_bytes())
    log.identity(tmp_path)
    (tmp_path/'geofirex/focal_experiment.py').write_text('changed')
    with pytest.raises(ValueError, match='scientific source differs'): log.identity(tmp_path)


def test_final_checkpoint_requires_sha_complete_budget_and_identity(tmp_path):
    c = log.read(ROOT/'reports/focal_preparation_v1/focal_config.json')
    p = tmp_path/'final.pt'; p.write_bytes(b'SYNTHETIC ONLY')
    ident = dict(config_sha256=log.CONFIG_SHA,code_sha256=log.CODE_SHA,
                 experiment=c['experiment'],initial_state_sha256=c['initial_state_sha256'])
    log.write_new(p.with_suffix('.json'),dict(sha256=log.sha(p)))
    log.write_new(tmp_path/'training_complete.json',dict(status='COMPLETE',updates=23404,epochs=4,
        history=[{}, {}, {}, {}],checkpoint=p.name,checkpoint_sha256=log.sha(p),identity=ident))
    assert log.final_checkpoint(tmp_path,c)[0] == p
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checkpoint SHA'): log.final_checkpoint(tmp_path,c)


@pytest.mark.parametrize('evaluation_fails', [False, True])
def test_training_orchestration_with_fake_runner_saves_before_ap_and_on_failure(tmp_path, monkeypatch, evaluation_fails):
    # Entire runner and CUDA namespace are fakes. No model, optimizer or pixels.
    working = tmp_path/'working'; working.mkdir()
    workspace = tmp_path/'workspace'; workspace.mkdir()
    corpus = tmp_path/'corpus'; (corpus/'index').mkdir(parents=True)
    for name in ('focal_config.json','focal-code.zip','baseline_evaluation32.json'):
        (workspace/name).write_bytes((ROOT/'reports/focal_preparation_v1'/name).read_bytes())
    with zipfile.ZipFile(workspace/'focal-code.zip') as z: z.extractall(workspace)
    (workspace/'focal_run_logistics.py').write_bytes((ROOT/'scripts/focal_run_logistics.py').read_bytes())
    (workspace/'precheck_report_kaggle_PASS.json').write_bytes(
        (ROOT/'reports/focal_repro_precheck_v1/precheck_report_kaggle_PASS.json').read_bytes())
    for name in ('normalization.json','index.json'): log.write_new(corpus/'index'/name,{})
    log.write_new(corpus/'mount-verification.json',dict(all_hashes_verified=True))
    c = log.read(workspace/'focal_config.json')
    out = working/'fake-only-run'
    args = SimpleNamespace(workspace=workspace,corpus=corpus,out=out,authorization=tmp_path/'fake-auth')
    original_path = Path
    monkeypatch.setattr(log,'Path',lambda p: working if str(p)=='/kaggle/working' else original_path(p))
    monkeypatch.setattr(log,'disk_check',lambda *a:10**12)
    monkeypatch.setattr(log,'gate_authorization',lambda *a:{})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda _: 'Tesla T4'))
    fake_exp = SimpleNamespace(verify_authorization=lambda *a:None)
    def evaluate(*a, **kw):
        assert list(working.glob('*-TRAINING-SAFE-*.zip')), 'Final checkpoint must be archived before AP'
        if evaluation_fails: raise RuntimeError('synthetic AP failure')
        (out/'development32').mkdir()
        log.write_new(out/'development32/evaluation.json',dict(status='COMPLETE'))
    fake_exp.evaluate = evaluate
    def fake_run(_):
        out.mkdir()
        p = out/'checkpoint_e5_u0023404_final.pt'; p.write_bytes(b'NOT A MODEL')
        ident = dict(config_sha256=log.CONFIG_SHA,code_sha256=log.CODE_SHA,
                     experiment=c['experiment'],initial_state_sha256=c['initial_state_sha256'])
        log.write_new(p.with_suffix('.json'),dict(sha256=log.sha(p)))
        log.write_new(out/'training_complete.json',dict(status='COMPLETE',updates=23404,epochs=4,
            history=[{}, {}, {}, {}],checkpoint=p.name,checkpoint_sha256=log.sha(p),identity=ident))
        fake_exp.evaluate()
        log.write_new(out/'paired_comparison.json',dict(candidate=dict(threshold=.78),boundary_inconclusive=False))
        log.write_new(out/'complete.json',dict(status='COMPLETE'))
        return dict(status='COMPLETE')
    fake_exp.run = fake_run
    package = ModuleType('geofirex'); package.focal_experiment=fake_exp
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setitem(sys.modules,'geofirex',package)
    if evaluation_fails:
        with pytest.raises(RuntimeError,match='synthetic AP'): log.train(args)
        assert not log.is_complete(out)
        assert list(working.glob('*-PARTIAL-*.zip'))
    else:
        assert log.train(args)['status']=='COMPLETE'
        assert log.is_complete(out)
        assert log.read(out/'selection32.json')['threshold']==.78
        assert list(working.glob('*-COMPLETE-*.zip'))
    assert (working/'GEOFIREX_FOCAL_SINGLE_RUN.lock.json').exists()
    assert fake_exp.evaluate is evaluate
