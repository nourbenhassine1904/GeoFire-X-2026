"""Light CPU checks of a diagnostic notebook that has no training entry point."""
import ast
import base64
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/focal_repro_precheck_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


@pytest.fixture
def probe(): return module('repro_probe', ROOT/'scripts/focal_repro_precheck.py')


@pytest.fixture
def config(): return json.loads((ROOT/'reports/focal_preparation_v1/focal_config.json').read_text())


def test_hash_ast_is_exactly_the_historical_function():
    builder = module('repro_builder', ROOT/'scripts/prepare_focal_repro_precheck.py')
    with zipfile.ZipFile(ROOT/'reports/loss_ablation_20261112_v1/loss-ablation-code.zip') as z:
        historical = builder.extract(z.read('geofirex/loss_ablation.py').decode(), 'model_hash')
    current = builder.extract((ROOT/'scripts/focal_repro_precheck.py').read_text(), 'model_hash')
    assert ast.dump(ast.parse(current)) == ast.dump(ast.parse(historical))


def test_rng_trace_serialization_controls_and_no_forward_or_checkpoint(probe, config, monkeypatch):
    def forbidden(*a, **kw): raise AssertionError('No training/forward/checkpoint/CUDA transfer')
    from geofirex.benchmark_models import StackedUNet
    monkeypatch.setattr(StackedUNet, 'forward', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    before_torch = torch.get_rng_state().clone()
    before_np = np.random.get_state(); before_python = random.getstate()
    result = probe.initialization(config, dict(default_dtype='torch.float32', default_device='cpu',
                                              cuda_available=False), cuda_transfer=False)
    assert result['status'] == 'BLOCKED'  # no historical CUDA boundary on this CPU test
    assert result['cpu_sha256'] == result['repeat_sha256'] == result['numpy_seed_control_sha256']
    assert result['cpu_sha256'] == result['independent_bytes_sha256']
    assert result['cpu_sha256'] == result['legacy_reset_current_backend_sha256']
    assert result['cpu_sha256'] != result['one_torch_draw_control_sha256']
    assert result['hash_does_not_consume_rng'] and result['numpy_rng_unchanged_by_model']
    assert result['parameters'] == 123977 and result['head_bias'] == -3
    conv = [r for r in result['initializer_trace'] if r['kind'] == 'Conv2d']
    norm = [r for r in result['initializer_trace'] if r['kind'] == 'GroupNorm']
    assert len(conv) == 15 and all(r['rng_consumed'] for r in conv)
    assert len(norm) == 14 and not any(r['rng_consumed'] for r in norm)
    assert len(result['tensors']) == 44 and all(r['dtype'] == 'torch.float32' for r in result['tensors'])
    assert result['forward_calls'] == result['optimizer_steps'] == result['checkpoint_loads'] == 0
    assert torch.equal(before_torch, torch.get_rng_state()) and random.getstate() == before_python
    after_np = np.random.get_state()
    assert np.array_equal(before_np[1], after_np[1]) and before_np[2:] == after_np[2:]


@pytest.mark.parametrize('dtype,device', [('torch.float64','cpu'), ('torch.float32','cuda')])
def test_ambient_defaults_are_not_silently_changed(probe, config, dtype, device):
    result = probe.initialization(config, dict(default_dtype=dtype, default_device=device))
    assert result['status'] == 'BLOCKED' and len(result['causes']) == 1


def test_decision_does_not_authorize_training_even_if_every_prerequisite_passes(probe, config):
    env = dict(torch=config['torch_version'], numpy=config['numpy_version'], cuda_available=True)
    r = probe.decision(config, env, dict(status='PASS'), dict(status='PASS'))
    assert r['status'] == 'PASS' and not r['training_authorized'] and not r['RUN_TRAINING']
    assert not r['authorization_generated'] and r['optimizer_steps'] == 0


@pytest.mark.parametrize('failure', ['torch','numpy','gpu','initialization','corpus'])
def test_one_failed_prerequisite_blocks_pass(probe, config, failure):
    env = dict(torch=config['torch_version'], numpy=config['numpy_version'], cuda_available=True)
    initial = dict(status='PASS'); corpus = dict(status='PASS')
    if failure in ('torch','numpy'): env[failure] = 'incompatible'
    if failure == 'gpu': env['cuda_available'] = False
    if failure == 'initialization': initial['status'] = 'BLOCKED'
    if failure == 'corpus': corpus['status'] = 'BLOCKED'
    r = probe.decision(config, env, initial, corpus)
    assert r['status'] == 'BLOCKED' and r['causes']


def test_missing_mounts_rejected_before_any_payload_read(probe, config):
    with pytest.raises(ValueError, match='eleven'): probe.verify_corpus([], config, None)


def test_no_training_code_exists_in_notebook_or_embedded_payload():
    builder = module('repro_builder', ROOT/'scripts/prepare_focal_repro_precheck.py')
    notebook = json.loads((OUT/'GEOFIREX_FOCAL_PRECHECK_ONLY.ipynb').read_text(encoding='utf-8'))
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(code) == 5
    for cell in code:
        assert cell['execution_count'] is None and cell['outputs'] == []
        builder.no_training(cell['source']); compile(cell['source'], 'diagnostic-cell', 'exec')
    tree = ast.parse(code[0]['source'])
    assignments = {n.targets[0].id:n.value for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert assignments['RUN_TRAINING'].value is False and assignments['AUTHORIZATION_FILE'].value is None
    payload = base64.b64decode(assignments['payload'].args[0].value)
    build = json.loads((OUT/'build.json').read_text())
    assert hashlib.sha256(payload).hexdigest() == build['code_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert not any(n.endswith(('focal_experiment.py','loss_ablation.py','fast_compact.py')) for n in z.namelist())
        assert z.read('focal_config.json') == (ROOT/'reports/focal_preparation_v1/focal_config.json').read_bytes()
        for name in build['sources']:
            source = z.read(name).decode(); builder.no_training(source); compile(source,name,'exec')
            assert hashlib.sha256(source.encode()).hexdigest() == build['sources'][name]
    assert hashlib.sha256((ROOT/'reports/focal_preparation_v1/GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb').read_bytes()).hexdigest() == build['existing_focal_notebook_sha256']


def test_existing_output_is_never_overwritten():
    builder = module('repro_builder', ROOT/'scripts/prepare_focal_repro_precheck.py')
    with pytest.raises(FileExistsError): builder.build(OUT)
