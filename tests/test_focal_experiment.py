"""CPU/synthetic tests only; no optimizer step, real pixels, CUDA or evaluation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from geofirex import focal_experiment as experiment
from geofirex.benchmark_models import make_model
from geofirex.fast_compact import FastCompactTiles, LocalitySampler, DTYPE
from geofirex.focal_loss import BinaryFocalWithLogits
from geofirex.loss_ablation import model_hash, THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'reports/unet_confirmation_20261112_v1/inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip'


@pytest.fixture
def baseline():
    # Read a saved JSON, never deserialize a checkpoint/prediction array.
    with zipfile.ZipFile(ARCHIVE) as z:
        return json.loads(z.read('B2_posweight_10/calibration32/evaluation.json'))


@pytest.fixture
def config(baseline):
    campaign = json.loads((ROOT / 'configs/scientific_improvements_v1.json').read_text())
    return dict(schema='geofirex-one-focal-v1', experiment='A_focal_gamma2_pos10_seed42_v1',
                model='B2', channels=8, context=3, seed=42, epochs=4, batch_size=16,
                train_tiles=93603, updates=23404, training_authorized=False,
                promotion_to_studio=False, runs=1, max_training_seconds=2700,
                loss=dict(name='binary_focal_logits', gamma=2., positive_weight=10.,
                          negative_weight=1., dice_weight=0., reduction='valid_pixel_mean'),
                optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, gradient_clip=1.),
                events=baseline['events'], thresholds=THRESHOLDS.tolist(),
                acceptance=campaign['predeclared_acceptance_on_development32'],
                locked_test_years=[2021], selection_scope='development32',
                confirmation_scope='observed169_descriptive_only', training_workers=4,
                validation_batch_size=32, torch_version='2.11.0+cu130', numpy_version='2.1.3',
                **{k:campaign['historical_reference'][k] for k in
                   ('initial_state_sha256', 'index_sha256', 'normalization_sha256')})


@pytest.mark.parametrize('gamma', [0., 2.])
def test_formula_and_valid_pixel_mean(gamma):
    z = torch.tensor([-4., -1., 0., 2., 5.], dtype=torch.float64)
    y = torch.tensor([1., 0., 1., 0., 1.], dtype=torch.float64)
    v = torch.tensor([1, 1, 0, 1, 1])
    p = z.sigmoid()
    terms = 10*y*(1-p).pow(gamma)*F.softplus(-z) + (1-y)*p.pow(gamma)*F.softplus(z)
    result = BinaryFocalWithLogits(gamma)(z, y, v)
    assert torch.allclose(result, terms[v.bool()].mean(), atol=1e-12)
    if gamma == 0:
        bce = F.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor(10.), reduction='none')
        assert torch.allclose(result, bce[v.bool()].mean())


def test_batch_without_fire_has_finite_nonzero_negative_gradients():
    z = torch.tensor([-3., 0., 3.], requires_grad=True)
    result = BinaryFocalWithLogits()(z, torch.zeros_like(z), torch.ones_like(z))
    result.backward()
    assert result > 0 and torch.isfinite(z.grad).all() and (z.grad > 0).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_extremes_finite_gradients_and_no_probability_clamp(dtype):
    z = torch.tensor([-1000., 1000., -1000., 1000.], dtype=dtype, requires_grad=True)
    y = torch.tensor([1., 0., 0., 1.], dtype=dtype)
    loss = BinaryFocalWithLogits()(z, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad[0] < 0 and z.grad[1] > 0


def test_masked_nan_padding_and_all_invalid_gradient_zero():
    z = torch.tensor([0., float('nan'), float('inf')], requires_grad=True)
    y = torch.tensor([1., float('nan'), float('nan')])
    loss = BinaryFocalWithLogits()(z, y, torch.tensor([1, 0, 0])); loss.backward()
    assert torch.isfinite(loss) and z.grad[1:].eq(0).all()
    z.grad = None
    empty = BinaryFocalWithLogits()(z, y, torch.zeros(3)); empty.backward()
    assert empty == 0 and z.grad.eq(0).all()


def test_invalid_images_do_not_change_denominator():
    z = torch.zeros(1, 1, 2, 2); y = torch.ones_like(z); v = torch.ones_like(z)
    one = BinaryFocalWithLogits()(z, y, v)
    two = BinaryFocalWithLogits()(torch.cat([z, z+10]), torch.cat([y, y]), torch.cat([v, v*0]))
    assert torch.equal(one, two)


def test_focal_finite_difference_gradient():
    z = torch.tensor([-2., .1, 3.], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1., 0., 1.], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: BinaryFocalWithLogits()(x, y), (z,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize('bad', ['shape', 'target', 'mask', 'logit'])
def test_invalid_valid_data_rejected(bad):
    z = torch.zeros(3); y = torch.ones(3); v = torch.ones(3)
    if bad == 'shape': y = y[:2]
    if bad == 'target': y[0] = .5
    if bad == 'mask': v[0] = .5
    if bad == 'logit': z[0] = float('nan')
    with pytest.raises(ValueError): BinaryFocalWithLogits()(z, y, v)


@pytest.mark.parametrize('kwargs', [dict(gamma=-1), dict(gamma=float('nan')),
                                   dict(positive_weight=0), dict(negative_weight=float('inf'))])
def test_invalid_loss_parameters_rejected(kwargs):
    with pytest.raises(ValueError): BinaryFocalWithLogits(**kwargs)


def test_same_model_fresh_deterministic_initialization_and_tiny_forward_gradients(config):
    # No checkpoint is loaded and no optimizer is constructed/stepped.
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(2)
        torch.manual_seed(42); model = make_model('B2', 8, 3)
        first = model_hash(model)
        torch.manual_seed(42); other = make_model('B2', 8, 3)
        assert first == model_hash(other)
        if first != config['initial_state_sha256']:
            # This local probe is NOT qualification for the historical GPU env.
            # The production guard must reject the mismatch, never update it.
            torch.manual_seed(42)
            with pytest.raises(ValueError, match='initialization differs'):
                experiment.fresh_paired_model(config)
        else:
            torch.manual_seed(42)
            assert model_hash(experiment.fresh_paired_model(config)) == first
        assert sum(p.numel() for p in model.parameters()) == 123977
        x = torch.randn(1, 3, 8, 9, 11)
        logits = model(x)
        assert logits.shape == (1, 1, 9, 11)
        assert torch.equal(model.predict_proba(x), logits.sigmoid())
        loss = BinaryFocalWithLogits()(logits, torch.zeros_like(logits))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert first == model_hash(model)  # backward does not update any weights
        with pytest.raises(ValueError): model(torch.zeros(1, 1, 8, 9, 11))


def test_sampler_determinism_exhaustiveness_and_resume_without_dataset_reads():
    class Metadata:
        records = np.array([(i % 3, i, i % 2, i % 5, i % 3) for i in range(1031)], dtype=DTYPE)
        def __len__(self): return len(self.records)
        def __getitem__(self, i): raise AssertionError('Pixels forbidden')
    ds = Metadata(); sampler = LocalitySampler(ds, seed=42)
    order = sampler.order()
    assert np.array_equal(np.sort(order), np.arange(len(ds)))
    assert np.array_equal(order, LocalitySampler(ds, seed=42).order())
    sampler.start = 16
    assert list(sampler) == order[16:].tolist()
    sampler.epoch = 1
    assert not np.array_equal(sampler.order(), order)


def test_metadata_default_path_does_not_touch_pixels_model_checkpoint_or_cuda(monkeypatch, config, tmp_path):
    def forbidden(*a, **kw): raise AssertionError('Training/inference/pixels/CUDA forbidden')
    monkeypatch.setattr(FastCompactTiles, '__getitem__', forbidden)
    monkeypatch.setattr(experiment, 'make_model', forbidden)
    monkeypatch.setattr(experiment, 'evaluate', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    p = tmp_path / 'config.json'; p.write_text(json.dumps(config))
    args = SimpleNamespace(config=p, corpus=ROOT/'reports/mission_20261111/compact_full',
                           index=ROOT/'reports/gpu_acceleration_20261111_v1/index', execute_training=False)
    report = experiment.run(args)
    assert report['real_pixels_opened'] == 0 and not report['gpu_queried']
    assert len(report['sampler_order_sha256']) == 4
    args.execute_training = True; args.authorization = None
    with pytest.raises(ValueError, match='authorization'): experiment.run(args)


@pytest.mark.parametrize('key,value', [('runs', 2), ('context', 1), ('seed', 43), ('epochs', 5),
                                      ('training_authorized', True), ('promotion_to_studio', True),
                                      ('locked_test_years', []), ('selection_scope', 'observed169')])
def test_frozen_experiment_rejects_scope_changes(config, key, value):
    config[key] = value
    with pytest.raises(ValueError): experiment.validate_config(config)


def test_frozen_loss_and_threshold_gates_cannot_drift(config):
    experiment.validate_config(config)
    config['loss']['gamma'] = 1
    with pytest.raises(ValueError): experiment.validate_config(config)
    config['loss']['gamma'] = 2
    config['thresholds'] = [.5]
    with pytest.raises(ValueError): experiment.validate_config(config)


def test_saved_baseline_comparison_does_not_claim_an_improvement(baseline, config):
    # No evaluation: same saved outputs compared to themselves.
    reference = deepcopy(baseline)
    # Its original selected .78 may be the same best grid threshold.
    compared = experiment.compare_saved(reference, baseline, config)
    assert not compared['acceptance'] and not compared['checks']['ap']
    assert not compared['promotion_to_studio'] and not compared['test2021_evaluated']
    assert all(r['dice_delta'] == 0 for r in compared['paired_event_differences'])
    assert compared['baseline']['new_fire']['positive_pixels'] > 0
    assert compared['baseline']['predicted_observed_ratio'] > 0
    candidate = deepcopy(baseline); candidate['events'][0] = '2021/fire_1'
    with pytest.raises(ValueError): experiment.compare_saved(candidate, baseline, config)


def test_comparison_rejects_changed_pixel_domain(baseline, config):
    candidate = deepcopy(baseline)
    for r in candidate['new_fire_candidates']: r['valid_pixels'] -= 1
    with pytest.raises(ValueError, match='domain'): experiment.compare_saved(candidate, baseline, config)


def test_threshold_ties_and_boundary_are_predefined(config, baseline):
    curves = [dict(threshold=t, positive_pixels=1, dice=.5, precision=p)
              for t, p in [(.1, .3), (.78, .5), (.9, .5)]]
    assert experiment.select_threshold(curves)['threshold'] == .9
    candidate = deepcopy(baseline)
    candidate['curves'][-1]['dice'] = 1.
    report = experiment.compare_saved(candidate, baseline, config)
    assert report['boundary_inconclusive'] and not report['acceptance']


def test_preparation_does_not_overwrite_existing_directory(tmp_path):
    spec = importlib.util.spec_from_file_location('focal_builder', ROOT/'scripts/prepare_focal_experiment.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    # Existing directory guard is reached before opening scientific inputs.
    with pytest.raises((ValueError, FileExistsError)): builder.build(tmp_path)


@pytest.mark.parametrize('bad', ['missing', 'runs', 'config', 'code', 'time', 'backup'])
def test_authorization_refuses_incomplete_or_other_experiment_before_cuda(tmp_path, monkeypatch, config, bad):
    from geofirex.calibration_io import sha
    cp = tmp_path/'config.json'; cp.write_text(json.dumps(config))
    bundle = tmp_path/'code.zip'; bundle.write_bytes(b'synthetic authorization fixture only')
    auth = dict(training_authorized=True, runs=1, experiment=config['experiment'],
                config_sha256=sha(cp), code_sha256=sha(bundle), max_training_seconds=2700,
                checkpoint_backup_destination='synthetic-example-no-transfer')
    if bad == 'runs': auth['runs'] = 2
    if bad == 'config': auth['config_sha256'] = 'different'
    if bad == 'code': auth['code_sha256'] = 'different'
    if bad == 'time': auth['max_training_seconds'] = 5400
    if bad == 'backup': auth['checkpoint_backup_destination'] = ''
    ap = tmp_path/'authorization.json'; ap.write_text(json.dumps(auth))
    args = SimpleNamespace(authorization=None if bad == 'missing' else ap, config=cp, code_archive=bundle)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA must not be queried'))
    with pytest.raises(ValueError, match='authorization|Authorization'):
        experiment.verify_authorization(args, config)


def test_historical_environment_is_a_hard_gate(tmp_path, monkeypatch, config):
    from geofirex.calibration_io import sha
    cp = tmp_path/'config.json'; cp.write_text(json.dumps(config))
    bundle = tmp_path/'code.zip'; bundle.write_bytes(b'synthetic fixture')
    ap = tmp_path/'authorization.json'
    ap.write_text(json.dumps(dict(training_authorized=True, runs=1, experiment=config['experiment'],
        config_sha256=sha(cp), code_sha256=sha(bundle), max_training_seconds=2700,
        checkpoint_backup_destination='synthetic-example-no-transfer')))
    monkeypatch.setattr(torch, '__version__', 'deliberately-incompatible')
    with pytest.raises(ValueError, match='environment'):
        experiment.verify_authorization(SimpleNamespace(authorization=ap, config=cp, code_archive=bundle), config)


def test_prepared_notebook_bundle_is_pinned_compilable_and_disabled(monkeypatch):
    import ast
    import base64
    import hashlib
    report = ROOT/'reports/focal_preparation_v1'
    assert report.is_dir(), 'Run offline preparation before delivery verification'
    nb = json.loads((report/'GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb').read_text(encoding='utf-8'))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    for cell in cells:
        assert cell['execution_count'] is None and cell['outputs'] == []
        compile(cell['source'], 'prepared-focal-notebook', 'exec')
    # Execute only the disabled launch/export cells; no bootstrap/mount scan.
    env = {'Path':Path}
    exec(cells[-2]['source'], env)
    exec(cells[-1]['source'], env)
    assert env['RUN_TRAINING'] is False and env['AUTHORIZATION_FILE'] is None and env['OUT'] is None
    build = json.loads((report/'build.json').read_text())
    tree = ast.parse(cells[0]['source'])
    encoded = next(node.value.args[0].value for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'payload')
    assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == build['code_sha256']
    from geofirex.calibration_io import sha
    assert sha(report/'focal-code.zip') == build['code_sha256']
    assert sha(report/'focal_config.json') == build['config_sha256']
    assert not build['training_executed'] and not build['gpu_queried'] and not build['submitted']
    with zipfile.ZipFile(report/'focal-code.zip') as z:
        c = json.loads(z.read('focal_config.json')); experiment.validate_config(c)
        assert z.read('baseline_evaluation32.json') == (report/'baseline_evaluation32.json').read_bytes()
        for name, digest in build['bundled_source_sha256'].items():
            source = z.read(name)
            assert hashlib.sha256(source).hexdigest() == digest
            compile(source, name, 'exec')
            local = (ROOT/'src'/name).read_text(encoding='utf-8')
            if name == 'geofirex/fast_compact.py':
                local = local.replace('from geofirex.mission_streaming import save,sha',
                                      'from geofirex.calibration_io import save,sha')
            assert source.decode('utf-8') == local
    verification = json.loads((report/'partitions_verified.json').read_text())
    assert verification['status'] == 'PASS' and len(verification['partitions']) == 11
    assert verification['objects'] == 459 and verification['arrays_decoded'] == 0
    assert len({p['identity'] for p in verification['partitions']}) == 11
