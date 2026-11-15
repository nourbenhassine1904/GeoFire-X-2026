"""One prepared Focal experiment. Default path reads metadata only.

Execution requires an explicit flag AND a separately supplied authorization.
No historical checkpoint initialization, public weights, search or 169 inference.
"""
import argparse
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from geofirex.benchmark_models import make_model
from geofirex.calibration_io import save, sha
from geofirex.compact_calibration import evaluate, event_ids, macro
from geofirex.fast_compact import FastCompactTiles, LocalitySampler
from geofirex.focal_loss import BinaryFocalWithLogits
from geofirex.loss_ablation import model_hash, select_threshold


def validate_config(c):
    fixed = dict(schema='geofirex-one-focal-v1', model='B2', channels=8, context=3,
                 seed=42, epochs=4, batch_size=16, train_tiles=93603, updates=23404,
                 training_authorized=False, promotion_to_studio=False, runs=1,
                 loss=dict(name='binary_focal_logits', gamma=2., positive_weight=10.,
                           negative_weight=1., dice_weight=0., reduction='valid_pixel_mean'),
                 optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, gradient_clip=1.),
                 locked_test_years=[2021], selection_scope='development32',
                 confirmation_scope='observed169_descriptive_only')
    for k, v in fixed.items():
        if c.get(k) != v:
            raise ValueError('Frozen Focal contract differs: ' + k)
    events = c['events']
    if len(events) != 32 or len(set(events)) != 32 or any(
            not e.startswith('2020/fire_') for e in events):
        raise ValueError('Exactly the historical 32 development events required')
    from geofirex.loss_ablation import THRESHOLDS
    if c['thresholds'] != THRESHOLDS.tolist():
        raise ValueError('Threshold grid changed')
    gates = dict(all_required=True, delta_pooled_ap_at_least=.005,
                 delta_pooled_dice_at_least=-.005, delta_new_fire_recall_at_least=.02,
                 new_fire_fp_ratio_at_most=1.15, delta_macro_event_dice_at_least=-.01)
    if any(c['acceptance'].get(k) != v for k, v in gates.items()):
        raise ValueError('Predeclared acceptance gates changed')
    if c['training_workers'] != 4 or c['validation_batch_size'] != 32:
        raise ValueError('Historical loader policy changed')
    if (c['max_training_seconds'] != 2700 or c['torch_version'] != '2.11.0+cu130'
            or c['numpy_version'] != '2.1.3'):
        raise ValueError('Frozen environment/time cap changed')


def preflight(config, corpus, index):
    """No __getitem__, tensor creation, model initialization or CUDA query."""
    c = config if isinstance(config, dict) else json.loads(Path(config).read_text(encoding='utf-8'))
    validate_config(c)
    index = Path(index)
    for name, key in [('index.json', 'index_sha256'), ('normalization.json', 'normalization_sha256')]:
        if sha(index / name) != c[key]:
            raise ValueError('Certified metadata differs: ' + name)
    meta = json.loads((index / 'index.json').read_text(encoding='utf-8'))
    if (len(meta['archives']) != 451 or len(set(meta['archives'])) != 451
            or any('/2021/' in '/' + p or not p.endswith('.zip') for p in meta['archives'])
            or any(not e.startswith(('2018/fire_', '2019/fire_', '2020/fire_'))
                   for e in meta['events'].values())):
        raise ValueError('Unexpected/locked compact archive metadata')
    train = val = None
    try:
        train = FastCompactTiles(corpus, index, 'train')
        val = FastCompactTiles(corpus, index, 'val')
        train_ids, val_ids = event_ids(train), event_ids(val)
        if (len(train), len(set(train_ids)), len(val), len(set(val_ids))) != (93603, 250, 78771, 201):
            raise ValueError('Corpus coverage differs')
        if any(e[:4] not in ('2018', '2019') for e in train_ids) or any(
                not e.startswith('2020/') for e in val_ids):
            raise ValueError('Annual split leakage')
        if not set(c['events']) <= set(val_ids) or int(np.isin(val_ids, c['events']).sum()) != 13759:
            raise ValueError('Development32 domain differs')
        # Pin the exact stream ordering, not a label-selected diagnostic subset.
        import hashlib
        orders = [hashlib.sha256(LocalitySampler(train, seed=42).order().tobytes()).hexdigest()]
        sampler = LocalitySampler(train, seed=42)
        for epoch in range(1, 4):
            sampler.epoch = epoch
            orders.append(hashlib.sha256(sampler.order().tobytes()).hexdigest())
        if c.get('sampler_order_sha256') and c['sampler_order_sha256'] != orders:
            raise ValueError('Sampler order differs')
        return dict(status='METADATA_VALID', train_tiles=len(train), development_tiles=13759,
                    events=c['events'], sampler_order_sha256=orders,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    runtime_matches_historical=(torch.__version__ == c['torch_version']
                                                and np.__version__ == c['numpy_version']),
                    real_pixels_opened=0, gpu_queried=False, training_executed=False)
    finally:
        if train is not None: train.close()
        if val is not None: val.close()


def row_at(rows, threshold):
    found = [r for r in rows if abs(r['threshold'] - threshold) < 1e-9]
    if len(found) != 1:
        raise ValueError('Missing/duplicate fixed threshold')
    return found[0]


def summarize(result, threshold):
    pooled = row_at(result['curves'], threshold)
    new = row_at(result['new_fire_candidates'], threshold)
    old = row_at(result['observed_fire_candidates'], threshold)
    per = [r for r in result['per_event'] if abs(r['threshold'] - threshold) < 1e-9]
    ratio = lambda r: (r['tp'] + r['fp']) / r['positive_pixels'] if r['positive_pixels'] else None
    return dict(threshold=threshold, pooled=pooled, new_fire=new, already_active=old,
                predicted_observed_ratio=ratio(pooled), new_predicted_observed_ratio=ratio(new),
                macro_event=macro(per), per_event=per)


def compare_saved(candidate, baseline, c):
    """Compare completed saved outputs; never infer the historical baseline."""
    if (candidate['status'] != 'COMPLETE' or baseline['status'] != 'COMPLETE'
            or candidate['events'] != c['events'] or baseline['events'] != c['events']
            or candidate['tiles'] != baseline['tiles'] or candidate['tiles'] != 13759):
        raise ValueError('Comparison event/tile domain differs')
    selected = select_threshold(candidate['curves'])
    a, b = summarize(candidate, selected['threshold']), summarize(baseline, .78)
    # Counts do not prove pixel identity alone: certified archive/index/source
    # identities must ALSO match, as enforced before an authorized execution.
    for field in ('pooled', 'new_fire', 'already_active'):
        for k in ('valid_pixels', 'positive_pixels'):
            if a[field][k] != b[field][k]:
                raise ValueError('Comparison pixel-domain counts differ')
    bm = {r['event_id']: r for r in b['per_event']}
    if (len(bm) != 32 or len(a['per_event']) != 32
            or {r['event_id'] for r in a['per_event']} != set(c['events'])
            or set(bm) != set(c['events'])):
        raise ValueError('Incomplete per-event comparison')
    differences = []
    for r in a['per_event']:
        q = bm[r['event_id']]
        if any(r[k] != q[k] for k in ('tiles', 'valid_pixels', 'positive_pixels')):
            raise ValueError('Per-event comparison domain differs')
        differences.append(dict(event_id=r['event_id'], dice_delta=r['dice'] - q['dice']))
    g = c['acceptance']
    checks = dict(ap=a['pooled']['average_precision'] - b['pooled']['average_precision'] >= g['delta_pooled_ap_at_least'],
                  dice=a['pooled']['dice'] - b['pooled']['dice'] >= g['delta_pooled_dice_at_least'],
                  new_recall=a['new_fire']['recall'] - b['new_fire']['recall'] >= g['delta_new_fire_recall_at_least'],
                  new_fp=a['new_fire']['fp'] <= g['new_fire_fp_ratio_at_most'] * b['new_fire']['fp'],
                  macro_dice=a['macro_event']['dice']['mean'] - b['macro_event']['dice']['mean'] >= g['delta_macro_event_dice_at_least'])
    boundary = selected['threshold'] in (min(c['thresholds']), max(c['thresholds']))
    return dict(scope='reused development32; exploratory, not an independent test',
                candidate=a, baseline=b, paired_event_differences=differences, checks=checks,
                boundary_inconclusive=boundary, acceptance=all(checks.values()) and not boundary,
                promotion_to_studio=False, observed169_evaluated=False, test2021_evaluated=False)


def verify_authorization(args, c):
    if not args.authorization:
        raise ValueError('Explicit future authorization file required; no execution authorized now')
    auth = json.loads(Path(args.authorization).read_text(encoding='utf-8'))
    if (auth.get('training_authorized') is not True or auth.get('runs') != 1
            or auth.get('experiment') != c['experiment'] or auth.get('config_sha256') != sha(args.config)
            or auth.get('code_sha256') != sha(args.code_archive)
            or auth.get('max_training_seconds') != c['max_training_seconds']
            or not auth.get('checkpoint_backup_destination')):
        raise ValueError('Authorization does not match this single frozen experiment/budget/backup')
    if torch.__version__ != c['torch_version'] or np.__version__ != c['numpy_version']:
        raise ValueError('Historical torch/numpy environment not reproduced')
    return auth


def fresh_paired_model(c):
    """CPU construction only; the caller seeds42. Never relax this digest."""
    network = make_model('B2', 8, 3)
    if model_hash(network) != c['initial_state_sha256']:
        raise ValueError('Fresh initialization differs from paired historical initialization')
    return network


def run(args):
    report = preflight(args.config, args.corpus, args.index)
    if not args.execute_training:
        return report
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    auth = verify_authorization(args, c)  # before CUDA, initialization or pixels
    baseline_path = Path(args.baseline)
    if sha(baseline_path) != c['baseline_evaluation_sha256']:
        raise ValueError('Saved historical reference differs')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if baseline['events'] != c['events'] or baseline['tiles'] != 13759:
        raise ValueError('Historical comparison scope differs')
    # The notebook first verifies every byte of the eleven read-only mounts.
    mounted = json.loads((Path(args.corpus) / 'mount-verification.json').read_text())
    if mounted.get('all_hashes_verified') is not True:
        raise ValueError('Full read-only mount verification required')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for the separately authorized experiment')
    out = Path(args.out).resolve()
    if out.is_relative_to(Path(args.corpus).resolve()) or out.is_relative_to(Path(args.index).resolve()):
        raise ValueError('Output must be separate from immutable inputs')
    out.mkdir(parents=True, exist_ok=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    network = fresh_paired_model(c)  # ALWAYS fresh; no historical torch.load
    network.to('cuda')
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.0001)
    criterion = BinaryFocalWithLogits()
    train = FastCompactTiles(args.corpus, args.index, 'train')
    val = FastCompactTiles(args.corpus, args.index, 'val')
    generator = torch.Generator().manual_seed(43)
    state_id = uuid.uuid4().hex
    epoch, cursor, updates = 1, 0, 0
    history = []; loss_sum = 0.; loss_pixels = 0
    identity = dict(config_sha256=sha(args.config), code_sha256=sha(args.code_archive),
                    experiment=c['experiment'], initial_state_sha256=c['initial_state_sha256'])
    try:
        if args.resume:
            if not args.resume_sha256 or sha(args.resume) != args.resume_sha256:
                raise ValueError('Own Focal resume requires its verified SHA256')
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Resume is not this exact Focal experiment')
            epoch, cursor, updates = state['epoch'], state['cursor'], state['updates']
            if not (1 <= epoch <= 5 and 0 <= cursor <= len(train)):
                raise ValueError('Invalid resume cursor')
            if cursor not in (0, len(train)) and cursor % 16:
                raise ValueError('Resume cursor is not a batch boundary')
            if updates != (epoch - 1) * 5851 + (cursor + 15) // 16 or updates > 23404:
                raise ValueError('Resume update budget differs')
            network.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            for values in optimizer.state.values():
                for k, v in values.items():
                    if isinstance(v, torch.Tensor) and k != 'step': values[k] = v.to('cuda')
            history, loss_sum, loss_pixels = state['history'], state['loss_sum'], state['loss_pixels']
            state_id = state['run_id']
            torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
            generator.set_state(state['loader_rng'])
        save(out / 'run.json', dict(config=c, identity=identity, run_id=state_id,
                                   authorization=auth, resume_sha256=args.resume_sha256,
                                   environment=dict(torch=torch.__version__, numpy=np.__version__,
                                                    gpu=torch.cuda.get_device_name(0))))
        sampler = LocalitySampler(train, seed=42)
        loader = DataLoader(train, batch_size=16, sampler=sampler, num_workers=4,
                            pin_memory=True, persistent_workers=True, prefetch_factor=2, generator=generator)
        started = time.monotonic()
        def checkpoint(reason):
            path = out / f'checkpoint_e{epoch}_u{updates:07d}_{reason}.pt'
            with path.open('xb') as stream:
                torch.save(dict(identity=identity, run_id=state_id, config=c,
                                model=network.state_dict(), optimizer=optimizer.state_dict(),
                                epoch=epoch, cursor=cursor, updates=updates, history=history,
                                loss_sum=loss_sum, loss_pixels=loss_pixels,
                                torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                                python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all(),
                                loader_rng=generator.get_state(), sampler=dict(seed=42, epoch=epoch-1, start=cursor)), stream)
            save(path.with_suffix('.json'), dict(path=path.name, bytes=path.stat().st_size,
                 sha256=sha(path), identity=identity, epoch=epoch, cursor=cursor, updates=updates,
                 restore='same experiment only, --resume and --resume-sha256, always NEW out'))
            return path
        with (out / 'steps.jsonl').open('x', encoding='utf-8') as log:
            while epoch <= 4:
                sampler.epoch = epoch - 1; sampler.start = cursor; network.train()
                for x, y, valid in loader:
                    if time.monotonic() - started >= c['max_training_seconds']:
                        path = checkpoint('paused')
                        save(out / 'paused.json', dict(status='PAUSED_NOT_COMPLETE', checkpoint=path.name,
                                                     updates=updates, evaluation_executed=False))
                        return dict(status='PAUSED_NOT_COMPLETE', updates=updates)
                    x, y, valid = [t.to('cuda', non_blocking=True) for t in (x, y, valid)]
                    if not valid.bool().any():
                        raise ValueError('Unexpected all-invalid training batch; no silent budget change')
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(network(x), y, valid)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite Focal loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); cursor += len(x); updates += 1
                    n = int(valid.sum()); loss_sum += float(loss.detach()) * n; loss_pixels += n
                    log.write(json.dumps(dict(epoch=epoch, updates=updates, cursor=cursor,
                                              loss=float(loss.detach()))) + '\n'); log.flush()
                    if updates % 100 == 0: checkpoint('periodic')
                history.append(dict(epoch=epoch, updates=updates, train_loss=loss_sum / loss_pixels))
                epoch += 1; cursor = 0; loss_sum = 0.; loss_pixels = 0
                checkpoint('epoch_complete')
        if updates != 23404: raise ValueError('Fixed training budget differs')
        final = checkpoint('final')
        save(out / 'training_complete.json', dict(status='COMPLETE', updates=updates,
             epochs=4, seconds=time.monotonic()-started, checkpoint=final.name,
             checkpoint_sha256=sha(final), history=history, identity=identity))
        # Exactly one final evaluation; no validation/epoch selection during fit.
        network.eval()
        eval_args = argparse.Namespace(batch_size=32, workers=4, device='cuda')
        result = evaluate(network, val, c['events'], c['thresholds'], eval_args,
                          out / 'development32', 'Focal development32', set())
        if sha(baseline_path) != c['baseline_evaluation_sha256']:
            raise ValueError('Saved historical reference differs')
        comparison = compare_saved(result, baseline, c)
        save(out / 'paired_comparison.json', comparison)
        save(out / 'complete.json', dict(status='COMPLETE', checkpoint=final.name,
             acceptance=comparison['acceptance'], promotion_to_studio=False,
             training_runs=1, test_2021_pixels_opened=0, observed169_pixels_opened=0))
        return dict(status='COMPLETE', acceptance=comparison['acceptance'])
    finally:
        train.close(); val.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'corpus', 'index'): p.add_argument('--' + name, required=True)
    for name in ('out', 'authorization', 'code-archive', 'baseline', 'resume', 'resume-sha256'):
        p.add_argument('--' + name)
    p.add_argument('--execute-training', action='store_true')
    args = p.parse_args()
    if args.execute_training and not all((args.out, args.code_archive, args.baseline)):
        p.error('Execution requires NEW output, pinned code archive and saved baseline')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__': main()
