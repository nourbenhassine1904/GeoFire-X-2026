"""Hard-time-bounded profile. Discarded model; never a model-selection result."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not 1 <= a.seconds <= 900 or not 1 <= a.steps <= 100 or not 0 <= a.warmup <= 10:
        p.error('Limits: 900 seconds, 100 measured steps, 10 warmup steps')
    return a


def worker(a):
    import numpy as np
    import psutil
    import shutil
    import torch
    from torch.utils.data import DataLoader
    from geofirex.corpus import json_new
    from geofirex.gpu_operational import load_training_normalizer, spread_windows
    from geofirex.runner import choose_device, load_config, runner_model, seed_all
    from geofirex.runner_data import FoldData, TemporalStore, TrainTiles
    from geofirex.runner_evaluation import evaluate
    from geofirex.unet import SegmentationLoss
    from geofirex.unet_data import tile_coordinates
    out = Path(a.out)
    started = time.perf_counter()
    c = load_config(a.config)
    if c['corpus_scope'] != 'train_validation' or c['device'] != 'cuda':
        raise ValueError('This probe requires locked train/validation scope and CUDA')
    device = choose_device(c['device'])
    torch.set_num_threads(c['cpu_threads'])
    seed_all(c['seed'])

    def sync():
        torch.cuda.synchronize()

    def log(kind, **value):
        if shutil.disk_usage(out).free < 5_000_000_000:
            raise RuntimeError('Less than 5 GB free for profile outputs')
        if psutil.Process().memory_info().rss > 8*1024**3:
            raise RuntimeError('Profile exceeded 8 GiB process RSS')
        if torch.cuda.max_memory_reserved() > 12*1024**3:
            raise RuntimeError('Profile exceeded 12 GiB CUDA reservation')
        with (out/'progress.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(kind=kind, seconds=time.perf_counter()-started, **value))+'\n')
            f.flush()
            os.fsync(f.fileno())

    log('start', model=c['model'])
    data = FoldData(c)
    norm = load_training_normalizer(c, data)
    stores = {}
    for role, count in (('train', 16), ('val', 8)):
        indices = spread_windows(data.windows(role), count)
        stores[role] = TemporalStore([data.windows(role)[i] for i in indices], norm,
                                    [data.groups[role][5][i] for i in indices], int(c['reader_cache_mb']*1024**2))
    tiles = TrainTiles(stores['train'], c['patch_size'], c['batch_size']*(a.steps+a.warmup),
                       c['seed'], c['positive_tile_fraction'])
    prior = tiles.prior()
    weight = min(prior['negative_positive_ratio'], c['max_pos_weight']) if c['pos_weight'] == 'auto' else c['pos_weight']
    criterion = SegmentationLoss(c['dice_weight'], weight).to(device)
    model = runner_model(c, norm).to(device)
    trainable = c['model'] != 'B0'
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay']) if trainable else None
    amp = bool(c['amp'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loader = iter(DataLoader(tiles, batch_size=c['batch_size'], shuffle=False, num_workers=c['workers'], pin_memory=True))
    log('prepared', events={r: [w.event_id for w in s.windows] for r, s in stores.items()},
        prior=prior, setup_seconds=time.perf_counter()-started)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(a.warmup+a.steps):
        t = time.perf_counter()
        batch = next(loader)
        read = time.perf_counter()-t
        t = time.perf_counter()
        x, y, valid = [b.to(device, non_blocking=True) for b in batch]
        sync()
        transfer = time.perf_counter()-t
        t = time.perf_counter()
        if trainable:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
                loss = criterion(model(x).float(), y, valid)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite profiling loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update = int(scaler.get_scale() >= scale)
        else:
            with torch.inference_mode():
                model.predict_proba(x)
            update = 0
        sync()
        compute = time.perf_counter()-t
        if step >= a.warmup:
            rows.append(dict(read_seconds=read, h2d_seconds=transfer, compute_seconds=compute,
                             updates=update, patches=len(x)))
        log('step', step=step, measured=step >= a.warmup)
    t = time.perf_counter()
    # Fixed threshold for profiling: never tune or select on this subset.
    validation = evaluate(model, stores['val'], c, device, out/'validation-profile', threshold=.5,
                          criterion=criterion if trainable else None)
    validation_seconds = time.perf_counter()-t
    ntiles = sum(len(tile_coordinates(*s, c['patch_size'])) for s in stores['val'].shapes)
    elapsed = sum(sum(r[k] for k in ('read_seconds', 'h2d_seconds', 'compute_seconds')) for r in rows)
    updates, patches = sum(r['updates'] for r in rows), sum(r['patches'] for r in rows)
    result = dict(purpose='Discarded throughput probe; subset validation fixed threshold; not benchmark quality',
                  gpu=torch.cuda.get_device_name(), model=c['model'], input_hashes=data.hashes,
                  normalizer_sha256=c['normalization_sha256'], train_events_available=250, validation_events_available=201,
                  profiled_events={r: [w.event_id for w in s.windows] for r, s in stores.items()},
                  measured_steps=len(rows), warmup_steps=a.warmup, timings=rows,
                  updates_per_second=updates/elapsed, patches_per_second=patches/elapsed,
                  projected_epoch_train_seconds=c['samples_per_epoch']*elapsed/patches if trainable else 0,
                  validation_seconds=validation_seconds, validation_profile_tiles=ntiles,
                  validation_timings=validation['timings'],
                  validation_linear_projection_seconds=validation_seconds*78771/ntiles,
                  projection_limit='Cold/warm caches, full-scene variation and global AP merges may change full-validation cost; not a bound',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  total_seconds=time.perf_counter()-started, test_inputs=0, checkpoint_written=False)
    json_new(out/'profile.json', result)
    log('finished')


if __name__ == '__main__':
    a = arguments()
    if a.worker:
        worker(a)
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, __file__, '--worker', '--config', a.config, '--out', str(out),
                   '--seconds', str(a.seconds), '--steps', str(a.steps), '--warmup', str(a.warmup)]
        from geofirex.job_control import supervise
        with (out/'worker.log').open('x', encoding='utf-8') as log:
            result = supervise(command, seconds=a.seconds, log=log)
        (out/'supervisor.json').write_text(json.dumps(result), encoding='utf-8')
        raise SystemExit(0 if result['status'] == 'FINISHED' else 1)
