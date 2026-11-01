"""Authorized one-session sample pilot: global 3600s cap, no acquisition.

Attach the immutable code package, certified sample and CPU integrity output.
Uses free Kaggle T4 only; local controller verifies quota/pay-to-scale before push.
"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time

START = time.monotonic()
OUT = Path('/kaggle/working/sample-pilot')
OUT.mkdir()
REPORT = {'qualification': 'pilot', 'limit_seconds': 3600, 'models': [], 'preflight': [],
          'test_2021': 'locked', 'zenodo_bytes': 0, 'gpu_training_started': False}


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False))


def remaining(deadline):
    return max(0, deadline-(time.monotonic()-START))


def require_time(deadline):
    if remaining(deadline) <= 0:
        raise TimeoutError('Global pilot budget exhausted')


def main():
    # 300s setup + 300s B0 + 3*900s models + 300s preservation = 3600s TOTAL.
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Global 3500s driver deadline')))
    signal.alarm(3500)
    sources = list(Path('/kaggle/input').rglob('code_manifest.json'))
    if not sources:
        # Kaggle may preserve the uploaded code ZIP as one member.
        import zipfile
        archives = list(Path('/kaggle/input').rglob('geofirex-sample-pilot-launch.zip'))
        assert len(archives) == 1, 'Code package missing or ambiguous'
        with zipfile.ZipFile(archives[0]) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall('/kaggle/working/code-mount')
        sources = list(Path('/kaggle/working/code-mount').rglob('code_manifest.json'))
    assert len(sources) == 1
    code = Path('/kaggle/working/GeoFire-X-2026')
    shutil.copytree(sources[0].parent, code)
    os.chdir(code)
    hashes = json.loads((code/'code_manifest.json').read_text())
    assert all(hashlib.sha256((code/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    REPORT['code_manifest_sha256'] = hashlib.sha256((code/'code_manifest.json').read_bytes()).hexdigest()
    sys.path.insert(0, str(code/'src'))
    os.environ['PYTHONPATH'] = str(code/'src')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    missing = [n for n in ('mlflow', 'yaml', 'rasterio', 'sklearn', 'matplotlib') if importlib.util.find_spec(n) is None]
    if missing:
        # Never install/replace the CUDA torch build; keep installation inside setup budget.
        packages = {'mlflow': 'mlflow==3.5.1', 'yaml': 'pyyaml', 'rasterio': 'rasterio',
                    'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib'}
        with (OUT/'dependencies.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                            *[packages[n] for n in missing]], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(180, remaining(240)))
    import numpy as np
    import torch
    import yaml
    from geofirex.runner import load_config, runner_model, seed_all
    from geofirex.runner_data import FoldData, fit_fold_normalizer, TemporalStore, TrainTiles
    from geofirex.kaggle_persistence import snapshot_training, verify_package
    from geofirex.corpus import verify_member, sha256_file
    from geofirex.unet import SegmentationLoss
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'T4 CUDA unavailable'
    REPORT['gpu'] = torch.cuda.get_device_name(0)
    REPORT['allocated_gpus'] = torch.cuda.device_count()
    REPORT['torch'] = str(torch.__version__)
    torch.set_num_threads(2)
    assert shutil.disk_usage('/kaggle/working').free >= 8*1024**3
    proofs = list(Path('/kaggle/input').rglob('sample-integrity.json'))
    assert len(proofs) == 1
    proof = json.loads(proofs[0].read_text())
    assert proof['success'] and len(proof['files']) == 108
    tiffs = list(Path('/kaggle/input').rglob('*.tif'))
    assert len(tiffs) == 108
    prepared = OUT/'prepared-configs'; prepared.mkdir()
    index = code/'data/manifests/real/corpus_zip_index_v1.json'
    entries = {e['path']: e for e in json.loads(index.read_text())['entries']}
    rows, roots = [], set()
    for row in proof['files']:
        matches = [p for p in tiffs if p.as_posix().endswith('/'+row['path'])]
        assert len(matches) == 1
        p = matches[0]; e = entries[row['path']]
        assert sha256_file(p) == row['sha256'] and verify_member(p, e)
        import rasterio
        with rasterio.open(p) as ds:
            rows.append(dict(event_id=e['event_id'], date=e['date'], relative_path=row['path'],
                             height=ds.height, width=ds.width))
        roots.add(str(p.parents[2]))
    with (prepared/'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    base = yaml.safe_load((code/'configs/runner_sample_gpu_pilot.yaml').read_text())
    base.update(manifest=str(prepared/'manifest.csv'), data_roots=sorted(roots), index=str(index),
                output_root=str(OUT/'runs'), amp=False)
    configs = {}
    for model in ('B0', 'B1', 'B2', 'B3'):
        kwargs = {'architecture': 'spatial_cnn_v2', 'width': 16} if model == 'B1' else (
            {'upsampling': 'nearest_v1'} if model in ('B2', 'B3') else {})
        cfg = {**base, 'model': model, 'model_kwargs': kwargs}
        if model == 'B0': cfg['max_run_seconds'] = 0
        p = prepared/(model+'.yaml'); p.write_text(yaml.safe_dump(cfg))
        configs[model] = load_config(p)
    data = FoldData(configs['B1'])
    assert [len(data.groups[r][5]) for r in ('train', 'val', 'test')] == [41, 42, 0]
    audit = data.audit({'train', 'val'})
    REPORT['corpus_counts'] = audit['corpus_counts']
    save('input-audit.json', audit)
    norm = fit_fold_normalizer(data.groups['train'][5])
    store = TemporalStore(data.windows('train'), norm, data.groups['train'][5], 256*1024**2)
    tiles = TrainTiles(store, 64, 2048, 42, .5); prior = tiles.prior()
    coordinate = tiles.positive_tiles[0]
    x, y, valid, _ = store.tile(*coordinate, 64)
    tensors = [torch.from_numpy(a.copy())[None].repeat(2, *([1]*a.ndim)).cuda() for a in (x, y, valid)]
    criterion = SegmentationLoss(.5, min(100, prior['negative_positive_ratio'])).cuda()
    # ALL architectures must pass deterministic FP32 backward before any real run.
    amp_compatible = True
    for name in ('B1', 'B2', 'B3'):
        for amp in (False, True):
            seed_all(42)
            model = runner_model(configs[name], norm).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            scaler = torch.amp.GradScaler('cuda', enabled=amp)
            try:
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', enabled=amp, dtype=torch.float16):
                        loss = criterion(model(tensors[0]).float(), tensors[1], tensors[2])
                    assert torch.isfinite(loss)
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    scaler.step(opt); scaler.update()
                torch.cuda.synchronize()
                REPORT['preflight'].append(dict(model=name, amp=amp, passed=True, loss=float(loss.detach().cpu())))
            except Exception as error:
                REPORT['preflight'].append(dict(model=name, amp=amp, passed=False, error=type(error).__name__+': '+str(error)[:300]))
                if not amp: raise
                amp_compatible = False
            finally:
                del model, opt, scaler
                torch.cuda.empty_cache()
    REPORT['amp'] = amp_compatible
    for name, cfg in configs.items():
        cfg['amp'] = amp_compatible
        # Write original portable config values, then retain the resolved version for provenance.
        (prepared/(name+'.yaml')).write_text(yaml.safe_dump(cfg))
    save('preflight.json', REPORT)
    require_time(300)
    REPORT['preparation_seconds'] = time.monotonic()-START
    REPORT['gpu_training_started'] = True
    power = (OUT/'gpu-power.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used',
                                '--format=csv,noheader,nounits', '-l', '1'], stdout=power, stderr=subprocess.DEVNULL)
    try:
        for name in ('B0', 'B1', 'B2', 'B3'):
            require_time(3300)
            phase_start = time.monotonic()
            limit = min(300 if name == 'B0' else 900, remaining(3300))
            record = {'model': name, 'limit_seconds': limit}
            before = set((OUT/'runs').glob(name+'-*-train-*'))
            with (OUT/(name+'.log')).open('w') as log:
                try:
                    subprocess.run([sys.executable, 'scripts/run_benchmark.py', 'train', '--config', str(prepared/(name+'.yaml'))],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=limit)
                    record['process_status'] = 'completed'
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    record['process_status'] = type(error).__name__
            record['wall_seconds'] = time.monotonic()-phase_start
            runs = set((OUT/'runs').glob(name+'-*-train-*'))-before
            if len(runs) == 1:
                run = runs.pop(); record['run'] = run.name
                if (run/'summary.json').exists():
                    s = json.loads((run/'summary.json').read_text()); record['summary'] = s
                    metrics = json.loads((run/f"validation_{s['best_epoch']:04d}.json").read_text())
                    record['metrics'] = metrics['micro']
                    record['epochs'] = len(s['history'])
                    record['updates'] = s['optimizer_updates']
                    record['training_seconds'] = sum(h['train_seconds'] for h in s['history'])
                    record['complete_budget'] = name == 'B0' or record['epochs'] == 10
                else:
                    checkpoints = sorted(run.glob('epoch_*.pt'))
                    if checkpoints:
                        state = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
                        record.update(checkpointed_epochs=state['epoch'], checkpointed_updates=sum(h['optimizer_updates'] for h in state.get('history', [])))
                    record['complete_budget'] = False
            REPORT['models'].append(record)
            save('campaign-progress.json', REPORT)
    finally:
        monitor.terminate(); monitor.wait(timeout=10); power.close()
    sums = [r['summary'] for r in REPORT['models'] if 'summary' in r]
    if sums:
        assert len({s['comparison_hash'] for s in sums}) == 1
        assert len({s['evaluation_target_hash'] for s in sums}) == 1
        # All four preprocessing fits must be identical, beyond just the target hash.
        normalizations = [json.loads((OUT/'runs'/r['run']/'normalization.json').read_text()) for r in REPORT['models'] if 'summary' in r]
        assert all(n == normalizations[0] for n in normalizations)
    for record in REPORT['models']:
        if 'summary' in record:
            package = OUT/'snapshots'/record['model']
            inventory = snapshot_training(OUT/'runs'/record['run'], package, tracking_db=OUT/'runs/tracking.db')
            record['snapshot'] = verify_package(package, inventory)
    # Choose only a demonstration candidate on validation, never unlock test 2021.
    candidates = [r for r in REPORT['models'] if r['model'] != 'B0' and 'summary' in r]
    if candidates:
        best = max(candidates, key=lambda r: r['metrics']['average_precision'])
        REPORT['demo_candidate'] = best['model']
        REPORT['selection_qualification'] = 'validation-only pilot; unequal/early-stopped budgets may bias comparison'


try:
    main()
except Exception as error:
    REPORT['error'] = type(error).__name__+': '+str(error)[:500]
    print(REPORT['error'], flush=True)
finally:
    REPORT['wall_seconds_before_preservation'] = time.monotonic()-START
    samples = {}
    if (OUT/'gpu-power.csv').exists():
        for row in csv.reader((OUT/'gpu-power.csv').read_text().splitlines()):
            try:
                stamp = dt.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
                samples.setdefault(row[1].strip(), []).append((stamp, float(row[2])))
            except (ValueError, IndexError): pass
    REPORT['energy_kwh_by_gpu'] = {g: sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(v,v[1:]))/3600000 for g,v in samples.items() if len(v)>1}
    REPORT['energy_scope'] = 'GPU board power sampled during B0/B1/B2/B3; excludes setup, CPU/RAM/PUE'
    save('campaign.json', REPORT)
    # Full durable output tree + compact archive; no existing file is removed.
    files = [{'path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ('.bin', '.tmp')]
    save('output-manifest.json', {'files': files})
    with tarfile.open('/kaggle/working/sample-pilot-preserved.tar.gz', 'w:gz') as archive:
        for entry in files:
            archive.add(OUT/entry['path'], arcname=entry['path'], recursive=False)
        archive.add(OUT/'output-manifest.json', arcname='output-manifest.json')
    end = {'wall_seconds': time.monotonic()-START, 'within_3600_seconds': time.monotonic()-START < 3600,
           'archive_sha256': hashlib.sha256(Path('/kaggle/working/sample-pilot-preserved.tar.gz').read_bytes()).hexdigest(),
           'files': len(files)}
    Path('/kaggle/working/pilot-completion.json').write_text(json.dumps(end, indent=2))
    print(json.dumps(end), flush=True)
