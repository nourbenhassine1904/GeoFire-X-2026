"""Portable train/validation deployment. No acquisition or implicit publication."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from geofirex.corpus import json_new, sha256_file
from geofirex.feature_profiles import feature_profile
from geofirex.runner_data import FoldData, TemporalStore, digest
from geofirex.unet_data import Normalizer


def training_tokens(data):
    return {data.path_tokens[p] for w in data.groups['train'][5] for p in w.context_paths}


def load_training_normalizer(config, data):
    """Pin statistics, feature contract, input tables and exact train-only fit set."""
    path = Path(config['normalization_artifact'])
    if sha256_file(path) != config['normalization_sha256']:
        raise ValueError('Normalization artifact SHA256 mismatch')
    proof = json.loads(path.read_text(encoding='utf-8'))
    if proof['schema'] != 'train-normalizer-portable-v1':
        raise ValueError('Unsupported normalization certificate')
    if proof['bindings'] != {k: data.hashes[k] for k in ('manifest', 'window_manifest', 'split', 'index')}:
        raise ValueError('Normalizer input table bindings differ')
    value = proof['normalization']
    tokens = value['fit_paths']
    expected = training_tokens(data)
    if len(tokens) != len(set(tokens)) or set(tokens) != expected:
        raise ValueError('Normalizer fit set differs from train maximum-context inputs')
    if any(data.assignment[t.rsplit('/', 1)[0]] != 'train' for t in tokens):
        raise ValueError('Held-out normalization input')
    profile = feature_profile(config)
    if (value['continuous'] != list(profile['continuous']) or value['include_landcover'] != profile['landcover']
            or value['clip'] != config['normalization_clip'] or proof['feature_policy'] != config['feature_policy']):
        raise ValueError('Normalizer feature contract differs')
    for key in ('mean', 'std', 'finite_count'):
        a = np.asarray(value[key])
        if a.shape != (len(value['continuous']),) or not np.isfinite(a).all() or (key != 'mean' and (a <= 0).any()):
            raise ValueError('Invalid normalization statistics')
    paths = {token: path for path, token in data.path_tokens.items()}
    return Normalizer(**{**value, 'fit_paths': [paths[t] for t in tokens]})


def event_lots(rows, max_bytes=5_000_000_000):
    """Deterministic complete-event partitions; never mix train and validation."""
    lots = []
    for role in ('train', 'val'):
        groups = {}
        for r in rows:
            if r['split'] == role:
                groups.setdefault(r['event_id'], []).append(r)
            elif r['split'] not in ('train', 'val', 'test'):
                raise ValueError('Unknown split')
        current, used = [], 0
        for event, members in sorted(groups.items()):
            n = sum(r['bytes'] for r in members)
            if n > max_bytes:
                raise ValueError(f'Complete event exceeds lot ceiling: {event}')
            if current and used+n > max_bytes:
                lots.append((role, current))
                current, used = [], 0
            current += members
            used += n
        if current:
            lots.append((role, current))
    return lots


def prepare(prepared, index, raw_root, destination, *, owner):
    """Small metadata/code preparation only. Raw TIFFs are never copied or read."""
    prepared, destination = Path(prepared), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    bundle = destination/'bundle'
    metadata = bundle/'metadata'
    metadata.mkdir(parents=True)
    observations = pq.read_table(prepared/'observations.parquet')
    rows = observations.to_pylist()
    selected = [r for r in rows if r['split'] in ('train', 'val')]
    if len(selected) != 9490 or len({r['event_id'] for r in selected}) != 451:
        raise ValueError('Expected certified 451-event / 9490-observation train/validation corpus')
    if any(r['status'] != 'ok' for r in selected):
        raise ValueError('Uncertified observation')
    windows = pq.read_table(prepared/'windows.parquet')
    wr = windows.to_pylist()
    if any(r['split'] not in ('train', 'val') for r in wr):
        raise ValueError('Test windows forbidden')
    pq.write_table(pa.Table.from_pylist(selected, schema=observations.schema), metadata/'observations.parquet')
    shutil.copy2(prepared/'windows.parquet', metadata/'windows.parquet')
    shutil.copy2(prepared/'fold_00.csv', metadata/'fold_00.csv')
    shutil.copy2(index, metadata/'index.json')
    original = json.loads((prepared/'normalization.json').read_text(encoding='utf-8'))
    cpu = json.loads((prepared/'cpu-validation.json').read_text(encoding='utf-8'))
    source_sha = sha256_file(prepared/'normalization.json')
    reuse = cpu.get('normalization_reuse', {})
    if (source_sha != reuse.get('sha256') or cpu['normalization_train_events'] != 250
            or cpu['normalization_unique_train_contexts'] != 4948 or cpu['test_paths_used'] != 0):
        raise ValueError('Normalizer provenance not certified for all 250 train events')
    old_hashes = reuse['original_input_hashes']
    for key, path in dict(manifest=prepared/'observations.parquet', window_manifest=prepared/'windows.parquet',
                          split=prepared/'fold_00.csv', index=Path(index)).items():
        if sha256_file(path) != old_hashes[key]:
            raise ValueError('Prepared inputs changed since original normalizer fit')
    mapping = {str((Path(raw_root)/r['relative_path']).resolve()): r['relative_path'] for r in selected}
    tokens = sorted(mapping[p] for p in original['fit_paths'])
    expected = {p for w in wr if w['split'] == 'train' and w['context'] == 5 for p in w['context_paths']}
    if set(tokens) != expected or len(tokens) != 4948 or len({t.rsplit('/', 1)[0] for t in tokens}) != 250:
        raise ValueError('Source normalizer fit paths do not cover exactly the training inputs')
    binding_paths = dict(manifest='observations.parquet', window_manifest='windows.parquet',
                         split='fold_00.csv', index='index.json')
    proof = dict(schema='train-normalizer-portable-v1', feature_policy='conservative_fire_terrain',
                 normalization={**original, 'fit_paths': tokens},
                 bindings={k: sha256_file(metadata/p) for k, p in binding_paths.items()},
                 provenance=dict(source_sha256=source_sha, original_input_hashes=old_hashes,
                                 train_events=250, fit_observations=4948, validation_inputs=0, test_inputs=0,
                                 original_fit_seconds=reuse['original_fit_seconds']))
    json_new(metadata/'normalization.json', proof)
    indexed = {(e['event_id'], e['date']): e for e in json.loads(Path(index).read_text())['entries']}
    lots = []
    for i, (role, members) in enumerate(event_lots(selected), 1):
        slug = f'geofirex-wsts-fold0-tv-{i:03d}-v1'
        lot = dict(number=i, role=role, slug=slug, handle=f'{owner}/{slug}',
                   events=sorted({r['event_id'] for r in members}),
                   bytes=sum(r['bytes'] for r in members), observations=len(members),
                   files=[dict(path=r['relative_path'], bytes=r['bytes'], sha256=r['sha256'],
                               crc32=indexed[(r['event_id'], r['date'])]['crc32'])
                          for r in members])
        lot['identity'] = digest(lot)
        json_new(metadata/f'lot-{i:03d}.json', lot)
        lots.append({k: v for k, v in lot.items() if k != 'files'})
    plan = dict(schema='gpu-deployment-v1', raw_bytes=sum(r['bytes'] for r in selected),
                observations=9490, events=451, test_files=0, lots=lots,
                max_lot_raw_bytes=5_000_000_000, upload_budget_bytes=36_000_000_000,
                upload_budget_policy='one attempt per immutable lot; no automatic retries; archive sizes checked before upload',
                local_staging_reserve_bytes=20_000_000_000, gpu_profile_seconds_per_model=900,
                proposed_profile_seconds_total=3600, proposed_training_seconds_per_model=7200,
                output_ceiling_bytes=15_000_000_000, normalizer_sha256=sha256_file(metadata/'normalization.json'))
    json_new(bundle/'deployment.json', plan)
    prefix = bundle.resolve().relative_to(Path.cwd().resolve()).as_posix()
    configs = bundle/'configs'
    configs.mkdir()
    for model in ('B0', 'B1', 'B2', 'B3'):
        c = yaml.safe_load((prepared/'gpu_configs'/f'{model}_T3_seed42.yaml').read_text())
        c.update(corpus_scope='train_validation', data_root=f'/kaggle/input/{lots[0]["slug"]}/rasters',
                 data_roots=[f'/kaggle/input/{lot["slug"]}/rasters' for lot in lots],
                 **{k: f'{prefix}/metadata/{p}' for k, p in binding_paths.items()},
                 normalization_artifact=f'{prefix}/metadata/normalization.json',
                 normalization_sha256=plan['normalizer_sha256'], experiment='geofirex-fold0-expanded-operational-v1',
                 output_root='/kaggle/working/geofirex-runs-v1',
                 tracking_uri='sqlite:////kaggle/working/geofirex-runs-v1/tracking.db',
                 artifact_root='file:///kaggle/working/geofirex-runs-v1/mlflow_artifacts',
                 storage_policy='ephemeral_export', checkpoint_dataset='',
                 evaluation_scratch_root='/kaggle/temp/geofirex-ap-v1', evaluation_cleanup_runs=True,
                 epochs=1, samples_per_epoch=93616, early_stopping_patience=0, max_run_seconds=7000)
        (configs/f'{model}_T3_seed42.yaml').write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    return plan


def spread_windows(windows, count):
    groups = {}
    for i, w in enumerate(windows):
        groups.setdefault(w.event_id, []).append(i)
    events = sorted(groups)
    chosen = [events[i] for i in np.linspace(0, len(events)-1, min(count, len(events)), dtype=int)]
    return [groups[e][len(groups[e])//2] for e in chosen]


def cpu_integration(config, output):
    """Real Parquet -> relocated normalizer -> eight events -> four model forwards."""
    from geofirex.runner import runner_model, seed_all
    torch.set_num_threads(config['cpu_threads'])
    seed_all(config['seed'])
    start = time.perf_counter()
    data = FoldData(config)
    norm = load_training_normalizer(config, data)
    batches = []
    for role in ('train', 'val'):
        indices = spread_windows(data.windows(role), 4)
        ws = [data.windows(role)[i] for i in indices]
        ds = [data.groups[role][5][i] for i in indices]
        store = TemporalStore(ws, norm, ds, int(config['reader_cache_mb']*1024**2))
        start_read = time.perf_counter()
        from torch.utils.data import DataLoader, Dataset
        class RealTiles(Dataset):
            def __len__(self):
                return len(ws)*4
            def __getitem__(self, i):
                shape = store.shapes[i % len(ws)]
                r = min((i//len(ws))*16, max(0, shape[0]-64))
                c = min((i//len(ws))*16, max(0, shape[1]-64))
                return tuple(torch.from_numpy(a.copy()) for a in store.tile(i % len(ws), r, c, 64)[:3])
        loader = DataLoader(RealTiles(), batch_size=4, num_workers=0)
        model_outputs = {}
        read_seconds = 0.
        ready = time.perf_counter()
        for batchno, (x, y, valid) in enumerate(loader):
            read_seconds += time.perf_counter()-ready
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or valid.sum() <= 0:
                raise ValueError('Nonfinite or invalid real integration batch')
            if batchno == 0:
                for model in ('B0', 'B1', 'B2', 'B3'):
                    kwargs = {'B0': {}, 'B1': dict(architecture='spatial_cnn_v2', width=16),
                              'B2': dict(base_channels=8, depth=3, upsampling='nearest_v1'),
                              'B3': dict(upsampling='nearest_v1')}[model]
                    net = runner_model({**config, 'model': model, 'model_kwargs': kwargs}, norm).eval()
                    with torch.inference_mode():
                        prediction = net.predict_proba(x)
                    if prediction.shape != y.shape or not torch.isfinite(prediction).all():
                        raise ValueError('Invalid model output')
                    model_outputs[model] = list(prediction.shape)
            ready = time.perf_counter()
        batches.append(dict(split=role, events=[w.event_id for w in ws], patches=len(loader.dataset),
                            x_shape=list(x.shape), y_shape=list(y.shape), read_seconds=read_seconds,
                            patches_per_read_second=len(loader.dataset)/read_seconds, outputs=model_outputs,
                            total_seconds=time.perf_counter()-start_read))
    report = dict(status='PASSED_REAL_CPU_INTEGRATION', input_hashes=data.hashes,
                  normalizer_sha256=config['normalization_sha256'], train_events=250, validation_events=201,
                  targets={r: len(data.windows(r)) for r in ('train', 'val', 'test')},
                  test_rows=sum(data.assignment[r['event_id']] == 'test' for r in data.rows),
                  fit_files=len(norm.fit_paths), fit_events=len({data.path_tokens[p].rsplit('/', 1)[0] for p in norm.fit_paths}),
                  channels=norm.channel_names, batches=batches, seconds=time.perf_counter()-start,
                  optimizer_updates=0, raw_writes=0, network_bytes=0)
    json_new(output, report)
    return report
