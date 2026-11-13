"""Embedded CPU certification runtime. Inputs supplied by the notebook builder."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import zlib


LOT001_2017_CONTEXT = frozenset({
    '2018/fire_21458836/2017-12-29.tif',
    '2018/fire_21458836/2017-12-30.tif',
    '2018/fire_21458836/2017-12-31.tif',
})


def validate_observation_path(name, split, *, allow_2017_context=False):
    """Split follows the event; calendar dates are validated independently.

    Four leading days are part of WildfireSpreadTS. The three pinned 2017
    context files require explicit approval; this never permits 2020/2021.
    """
    p = PurePosixPath(name)
    if (p.is_absolute() or len(p.parts) != 3 or '..' in p.parts or ':' in name
            or '\\' in name or p.parts[0] not in ('2018','2019') or p.suffix != '.tif'):
        raise ValueError('Unsafe or non-train canonical raster path: '+name)
    if split.get(str(p.parent)) != 'train':
        raise ValueError('Observation event is not assigned to train: '+name)
    year = dt.date.fromisoformat(p.stem).year
    if year in (2020, 2021):
        raise ValueError(f'Forbidden observation year {year} in train lot: {name}')
    if year not in (2018, 2019):
        if not (allow_2017_context and name in LOT001_2017_CONTEXT):
            raise ValueError(f'Observation year {year} outside approved train dates: {name}')
    return year


def validate_lot_dates(reference, split, *, allow_2017_context=False):
    years, mismatches = {}, []
    for entry in reference['files']:
        name = entry['path']
        year = validate_observation_path(name, split, allow_2017_context=allow_2017_context)
        years[str(year)] = years.get(str(year), 0)+1
        folder_year = int(PurePosixPath(name).parts[0])
        if year != folder_year:
            mismatches.append(dict(path=name, folder_year=folder_year, observation_year=year))
    return dict(observation_year_counts=dict(sorted(years.items())),
                folder_date_year_mismatches=mismatches,
                approved_2017_context=allow_2017_context, forbidden_2020_2021_observations=0)


def run(reference, observations, normalization, split, out, *, allow_2017_context=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(status='RUNNING', files_certified=0, bytes_certified=0,
                  dataset=reference['handle'], identity=reference['identity'],
                  anomalies=[], gpu_used=False, training=False)
    (out/'progress.json').write_text(json.dumps(result, indent=2))
    import psutil
    proc = psutil.Process()
    metrics = dict(peak_rss_bytes=0, minimum_available_bytes=psutil.virtual_memory().available)
    stopped = threading.Event()
    deadline = started + 1800

    def guard():
        rss, available = proc.memory_info().rss, psutil.virtual_memory().available
        metrics['peak_rss_bytes'] = max(metrics['peak_rss_bytes'], rss)
        metrics['minimum_available_bytes'] = min(metrics['minimum_available_bytes'], available)
        if time.monotonic() > deadline:
            raise TimeoutError('Certification exceeded 1800 seconds')
        if rss > 3*1024**3 or available < 2*1024**3:
            raise MemoryError('Certification resource limit reached')

    def watch():
        while not stopped.wait(1):
            try:
                guard()
            except Exception as e:
                (out/'resource-stop.json').write_text(json.dumps(dict(status='RESOURCE_STOP', error=str(e), **metrics)))
                os._exit(75)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        guard()
        if os.environ.get('KAGGLE_ACCELERATOR_TYPE', '').lower() not in ('', 'none', 'cpu'):
            raise RuntimeError('Accelerated session refused')
        if list(Path('/dev').glob('nvidia[0-9]*')):
            raise RuntimeError('GPU device present; CPU session required')
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                          OPENBLAS_NUM_THREADS='2', GDAL_NUM_THREADS='1')
        from geofirex.runner_data import digest
        if digest({k:v for k,v in reference.items() if k != 'identity'}) != reference['identity']:
            raise ValueError('Pinned local manifest identity invalid')
        inventories = list(Path('/kaggle/input').rglob('lot-inventory.json'))
        if len(inventories) != 1:
            raise ValueError('Exactly one attached lot inventory required')
        inventory_path = inventories[0]
        published = json.loads(inventory_path.read_text())
        if published != reference:
            raise ValueError('Attached inventory differs from local canonical manifest')
        root = inventory_path.parent/'rasters'
        expected = {f['path']:f for f in reference['files']}
        if len(expected) != 1269 or len(reference['files']) != 1269 or reference['observations'] != 1269:
            raise ValueError('Expected exactly 1269 distinct manifest files')
        if reference['role'] != 'train' or len(reference['events']) != 57 or len(set(reference['events'])) != 57:
            raise ValueError('Expected exactly 57 distinct train events')
        event_set = {str(PurePosixPath(p).parent) for p in expected}
        if event_set != set(reference['events']) or any(split.get(e) != 'train' for e in event_set):
            raise ValueError('Event set or annual train assignment mismatch')
        if sum(f['bytes'] for f in expected.values()) != reference['bytes']:
            raise ValueError('Manifest byte total mismatch')
        date_policy = validate_lot_dates(reference, split, allow_2017_context=allow_2017_context)
        all_attached = [p for p in Path('/kaggle/input').rglob('*') if p.is_file()]
        if any('2021' in p.parts or (p.suffix.lower() in ('.tif','.tiff') and p.stem.startswith('2021')) for p in all_attached):
            raise ValueError('Held-out 2021 data attached')
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if actual != set(expected) or len(all_attached) != 1270:
            raise ValueError(f'Attached file inventory differs: missing={len(set(expected)-actual)}, extra={len(actual-set(expected))}, total={len(all_attached)}')
        result['inventory'] = dict(manifest_files=len(published['files']), actual_geotiffs=len(actual),
            total_attached_files=len(all_attached), events=57, train_events=57,
            years=sorted({e.split('/')[0] for e in event_set}), heldout_2021_files=0,
            preview_reported_elements=617,
            preview_is_complete_manifest=False if len(published['files']) != 617 else True,
            json_top_level_keys=len(published), manifest_matches_local=True,
            inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(), **date_policy)
        with (out/'verified.jsonl').open('x', encoding='utf-8') as log:
            for i, entry in enumerate(reference['files'], 1):
                guard()
                path = root/entry['path']
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != entry['bytes']:
                    raise ValueError('Path or size mismatch: '+entry['path'])
                sha, crc, n = hashlib.sha256(), 0, 0
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(1024**2), b''):
                        guard()
                        sha.update(block)
                        crc = zlib.crc32(block, crc)
                        n += len(block)
                row = dict(path=entry['path'], bytes=n, sha256=sha.hexdigest(), crc32=f'{crc & 0xffffffff:08x}')
                if row != entry:
                    raise ValueError('SHA256/CRC32/size mismatch: '+entry['path'])
                log.write(json.dumps(row)+'\n')
                log.flush()
                result.update(files_certified=i, bytes_certified=result['bytes_certified']+n)
                if i % 50 == 0 or i == 1269:
                    (out/'progress.json').write_text(json.dumps(dict(**result, resources=metrics), indent=2))
                    print(f'Hash certification {i}/1269; {result["bytes_certified"]} bytes', flush=True)

        import numpy as np
        import rasterio
        import torch
        from rasterio.windows import Window
        from torch.utils.data import DataLoader
        from geofirex.unet_data import Normalizer, validate_layout
        from geofirex.benchmark import aligned_windows
        from geofirex.runner_data import FoldData, TemporalStore, TrainTiles
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        # Exercise the actual repository resolver on all 1269 portable tokens.
        resolver = FoldData.__new__(FoldData)
        resolver.config = {'corpus_scope':'train_validation'}
        resolver.root, resolver.roots = root.resolve(), [root.resolve()]
        for name in expected:
            if resolver.resolve_path(name) != (root/name).resolve():
                raise ValueError('Repository DataLoader path resolution differs')
        result['path_compatibility'] = dict(resolver='geofirex.runner_data.FoldData.resolve_path',
            resolved_files=1269, data_root=str(root), data_roots=[str(root)], windows_drive_dependency=False)
        samples = []
        indices = (0, len(reference['files'])//4, len(reference['files'])//2, 3*len(reference['files'])//4, -1)
        with rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_NUM_THREADS='1'):
            for i in indices:
                guard()
                name = reference['files'][i]['path']
                with rasterio.open(root/name) as ds:
                    validate_layout(ds)
                    if ds.crs is None:
                        raise ValueError('GeoTIFF CRS missing')
                    a = ds.read(window=Window(0,0,min(64,ds.width),min(64,ds.height)))
                    if a.shape[0] != 23 or np.isinf(a).any():
                        raise ValueError('Raster sample layout/infinite values invalid')
                    samples.append(dict(path=name, bands=ds.count, width=ds.width, height=ds.height,
                        crs=str(ds.crs), transform=list(ds.transform), tile_shape=list(a.shape),
                        nan_values=int(np.isnan(a).sum()), driver=ds.driver))
            result['rasterio_reads'] = samples
            if {r['relative_path'] for r in observations} != set(expected):
                raise ValueError('Local observation QC table differs from lot')
            relocated = [{**r,'relative_path':str(resolver.resolve_path(r['relative_path']))} for r in observations]
            groups, rejected = aligned_windows(relocated, split, 'train', contexts=(1,3,5))
            if any(not groups[t] for t in (1,3,5)):
                raise ValueError('No valid aligned temporal window')
            normalizer = Normalizer(**normalization['normalization'])
            loader_reads = []
            for context in (1,3,5):
                guard()
                w = groups[context][0]
                maxw = groups[5][0]
                geometry = None
                for path in (*maxw.context_paths, maxw.target_path):
                    with rasterio.open(path) as ds:
                        validate_layout(ds)
                        current = (ds.shape, str(ds.crs), tuple(ds.transform))
                        if geometry is not None and current != geometry:
                            raise ValueError('Temporal raster geometry mismatch')
                        geometry = current
                store = TemporalStore([w], normalizer, domain_windows=[maxw], cache_bytes=0)
                dataset = TrainTiles(store, size=64, samples=1, seed=42)
                x,y,valid = next(iter(DataLoader(dataset,batch_size=1,num_workers=0,pin_memory=False)))
                if list(x.shape) != [1,context,len(normalizer.channel_names),64,64] or list(y.shape) != [1,1,64,64]:
                    raise ValueError('Actual DataLoader tensor shape mismatch')
                if any(t.device.type != 'cpu' or t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (x,y,valid)):
                    raise ValueError('Actual DataLoader dtype/device/finite check failed')
                if not set(torch.unique(y).tolist()) <= {0.,1.} or not set(torch.unique(valid).tolist()) <= {0.,1.}:
                    raise ValueError('Target or validity is not binary')
                loader_reads.append(dict(context_days=context, event=w.event_id, context_dates=list(w.context_dates),
                    target_date=w.target_date, X=list(x.shape), Y=list(y.shape), validity=list(valid.shape),
                    dtype=str(x.dtype), device=str(x.device), workers=0, cache_bytes=0, finite=True))
            # Raw 23-band contract, spatially bounded, before feature normalization.
            w = groups[3][0]
            raw_days = []
            for path in w.context_paths:
                with rasterio.open(path) as ds:
                    a = ds.read(window=Window(0,0,min(64,ds.width),min(64,ds.height))).astype(np.float32)
                    a[22] = (~np.isnan(a[22])).astype(np.float32)
                    a[np.isnan(a)] = 0.
                    raw_days.append(a)
            raw = torch.from_numpy(np.stack(raw_days)).unsqueeze(0)
            if raw.shape[:3] != (1,3,23) or not torch.isfinite(raw).all():
                raise ValueError('Raw temporal contract failed')
            result['raw_temporal_window'] = dict(X=list(raw.shape), device=str(raw.device), dtype=str(raw.dtype))
        guard()
        result.update(status='LOT001_REMOTE_CPU_CERTIFIED', events=57, role='train',
            loader_reads=loader_reads, aligned_windows={str(k):len(v) for k,v in groups.items()},
            window_rejections=rejected, versions=dict(python=sys.version, rasterio=rasterio.__version__,
                torch=torch.__version__, numpy=np.__version__, psutil=psutil.__version__),
            resources=dict(**metrics, elapsed_seconds=time.monotonic()-started,
                deadline_seconds=1800, rss_limit_bytes=3*1024**3, min_available_bytes=2*1024**3,
                block_bytes=1024**2, gdal_cache_bytes=64*1024**2),
            scope='Only existing attached lot 001; no other lots or training certified')
        (out/'certificate.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as e:
        result.update(status='FAILED_STOP', error_type=type(e).__name__, error=str(e), resources=metrics)
        result['anomalies'].append(str(e))
        (out/'failure.json').write_text(json.dumps(result, indent=2))
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
