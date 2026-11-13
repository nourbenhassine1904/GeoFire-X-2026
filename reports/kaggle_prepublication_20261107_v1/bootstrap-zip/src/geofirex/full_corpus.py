"""Full-corpus integration, read-only rasters, resumable streaming certificates.

Test 2021 QC is storage integrity only. It never supplies windows or statistics
to the training preparation functions below.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zlib
import shutil

import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.windows import Window as RasterWindow
from rasterio.warp import transform_bounds

from geofirex.benchmark import event_assignments
from geofirex.cloud_corpus import fingerprint
from geofirex.corpus import json_new, publication_record, safe_entry, sha256_file
from geofirex.unet_data import validate_layout


def local_resources(root):
    import platform
    result = dict(platform=platform.platform(), logical_cpus=os.cpu_count(),
                  workspace_free_bytes=shutil.disk_usage('.').free,
                  corpus_volume_free_bytes=shutil.disk_usage(root).free)
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes as w

        class Memory(ctypes.Structure):
            _fields_ = [('length', w.DWORD), ('load', w.DWORD)] + [
                (n, ctypes.c_ulonglong) for n in ('total_physical', 'available_physical',
                'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'extended')]

        class Counters(ctypes.Structure):
            _fields_ = [('cb', w.DWORD), ('faults', w.DWORD)] + [
                (n, ctypes.c_size_t) for n in ('peak_working_set', 'working_set',
                'peak_paged_pool', 'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile')]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = w.HANDLE
        mem, counters = Memory(), Counters()
        mem.length, counters.cb = ctypes.sizeof(mem), ctypes.sizeof(counters)
        if kernel.GlobalMemoryStatusEx(ctypes.byref(mem)):
            result.update(ram_total_bytes=mem.total_physical, ram_available_bytes=mem.available_physical)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Counters), w.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            result.update(process_working_set_bytes=counters.working_set, process_peak_working_set_bytes=counters.peak_working_set)
    return result


def file_identity(path):
    s = Path(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino)


def audit_file(entry, root, block_size=256):
    """One checksum pass, then bounded all-band raster blocks; no writes."""
    safe_entry(entry['path'])
    path = (Path(root) / entry['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Raster escapes root')
    before = file_identity(path)
    if before['bytes'] != entry['geotiff_bytes']:
        raise ValueError('Indexed size mismatch')
    crc, sha = 0, hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            crc = zlib.crc32(block, crc)
            sha.update(block)
    if f'{crc & 0xffffffff:08x}' != entry['crc32']:
        raise ValueError('Indexed CRC mismatch')
    role = event_assignments([entry['event_id']], 0)[entry['event_id']]
    with rasterio.open(path) as ds:
        validate_layout(ds)
        if not ds.crs or min(ds.shape) < 1 or not np.isfinite(tuple(ds.transform)).all() or ds.transform.determinant == 0:
            raise ValueError('Invalid raster geometry')
        record = dict(event_id=entry['event_id'], date=entry['date'], year=entry['year'],
                      split=role, relative_path=str(path), indexed_path=entry['path'],
                      status='ok', bytes=before['bytes'], crc32=entry['crc32'], sha256=sha.hexdigest(),
                      identity=before, height=ds.height, width=ds.width, num_channels=ds.count,
                      dtype=ds.dtypes[0], crs=str(ds.crs), transform=list(ds.transform),
                      band_names=list(ds.descriptions), pixel_count=ds.height*ds.width,
                      geo_partition=str(ds.crs).replace(':', '_'),
                      sidecars={Path(p).name: sha256_file(p) for p in ds.files if Path(p).resolve() != path},
                      test_locked=role == 'test', positive_pixels=0, valid_pixels=0,
                      invalid_pixels=0, no_detection_pixels=0,
                      qc_scope='storage_integrity_only' if role == 'test' else 'train_validation')
        bounds = transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)
        record.update(zip(('west', 'south', 'east', 'north'), bounds))
        record.update(raster_center_longitude=(bounds[0]+bounds[2])/2,
                      raster_center_latitude=(bounds[1]+bounds[3])/2)
        finite, nan, inf = [np.zeros(23, np.int64) for _ in range(3)]
        minima, maxima = np.full(23, np.inf), np.full(23, -np.inf)
        explicit_mask = any(MaskFlags.per_dataset in f or MaskFlags.alpha in f for f in ds.mask_flag_enums)
        nodata = ds.nodatavals[22]
        for row in range(0, ds.height, block_size):
            for col in range(0, ds.width, block_size):
                window = RasterWindow(col, row, min(block_size, ds.width-col), min(block_size, ds.height-row))
                raw = ds.read(window=window)
                finite += np.isfinite(raw).sum(axis=(1, 2))
                nan += np.isnan(raw).sum(axis=(1, 2))
                inf += np.isinf(raw).sum(axis=(1, 2))
                for band in range(23):
                    values = raw[band][np.isfinite(raw[band])]
                    if values.size:
                        minima[band] = min(minima[band], float(values.min()))
                        maxima[band] = max(maxima[band], float(values.max()))
                valid = ds.dataset_mask(window=window) > 0 if explicit_mask else np.ones(raw.shape[-2:], bool)
                if nodata is not None and np.isfinite(nodata):
                    valid &= raw[22] != nodata
                record['positive_pixels'] += int((np.isfinite(raw[22]) & valid).sum())
                record['valid_pixels'] += int(valid.sum())
        if inf[22]:
            raise ValueError('Infinite active-fire code')
        record.update(finite_counts=finite.tolist(), nan_counts=nan.tolist(), inf_counts=inf.tolist(),
                      band_min=[float(v) if np.isfinite(v) else None for v in minima],
                      band_max=[float(v) if np.isfinite(v) else None for v in maxima])
        record['invalid_pixels'] = record['pixel_count'] - record['valid_pixels']
        record['no_detection_pixels'] = record['valid_pixels'] - record['positive_pixels']
    if file_identity(path) != before:
        raise ValueError('Raster changed during independent audit')
    return record


def summarize_events(records, index):
    expected, actual = defaultdict(list), defaultdict(list)
    for e in index['entries']:
        expected[e['event_id']].append(e)
    for r in records:
        actual[r['event_id']].append(r)
    result = []
    for event, entries in sorted(expected.items()):
        group = actual[event]
        geometries = {(r['height'], r['width'], r['crs'], tuple(r['transform'])) for r in group}
        expected_dates = {e['date'] for e in entries}
        observed = {r['date'] for r in group}
        dates = sorted(map(date.fromisoformat, observed))
        result.append(dict(event_id=event, split=event_assignments([event], 0)[event],
                           expected_observations=len(entries), observations=len(group),
                           geometry_consistent=len(geometries) == 1,
                           complete=observed == expected_dates and len(geometries) == 1,
                           missing_dates=sorted(expected_dates-observed),
                           gaps=[dict(start=a.isoformat(), end=b.isoformat(), missing_days=(b-a).days-1)
                                 for a, b in zip(dates, dates[1:]) if (b-a).days != 1]))
    return result


def _audit_corpus(index, root, output, *, limit=None, stop_requested=lambda: False):
    """Resume only independently successful entries whose file identities match."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specification = dict(schema_version=1, root=str(root), index_identity=fingerprint(index),
                         block_size=256, fold=0, test_policy='storage QC only; never model preparation')
    identity_path = output / 'audit-identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != specification:
            raise ValueError('Existing audit belongs to another corpus or contract')
    else:
        json_new(identity_path, specification)
    expected = {e['path'] for e in index['entries']}
    found = {p.relative_to(root).as_posix() for p in root.rglob('*.tif') if p.is_file()}
    missing, unexpected = sorted(expected-found), sorted(found-expected)
    if missing or unexpected:
        report = dict(status='INVENTORY_MISMATCH', missing=missing, unexpected=unexpected)
        json_new(output / f'inventory-error-{uuid.uuid4().hex}.json', report)
        raise ValueError(f'Physical inventory mismatch: {len(missing)} missing / {len(unexpected)} unexpected')
    cached = {}
    for log in sorted(output.glob('audit-attempt-*.jsonl')):
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break  # Interrupted tail is retained, never reused as proof.
                item = json.loads(line)
                if item['kind'] == 'audit_ok':
                    cached[item['record']['indexed_path']] = item['record']
    records, errors, processed, reused = [], [], 0, 0
    journal = output / f'audit-attempt-{time.time_ns()}-{uuid.uuid4().hex}.jsonl'
    with journal.open('x'):
        pass
    started = time.perf_counter()
    with rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_PAM_ENABLED='NO'):
        for ordinal, entry in enumerate(index['entries'], 1):
            if stop_requested() or (limit is not None and processed >= limit):
                break
            prior = cached.get(entry['path'])
            if prior and prior['identity'] == file_identity(root/entry['path']) and prior['crc32'] == entry['crc32']:
                records.append(prior)
                reused += 1
                continue
            publication_record(journal, 'audit_start', ordinal=ordinal, member=entry['path'])
            try:
                record = audit_file(entry, root)
                publication_record(journal, 'audit_ok', ordinal=ordinal, record=record)
                records.append(record)
            except (OSError, ValueError, rasterio.errors.RasterioError) as error:
                item = dict(member=entry['path'], error_type=type(error).__name__, message=str(error))
                publication_record(journal, 'audit_error', ordinal=ordinal, **item)
                errors.append(item)
            processed += 1
            if ordinal % 25 == 0 or errors:
                elapsed = time.perf_counter()-started
                print(f'AUDIT {ordinal}/{len(expected)} verified={len(records)} errors={len(errors)} elapsed={elapsed:.1f}s', flush=True)
    events = summarize_events(records, index)
    summary = dict(status='CERTIFIED' if len(records) == len(expected) and all(e['complete'] for e in events) and not errors else 'INCOMPLETE',
                   verified_observations=len(records), verified_events=sum(e['complete'] for e in events),
                   expected_observations=len(expected), expected_events=len(events), errors=errors,
                   processed_this_attempt=processed, reused_certificates=reused,
                   seconds=time.perf_counter()-started, verified_bytes=sum(r['bytes'] for r in records),
                   network_bytes=0, raw_writes=0, journal=str(journal),
                   split_events=dict(Counter(e['split'] for e in events if e['complete'])),
                   split_observations=dict(Counter(r['split'] for r in records)))
    suffix = str(time.time_ns())
    json_new(output/f'audit-summary-{suffix}.json', summary)
    if summary['status'] == 'CERTIFIED':
        json_new(output/f'certified-observations-{suffix}.json', records)
        json_new(output/f'certified-events-{suffix}.json', events)
    return summary


def audit_corpus(index, root, output, *, limit=None, stop_requested=lambda: False):
    from geofirex.acquisition_session import session_lock
    if Path(output).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Audit artifacts must remain outside the raw corpus')
    with session_lock(Path(output)/'.audit-session'):
        return _audit_corpus(index, root, output, limit=limit, stop_requested=stop_requested)


def canonical_tables(records, index):
    """Portable metadata and both natural and T=1/3/5-aligned window counts."""
    from geofirex.benchmark import aligned_windows
    from geofirex.windows import build_windows
    if {r['indexed_path'] for r in records} != {e['path'] for e in index['entries']}:
        raise ValueError('Preparation requires the entire certified inventory')
    events = summarize_events(records, index)
    if not all(e['complete'] for e in events) or any(r['status'] != 'ok' for r in records):
        raise ValueError('Incomplete or spatially inconsistent corpus')
    observations = [{**r, 'relative_path': r['indexed_path']} for r in records]
    assignment = event_assignments(sorted({r['event_id'] for r in records}), 0)
    seen_hashes = defaultdict(set)
    for r in records:
        seen_hashes[r['sha256']].add(r['split'])
    if any(len(roles) > 1 for roles in seen_hashes.values()):
        raise ValueError('Identical raster bytes cross annual splits')
    windows, counts = [], {}
    for role in ('train', 'val'):
        selected = [r for r in observations if r['split'] == role]
        groups, rejected = aligned_windows(observations, assignment, role)
        counts[role] = {}
        for context in (1, 3, 5):
            natural, reasons = build_windows(selected, context)
            counts[role][str(context)] = dict(natural=len(natural), aligned=len(groups[context]), rejections=reasons)
            windows.extend(dict(**asdict(w), context=context, split=role) for w in groups[context])
    return observations, windows, counts


def compare_extraction_certificate(root, records):
    """Compare freshly computed SHA-256 to the prior extraction receipt."""
    certificates = list((Path(root)/'.extraction-session').glob('complete-*.json'))
    if not certificates:
        return dict(status='NO_EXTRACTION_RECEIPT', independently_checked_crc_members=len(records))
    certificate = max(certificates, key=lambda p: p.stat().st_mtime_ns)
    prior = json.loads(certificate.read_text(encoding='utf-8'))
    expected = {r['path']: r for r in prior['files']}
    differences = [r['indexed_path'] for r in records if r['indexed_path'] not in expected or
                   (r['sha256'], r['crc32'], r['bytes']) !=
                   (expected[r['indexed_path']]['sha256'], expected[r['indexed_path']]['crc32'], expected[r['indexed_path']]['bytes'])]
    if differences or len(expected) != len(records):
        raise ValueError(f'Independent SHA/CRC differs from extraction receipt: {differences[:10]}')
    return dict(status='EXACT_SHA256_CRC_SIZE_MATCH', members=len(records), certificate=str(certificate),
                certificate_sha256=sha256_file(certificate),
                archive_md5=prior['archive'].get('md5'), archive_md5_status=prior['archive'].get('md5_status'),
                archive_note='Previously certified; no repeat read of the ZIP in this integration')


def spark_tables(observations, expected_windows, stage):
    """Spark SQL makes the window rows consumed by FoldData, not a side demo.

    PyArrow materializes collected bounded metadata to avoid Windows Hadoop
    native write utilities. Spark performs the lags/annual checks itself.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession, Window as SQLWindow, functions as F
    from geofirex.scientific_scaleup import write_metadata_tables
    stage = Path(stage)
    input_dir = stage / 'spark-input'
    input_dir.mkdir()
    write_metadata_tables(observations, [], input_dir)
    spark = (SparkSession.builder.master('local[2]').appName('GeoFireX-full-corpus-fold0-v1')
             .config('spark.ui.enabled', 'false').config('spark.sql.shuffle.partitions', '2')
             .config('spark.driver.memory', '512m').config('spark.executor.memory', '512m')
             .config('spark.driver.extraJavaOptions', '-XX:MaxDirectMemorySize=128m -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    started = time.perf_counter()
    try:
        frame = spark.read.parquet(str(input_dir / 'observations.parquet'))
        if frame.groupBy('event_id', 'date').count().filter('count > 1').count():
            raise ValueError('Duplicate Spark observation key')
        derived = F.when(F.col('year').isin(2018, 2019), F.lit('train')).when(F.col('year') == 2020, F.lit('val')).otherwise(F.lit('test'))
        if frame.filter(F.col('split') != derived).count():
            raise ValueError('Spark annual assignments mismatch')
        annual = [r.asDict() for r in frame.groupBy('year', 'split').agg(
            F.count('*').alias('observations'), F.countDistinct('event_id').alias('events')).orderBy('year').collect()]
        # Test rows are removed BEFORE constructing temporal lags or targets.
        selected = frame.filter(F.col('split').isin('train', 'val'))
        order = SQLWindow.partitionBy('event_id').orderBy('date')
        lagged = selected
        for offset in range(1, 6):
            lagged = lagged.withColumn(f'd{offset}', F.lag('date', offset).over(order))
            lagged = lagged.withColumn(f'p{offset}', F.lag('relative_path', offset).over(order))
        consecutive = F.lit(True)
        for offset in range(1, 6):
            consecutive = consecutive & (F.datediff(F.to_date('date'), F.to_date(f'd{offset}')) == offset)
        aligned = lagged.filter(consecutive).cache()
        rows = []
        for context in (1, 3, 5):
            table = aligned.select('event_id', F.col('date').alias('target_date'),
                F.col('relative_path').alias('target_path'), 'split', F.lit(context).cast('long').alias('context'),
                F.array(*[F.col(f'd{i}') for i in range(context, 0, -1)]).alias('context_dates'),
                F.array(*[F.col(f'p{i}') for i in range(context, 0, -1)]).alias('context_paths'))
            rows.extend(r.asDict(recursive=True) for r in table.orderBy('split', 'event_id', 'target_date').collect())
        key = lambda r: (r['split'], r['context'], r['event_id'], r['target_date'])
        rows.sort(key=key)
        reference = [{**r, 'context_dates': list(r['context_dates']), 'context_paths': list(r['context_paths'])} for r in expected_windows]
        if rows != sorted(reference, key=key):
            raise ValueError('Spark windows differ from independent canonical builder')
        # The final Parquet rows come from Spark's actual output.
        result_obs = [r.asDict(recursive=True) for r in frame.orderBy('event_id', 'date').collect()]
        for row in result_obs:
            row['sidecars'] = json.loads(row['sidecars_json'])
        write_metadata_tables(result_obs, rows, stage)
        pq.write_table(pa.Table.from_pylist(annual), stage / 'annual-inventory.parquet', compression='snappy')
        return dict(status='SPARK_EXECUTED', spark_version=spark.version, master='local[2]',
                    distributed_cluster=False, observations=frame.count(), window_rows=len(rows),
                    independent_window_comparison='exact_match', annual=annual,
                    materialization='Spark SQL results -> PyArrow Parquet; no Hadoop native writer',
                    seconds=time.perf_counter()-started)
    finally:
        spark.stop()


def prepare_corpus(index, root, audit_output, destination, *, engine='spark'):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from geofirex.real_manifest import FIELDNAMES
    from geofirex.scientific_scaleup import write_metadata_tables, feature_contracts
    if engine == 'spark':
        import importlib.util
        if importlib.util.find_spec('pyspark') is None:
            raise RuntimeError('PySpark absent: no download is started; install only with authorization or explicitly choose --engine python')
    audit_output, destination = Path(audit_output), Path(destination)
    certificates = sorted(audit_output.glob('certified-observations-*.json'))
    if not certificates:
        raise ValueError('No independent complete corpus certificate')
    records = json.loads(certificates[-1].read_text(encoding='utf-8'))
    spec = json.loads((audit_output/'audit-identity.json').read_text())
    if spec['root'] != str(Path(root).resolve()) or spec['index_identity'] != fingerprint(index):
        raise ValueError('Certificate root/index differs')
    for r in records:
        if file_identity(Path(root)/r['indexed_path']) != r['identity']:
            raise ValueError(f"Certified raster changed: {r['indexed_path']}")
    observations, windows, counts = canonical_tables(records, index)
    extraction_comparison = compare_extraction_certificate(root, records)
    destination.mkdir(parents=True, exist_ok=False)
    if engine == 'spark':
        processing = spark_tables(observations, windows, destination)
    elif engine == 'python':
        write_metadata_tables(observations, windows, destination)
        processing = dict(status='PYARROW_ONLY', note='Spark not executed')
    else:
        raise ValueError('Unknown metadata engine')
    rich = [{**{k: v for k, v in r.items() if k not in ('identity', 'sidecars')},
             'identity_json': json.dumps(r['identity'], sort_keys=True),
             'sidecars_json': json.dumps(r['sidecars'], sort_keys=True)} for r in observations]
    pq.write_table(pa.Table.from_pylist(rich), destination/'observations-qc.parquet', compression='snappy')
    event_records = summarize_events(records, index)
    # JSON carries potentially empty gap lists safely; Parquet has scalar inventory.
    pq.write_table(pa.Table.from_pylist([{k: v for k, v in e.items() if k not in ('missing_dates', 'gaps')} for e in event_records]),
                   destination/'events.parquet', compression='snappy')
    json_new(destination/'events.json', event_records)
    with (destination/'manifest.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for r in observations:
            writer.writerow({**r, 'band_names': ';'.join(r['band_names']), 'fire_id': r['event_id'].split('/')[1],
                             'observation_type': 'daily_geotiff'})
    with (destination/'fold_00.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['event_id', 'split'])
        writer.writerows(sorted(event_assignments([e['event_id'] for e in event_records], 0).items()))
    json_new(destination/'feature-contracts.json', feature_contracts())
    json_new(destination/'window-counts.json', counts)
    json_new(destination/'processing.json', processing)
    json_new(destination/'extraction-comparison.json', extraction_comparison)
    result = dict(status='PREPARED', events=len(event_records), observations=len(observations),
                  train_validation_events=sum(e['split'] != 'test' for e in event_records),
                  window_counts=counts, window_rows=len(windows), processing=processing, extraction_comparison=extraction_comparison,
                  test_policy='2021 storage inventory/QC only; excluded from window tables, normalization and selection',
                  certificate_sha256=sha256_file(certificates[-1]), index_identity=fingerprint(index),
                  artifacts={p.name: sha256_file(p) for p in destination.iterdir() if p.is_file()})
    json_new(destination/'preparation.json', result)
    return result


def patch_inventory(records, window_rows, root, size=64):
    """Exact nonempty joint T=5 footprints, including partial edge patches.

    Fully valid source observations prove every grid patch usable without
    rereading their labels. Only explicitly masked observations need mask I/O.
    """
    from geofirex.runner_data import read_tile
    lookup = {r['indexed_path']: r for r in records}
    result = {role: dict(examples=0, geometric_patches=0, usable_patches=0,
                        empty_patches=0, valid_joint_pixels=0, masked_examples=0)
              for role in ('train', 'val')}
    for w in window_rows:
        if w['context'] != 5:
            continue
        if w['split'] not in result or w['event_id'].startswith('2021/'):
            raise ValueError('Locked test window in patch inventory')
        paths = [*w['context_paths'], w['target_path']]
        group = [lookup[p] for p in paths]
        target = group[-1]
        height, width = target['height'], target['width']
        count = ((height+size-1)//size)*((width+size-1)//size)
        summary = result[w['split']]
        summary['examples'] += 1
        summary['geometric_patches'] += count
        if all(r['valid_pixels'] == r['pixel_count'] for r in group):
            summary['usable_patches'] += count
            summary['valid_joint_pixels'] += height*width
            continue
        summary['masked_examples'] += 1
        for row in range(0, height, size):
            for col in range(0, width, size):
                mask = None
                for path in paths:
                    _, valid = read_tile(Path(root)/path, row, col, size)
                    mask = valid.astype(bool) if mask is None else mask & valid.astype(bool)
                n = int(mask.sum())
                summary['valid_joint_pixels'] += n
                summary['usable_patches' if n else 'empty_patches'] += 1
    return dict(patch_size=size, domain='joint maximum context T=5 + target, identical for T=1/3/5',
                positive_sampling='not used for this inventory; validation never filtered by labels', by_split=result)


def configure_gpu(prepared, root, index_path, patch_counts, byte_counts=None):
    import math
    import yaml
    from geofirex.runner import load_config, runner_model
    from geofirex.unet_data import Normalizer
    prepared = Path(prepared)
    configs = prepared / 'gpu_configs'
    configs.mkdir()
    namespace = 'geofirex-full-corpus-fold0-v1'
    base = yaml.safe_load(Path('configs/runner_sample_gpu_pilot.yaml').read_text())
    samples = math.ceil(patch_counts['by_split']['train']['geometric_patches']/16)*16
    base.update(qualification='scientific', experiment=namespace,
                data_root=str(Path(root).resolve()), data_roots=[],
                manifest=str((prepared/'observations.parquet').resolve()),
                window_manifest=str((prepared/'windows.parquet').resolve()),
                split=str((prepared/'fold_00.csv').resolve()), index=str(Path(index_path).resolve()),
                output_root=f'artifacts/{namespace}', tracking_uri=f'sqlite:///artifacts/{namespace}/tracking.db',
                artifact_root=f'file:./artifacts/{namespace}/mlflow_artifacts',
                storage_policy='persistent', workers=0, reader_cache_mb=64,
                epochs=10, samples_per_epoch=samples, context=3, batch_size=16, eval_batch_size=16,
                max_run_seconds=7200, early_stopping_patience=3)
    runs = []
    norm = Normalizer([12, 14], [0., 0.], [1., 1.], [1, 1], [])
    for context in (1, 3, 5):
        for model in ('B0', 'B1', 'B2', 'B3'):
            config = {**base, 'model': model, 'context': context, 'seed': 42}
            config.pop('model_kwargs', None)
            if model == 'B1':
                config['model_kwargs'] = dict(architecture='spatial_cnn_v2', width=16)
            elif model in ('B2', 'B3'):
                config['model_kwargs'] = dict(upsampling='nearest_v1')
            path = configs/f'{model}_T{context}_seed42.yaml'
            with path.open('x', encoding='utf-8') as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            validated = load_config(path)
            instance = runner_model(validated, norm)
            runs.append(dict(config=str(path), context=context, model=model, seed=42,
                             parameters=sum(p.numel() for p in instance.parameters()),
                             maximum_optimizer_updates=0 if model == 'B0' else 10*math.ceil(samples/16),
                             sampled_patches_per_epoch=0 if model == 'B0' else samples,
                             full_grid_coverage_guaranteed=False,
                             status='PREPARED_NOT_AUTHORIZED_NOT_SUBMITTED'))
    budget = dict(status='NOT_AUTHORIZED_NOT_SUBMITTED', namespace=namespace, runs=runs,
                  next_step='Discarded T=3 throughput probes, 20 measured + 3 warmup steps for B1/B2/B3; separately authorize <= 0.5 GPU hours total including preparation',
                  proposed_first_T3_campaign_gpu_hours=6., proposed_per_trainable_run_gpu_hours=2.,
                  B0='Persistence, zero optimizer updates; prefer CPU, not included in GPU-hour allocation',
                  T1_T5='Prepared configs only; no GPU budget granted or assumed',
                  timing_caveat='Provisional resource envelopes, not measured full-corpus GPU runtimes. Profile full validation before approval.',
                  runner_time_limit='max_run_seconds is cooperative at epoch boundaries, not a hard wall-time cap; external job timeout required',
                  machine=dict(gpu_vram_gb=16, ram_gb=16, cpu_threads=2, loader_workers=0,
                               reader_cache_mb_per_process=64, minimum_free_disk_bytes=20*10**9),
                  transfer_policy='No automatic cloud publication or copying; mount an already authorized corpus and relocate data_root/window paths',
                  modalities='Only conservative_fire_terrain; other existing profiles retain their availability-evidence lock')
    if byte_counts is not None:
        budget['corpus_bytes_by_split'] = byte_counts
        budget['minimum_train_validation_volume_bytes'] = byte_counts['train'] + byte_counts['val'] + 20*10**9
    json_new(prepared/'gpu-budget.json', budget)
    return budget


def reuse_normalizer(source, prepared, data, profile):
    """Reuse a certified train-only fit only for equivalent, unchanged inputs."""
    import pyarrow.parquet as pq
    from geofirex.unet_data import Normalizer
    source, prepared = Path(source), Path(prepared)
    prior = json.loads((source/'cpu-validation.json').read_text())
    for name, key in (('observations.parquet', 'manifest'), ('windows.parquet', 'window_manifest')):
        if sha256_file(source/name) != prior['input_hashes'][key]:
            raise ValueError('Prior normalization input table changed')
        if not pq.read_table(source/name).equals(pq.read_table(prepared/name)):
            raise ValueError('Spark and prior normalization input tables differ')
    if any(data.hashes[k] != prior['input_hashes'][k] for k in ('index', 'split', 'targets')):
        raise ValueError('Normalization fold/index/targets differ')
    receipts = [json.loads(p.read_text()) for p in source.glob('delivery*.json')]
    checksum = sha256_file(source/'normalization.json')
    if not any(r['artifact_hashes'].get('normalization.json') == checksum for r in receipts):
        raise ValueError('Prior normalization SHA-256 has no certified receipt')
    norm = Normalizer(**json.loads((source/'normalization.json').read_text()))
    if (norm.continuous != profile['continuous'] or norm.include_landcover != profile['landcover']
            or norm.clip != data.config['normalization_clip']
            or not np.isfinite(norm.mean).all() or not np.isfinite(norm.std).all()
            or any(s <= 0 for s in norm.std) or any(n <= 0 for n in norm.finite_count)):
        raise ValueError('Cached normalization profile/statistics invalid')
    return norm, dict(source=str(source/'normalization.json'), sha256=checksum,
                      original_fit_seconds=prior['normalization_seconds'],
                      original_input_hashes=prior['input_hashes'])


def validate_loaders(prepared, root, index_path, audit_output, *, reuse_from=None):
    """Fit complete training-only normalization; bounded real CPU I/O/forward."""
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader
    from geofirex.feature_profiles import feature_profile
    from geofirex.runner import load_config, runner_model
    from geofirex.runner_data import FoldData, TemporalStore, TrainTiles, fit_fold_normalizer
    prepared = Path(prepared)
    records = json.loads(sorted(Path(audit_output).glob('certified-observations-*.json'))[-1].read_text())
    window_rows = pq.read_table(prepared/'windows.parquet').to_pylist()
    patches = patch_inventory(records, window_rows, root)
    json_new(prepared/'patch-inventory.json', patches)
    configure_gpu(prepared, root, index_path, patches,
                  {role: sum(r['bytes'] for r in records if r['split'] == role) for role in ('train', 'val', 'test')})
    config = load_config(prepared/'gpu_configs/B1_T3_seed42.yaml', dict(device='cpu'))
    torch.set_num_threads(config['cpu_threads'])
    data = FoldData(config)
    profile = feature_profile(config)
    norm_started = time.perf_counter()
    normalization_reuse = None
    if reuse_from:
        norm, normalization_reuse = reuse_normalizer(reuse_from, prepared, data, profile)
        if any(file_identity(Path(root)/r['indexed_path']) != r['identity'] for r in records):
            raise ValueError('Certified raw input changed before normalization reuse')
    else:
        norm = fit_fold_normalizer(data.groups['train'][5], continuous=profile['continuous'], landcover=profile['landcover'])
    allowed = {p for w in data.groups['train'][5] for p in w.context_paths}
    if set(norm.fit_paths) != allowed or any(data.assignment[data.path_tokens[p].rsplit('/', 1)[0]] != 'train' for p in norm.fit_paths):
        raise ValueError('Normalization paths differ from unique training contexts')
    json_new(prepared/'normalization.json', norm.to_dict())
    norm_seconds = time.perf_counter()-norm_started
    results = []
    for role in ('train', 'val'):
        # Metadata-only, deterministic spread across events, never label selection.
        events = sorted({w.event_id for w in data.groups[role][5]})
        chosen = [events[i*(len(events)-1)//3] for i in range(4)]
        keys = {(event, next(w.target_date for w in data.groups[role][5] if w.event_id == event)) for event in chosen}
        for context in (1, 3, 5):
            windows = [w for w in data.groups[role][context] if (w.event_id, w.target_date) in keys]
            domain = [w for w in data.groups[role][5] if (w.event_id, w.target_date) in keys]
            store = TemporalStore(windows, norm, domain, cache_bytes=64*1024**2)
            tiles = TrainTiles(store, 64, 48, 42, positive_fraction=0.)
            sampled_events = {windows[tiles.coordinate(i)[0]].event_id for i in range(len(tiles))}
            loader = DataLoader(tiles, batch_size=4, num_workers=0, shuffle=False)
            timings, count = [], 0
            iterator = iter(loader)
            for _ in range(12):
                start = time.perf_counter()
                x, y, valid = next(iterator)
                timings.append(time.perf_counter()-start)
                if x.shape[1:] != (context, 8, 64, 64) or y.shape[1:] != (1, 64, 64) or y.shape != valid.shape:
                    raise ValueError('DataLoader tensor shape mismatch')
                if not all(torch.isfinite(a).all() for a in (x, y, valid)) or not all(((a == 0) | (a == 1)).all() for a in (y, valid)):
                    raise ValueError('Invalid DataLoader values')
                count += x.shape[0]
            results.append(dict(split=role, context=context, events=sorted(sampled_events), patches=count,
                                first_batch_seconds=timings[0], seconds=sum(timings),
                                patches_per_second=count/sum(timings), p50_batch_seconds=float(np.median(timings)),
                                p95_batch_seconds=float(np.quantile(timings, .95)),
                                x_shape=list(x.shape), y_shape=list(y.shape), valid_shape=list(valid.shape),
                                cache_hits=store.reader.hits, cache_misses=store.reader.misses,
                                scope='bounded read/encoding CPU microbenchmark, sampled I/O only; no quality metric'))
    # Actual existing B0/B1/B2/B3 CPU forward contracts, no optimizer/backprop.
    forward = []
    t3_train = data.groups['train'][3][:1]
    store = TemporalStore(t3_train, norm, data.groups['train'][5][:1])
    batch = TrainTiles(store, 64, 1, 42)[0][0][None]
    for name in ('B0', 'B1', 'B2', 'B3'):
        c = load_config(prepared/f'gpu_configs/{name}_T3_seed42.yaml', dict(device='cpu'))
        model = runner_model(c, norm).eval()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(batch)
        if prediction.shape != (1, 1, 64, 64) or not torch.isfinite(prediction).all():
            raise ValueError(f'Invalid model I/O contract: {name}')
        forward.append(dict(model=name, output_shape=list(prediction.shape), seconds=time.perf_counter()-started,
                            parameters=sum(p.numel() for p in model.parameters()), device='cpu'))
    report = dict(status='VALIDATED_CPU_ONLY', normalization_seconds=norm_seconds,
                  normalization_mode='REUSED_VERIFIED_TRAIN_ONLY' if normalization_reuse else 'FIT_TRAIN_ONLY',
                  normalization_reuse=normalization_reuse,
                  normalization_unique_train_contexts=len(norm.fit_paths),
                  normalization_train_events=len({data.path_tokens[p].rsplit('/', 1)[0] for p in norm.fit_paths}),
                  normalizer_channels=norm.channel_names, channels=len(norm.channel_names), raw_bands=23,
                  test_paths_used=0, batches=results, forward_contracts=forward, patch_inventory=patches,
                  gpu_training=False, optimizer_updates=0, windows_source=str(prepared/'windows.parquet'),
                  input_hashes=data.hashes, torch_threads=2, loader_workers=0, reader_cache_mb=64,
                  resources=local_resources(root),
                  cache_caveat='OS cache not flushed; first batch is not a certified cold-disk measurement')
    json_new(prepared/'cpu-validation.json', report)
    return report


def delivery_report(prepared, audit_output):
    """Freeze hashes and actual measured outcomes after a successful CPU check."""
    import pyarrow.parquet as pq
    prepared, audit_output = Path(prepared), Path(audit_output)
    raw_root = json.loads((audit_output/'audit-identity.json').read_text())['root']
    generation = 1
    while True:
        suffix = '' if generation == 1 else f'-v{generation}'
        delivery_path = prepared/f'delivery{suffix}.json'
        report_path = prepared/f'REPORT{suffix}.md'
        if not delivery_path.exists() and not report_path.exists():
            break
        generation += 1
    preparation = json.loads((prepared/'preparation.json').read_text())
    cpu = json.loads((prepared/'cpu-validation.json').read_text())
    budget = json.loads((prepared/'gpu-budget.json').read_text())
    audit = json.loads(sorted(audit_output.glob('audit-summary-*.json'))[-1].read_text())
    records = json.loads(sorted(audit_output.glob('certified-observations-*.json'))[-1].read_text())
    windows = pq.read_table(prepared/'windows.parquet', columns=['event_id', 'split']).to_pylist()
    used_events = {role: len({w['event_id'] for w in windows if w['split'] == role}) for role in ('train', 'val')}
    qc = {}
    for role in ('train', 'val', 'test'):
        group = [r for r in records if r['split'] == role]
        qc[role] = dict(observations=len(group), raw_bands=sorted({r['num_channels'] for r in group}),
                        invalid_footprint_pixels=sum(r['invalid_pixels'] for r in group),
                        band_nan_counts=np.sum([r['nan_counts'] for r in group], axis=0).tolist(),
                        band_inf_counts=np.sum([r['inf_counts'] for r in group], axis=0).tolist(),
                        scope='storage integrity only, no model use' if role == 'test' else 'training preparation')
    source_paths = ['src/geofirex/full_corpus.py', 'src/geofirex/runner.py', 'src/geofirex/runner_data.py',
                    'src/geofirex/scientific_scaleup.py',
                    'scripts/integrate_full_corpus.py', 'scripts/run_benchmark.py', 'tests/test_full_corpus.py',
                    'scripts/install_local_spark_bounded.py', 'tests/test_local_spark_install.py']
    byte_counts = {role: sum(r['bytes'] for r in records if r['split'] == role) for role in ('train', 'val', 'test')}
    budget['corpus_bytes_by_split'] = byte_counts
    budget['minimum_train_validation_volume_bytes'] = byte_counts['train'] + byte_counts['val'] + 20*10**9
    import xml.etree.ElementTree as ET
    test_reports = {}
    for path in (audit_output/name for name in ('tests_delivery.xml', 'tests_contracts_final.xml',
                 'tests_spark_installer.xml', 'tests_spark_execution_v1.xml', 'tests_spark_reuse.xml', 'tests_spark_final.xml')):
        if path.is_file():
            tree = ET.parse(path).getroot()
            suite = tree if tree.tag == 'testsuite' else tree.find('testsuite')
            fields = {k: int(suite.get(k, '0')) for k in ('tests', 'errors', 'failures', 'skipped')}
            fields['passed'] = fields['tests']-fields['errors']-fields['failures']-fields['skipped']
            test_reports[path.name] = dict(**fields, sha256=sha256_file(path))
    installation_path = audit_output/'spark-installation.json'
    installation = json.loads(installation_path.read_text()) if installation_path.exists() else None
    receipt = dict(status='FULL_CORPUS_INTEGRATED' if preparation['processing']['status'] == 'SPARK_EXECUTED' else 'CORPUS_USABLE_SPARK_PENDING',
                   audit=audit, preparation=preparation, cpu_validation=cpu, window_events_used=used_events,
                   qc_by_split=qc, gpu_budget=budget, tests=test_reports, installation=installation, resources=local_resources(raw_root),
                   artifact_hashes={p.relative_to(prepared).as_posix(): sha256_file(p)
                                    for p in prepared.rglob('*') if p.is_file()},
                   source_hashes={p: sha256_file(p) for p in source_paths},
                   forbidden_actions=dict(corpus_acquisition=False, gpu_training=False, model_test_2021=False,
                                          cloud_publication=False, commit=False, push=False))
    json_new(delivery_path, receipt)
    patches = cpu['patch_inventory']['by_split']
    lines = ['# Corpus complet WildfireSpreadTS — fold 0', '',
             f"Certification indépendante : **{audit['verified_events']} événements / {audit['verified_observations']} GeoTIFF**.",
             f"CRC, SHA-256, lecture des 23 bandes, géométrie et dates ; {len(audit['errors'])} erreur(s).",
             f"Comparaison SHA-256 au certificat d’extraction : {preparation['extraction_comparison']['status']}.", '',
             'Train 2018–2019 : 250 événements ; validation 2020 : 201 ; test 2021 : 156, verrouillé.', '',
             f"Événements réellement représentés dans les fenêtres : train {used_events['train']}, validation {used_events['val']}.", '',
             '| Split | Fenêtres T1 naturelles | T3 naturelles | T5 naturelles | Cibles alignées réellement chargées par T | Patches utilisables à 64 px |',
             '|---|---:|---:|---:|---:|---:|']
    for role in ('train', 'val'):
        c = preparation['window_counts'][role]
        lines.append(f"| {role} | {c['1']['natural']} | {c['3']['natural']} | {c['5']['natural']} | {c['3']['aligned']} | {patches[role]['usable_patches']} |")
    lines += ['', f"Parquet fenêtres : {preparation['window_rows']} lignes. Traitement : {preparation['processing']['status']}.",
              'Les fenêtres alignées partagent les cibles et le masque conjoint T5 ; aucune cible de 2021 dans ces tables.',
              f"Normalisation : {cpu['normalization_unique_train_contexts']} fichiers contextes uniques de {cpu['normalization_train_events']} événements train ; 8 canaux encodés.", '',
              f"Mode du normalisateur : {cpu.get('normalization_mode', 'FIT_TRAIN_ONLY')}.", '',
              '## Mesures CPU de lecture et encodage', '',
              '| Split | T | Événements échantillonnés | Patches | Secondes | Patches/s |',
              '|---|---:|---:|---:|---:|---:|']
    for b in cpu['batches']:
        lines.append(f"| {b['split']} | {b['context']} | {len(b['events'])} | {b['patches']} | {b['seconds']:.3f} | {b['patches_per_second']:.2f} |")
    if preparation['processing']['status'] == 'SPARK_EXECUTED':
        processing = preparation['processing']
        lines += ['', f"Spark {processing['spark_version']}, {processing['master']} : {processing['seconds']:.2f} secondes pour SQL/collect/matérialisation, hors démarrage JVM et contrôles préalables des certificats."]
    if installation:
        lines += ['', f"Dépendances installées : {installation['versions']}. Corps HTTP consommés : {installation['confirmed_body_bytes']} / {installation['cap_bytes']} octets ; réservations conservatrices : {installation['reserved_body_bytes']} octets.",
                  'Aucune donnée supplémentaire du corpus téléchargée.']
    lines += ['', 'Mesures bornées, workers=0, deux threads CPU, cache lecteur 64 Mio. Le cache système n’a pas été vidé.',
              'Les sorties B0/B1/B2/B3 ont été contrôlées sur CPU, sans gradient ni mise à jour.', '',
              '## GPU — commandes futures, aucune exécution autorisée', '',
              'Proposition : calibration B1/B2/B3 T3, 20 étapes + 3 préchauffages, enveloppe totale 0,5 heure GPU incluant la préparation.',
              'Puis première campagne T3 seed42 : plafond proposé 2 heures par modèle entraînable, soit 6 heures GPU. B0 peut fonctionner sur CPU.',
              'Ces enveloppes demandent approbation et calibration réelle ; aucun temps GPU de ce corpus n’a été mesuré.',
              'Le garde-fou du runner agit aux frontières des époques : prévoir aussi un timeout externe dur.',
              'Prévoir GPU 16 Go, RAM 16 Go et volume libre pour le corpus train/validation, les sorties et une réserve de 20 Go.', '',
              '```powershell']
    for name in ('B1', 'B2', 'B3'):
        lines.append(f".\\.venv\\Scripts\\python.exe scripts/measure_gpu_throughput.py --config {prepared.as_posix()}/gpu_configs/{name}_T3_seed42.yaml --steps 20 --warmup 3 --out artifacts/full-corpus-probes-v1/{name}.json")
    lines += ['```', '', 'Baseline B0 sur CPU (commande préparée, non exécutée) :', '', '```powershell',
              f".\\.venv\\Scripts\\python.exe scripts/run_benchmark.py train --config {prepared.as_posix()}/gpu_configs/B0_T3_seed42.yaml --device cpu",
              '```', '', 'Après calibration et autorisation de la campagne :', '', '```powershell']
    for name in ('B1', 'B2', 'B3'):
        lines.append(f".\\.venv\\Scripts\\python.exe scripts/run_benchmark.py train --config {prepared.as_posix()}/gpu_configs/{name}_T3_seed42.yaml")
    lines += ['```', '', 'Les chemins configurés ciblent le PC actuel. Pour une autre machine : monter le corpus déjà autorisé,',
              'adapter les chemins de configuration ou les options --data-root/--manifest/--window-manifest/--split/--index.',
              'Les chemins stockés dans les tables restent relatifs au corpus. Aucune publication cloud n’est déclenchée.', '',
              'Nouveau namespace MLflow : `geofirex-full-corpus-fold0-v1`. Les autres profils de variables gardent leur verrou de disponibilité.',
              f"Tests : {json.dumps(test_reports, ensure_ascii=False)}.",
              'Les résultats, checkpoints et démonstrations historiques n’ont pas été modifiés.']
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    return dict(status=receipt['status'], report=str(report_path), delivery=str(delivery_path), windows=preparation['window_counts'], patches=patches)
