"""Offline scale-up contracts. Pixels are read only for acquired train/val data."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
import hashlib
import json

import numpy as np
import rasterio

from geofirex.benchmark import aligned_windows, event_assignments
from geofirex.cloud_corpus import validate_index, seal, fingerprint, check_seal
from geofirex.corpus import sha256_file, verify_member
from geofirex.runner_data import _read_selected, RasterReader
from geofirex.unet_data import tile_coordinates, validate_layout
from geofirex.windows import Window, build_windows


def feature_contracts():
    """Versioned channel identities without granting feature availability."""
    from geofirex.feature_profiles import PROFILES
    from geofirex.unet_data import Normalizer
    contracts = {}
    for name, profile in PROFILES.items():
        bands = list(profile['continuous'])
        normalizer = Normalizer(bands,[0.]*len(bands),[1.]*len(bands),[0]*len(bands),[],profile['landcover'])
        contracts[name] = seal(dict(schema_version=1, policy=name, raw_continuous_indices=bands,
                                    channel_names=normalizer.channel_names,
                                    channels=len(normalizer.channel_names),
                                    availability='historical conditional end-of-day-J assumption' if name == 'conservative_fire_terrain'
                                    else 'BLOCKED until reviewed availability evidence for every selected band',
                                    normalization='fit on unique training context paths only; dummy values here are schema only'))
    return contracts


def write_metadata_tables(observations, windows, output):
    """Stable nullable Parquet schemas, also for empty tables and failed reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    observation_schema = pa.schema([
        (name, pa.string()) for name in ('event_id','date','split','relative_path','status','crs','sha256','error','sidecars_json')
    ] + [(name, pa.int64()) for name in ('year','height','width','bytes','positive_pixels','valid_pixels','invalid_pixels','no_detection_pixels')]
      + [('transform',pa.list_(pa.float64()))])
    window_schema = pa.schema([(name,pa.string()) for name in ('event_id','target_date','target_path','split')]
                             + [('context',pa.int64()),('context_dates',pa.list_(pa.string())),
                                ('context_paths',pa.list_(pa.string()))])
    projection = [{**r, 'sidecars_json':json.dumps(r.get('sidecars',{}),sort_keys=True)} for r in observations]
    for name, rows, schema in [('observations',projection,observation_schema),('windows',windows,window_schema)]:
        path = Path(output)/f'{name}.parquet'
        if path.exists():
            raise FileExistsError(path)
        table = pa.Table.from_pylist(rows,schema=schema)
        pq.write_table(table,path,compression='snappy')
        if not pq.read_table(path).equals(table):
            raise RuntimeError(f'Parquet roundtrip failed: {name}')


def audit_observations(rows, index, root, *, block_size=256):
    """CRC, SHA and block QC; NaN fire = no detection, finite code = detection.

    Missing/corrupt observations have null counts, never zero counts. Explicit
    GDAL masks/finite nodata are interpreted by the training reader itself.
    """
    if block_size < 1:
        raise ValueError("Positive block size required")
    expected = {(e['event_id'], e['date']): e for e in index['entries']}
    if len(expected) != len(index['entries']):
        raise ValueError('Duplicate indexed observation')
    assignment = event_assignments(sorted({e[0] for e in expected}), 0)
    seen, paths, records = set(), set(), []
    for row in rows:
        key = (row['event_id'], row['date'])
        path = (Path(root) / row['relative_path']).resolve()
        if key in seen or path in paths:
            raise ValueError("Duplicate observation or reused path")
        if key not in expected:
            raise ValueError("Observation absent from pinned index")
        date.fromisoformat(row['date'])
        if assignment[key[0]] == 'test':
            raise ValueError("Test 2021 pixel audit is locked")
        seen.add(key)
        paths.add(path)
        record = dict(event_id=key[0], date=key[1], year=int(key[0][:4]),
                      split=assignment[key[0]], relative_path=str(path),
                      status='missing', positive_pixels=None, valid_pixels=None,
                      invalid_pixels=None, no_detection_pixels=None)
        if path.is_file():
            try:
                if not verify_member(path, expected[key]):
                    raise ValueError('Indexed size/CRC mismatch')
                with rasterio.open(path) as ds:
                    validate_layout(ds)
                    if ds.crs is None:
                        raise ValueError('Missing CRS')
                    if ds.shape != (int(row['height']), int(row['width'])):
                        raise ValueError('Manifest dimensions mismatch')
                    record.update(height=ds.height, width=ds.width, crs=str(ds.crs),
                                  transform=list(ds.transform), sha256=sha256_file(path),
                                  bytes=path.stat().st_size,
                                  sidecars={Path(p).name: sha256_file(p) for p in ds.files
                                            if Path(p).resolve() != path})
                positive = valid_count = pixels = 0
                for r, c in tile_coordinates(record['height'], record['width'], block_size):
                    raw, valid = _read_selected(path, r, c, block_size)
                    mask = valid[0].astype(bool)
                    positive += int((np.isfinite(raw[3]) & mask).sum())
                    valid_count += int(mask.sum())
                    pixels += mask.size
                record.update(status='ok', positive_pixels=positive, valid_pixels=valid_count,
                              invalid_pixels=pixels-valid_count,
                              no_detection_pixels=valid_count-positive)
            except (OSError, ValueError, rasterio.errors.RasterioError) as error:
                record.update(status='corrupt', error=f'{type(error).__name__}: {error}')
        records.append(record)
    return records


def summarize_audit(records, index):
    """Windows use only readable, same-grid observations and canonical rules."""
    by_event = defaultdict(list)
    for r in records:
        by_event[r['event_id']].append(r)
    events, clean = [], []
    expected = defaultdict(set)
    for e in index['entries']:
        expected[e['event_id']].add(e['date'])
    for event, group in sorted(by_event.items()):
        ok = [r for r in group if r['status'] == 'ok']
        geometries = {(r['height'], r['width'], r['crs'], tuple(r['transform'])) for r in ok}
        consistent = len(geometries) == 1
        if consistent:
            clean.extend(ok)
        dates = sorted(date.fromisoformat(r['date']) for r in ok)
        events.append(dict(event_id=event, year=int(event[:4]), split=group[0]['split'],
                           expected_observations=len(expected[event]), manifest_observations=len(group),
                           audited_observations=len(ok), geometry_consistent=consistent,
                           complete=consistent and {r['date'] for r in ok} == expected[event],
                           missing_dates=sorted(expected[event]-{r['date'] for r in ok}),
                           gaps=[dict(start=a.isoformat(), end=b.isoformat(), missing_days=(b-a).days-1)
                                 for a, b in zip(dates, dates[1:]) if (b-a).days != 1],
                           **pixel_summary(ok)))
    # Same content across different splits is rejected, even with distinct paths.
    hashes = defaultdict(set)
    for r in clean:
        hashes[r['sha256']].add(r['split'])
    if any(len(roles) > 1 for roles in hashes.values()):
        raise ValueError('Identical raster content crosses splits')
    assignment = event_assignments(sorted(by_event), 0)
    windows, counts = [], {}
    reader = RasterReader(cache_bytes=64*1024**2)
    for role in ('train', 'val'):
        selected = [r for r in clean if r['split'] == role]
        counts[role] = {}
        for t in (1, 3, 5):
            ws, rejected = build_windows(selected, t)
            lookup = {(r['event_id'], r['date']): r for r in selected}
            targets = [lookup[(w.event_id, w.target_date)] for w in ws]
            counts[role][str(t)] = dict(windows=len(ws), rejections=rejected,
                                       target_observation_counts=pixel_summary(targets))
        groups, _ = aligned_windows(clean, assignment, role)
        domain = []
        for w in groups[5]:
            target = lookup[(w.event_id, w.target_date)]
            positive = valid_pixels = pixels = 0
            for r, c in tile_coordinates(target['height'], target['width'], 256):
                raw, valid = reader.tile(w.target_path, r, c, 256)
                mask = valid[0].astype(bool)
                for path in w.context_paths:
                    _, context_mask = reader.tile(path, r, c, 256)
                    mask &= context_mask[0].astype(bool)
                positive += int((np.isfinite(raw[22]) & mask).sum())
                valid_pixels += int(mask.sum())
                pixels += mask.size
            domain.append(dict(event_id=w.event_id, target_date=w.target_date,
                               positive_pixels=positive, valid_pixels=valid_pixels,
                               invalid_pixels=pixels-valid_pixels))
        counts[role]['aligned_joint_domain'] = dict(**pixel_summary(domain), targets=domain)
        for t, ws in groups.items():
            windows.extend(dict(**asdict(w), context=t, split=role) for w in ws)
    return dict(schema_version=1, scope='acquired train/val rasters only; local CPU',
                fire_convention='finite detection code; NaN=no detection; explicit masks/finite nodata=invalid',
                statuses=dict(Counter(r['status'] for r in records)), events=events,
                by_split={role: pixel_summary([r for r in records if r['split'] == role and r['status'] == 'ok'])
                          for role in ('train', 'val')}, window_counts=counts,
                target_count_warning='Per-target counts; joint context footprint is applied by TemporalStore during evaluation',
                observations=records, aligned_windows=windows)


def pixel_summary(records):
    valid = sum(r['valid_pixels'] for r in records)
    positive = sum(r['positive_pixels'] for r in records)
    invalid = sum(r['invalid_pixels'] for r in records)
    return dict(observations=len(records), positive_days=sum(r['positive_pixels'] > 0 for r in records),
                zero_detection_days=sum(r['valid_pixels'] > 0 and r['positive_pixels'] == 0 for r in records),
                entirely_invalid_days=sum(r['valid_pixels'] == 0 for r in records),
                positive_pixels=positive, valid_pixels=valid, invalid_pixels=invalid,
                positive_percent=100*positive/valid if valid else None,
                invalid_percent=100*invalid/(valid+invalid) if valid+invalid else None)


def load_window_manifest(records, observations, assignment, role, context):
    """Consume JSON/Parquet rows directly as existing TemporalStore Windows.

    Recompute canonical metadata windows to reject tampering, leakage and drift.
    """
    if role not in ('train', 'val') or context not in (1, 3, 5):
        raise ValueError('Only train/val T=1/3/5 allowed')
    if assignment != event_assignments(list(assignment),0):
        raise ValueError('Window metadata must retain annual fold 0 assignments')
    geometries = {}
    for row in observations:
        if row.get('status') != 'ok' or not row.get('sha256'):
            raise ValueError('Window metadata requires successful raster QC and SHA256')
        geometry = (row['height'],row['width'],row['crs'],tuple(row['transform']))
        if geometries.setdefault(row['event_id'],geometry) != geometry:
            raise ValueError('Window metadata has inconsistent event geometry')
    groups, _ = aligned_windows(observations, assignment, role)
    expected = groups[context]
    selected = [r for r in records if r['split'] == role and r['context'] == context]
    actual = [Window(r['event_id'], tuple(r['context_dates']), tuple(r['context_paths']),
                     r['target_date'], r['target_path']) for r in selected]
    if actual != expected:
        raise ValueError('Window manifest differs from canonical aligned windows')
    return actual


def acquisition_stages(index, verified_paths, targets=(80, 200, 451), seed=42):
    """Nested year-stratified selection independent of labels/model performance.

    Exact ZIP payload sizes are separate from proposed session caps. Reuse is
    credited only from CRC-verified local paths, never cancelled remote jobs.
    """
    validate_index(index)
    groups = defaultdict(list)
    for e in index['entries']:
        if e['year'] in (2018, 2019, 2020):
            groups[e['event_id']].append(e)
    verified = set(verified_paths)
    if not verified <= {e['path'] for e in index['entries']}:
        raise ValueError('Unknown verified path')
    selected = {event for event, entries in groups.items() if all(e['path'] in verified for e in entries)}
    by_year = {y: [e for e in groups if int(e[:4]) == y] for y in (2018, 2019, 2020)}
    order = {y: sorted(es, key=lambda e: hashlib.sha256(f'{seed}:{e}'.encode()).hexdigest())
             for y, es in by_year.items()}
    result, previous_paths = [], set()
    for n in targets:
        if n < len(selected) or n > len(groups):
            raise ValueError('Stage cannot shrink or exceed train/val corpus')
        while len(selected) < n:
            year = min((y for y in by_year if any(e not in selected for e in order[y])),
                       key=lambda y: (sum(int(e[:4]) == y for e in selected)/len(by_year[y]), y))
            selected.add(next(e for e in order[year] if e not in selected))
        entries = sorted((e for event in selected for e in groups[event]), key=lambda e: e['path'])
        missing = [e for e in entries if e['path'] not in verified]
        payload = sum(e['compressed_bytes'] for e in missing)
        extracted = sum(e['geotiff_bytes'] for e in entries)
        new_bytes = sum(e['geotiff_bytes'] for e in missing)
        # Directory allowance + request/header/retry margin; hard cap, not a guarantee.
        cap = (payload*115+99)//100 + 8*1024**2
        incremental = [e for e in missing if e['path'] not in previous_paths]
        result.append(seal(dict(schema_version=1, kind='scientific_scaleup_plan', approved=False,
                               target_events=n, events=sorted(selected), entries=entries,
                               observations=len(entries), by_year={str(y): sum(e['year'] == y for e in entries)
                                                                  for y in by_year},
                               events_by_year={str(y): sum(int(e[:4]) == y for e in selected) for y in by_year},
                               compressed_payload_bytes=sum(e['compressed_bytes'] for e in entries),
                               extracted_bytes=extracted, verified_reused_files=len(entries)-len(missing),
                               missing_files=len(missing), additional_payload_bytes=payload,
                               incremental_payload_bytes=sum(e['compressed_bytes'] for e in incremental),
                               incremental_extracted_bytes=sum(e['geotiff_bytes'] for e in incremental),
                               additional_extracted_bytes=new_bytes, proposed_transfer_cap_bytes=cap,
                               proposed_disk_bytes=new_bytes+extracted+20*10**9,
                               disk_policy='new TIFF + one conservative staging copy + 20 GB reserve; AP scratch/cache extra',
                               ideal_transfer_hours={str(mbps): payload*8/(mbps*10**6)/3600 for mbps in (10,50,100)},
                               selection=f'nested proportional years, SHA256 seed {seed}, preserve complete local events',
                               test_policy='2021 excluded', index_identity=fingerprint(index))))
        previous_paths = {e['path'] for e in entries}
    return result


def stage_download_entries(plan, index, roots):
    """Revalidate all local reuse immediately before a dry-run or acquisition."""
    check_seal(plan)
    if plan['index_identity'] != fingerprint(index):
        raise ValueError('Stage belongs to a different index')
    events = set(plan['events'])
    if any(int(e[:4]) not in (2018,2019,2020) for e in events):
        raise ValueError('Test acquisition excluded from scaleup stages')
    expected = sorted((e for e in index['entries'] if e['event_id'] in events),key=lambda e:e['path'])
    if expected != plan['entries'] or len(events) != plan['target_events']:
        raise ValueError('Stage must contain every indexed observation of selected events')
    missing, reused = [], []
    for entry in expected:
        found = []
        for root in roots:
            path = Path(root)/entry['path']
            if path.exists():
                if not verify_member(path,entry):
                    raise ValueError(f'Corrupt existing raster preserved: {path}')
                found.append(str(path.resolve()))
        if found:
            reused.append(dict(path=entry['path'],locations=found))
        else:
            missing.append(entry)
    return missing,reused
