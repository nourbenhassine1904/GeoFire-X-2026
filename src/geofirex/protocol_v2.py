"""Isolated, chronological V2 preparation. No model, optimizer or test reader.

The certified compact values and their frozen 2018/19 normalizer are reused.
Crossing events are embargoed in full, never divided between two roles.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from torch.utils.data import DataLoader

from geofirex.fast_compact import DTYPE, FastCompactTiles, LocalitySampler


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def event_intervals(observations):
    groups = defaultdict(list)
    paths, keys = set(), set()
    for row in observations:
        event, day, path = row['event_id'], row['date'], row['relative_path']
        if event.split('/')[0] not in ('2018', '2019', '2020', '2021'):
            raise ValueError('Unsupported event year')
        date.fromisoformat(day)
        if not path.startswith(event + '/') or path in paths or (event, day) in keys:
            raise ValueError('Reused path/date or event path mismatch')
        keys.add((event, day)); paths.add(path); groups[event].append(row)
    return {e: (min(r['date'] for r in rows), max(r['date'] for r in rows))
            for e, rows in groups.items()}


def chronological_split(intervals, train_fraction=.8):
    """Choose by dates/counts ONLY; no prevalence, label or model score tuning.

    Exhaust starts and end+1 boundaries; closest retained train ratio,
    then fewest embargoed events, then earliest cutoff. Half-open train side.
    Also expose the best full-retention solution, rather than hide its cost.
    """
    if not 0 < train_fraction < 1:
        raise ValueError('Invalid train fraction')
    pool = {e: v for e, v in intervals.items() if not e.startswith('2021/')}
    candidates = []
    boundaries = {start for start, end in pool.values()} | {
        (date.fromisoformat(end) + timedelta(days=1)).isoformat() for start, end in pool.values()}
    for cutoff in sorted(boundaries):
        train = sum(end < cutoff for start, end in pool.values())
        val = sum(start >= cutoff for start, end in pool.values())
        if not train or not val:
            continue
        candidates.append(dict(cutoff=cutoff, train=train, val=val,
                               purged=len(pool) - train - val,
                               train_fraction=train / (train + val),
                               ratio_error=abs(train / (train + val) - train_fraction)))
    if not candidates:
        raise ValueError('No nonempty chronological train/validation split')
    best = min(candidates, key=lambda r: (r['ratio_error'], r['purged'], r['cutoff']))
    cutoff = best['cutoff']
    roles = {e: ('test' if e.startswith('2021/') else
                 'train' if end < cutoff else 'val' if start >= cutoff else 'purged')
             for e, (start, end) in intervals.items()}
    clean = [r for r in candidates if not r['purged']]
    return roles, best, candidates, min(clean, key=lambda r: r['ratio_error']) if clean else None


def check_separation(observations, roles, cutoff):
    intervals = event_intervals(observations)
    if set(intervals) != set(roles):
        raise ValueError('Every event must be assigned exactly once')
    owners = defaultdict(set)
    for row in observations:
        event = row['event_id']; role = roles[event]
        if role not in ('train', 'val', 'purged', 'test'):
            raise ValueError('Invalid V2 role')
        start, end = intervals[event]
        if (event.startswith('2021/')) != (role == 'test'):
            raise ValueError('2021 must be wholly locked')
        if role in ('train', 'val'):
            if row['date'] >= '2021-01-01' or row['status'] != 'ok':
                raise ValueError('Future calendar pixels or uncertified observation')
            if (role == 'train' and end >= cutoff) or (role == 'val' and start < cutoff):
                raise ValueError('Chronological leakage')
            if row.get('sha256'):
                owners[row['sha256']].add(role)
        if role == 'purged' and not start < cutoff <= end:
            raise ValueError('Only cutoff-crossing events may be embargoed')
    shared = [h for h, values in owners.items() if len(values) > 1]
    if shared:
        raise ValueError('Identical source rasters cross train/validation')
    return dict(status='PASS', events=len(roles), roles=dict(Counter(roles.values())),
                train_last_date=max(end for e, (start, end) in intervals.items() if roles[e] == 'train'),
                validation_first_date=min(start for e, (start, end) in intervals.items() if roles[e] == 'val'),
                duplicate_event_dates=0, reused_source_paths=0, shared_train_val_source_sha256=0,
                test_pixels_opened=0,
                scope='Temporal and event separation; no claim of independent spatial regions')


def _safe_archive(logical, event):
    path = PurePosixPath(logical)
    if (not re.fullmatch(r'(?:2018|2019|2020)/fire_[0-9]+', event)
            or '\\' in logical or ':' in logical or path.is_absolute()
            or '..' in path.parts or not logical.endswith('/' + event + '.zip')):
        raise ValueError('Unsafe or mismatched archive path')
    if event.startswith('2021/') or '2021' in path.parts:
        raise ValueError('2021 locked')


def build_v2(prepared, old_index, corpus, publication, out):
    """Metadata-only derivative of the certified indices; no ZIP/NPZ opens."""
    import pyarrow.parquet as pq
    prepared, old_index, corpus, publication, out = map(Path, (prepared, old_index, corpus, publication, out))
    out.mkdir(parents=True, exist_ok=False)
    observations = pq.read_table(prepared / 'observations.parquet').to_pylist()
    intervals = event_intervals(observations)
    roles, best, candidates, clean = chronological_split(intervals)
    checks = check_separation(observations, roles, best['cutoff'])
    old = json.loads((old_index / 'index.json').read_text())
    normalizer = json.loads((old_index / 'normalization.json').read_text())
    if sha256(old_index / 'normalization.json') != old['normalization_sha256']:
        raise ValueError('Historical normalizer identity mismatch')
    if normalizer['continuous'] != [12, 14] or normalizer['include_landcover']:
        raise ValueError('Expected certified C8 profile')
    # Frozen statistics from a strict subset of V2 train are legitimate train-only
    # statistics. Do not invert clipped z-scores or silently call them a V2 refit.
    fitted = set()
    known_paths = {r['relative_path'] for r in observations if not r['event_id'].startswith('2021/')}
    for path in normalizer['fit_paths']:
        token = path.replace('\\', '/').split('/rasters/')[-1]
        event = '/'.join(token.split('/')[:2]); day = Path(token).stem
        if roles.get(event) != 'train' or day >= best['cutoff'] or event[:4] not in ('2018', '2019'):
            raise ValueError('Frozen normalizer uses data outside V2 train')
        if token not in known_paths:
            raise ValueError('Unknown normalization input')
        fitted.add(token)
    archives = old['archives']; events = old['events']
    if set(events.values()) != {e for e in roles if roles[e] != 'test'} or len(set(events.values())) != len(archives):
        raise ValueError('Certified compact event coverage differs')
    for archive in archives:
        _safe_archive(archive, events[archive])
    source = []
    for role in ('train', 'val'):
        path = old_index / (role + '.npy')
        if sha256(path) != old['index_sha256'][role]:
            raise ValueError('Historical index identity mismatch')
        source.append(np.load(path, allow_pickle=False))
    records = np.concatenate(source)
    if records.dtype != DTYPE:
        raise ValueError('Unexpected certified index schema')
    by_event = defaultdict(list)
    known_days = defaultdict(set)
    shapes = {}
    for row in observations:
        if roles[row['event_id']] == 'test':
            continue
        known_days[row['event_id']].add(date.fromisoformat(row['date']).toordinal())
        shape = row['height'], row['width']
        if row['event_id'] in shapes and shapes[row['event_id']] != shape:
            raise ValueError('Event geometry changed')
        shapes[row['event_id']] = shape
    seen, windows = set(), Counter()
    for i, rec in enumerate(records):
        a, day, row, col = map(int, (rec['archive'], rec['day'], rec['row'], rec['col']))
        event = events[archives[a]]; key = (event, day, row, col)
        if key in seen or any(day + d not in known_days[event] for d in range(-5, 1)):
            raise ValueError('Duplicate tile or inadmissible six-day window')
        h, w = shapes[event]
        if row % 64 or col % 64 or not 0 <= row < h or not 0 <= col < w or rec['category'] > 2:
            raise ValueError('Invalid grid/category')
        seen.add(key); windows[event, day] += 1; by_event[event].append(i)
    expected_windows = {(e, d) for e, days in known_days.items() for d in days
                        if all(d + k in days for k in range(-5, 1))}
    if set(windows) != expected_windows:
        raise ValueError('Compact omitted/added admissible windows')
    for (event, day), count in windows.items():
        h, w = shapes[event]
        if count != ((h + 63) // 64) * ((w + 63) // 64):
            raise ValueError('Incomplete tile coverage')
    # Exact logical/opaque routes preserve the existing eleven publications.
    routes = {}
    maps = sorted(publication.glob('p[0-9][0-9]/mount-map.json'))
    for path in maps:
        data = json.loads(path.read_text())
        for row in data['files']:
            if row['path'] in archives:
                if row['path'] in routes:
                    raise ValueError('Archive appears in multiple publications')
                routes[row['path']] = dict(partition=path.parent.name, **row)
    if set(routes) != set(archives):
        raise ValueError('Incomplete archive publication routes')
    index_dir = out / 'index'; index_dir.mkdir()
    summaries = {}
    for role in ('train', 'val'):
        ids = [i for i, rec in enumerate(records) if roles[events[archives[int(rec['archive'])]]] == role]
        selected = records[ids]
        with (index_dir / (role + '.npy')).open('xb') as stream:
            np.save(stream, selected, allow_pickle=False)
        summaries[role] = dict(events=sum(v == role for v in roles.values()), tiles=len(ids),
                               categories=dict(Counter({k: int((selected['category'] == v).sum())
                                                        for k, v in [('negative', 0), ('positive', 1), ('hard_negative', 2)]})))
    # Copy bytes of a small metadata file only, never the corpus arrays.
    with (index_dir / 'normalization.json').open('xb') as stream:
        stream.write((old_index / 'normalization.json').read_bytes())
    write_csv(out / 'cutoff_candidates.csv', candidates)
    rows = [dict(event_id=e, split=roles[e], first_date=intervals[e][0], last_date=intervals[e][1],
                 historical_split='test' if e.startswith('2021/') else 'val' if e.startswith('2020/') else 'train',
                 reason='crosses_cutoff_whole_event_embargo' if roles[e] == 'purged' else 'chronological',
                 observations=len(known_days.get(e, [])) if roles[e] != 'test' else '',
                 admissible_windows=sum(k[0] == e for k in windows) if roles[e] != 'test' else '',
                 tiles=len(by_event.get(e, [])) if roles[e] != 'test' else '') for e in sorted(roles)]
    write_csv(out / 'events.csv', rows)
    write_csv(out / 'windows.csv', [dict(event_id=e, split=roles[e], target_date=date.fromordinal(d).isoformat(),
              input_start=date.fromordinal(d-3).isoformat(), input_end=date.fromordinal(d-1).isoformat(),
              domain_start=date.fromordinal(d-5).isoformat(), tiles=n)
              for (e, d), n in sorted(windows.items())])
    checks.update(admissible_windows=len(windows), tiles=len(records),
                  exhaustive_grid_and_six_day_membership=True,
                  frozen_normalizer_fit_paths=len(fitted), frozen_normalizer_subset_of_v2_train=True)
    save(out / 'leakage_checks.json', checks)
    meta = dict(schema='geofirex-chronological-v2', archives=archives, events=events,
                event_roles=roles, cutoff=best['cutoff'], routes=routes, roles=summaries,
                normalization_policy='frozen_2018_2019_strict_subset_of_v2_train_no_refit',
                normalization_sha256=old['normalization_sha256'],
                index_sha256={r: sha256(index_dir / (r + '.npy')) for r in ('train', 'val')},
                events_csv_sha256=sha256(out / 'events.csv'),
                source_index_sha256=sha256(old_index / 'index.json'))
    save(index_dir / 'index.json', meta)
    inputs = [prepared / 'observations.parquet', prepared / 'windows.parquet', old_index / 'index.json',
              old_index / 'normalization.json', old_index / 'train.npy', old_index / 'val.npy',
              corpus / 'report.json', corpus / 'transfer-manifest.json', *maps]
    save(out / 'split_summary.json', dict(schema='geofirex-v2-preparation', selected=best,
         best_without_embargo=clean, roles=dict(Counter(roles.values())), usable_fraction=(best['train']+best['val'])/len(known_days),
         distributions=summaries, historical_normalizer_preserved=True,
         inputs={str(p): sha256(p) for p in inputs}, pixel_reads=0, training_executed=False))
    return meta


class V2CompactTiles(FastCompactTiles):
    """Reuse the certified decoder, without its historical annual assignment.

    Validate role membership before resolving a file or opening a ZIP. Resolve
    existing local ZIPs or published opaque .bin files directly, without copies.
    """
    def __init__(self, corpus, index, role, *, mounts=None, cache_mb=64, zip_handles=8):
        if role not in ('train', 'val'):
            raise ValueError('Only V2 train/val; 2021 and embargo locked')
        self.root = Path(corpus); self.index = Path(index); self.role = role
        self.meta = json.loads((self.index / 'index.json').read_text())
        if self.meta.get('schema') != 'geofirex-chronological-v2':
            raise ValueError('Expected V2 index')
        for name, digest in [('normalization.json', self.meta['normalization_sha256']),
                             (role + '.npy', self.meta['index_sha256'][role])]:
            if sha256(self.index / name) != digest:
                raise ValueError('V2 index/normalizer identity differs')
        if sha256(self.index.parent / 'events.csv') != self.meta['events_csv_sha256']:
            raise ValueError('V2 event manifest differs')
        self.normalization = json.loads((self.index / 'normalization.json').read_text())
        self.records = np.load(self.index / (role + '.npy'), mmap_mode='r', allow_pickle=False)
        if self.records.dtype != DTYPE or not len(self.records):
            raise ValueError('Nonempty certified record schema required')
        roots = [Path(p) for p in mounts] if mounts else [self.root]
        self.paths = [None] * len(self.meta['archives'])
        for a in np.unique(self.records['archive']):
            logical = self.meta['archives'][int(a)]; event = self.meta['events'][logical]
            _safe_archive(logical, event)
            if self.meta['event_roles'][event] != role:
                raise ValueError('V2 event role mismatch')
            days = self.records[self.records['archive'] == a]['day']
            cutoff = date.fromisoformat(self.meta['cutoff']).toordinal()
            if (days >= date(2021, 1, 1).toordinal()).any() or (role == 'train' and (days >= cutoff).any()) or (role == 'val' and (days - 5 < cutoff).any()):
                raise ValueError('V2 temporal boundary mismatch')
            route = self.meta['routes'][logical]; alias = route['alias']
            if PurePosixPath(alias).name != alias or '\\' in alias or ':' in alias:
                raise ValueError('Unsafe opaque alias')
            candidates = {p.resolve() for root in roots for p in (root / logical, root / alias) if p.is_file()}
            if len(candidates) != 1:
                raise ValueError('Missing/ambiguous V2 archive: ' + logical)
            path = candidates.pop()
            if path.stat().st_size != route['bytes']:
                raise ValueError('Archive size differs from certified publication')
            self.paths[int(a)] = str(path)
        self.budget = int(cache_mb * 1024**2); self.handle_limit = zip_handles
        if self.budget < 0 or zip_handles < 1:
            raise ValueError('Invalid bounded cache settings')
        self._reset()

    def _day(self, a, day, r, c):
        if a < 0 or a >= len(self.paths) or self.paths[a] is None or day >= date(2021, 1, 1).toordinal():
            raise ValueError('Forbidden compact read')
        return super()._day(a, day, r, c)


def make_v2_loaders(corpus, index, *, mounts=None, batch_size=16, workers=0):
    """CPU loaders only; construction performs metadata and file-size checks."""
    train = V2CompactTiles(corpus, index, 'train', mounts=mounts)
    val = V2CompactTiles(corpus, index, 'val', mounts=mounts)
    sampler = LocalitySampler(train, seed=42)
    options = dict(batch_size=batch_size, num_workers=workers, drop_last=False,
                   pin_memory=False, prefetch_factor=2 if workers else None)
    return (DataLoader(train, sampler=sampler, **options),
            DataLoader(val, shuffle=False, **options))


def sample_keys(dataset):
    """Immutable evaluation/overlay join keys in exact loader record order."""
    return [(dataset.event(i), date.fromordinal(int(r['day'])).isoformat(), int(r['row']), int(r['col']))
            for i, r in enumerate(dataset.records)]
