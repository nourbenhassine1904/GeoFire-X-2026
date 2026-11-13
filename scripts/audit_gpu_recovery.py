"""Bounded metadata recovery audit: no raster pixel reads and no network."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stored_zip_bytes(lot):
    """CPython seekable ZIP_STORED layout used by stage_expanded_lot.py.

    Local raster headers force ZIP64. Central sizes/offsets use ZIP64 at
    zipfile.ZIP64_LIMIT. This is a metadata prediction, never an upload receipt.
    """
    offset, central = 0, 0
    for member in lot['files']:
        name_bytes = len(('rasters/'+member['path']).encode('utf-8'))
        extra = (16 if member['bytes'] > zipfile.ZIP64_LIMIT else 0)
        extra += 8 if offset > zipfile.ZIP64_LIMIT else 0
        central += 46+name_bytes+(4+extra if extra else 0)
        offset += 30+name_bytes+20+member['bytes']
    inventory = json.dumps(lot, sort_keys=True).encode('utf-8')
    name_bytes = len('lot-inventory.json')
    extra = (16 if len(inventory) > zipfile.ZIP64_LIMIT else 0)+(8 if offset > zipfile.ZIP64_LIMIT else 0)
    central += 46+name_bytes+(4+extra if extra else 0)
    offset += 30+name_bytes+len(inventory)+(20 if len(inventory)*1.05 > zipfile.ZIP64_LIMIT else 0)
    zip64 = len(lot['files'])+1 > zipfile.ZIP_FILECOUNT_LIMIT or offset > zipfile.ZIP64_LIMIT or central > zipfile.ZIP64_LIMIT
    return offset+central+22+(76 if zip64 else 0)


def audit(bundle, raw_root, *, verify_sizes=True):
    import pyarrow.parquet as pq
    bundle, raw_root = Path(bundle), Path(raw_root)
    metadata = bundle/'metadata'
    plan = json.loads((bundle/'deployment.json').read_text(encoding='utf-8'))
    rows = pq.read_table(metadata/'observations.parquet').to_pylist()
    windows = pq.read_table(metadata/'windows.parquet').to_pylist()
    index = json.loads((metadata/'index.json').read_text(encoding='utf-8'))
    indexed = {e['path']: e for e in index['entries']}
    with (metadata/'fold_00.csv').open(encoding='utf-8', newline='') as f:
        assignment = {r['event_id']: r['split'] for r in csv.DictReader(f)}
    expected_role = lambda event: 'train' if event[:4] in ('2018', '2019') else 'val' if event[:4] == '2020' else 'test'
    assert len(assignment) == 607 and all(role == expected_role(event) for event, role in assignment.items())
    assert len(indexed) == 13607
    by_path = {r['relative_path']: r for r in rows}
    assert len(by_path) == len(rows) == 9490
    required = {e['path'] for e in index['entries'] if assignment[e['event_id']] in ('train', 'val')}
    assert set(by_path) == required
    events = {r['event_id'] for r in rows}
    assert len(events) == 451
    for token, r in by_path.items():
        assert token == f"{r['event_id']}/{r['date']}.tif"
        assert r['split'] == assignment[r['event_id']] in ('train', 'val') and r['status'] == 'ok'
        assert r['bytes'] == indexed[token]['geotiff_bytes']
    counts = collections.Counter((w['split'], w['context']) for w in windows)
    assert counts == {(role, c): n for role, n in [('train', 3948), ('val', 3287)] for c in (1, 3, 5)}
    targets = collections.defaultdict(set)
    train_tokens = set()
    window_keys = set()
    available = collections.defaultdict(set)
    for r in rows:
        available[r['event_id']].add(r['date'])
    for w in windows:
        event, date, context, role = w['event_id'], w['target_date'], w['context'], w['split']
        key = (role, event, date, context)
        assert key not in window_keys
        window_keys.add(key)
        end = dt.date.fromisoformat(date)
        days = [(end-dt.timedelta(days=d)).isoformat() for d in range(context, 0, -1)]
        assert w['target_path'] == f'{event}/{date}.tif'
        assert w['context_paths'] == [f'{event}/{day}.tif' for day in days]
        assert all(p in by_path and by_path[p]['split'] == role for p in [w['target_path']]+w['context_paths'])
        targets[(role, context)].add((event, date))
        if role == 'train' and context == 5:
            train_tokens.update(w['context_paths'])
    for role in ('train', 'val'):
        expected_targets = {(event, date) for event, dates in available.items() if assignment[event] == role
                            for date in dates if all((dt.date.fromisoformat(date)-dt.timedelta(days=d)).isoformat() in dates
                                                    for d in range(1, 6))}
        assert all(targets[(role, c)] == expected_targets for c in (1, 3, 5))
    proof = json.loads((metadata/'normalization.json').read_text(encoding='utf-8'))
    paths = dict(manifest='observations.parquet', window_manifest='windows.parquet', split='fold_00.csv', index='index.json')
    assert proof['bindings'] == {key: sha(metadata/name) for key, name in paths.items()}
    tokens = proof['normalization']['fit_paths']
    assert len(tokens) == len(set(tokens)) == 4948 and set(tokens) == train_tokens
    assert len({p.rsplit('/', 1)[0] for p in tokens}) == 250
    assert all(assignment[p.rsplit('/', 1)[0]] == 'train' for p in tokens)
    source = Path('reports/full_corpus_v1/prepared_spark_v1')
    original = json.loads((source/'normalization.json').read_text(encoding='utf-8'))
    assert sha(source/'normalization.json') == proof['provenance']['source_sha256']
    assert {k: v for k, v in original.items() if k != 'fit_paths'} == {
        k: v for k, v in proof['normalization'].items() if k != 'fit_paths'}
    assert {key: proof['provenance']['original_input_hashes'][key] for key in paths} == {
        key: sha(source/name) if key != 'index' else sha(metadata/name) for key, name in paths.items()}
    assert sha(metadata/'normalization.json') == plan['normalizer_sha256']
    lots, seen, seen_events = [], set(), set()
    for spec in plan['lots']:
        lot = json.loads((metadata/f"lot-{spec['number']:03d}.json").read_text(encoding='utf-8'))
        identity = lot.pop('identity')
        assert digest(lot) == identity == spec['identity']
        lot['identity'] = identity
        assert {k: v for k, v in lot.items() if k != 'files'} == spec
        assert lot['observations'] == len(lot['files'])
        assert sum(f['bytes'] for f in lot['files']) == lot['bytes'] <= 5_000_000_000
        assert set(lot['events']) == {f['path'].rsplit('/', 1)[0] for f in lot['files']}
        assert not seen_events.intersection(lot['events'])
        seen_events.update(lot['events'])
        for member in lot['files']:
            token = member['path']
            assert token not in seen and token in by_path
            r = by_path[token]
            assert r['bytes'] == member['bytes'] and r['sha256'] == member['sha256']
            assert indexed[token]['crc32'] == member['crc32'] and r['split'] == lot['role']
            path = raw_root/token
            if verify_sizes:
                assert path.is_file() and path.stat().st_size == member['bytes']
            seen.add(token)
        lots.append(dict(number=lot['number'], role=lot['role'], events=len(lot['events']),
                         files=lot['observations'], raw_bytes=lot['bytes'], identity=identity,
                         predicted_stored_zip_bytes=stored_zip_bytes(lot)))
    assert seen == set(by_path) and seen_events == events and len(lots) == 8
    raw_bytes = sum(r['bytes'] for r in rows)
    assert raw_bytes == plan['raw_bytes'] == 34_995_352_430
    predicted = sum(l['predicted_stored_zip_bytes'] for l in lots)
    return dict(status='PASSED_METADATA_AND_SIZE_AUDIT' if verify_sizes else 'PASSED_METADATA_AUDIT',
                raster_sizes_checked=9490 if verify_sizes else 0, raster_pixel_reads=0, network_bytes=0,
                official_index=dict(events=607, files=13607), present_train_validation=dict(events=451, files=len(rows)),
                windows={role: {str(c): counts[(role, c)] for c in (1, 3, 5)} for role in ('train', 'val')},
                normalizer=dict(sha256=plan['normalizer_sha256'], train_events=250, fit_files=4948, test_inputs=0, validation_inputs=0),
                test_rasters_required=0, lots=lots,
                upload=dict(raw_bytes=raw_bytes, proposed_ceiling_bytes=36_000_000_000,
                            raw_margin_bytes=36_000_000_000-raw_bytes, predicted_stored_zip_bytes=predicted,
                            predicted_margin_before_code_and_http_bytes=36_000_000_000-predicted,
                            conservative_archive_bound_bytes=raw_bytes+8*10_000_000,
                            smallest_lot_retry_bytes=min(l['predicted_stored_zip_bytes'] for l in lots),
                            archive_status='NOT_STAGED; predicted layout, not measured payload',
                            http_total_status='UNKNOWN until actual requests; durable shared BodyBudget required'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', default='reports/gpu_operational_v1/deployment-v3/bundle')
    p.add_argument('--raw-root', default='D:/GeoFire-X/WildfireSpreadTS/rasters')
    p.add_argument('--out', required=True)
    p.add_argument('--metadata-only', action='store_true', help='Do not stat or open any TIFF')
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('Choose a new report; historical artifacts are preserved')
    result = audit(a.bundle, a.raw_root, verify_sizes=not a.metadata_only)
    result['time'] = dt.datetime.now(dt.timezone.utc).isoformat()
    with Path(a.out).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
