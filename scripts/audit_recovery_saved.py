"""Low-memory verification of saved bindings and lot plans, without Torch/Arrow."""
import ast
from collections import Counter, defaultdict
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path

from audit_gpu_recovery import sha, digest, stored_zip_bytes


def audit():
    root = Path.cwd()
    base = root/'reports/gpu_operational_v1'
    bundle = base/'deployment-v3/bundle'
    metadata = bundle/'metadata'
    cpu = json.loads((base/'cpu-integration.json').read_text())
    proof = json.loads((metadata/'normalization.json').read_text())
    paths = dict(manifest='observations.parquet', window_manifest='windows.parquet', split='fold_00.csv', index='index.json')
    bindings = {key: sha(metadata/name) for key, name in paths.items()}
    assert bindings == proof['bindings'] == {key: cpu['input_hashes'][key] for key in paths}
    assert sha(metadata/'normalization.json') == cpu['normalizer_sha256']
    original = root/'reports/full_corpus_v1/prepared_spark_v1'
    assert sha(original/'normalization.json') == proof['provenance']['source_sha256']
    source = json.loads((original/'normalization.json').read_text())
    assert {k: v for k, v in source.items() if k != 'fit_paths'} == {
        k: v for k, v in proof['normalization'].items() if k != 'fit_paths'}
    index = json.loads((metadata/'index.json').read_text())
    entries = {e['path']: e for e in index['entries']}
    with (metadata/'fold_00.csv').open(newline='') as f:
        assignment = {r['event_id']: r['split'] for r in csv.DictReader(f)}
    assert len(entries) == 13607 and len(assignment) == 607
    assert all(role == ('train' if event[:4] in ('2018', '2019') else 'val' if event[:4] == '2020' else 'test')
               for event, role in assignment.items())
    dates = defaultdict(set)
    for e in entries.values():
        dates[e['event_id']].add(dt.date.fromisoformat(e['date']))
    counts, expected_fit = Counter(), set()
    for event, values in dates.items():
        role = assignment[event]
        if role == 'test':
            continue
        for end in values:
            days = [end-dt.timedelta(days=d) for d in range(1, 6)]
            if all(day in values for day in days):
                counts[role] += 1
                if role == 'train':
                    expected_fit.update(f'{event}/{day.isoformat()}.tif' for day in days)
    assert dict(counts) == {'train': 3948, 'val': 3287}
    tokens = proof['normalization']['fit_paths']
    assert len(tokens) == len(set(tokens)) == len(expected_fit) == 4948 and set(tokens) == expected_fit
    assert len({p.rsplit('/', 1)[0] for p in tokens}) == 250
    plan = json.loads((bundle/'deployment.json').read_text())
    lots, seen, events = [], set(), set()
    for summary in plan['lots']:
        lot = json.loads((metadata/f"lot-{summary['number']:03d}.json").read_text())
        identity = lot.pop('identity')
        assert digest(lot) == identity == summary['identity']
        lot['identity'] = identity
        assert {k: v for k, v in lot.items() if k != 'files'} == summary
        assert sum(f['bytes'] for f in lot['files']) == lot['bytes'] <= 5_000_000_000
        assert len(lot['files']) == lot['observations']
        assert not events.intersection(lot['events'])
        assert set(lot['events']) == {f['path'].rsplit('/', 1)[0] for f in lot['files']}
        events.update(lot['events'])
        for f in lot['files']:
            e = entries[f['path']]
            assert f['path'] not in seen and assignment[e['event_id']] == lot['role'] in ('train', 'val')
            assert e['geotiff_bytes'] == f['bytes'] and e['crc32'] == f['crc32']
            assert len(f['sha256']) == 64
            seen.add(f['path'])
        lots.append(dict(number=lot['number'], role=lot['role'], files=lot['observations'], events=len(lot['events']),
                         raw_bytes=lot['bytes'], predicted_zip_bytes=stored_zip_bytes(lot)))
    assert len(lots) == 8 and len(seen) == 9490 and len(events) == 451
    assert seen == {e['path'] for e in entries.values() if assignment[e['event_id']] in ('train', 'val')}
    raw_bytes = sum(lot['raw_bytes'] for lot in lots)
    assert raw_bytes == 34_995_352_430 == plan['raw_bytes']
    parsed = []
    for path in list((root/'scripts').glob('*.py'))+list((root/'src/geofirex').glob('*.py'))+list((root/'tests').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        parsed.append(path.relative_to(root).as_posix())
    return dict(status='PASSED_SAVED_BINDINGS_AND_LOT_PLANS',
        verification_scope='Saved Parquet bytes pinned to prior successful CPU receipt; independent index reconstruction. No fresh Parquet parse or TIFF integrity scan.',
        input_hashes=bindings, normalizer_sha256=cpu['normalizer_sha256'],
        windows=dict(counts), fit_files=4948, fit_train_events=250, test_inputs=0, lots=lots,
        raw_bytes=raw_bytes, raw_margin_to_proposed_36GB=36_000_000_000-raw_bytes,
        predicted_zip_bytes=sum(l['predicted_zip_bytes'] for l in lots),
        conservative_zip_bound=raw_bytes+8*10_000_000,
        archive_status='Not staged: ZIP bytes are a prediction, HTTP body totals unknown',
        missing_packages=[name for name in ('psutil', 'kaggle', 'kagglesdk', 'kagglehub') if importlib.util.find_spec(name) is None],
        syntax_files_checked=len(parsed), parsed_files=parsed, raw_pixel_reads=0, network_bytes=0)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = audit()
    with Path(a.out).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('parsed_files', 'lots')}, indent=2))
