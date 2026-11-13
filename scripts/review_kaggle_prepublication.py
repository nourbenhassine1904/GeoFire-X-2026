"""Small frozen-package/metadata review. Never stats or opens source rasters."""
import argparse
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib
import zipfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def review(frozen, credentials):
    frozen = Path(frozen)
    payload = frozen/'code-package/payload.zip'
    receipt = json.loads((payload.parent/'staging-receipt.json').read_text())
    raw = payload.read_bytes()
    assert len(raw) == receipt['payload_bytes'] and hashlib.sha256(raw).hexdigest() == receipt['sha256']
    secrets = []
    def collect(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items(): collect(v, k)
        elif isinstance(value, str) and len(value) > 16 and any(t in key.lower() for t in ('token', 'secret', 'password', 'key')):
            secrets.append(value.encode())
    collect(json.loads(Path(credentials).read_text()))
    with zipfile.ZipFile(payload) as z:
        assert z.testzip() is None
        names = z.namelist()
        manifest_raw = z.read('code-manifest.json')
        manifest = json.loads(manifest_raw)
        assert len(names) == len(set(names)) == len(manifest)+1
        assert set(names) == set(manifest) | {'code-manifest.json'}
        findings = []
        for name in names:
            p = PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
            assert p.suffix.lower() not in ('.tif', '.tiff', '.pt', '.pth', '.ckpt')
            assert not re.search(r'(^|/)(credentials\.json|kaggle\.json|\.env|id_rsa)(/|$)', name, re.I)
            data = z.read(name)
            if name in manifest: assert hashlib.sha256(data).hexdigest() == manifest[name]
            if any(secret in data for secret in secrets): findings.append(dict(file=name, kind='existing_credential_value'))
            if p.suffix in ('.py', '.json', '.toml', '.yaml', '.csv'):
                txt = data.decode('utf-8')
                for kind, pattern in (
                    ('windows_absolute_path', r'(?i)\b[A-Z]:[\\/]|[\x22\x27]\\\\\\\\[A-Za-z0-9_.-]+\\'),
                    ('private_key', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                    ('aws_access_key', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
                    ('kaggle_token_literal', r'KGAT_[A-Za-z0-9_-]{16,}')):
                    if re.search(pattern, txt): findings.append(dict(file=name, kind=kind))
        deps = tomllib.loads(z.read('pyproject.toml').decode())['project']
    assert not findings, findings
    bundle = frozen/'deployment/bundle'
    index = json.loads((bundle/'metadata/index.json').read_text())
    indexed = {x['path']: x for x in index['entries']}
    plan = json.loads((bundle/'deployment.json').read_text())
    events, paths, rows = set(), set(), []
    role_events = collections.defaultdict(set)
    for spec in plan['lots']:
        lot = json.loads((bundle/f"metadata/lot-{spec['number']:03d}.json").read_text())
        unsigned = {k: v for k, v in lot.items() if k != 'identity'}
        assert digest(unsigned) == lot['identity'] == spec['identity']
        for k in ('handle', 'role', 'events', 'bytes', 'observations'): assert spec[k] == lot[k]
        assert lot['role'] in ('train', 'val') and not (events & set(lot['events']))
        assert len(lot['files']) == lot['observations']
        assert sum(f['bytes'] for f in lot['files']) == lot['bytes']
        lot_paths = {f['path'] for f in lot['files']}
        assert len(lot_paths) == len(lot['files']) and not (paths & lot_paths)
        expected = {p for p, x in indexed.items() if x['event_id'] in lot['events']}
        assert lot_paths == expected  # Whole indexed events, not truncated slices.
        for f in lot['files']:
            p = PurePosixPath(f['path'])
            assert p.parts[0] in ('2018', '2019', '2020') and '..' not in p.parts
            assert f['bytes'] == indexed[f['path']]['geotiff_bytes'] and f['crc32'] == indexed[f['path']]['crc32']
            assert re.fullmatch('[0-9a-f]{64}', f['sha256'])
        events.update(lot['events']); paths.update(lot_paths); role_events[lot['role']].update(lot['events'])
        # Exact ZIP_STORED layout used by stage_expanded_lot (force_zip64 local headers).
        inv = json.dumps(lot, sort_keys=True).encode()
        offset, central = 0, 0
        for f in lot['files']:
            name_bytes = len(('rasters/'+f['path']).encode())
            central += 46+name_bytes+(12 if offset > zipfile.ZIP64_LIMIT else 0)
            offset += 30+name_bytes+20+f['bytes']
        central += 46+len('lot-inventory.json')+(12 if offset > zipfile.ZIP64_LIMIT else 0)
        offset += 30+len('lot-inventory.json')+len(inv)
        estimate = offset+central+22+(76 if offset > zipfile.ZIP64_LIMIT or central > zipfile.ZIP64_LIMIT else 0)
        rows.append(dict(number=lot['number'], handle=lot['handle'], role=lot['role'], events=len(lot['events']),
                         files=len(lot['files']), raw_bytes=lot['bytes'], predicted_zip_bytes=estimate, identity=lot['identity']))
    assert len(events) == 451 and len(paths) == 9490 and len(role_events['train']) == 250 and len(role_events['val']) == 201
    assert sum(r['raw_bytes'] for r in rows) == plan['raw_bytes'] == 34995352430
    norm_path = bundle/'metadata/normalization.json'
    norm = json.loads(norm_path.read_text())
    fit_paths = norm['normalization']['fit_paths']
    assert len(fit_paths) == len(set(fit_paths)) == 4948
    assert {str(PurePosixPath(p).parent) for p in fit_paths} == role_events['train']
    assert all(p.split('/')[0] in ('2018', '2019') for p in fit_paths)
    import pyarrow.parquet as pq
    parquets = {}
    for name in ('observations', 'windows'):
        table = pq.read_table(bundle/f'metadata/{name}.parquet')
        parquets[name] = dict(rows=table.num_rows, columns=table.column_names, schema=str(table.schema))
        for field in table.schema:
            if str(field.type) in ('string', 'large_string'):
                for value in table[field.name].to_pylist():
                    if value is not None:
                        assert not re.search(r'(?i)\b[A-Z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\', value)
        del table
    total = sum(r['predicted_zip_bytes'] for r in rows)+len(raw)
    return dict(status='FROZEN_CODE_AND_METADATA_REVIEW_PASSED', raw_rasters_opened=0, source_rasters_statted=0,
        package=dict(bytes=len(raw), sha256=receipt['sha256'], manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                     entries=len(names), manifest_files=len(manifest), crc_and_sha256=True, secret_path_scan_findings=findings,
                     secret_scan_scope='known local credential values + private-key/AWS/KGAT patterns; no static scanner proves all possible secrets absent'),
        dependencies=deps, lots=rows, complete_indexed_events=True, train_events=250, val_events=201,
        raw_bytes=plan['raw_bytes'], observations=9490, test_2021_rasters=0,
        normalizer=dict(sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(), fit_events=250, fit_files=4948, years=[2018,2019]),
        parquets=parquets,
        network=dict(predicted_initial_zip_bodies=total, actual_staged_zip_sizes_verified=False,
                     margin_at_36GB=36000000000-total, proposed_body_ceiling=50000000000,
                     initial_conservative_raw_plus_zip_overhead_and_code=plan['raw_bytes']+80000000+len(raw),
                     proposed_physical_network_planning_envelope=55000000000))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', required=True); p.add_argument('--credentials', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    result = review(a.frozen, a.credentials)
    with Path(a.out).open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'dependencies'}, indent=2))
