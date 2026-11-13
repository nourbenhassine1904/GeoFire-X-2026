"""Independent local artifact and historical-input verification; no TIFF opens."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import yaml

root = Path.cwd()
out = root/'reports/recovery_20261107_v3'
delivery = out/'finalized'
status = json.loads((delivery/'status.json').read_text())
receipt = json.loads((delivery/'code-package/staging-receipt.json').read_text())
payload = delivery/'code-package/payload.zip'
assert status['status'] == 'LOCAL_ESSENTIAL_TESTS_AND_PACKAGE_VALIDATED'
assert status['complete_reference_suite_validated'] is False
assert payload.stat().st_size == receipt['payload_bytes']
with payload.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256']
with zipfile.ZipFile(payload) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    manifest = json.loads(archive.read('code-manifest.json'))
    assert len(names) == len(set(names)) and set(names) == set(manifest)|{'code-manifest.json'}
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == sha for name, sha in manifest.items())
    assert not any(Path(name).suffix.lower() in ('.tif', '.tiff', '.pt', '.pth', '.ckpt') for name in names)
    assert not any('..' in Path(name).parts or name.startswith(('/', '\\')) for name in names)
    configs = {}
    for name in names:
        if '/bundle/configs/' not in name:
            continue
        c = yaml.safe_load(archive.read(name))
        assert c['corpus_scope'] == 'train_validation' and c['workers'] == 0 and c['cpu_threads'] == 2
        assert c['reader_cache_mb'] == 64 and c['evaluation_cleanup_runs'] is True
        assert c['evaluation_scratch_root'] == '/kaggle/temp/geofirex-ap-v1'
        assert 'D:' not in yaml.safe_dump(c)
        assert all(path.startswith('/kaggle/input/') for path in c['data_roots'])
        assert all(c[key] in names for key in ('manifest', 'window_manifest', 'split', 'index', 'normalization_artifact'))
        assert hashlib.sha256(archive.read(c['normalization_artifact'])).hexdigest() == c['normalization_sha256']
        configs[c['model']] = name
    assert set(configs) == {'B0', 'B1', 'B2', 'B3'}
notebook = json.loads((delivery/'kaggle-cpu-preflight/cpu-preflight.ipynb').read_text(encoding='utf-8'))
assert notebook['nbformat'] == 4 and len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
kernel = json.loads((delivery/'kaggle-cpu-preflight/kernel-metadata.json').read_text())
assert kernel['enable_gpu'] is False and kernel['enable_internet'] is False and kernel['is_private'] is True
assert len(kernel['dataset_sources']) == 9
historical = json.loads((out/'historical-inputs-before.json').read_text())
changed = []
for row in historical:
    with (root/row['path']).open('rb') as stream:
        current = hashlib.file_digest(stream, 'sha256').hexdigest()
    if current != row['sha256']:
        changed.append(row['path'])
assert not changed, changed
suite = json.loads((out/'essential-final/summary.json').read_text())
assert suite['status'] == 'ESSENTIAL_TESTS_PASSED'
passed = []
for index, result in enumerate(suite['results'], 1):
    xml = ET.parse(out/f'essential-final/test-{index:02d}.xml').find('testsuite')
    assert int(xml.get('tests')) == 1 and all(int(xml.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped'))
    passed.append(result['node'])
metrics = dict(tests_passed=len(passed), max_tree_rss_bytes=max(r['peak_tree_rss_bytes'] for r in suite['results']),
               max_private_commit_bytes=max(r['peak_sampled_private_commit_bytes'] for r in suite['results']),
               min_available_bytes=min(r['minimum_available_bytes'] for r in suite['results']),
               available_before=suite['results'][0]['before']['available'],
               available_after=suite['results'][-1]['after']['available'],
               total_test_process_seconds=sum(r['elapsed_seconds'] for r in suite['results']))
collection = (out/'reference-collection/output.log').read_text()
all_nodes = [line.strip() for line in collection.splitlines() if line.startswith('tests/') and '::' in line]
remaining = [node for node in all_nodes if node not in passed]
process_names = set()
for timeline in (out/'essential-final').glob('test-*/memory.jsonl'):
    for line in timeline.read_text().splitlines():
        for p in json.loads(line)['tree']['processes']:
            process_names.add(p.get('name', 'unknown'))
result = dict(status='PASSED_INDEPENDENT_DELIVERY_VERIFICATION', payload=receipt,
              archive_files=len(names), configs=configs, notebook_code_syntax='PASSED',
              historical_files_checked=len(historical), historical_files_changed=changed,
              metrics=metrics, passed_nodes=passed, reference_tests_collected=len(all_nodes),
              reference_nodes_not_run_in_final_matrix=remaining, owned_executable_names=sorted(process_names),
              raw_tiff_reads=0, uploads=0, gpu_jobs=0)
with (out/'delivery-verification.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in ('passed_nodes', 'reference_nodes_not_run_in_final_matrix', 'configs')}, indent=2))
