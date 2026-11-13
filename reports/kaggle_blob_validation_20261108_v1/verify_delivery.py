"""Offline artifact/accounting verification; no real ZIP contents or network read."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
campaign=Path('reports/kaggle_pilot_001_20261107_v1')
journal=campaign/'http-body.jsonl'
used=0
for line in journal.read_text().splitlines():
    row=json.loads(line)
    if row['previous']!=used or row['ceiling']!=6_000_000_000:raise ValueError('Budget chain changed')
    used+=row['reserved']
if used!=2739:raise ValueError('Real account budget changed during offline diagnosis')
stage=Path('artifacts/kaggle_publication_20261107_v1/lot-001')
receipt=json.loads((stage/'staging-receipt.json').read_text())
if (stage/'payload.zip').stat().st_size!=4829918684 or receipt['sha256']!='464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890':
    raise ValueError('Staged payload size or recorded SHA changed')
for name in ('lot-001-create-attempt-03','lot-001-upload-monitor-03'):
    if (campaign/name).exists():raise ValueError('Third publication attempt started')
monitor=json.loads((root/'targeted-tests-final/supervisor.json').read_text())
if monitor['status']!='FINISHED' or monitor['exit_code']!=0:raise ValueError('Targeted tests failed')
if 'Ran 33 tests' not in (root/'targeted-tests-final/output.log').read_text():raise ValueError('Targeted test count mismatch')
sources={}
for name in ('scripts/kaggle_blob_session.py','scripts/publish_kaggle_single_attempt.py','tests/test_kaggle_blob_session.py'):
    content=Path(name).read_bytes();ast.parse(content.decode('utf-8'));sources[name]=hashlib.sha256(content).hexdigest()
ps=Path('reports/kaggle_prepublication_20261107_v2/PILOTE_LOT_001.ps1')
sources[ps.as_posix()]=hashlib.sha256(ps.read_bytes()).hexdigest()
history={}
for suffix in ('01','02'):
    for path in (campaign/f'lot-001-create-attempt-{suffix}').glob('*.json*'):
        history[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
history[journal.as_posix()]=hashlib.sha256(journal.read_bytes()).hexdigest()
result=dict(status='OFFLINE_BLOB_VALIDATOR_VERIFIED_NO_NETWORK',tests_passed=33,
    real_start_blob_calls=0,real_put_calls=0,real_dataset_creates=0,
    cumulative_http_body_bytes=used,remaining_http_body_bytes=6_000_000_000-used,
    payload_bytes=4829918684,recorded_payload_sha256=receipt['sha256'],real_zip_contents_read=False,
    response_from_attempt_02_available=False,exact_prior_rejection_cause='UNKNOWN_FROM_RETAINED_LOGS',
    policy='HTTPS, existing GCS host family only, port 443/default, no userinfo/fragment/control chars, no redirects or retries',
    source_sha256=sources,preserved_history_sha256=history,
    minimum_available_bytes_during_final_tests=monitor['minimum_available_bytes'],
    final_tests_peak_rss_bytes=monitor['peak_tree_rss_bytes'])
with (root/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
