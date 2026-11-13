"""Small offline delivery checks. Does not read the 4.83 GB ZIP or use the network."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
diag=json.loads((root/'account-probes/diagnosis.json').read_text())
preflight=json.loads((root/'corrected-preflight/preflight.json').read_text())
if diag['identity']['username']!='nnnnbhss' or not diag['credentials_unchanged'] or not preflight['credentials_unchanged']:
    raise ValueError('Identity or credential preservation failed')
decision=preflight['decision']
if (decision['state']!='ABSENT_VERIFIED' or decision['direct_lookup_state']!='ACCESS_FORBIDDEN'
    or not decision['allow_create_new'] or decision['create_permission_verified']):
    raise ValueError('Classification qualifiers differ')
journal=Path('reports/kaggle_pilot_001_20261107_v1/http-body.jsonl').read_bytes()
if not journal.startswith((root/'account-probes/publication-ledger-before.jsonl').read_bytes()):
    raise ValueError('Historical ledger prefix changed')
used=0
for line in journal.decode().splitlines():
    row=json.loads(line)
    if row['previous']!=used or row['ceiling']!=6_000_000_000:raise ValueError('Budget history changed')
    used+=row['reserved']
if used!=2087 or used!=preflight['cumulative_body_bytes']:raise ValueError('Budget consumption mismatch')
for monitor in ('targeted-tests','corrected-preflight-monitor','readonly-monitor'):
    supervisor=json.loads((root/monitor/'supervisor.json').read_text())
    if supervisor['status']!='FINISHED' or supervisor['exit_code']!=0:raise ValueError('Validation process did not succeed')
if 'Ran 14 tests' not in (root/'targeted-tests/output.log').read_text():raise ValueError('Focused test receipt missing')
receipt=json.loads(Path('artifacts/kaggle_publication_20261107_v1/lot-001/staging-receipt.json').read_text())
zip_path=Path('artifacts/kaggle_publication_20261107_v1/lot-001/payload.zip')
if zip_path.stat().st_size!=4829918684 or receipt['sha256']!='464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890':
    raise ValueError('Staged size or recorded hash changed')
for name in ('lot-001-create-attempt-02','lot-001-upload-monitor-02'):
    if Path('reports/kaggle_pilot_001_20261107_v1',name).exists():raise ValueError('A second publication attempt was started')
sources={}
for name in ('scripts/publish_kaggle_single_attempt.py','scripts/kaggle_dataset_preflight.py',
             'scripts/diagnose_kaggle_403.py','tests/test_kaggle_dataset_preflight.py'):
    data=Path(name).read_bytes();ast.parse(data.decode('utf-8'));sources[name]=hashlib.sha256(data).hexdigest()
ps='reports/kaggle_prepublication_20261107_v2/PILOTE_LOT_001.ps1'
sources[ps]=hashlib.sha256(Path(ps).read_bytes()).hexdigest()
result=dict(status='DIAGNOSTIC_AND_FIX_VERIFIED_NO_PUBLICATION',classification=decision,
            targeted_tests_passed=14,cumulative_http_body_bytes=used,remaining=6_000_000_000-used,
            historical_budget_prefix_preserved=True,credentials_unchanged=True,zip_payload_bytes=zip_path.stat().st_size,
            zip_sha256_record_preserved=receipt['sha256'],zip_rehashed_this_diagnosis=False,
            second_publication_started=False,new_dataset_writes=0,new_notebook_jobs=0,source_sha256=sources)
with (root/'delivery-verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
