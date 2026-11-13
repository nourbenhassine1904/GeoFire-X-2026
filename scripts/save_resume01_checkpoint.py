"""Preserve reconciled lot002 state without changing historical attempt results."""
import json
from pathlib import Path

from kaggle_cpu_job import validate_receipt

base=Path('reports/kaggle_remaining_20261108_v1')
plan=json.loads((base/'plan.json').read_text())
resume=base/'resume-01'
reconciliation=json.loads((resume/'reconciliation-lot002/result.json').read_text())
presence=dict(**reconciliation['presence'],confirmed_after_indexing=True,
    source_reconciliation='resume-01/reconciliation-lot002/result.json')
target=resume/'execution/lot-002/publication/private-presence.json'
if not target.exists():
    with target.open('x',encoding='utf-8') as stream:json.dump(presence,stream,indent=2)
cpu=resume/'execution/lot-002/cpu'
controller=json.loads((cpu/'controller-result.json').read_text()) if (cpu/'controller-result.json').exists() else None
cert=None
if controller and controller['status']=='REMOTE_CPU_RECEIPT_VERIFIED':
    cert=validate_receipt(cpu,next(s for s in plan['lots'] if s['number']==2))
used=sum(json.loads(line)['reserved'] for line in Path(plan['journal']).read_text().splitlines())
checkpoint=dict(status='LOT002_PRIVATE_AND_CPU_CERTIFIED_REMAINING_PUBLICATIONS_STOPPED' if cert else 'LOT002_PRIVATE_CPU_CERTIFICATION_PENDING',
    lot001_closed=True,lot002=dict(handle=presence['handle'],private=True,version=presence['version'],upload_completed=True,
        upload_repetition_forbidden=True,archive_rebuilt=False,archive_verification='resume-01/zip002-verification/verification.json',
        cpu_certified=bool(cert),certified_files=cert['files'] if cert else 0,cpu_evidence=str(cpu)),
    lots003_to008='NOT_STARTED',cumulative_reserved_body_bytes=used,remaining_body_bytes=31_000_000_000-used,
    http_journal=str(plan['journal']),no_automatic_retry=True,gpu_started=False)
name='checkpoint-certified.json' if cert else 'checkpoint-private.json'
with (resume/name).open('x',encoding='utf-8') as stream:json.dump(checkpoint,stream,indent=2)
print(json.dumps(checkpoint,indent=2))
