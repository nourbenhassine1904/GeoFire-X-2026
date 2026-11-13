"""Write a new operational report from durable receipts; never relaunch any job."""
import json
from pathlib import Path
import sys

from kaggle_cpu_job import validate_receipt

base=Path('reports/kaggle_remaining_20261108_v1')
execution=Path(sys.argv[1]) if len(sys.argv)>1 else base/'execution'
plan=json.loads((base/'plan.json').read_text())


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def spent(path,key):
    if not path.is_file():return 0
    return sum(json.loads(line)[key] for line in path.read_text().splitlines())


summary=read(execution/'continuation-result.json') if (execution/'continuation-started.json').exists() else read(execution/'result.json')
if summary is None:raise RuntimeError('Execution still running; no final report yet')
budget=0
for line in (base/'http-body.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['previous']!=budget or row['ceiling']!=31_000_000_000:raise ValueError('Broken budget chain')
    budget+=row['reserved']
lots=[]
for spec in plan['lots']:
    folder=execution/spec['tag']
    stage=read(Path(spec['staging'])/'staging-receipt.json')
    publication=read(folder/'publication/result.json')
    presence=read(folder/'publication/private-presence.json')
    cpu=read(folder/'cpu/controller-result.json')
    cert=None
    if cpu and cpu['status']=='REMOTE_CPU_RECEIPT_VERIFIED':cert=validate_receipt(folder/'cpu',spec)
    state='NOT_STARTED'
    if folder.exists():state='STARTED_STOPPED'
    if stage:state='STAGED_NOT_PUBLISHED'
    if publication and publication.get('create_request_accepted'):state='CREATE_ACCEPTED_REMOTE_VERIFICATION_PENDING'
    if presence:state='PRIVATE_PRESENCE_CONFIRMED_CPU_PENDING'
    if cert:state='PRIVATE_AND_CPU_CERTIFIED'
    lots.append(dict(number=spec['number'],handle=spec['handle'],role=spec['role'],events_expected=spec['events'],
        private_remote_confirmed=bool(presence and presence.get('private') is True),
        events_certified=cert['events'] if cert else 0,files_expected=spec['files'],files_certified=cert['files'] if cert else 0,
        bytes_certified=cert['bytes'] if cert else 0,sha256_crc32='ALL_MATCH_CANONICAL' if cert else 'NOT_REMOTE_CERTIFIED',
        manifest_identity=spec['identity'],archive_sha256=stage['sha256'] if stage else None,
        archive_bytes=stage['payload_bytes'] if stage else None,
        confirmed_blob_body_bytes=publication.get('confirmed_blob_body_bytes',0) if publication else 0,
        reserved_http_body_bytes=spent(folder/'publication/network.jsonl','reserved_body_bytes')+spent(folder/'cpu/network.jsonl','body_bytes'),
        publication_error_stage=publication.get('error_stage') if publication else None,
        publication_error_type=publication.get('error_type') if publication else None,
        state=state))
result=dict(status=summary['status'],lots=lots,cumulative_reserved_http_body_bytes=budget,
    body_ceiling_bytes=31_000_000_000,remaining_budget_bytes=31_000_000_000-budget,
    lot001_unchanged=True,gpu_training_started=False,
    budget_note='Durable conservative HTTP body reservations, including preparation controls, not physical wire traffic')
with (execution/'deployment-report.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
lines=['# GeoFire-X — résultat de publication des lots 002–008','',f"État : **{summary['status']}**.",'',
    '| Lot | Rôle | Privé confirmé | Événements certifiés / prévus | GeoTIFF certifiés / prévus | SHA-256 / CRC32 | Corps HTTP réservés | État |',
    '| --- | --- | --- | ---: | ---: | --- | ---: | --- |']
for r in lots:
    lines.append(f"| {r['number']:03d} | {r['role']} | {'oui' if r['private_remote_confirmed'] else 'non confirmé'} | {r['events_certified']} / {r['events_expected']} | {r['files_certified']} / {r['files_expected']} | {r['sha256_crc32']} | {r['reserved_http_body_bytes']:,} | {r['state']} |")
lines+=['',f'Cumul du journal : **{budget:,} / 31 000 000 000 octets**. Les réservations ne sont pas une mesure physique du trafic.',
    '',f"Cause d'arrêt, le cas échéant : {summary.get('reason','aucune')}",
    '', 'Le lot 001 n’a été ni reconstruit, ni réuploadé, ni recertifié. Aucun GPU, entraînement, commit, push ou merge.',
    '', 'Les identités des manifestes, SHA-256 des archives, tailles et preuves détaillées figurent dans deployment-report.json et les sous-répertoires de chaque lot.']
with (execution/'RAPPORT_DEPLOIEMENT.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
