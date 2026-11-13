"""Small read-only progress view for the authorized sequential Kaggle deployment."""
import json
from pathlib import Path
import sys

base=Path('reports/kaggle_remaining_20261108_v1')
plan=json.loads((base/'plan.json').read_text())
execution=Path(sys.argv[1]) if len(sys.argv)>1 else base/'execution'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def last(path):
    if not path.is_file():return None
    with path.open('rb') as stream:
        stream.seek(0,2);stream.seek(max(0,stream.tell()-16384))
        lines=stream.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except ValueError:continue


budget=0
for line in (base/'http-body.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['previous']!=budget or row['ceiling']!=31_000_000_000:raise ValueError('Budget chain differs')
    budget+=row['reserved']
state=(read(execution/'continuation-result.json') or read(execution/'continuation-progress.json') or {'status':'RUNNING'}) if (execution/'continuation-started.json').exists() else (read(execution/'result.json') or read(execution/'progress.json') or {'status':'RUNNING'})
result=dict(state=state,
            cumulative_reserved_body_bytes=budget,remaining_budget_bytes=31_000_000_000-budget,lots=[])
for spec in plan['lots']:
    folder=execution/spec['tag']
    row=dict(lot=spec['number'],state='NOT_STARTED')
    if folder.exists():
        row['state']='STARTED'
        staged=Path(spec['staging'])
        for name in ('payload.zip.part','payload.zip'):
            if (staged/name).exists():row['archive_bytes_on_disk']=(staged/name).stat().st_size
        if (staged/'staging-receipt.json').is_file():row['state']='STAGED_INTEGRITY_VERIFIED'
        publication=read(folder/'publication/result.json')
        if publication:row['publication']=publication
        cert=read(folder/'cpu/certificate.json')
        if cert:row.update(state=cert['status'],certified_files=cert['files'],certified_bytes=cert['bytes'])
        for phase in ('stage','publish','cpu'):
            monitor=folder/(phase+'-monitor')
            complete=read(monitor/'supervisor.json')
            if complete and complete['status']!='FINISHED':row.update(state='STOPPED',phase=phase,monitor_status=complete['status'])
            if not complete:
                sample=last(monitor/'memory.jsonl')
                if sample:
                    row.update(active_phase=phase,elapsed_seconds=round(sample['seconds']),
                        available_ram_bytes=sample['system']['available'],tree_rss_bytes=sample['tree']['rss_bytes'])
        operation=last(folder/'publication/network.jsonl')
        if operation:row['last_network_operation']=operation
    result['lots'].append(row)
canary=read(execution/'cpu-canary/remote-status.json')
if canary:result['cpu_gate']=canary
print(json.dumps(result,indent=2))
