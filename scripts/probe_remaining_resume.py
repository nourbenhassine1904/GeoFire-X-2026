"""First gate for the single authorized resume; no ZIP reads or network."""
import datetime as dt
import json
from pathlib import Path

from run_bounded_local import memory

sample=memory()
required=4*1024**3
result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),memory=sample,
    required_start_bytes=required,status='RAM_GATE_PASSED' if sample and sample['available']>=required else 'PREFLIGHT_LOW_MEMORY',
    children_started=0,zip_bytes_read=0,network_requests=0)
base=Path('reports/kaggle_remaining_20261108_v1')
used=0
for line in (base/'http-body.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['previous']!=used or row['ceiling']!=31_000_000_000:raise ValueError('HTTP budget chain changed')
    used+=row['reserved']
result.update(cumulative_body_bytes=used,remaining_body_bytes=31_000_000_000-used)
if used!=7345:raise ValueError('Unexpected pre-resume HTTP counter; inspect state before any operation')
out=base/'resume-01'
out.mkdir(exist_ok=False)
with (out/'preflight.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['status']=='RAM_GATE_PASSED' else 1)
