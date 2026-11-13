"""First CPU certification of already-created lot002; no dataset writes/uploads."""
import json
from pathlib import Path
import sys

from run_memory_guarded import run

base=Path('reports/kaggle_remaining_20261108_v1')
presence=json.loads((base/'resume-01/reconciliation-lot002/result.json').read_text())
if presence['status']!='PRIVATE_REMOTE_PRESENCE_CONFIRMED' or presence['presence']['private'] is not True:
    raise ValueError('Existing private remote presence must be confirmed first')
output=base/'resume-01/execution/lot-002/cpu'
if output.exists():raise FileExistsError('CPU attempt exists; inspect without resubmission')
result=run([sys.executable,'scripts/kaggle_cpu_job.py','--plan',str(base/'plan.json'),'--tag','lot-002',
    '--out',str(output)],base/'resume-01/execution/lot-002/cpu-monitor',seconds=2400,
    start_bytes=4*1024**3,reserve_bytes=2*1024**3,job_bytes=512*1024**2,rss_bytes=512*1024**2)
raise SystemExit(0 if result['status']=='FINISHED' else 1)
