"""Bound the small metadata-only submission; keep 2 GiB system RAM free."""
import sys
from run_memory_guarded import run

from pathlib import Path
import json
first = Path('reports/kaggle_certification_20261108_v1/submission-attempt-01')
events = [json.loads(line) for line in (first/'network.jsonl').read_text().splitlines()]
if any(row['operation']=='SaveKernel' for row in events):
    raise RuntimeError('A prior SaveKernel exists; inspect remote state instead')
command = [sys.executable, 'scripts/submit_kaggle_cpu_certification.py',
    '--folder', 'reports/kaggle_certification_20261108_v1/cpu-certification/lot-001',
    '--credentials', 'C:/Users/nourb/.kaggle/credentials.json',
    '--journal', 'reports/kaggle_pilot_001_20261107_v1/http-body.jsonl',
    '--attempt', 'reports/kaggle_certification_20261108_v1/submission-attempt-02',
    '--max-http-body-bytes', '6000000000', '--stop-at-cumulative-body-bytes', '6000000000', '--execute']
result = run(command, 'reports/kaggle_certification_20261108_v1/submission-monitor-03',
    seconds=120, start_bytes=2*1024**3+128*1024**2, reserve_bytes=2*1024**3,
    job_bytes=256*1024**2, rss_bytes=256*1024**2)
raise SystemExit(0 if result['status']=='FINISHED' else 1)
