"""One authorized existing-ZIP verification, with the original RAM protections."""
import sys
from run_memory_guarded import run

result=run([sys.executable,'scripts/verify_staged_lot_zip.py',
    '--canonical','reports/kaggle_remaining_20261108_v1/training/bundle/metadata/lot-002.json',
    '--staged','artifacts/kaggle_remaining_20261108_v1/lot-002',
    '--evidence','reports/kaggle_remaining_20261108_v1/resume-01/zip002-verification'],
    'reports/kaggle_remaining_20261108_v1/resume-01/zip002-monitor',seconds=1800,
    start_bytes=4*1024**3,reserve_bytes=2*1024**3,job_bytes=512*1024**2,rss_bytes=512*1024**2)
raise SystemExit(0 if result['status']=='FINISHED' else 1)
