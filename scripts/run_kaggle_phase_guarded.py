"""One explicitly selected long Kaggle preparation phase; native tree and RAM limits.

No retries. This wrapper does not authorize its child operation.
"""
import argparse
from run_memory_guarded import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--seconds',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or not 60 <= a.seconds <= 14400: p.error('One command, 60..14400 seconds required')
    result=run(command,a.out,seconds=a.seconds,job_bytes=1024**3,rss_bytes=1024**3)
    raise SystemExit(0 if result['status']=='FINISHED' else 1)
