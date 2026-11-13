"""One isolated command: native Windows tree cap, memory timeline, no retries."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geofirex.job_control import ProcessTree
from run_bounded_local import memory


def run(command, out, *, seconds=90, start_bytes=3*1024**3, reserve_bytes=2*1024**3,
        job_bytes=2*1024**3, rss_bytes=2*1024**3, verbose=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    before = memory()
    result = dict(command=command, before=before, limit_seconds=seconds,
                  start_memory_bytes=start_bytes, reserve_memory_bytes=reserve_bytes,
                  job_commit_limit_bytes=job_bytes, rss_limit_bytes=rss_bytes,
                  time=dt.datetime.now(dt.timezone.utc).isoformat())
    if os.name != 'nt' or before is None:
        result.update(status='NATIVE_MONITOR_UNAVAILABLE', children_started=0)
    elif before['available'] < start_bytes:
        result.update(status='PREFLIGHT_LOW_MEMORY', children_started=0)
    else:
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                   PYTHONPATH=os.pathsep.join([str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]),
                   CUDA_VISIBLE_DEVICES='')
        started = time.monotonic()
        peak_rss = peak_private = peak_job = peak_processes = 0
        minimum_available = before['available']
        status, error, last_pids = 'FINISHED', None, []
        with (out/'output.log').open('x', encoding='utf-8') as log, (out/'memory.jsonl').open('x', encoding='utf-8') as timeline:
            tree = ProcessTree(command, stdout=log, stderr=log, env=env, memory_limit_bytes=job_bytes)
            try:
                while True:
                    system = memory()
                    usage = tree.memory_sample()
                    last_pids = [r['pid'] for r in usage['processes']]
                    peak_processes = max(peak_processes, len(last_pids))
                    peak_rss = max(peak_rss, usage['rss_bytes'])
                    peak_private = max(peak_private, usage['private_commit_bytes'])
                    peak_job = max(peak_job, usage['peak_job_commit_bytes'])
                    minimum_available = min(minimum_available, system['available'])
                    elapsed = time.monotonic()-started
                    timeline.write(json.dumps(dict(seconds=elapsed, system=system, tree=usage))+'\n')
                    timeline.flush()
                    if tree.proc.poll() is not None:
                        break
                    if system['available'] < reserve_bytes:
                        status = 'LOW_AVAILABLE_MEMORY'
                        break
                    if usage['rss_bytes'] > rss_bytes:
                        status = 'TREE_RSS_LIMIT'
                        break
                    if elapsed >= seconds:
                        status = 'HARD_TIME_LIMIT'
                        break
                    if shutil.disk_usage(out).free < 1024**3:
                        status = 'LOW_DISK_SPACE'
                        break
                    time.sleep(.25)
            except Exception as exception:
                status, error = 'MONITOR_ERROR', repr(exception)
            finally:
                try:
                    if tree.proc.poll() is None:
                        tree.stop()
                    tree.proc.wait(timeout=5)
                finally:
                    tree.close()  # Also owns descendants after parent exit.
            result.update(status=status if status != 'FINISHED' or tree.proc.returncode == 0 else 'FAILED',
                          exit_code=tree.proc.returncode, pid=tree.proc.pid, children_started=1,
                          last_observed_owned_pids=last_pids, elapsed_seconds=time.monotonic()-started,
                          peak_tree_rss_bytes=peak_rss, peak_sampled_private_commit_bytes=peak_private,
                          peak_job_commit_bytes=peak_job, peak_owned_processes=peak_processes,
                          minimum_available_bytes=minimum_available, monitor_error=error)
        time.sleep(.25)
    result['after'] = memory()
    with (out/'supervisor.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=int, default=90)
    p.add_argument('--job-mb', type=int, default=2048)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or not 1 <= a.seconds <= 300 or not 128 <= a.job_mb <= 2048:
        p.error('Command, deadline <=300s and native job cap 128..2048 MiB required')
    result = run(command, a.out, seconds=a.seconds, job_bytes=a.job_mb*1024**2)
    raise SystemExit(0 if result['status'] == 'FINISHED' else 1)
