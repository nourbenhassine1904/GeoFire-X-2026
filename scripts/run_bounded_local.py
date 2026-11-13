"""Run one local command with a deadline, bounded memory and exclusive logs.

No retry, Spark startup, network operation or historical artifact replacement.
Only processes created by this invocation may be stopped.
"""
import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.job_control import ProcessTree


def memory():
    if os.name != 'nt':
        return None
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
    value = Status()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return dict(total=value.total, available=value.available, load_percent=value.load,
                commit_limit_bytes=value.page_total, commit_available_bytes=value.page_available,
                commit_used_bytes=value.page_total-value.page_available)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=int, default=240)
    p.add_argument('--min-free-memory-mb', type=int, default=3072)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or not 1 <= a.seconds <= 600:
        p.error('A command and a deadline between 1 and 600 seconds are required')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    before = dict(memory=memory(), disk=shutil.disk_usage('.')._asdict())
    required_memory = max(1024**3, a.min_free_memory_mb*1024**2)
    if before['memory'] and before['memory']['available'] < required_memory:
        result = dict(status='PREFLIGHT_LOW_MEMORY', command=command, pid=None,
                      required_available_bytes=required_memory, before=before)
        with (out/'supervisor.json').open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return 1
    started = time.monotonic()
    peak_load = 0
    status = 'FINISHED'
    with (out/'output.log').open('x', encoding='utf-8') as log:
        tree = ProcessTree(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        proc = tree.proc
        try:
            while proc.poll() is None:
                current = memory()
                peak_load = max(peak_load, current['load_percent'] if current else 0)
                if current and current['available'] < 1024**3:
                    status = 'LOW_AVAILABLE_MEMORY'
                    break
                if shutil.disk_usage('.').free < 1024**3:
                    status = 'LOW_DISK_SPACE'
                    break
                remaining = a.seconds-(time.monotonic()-started)
                if remaining <= 0:
                    status = 'HARD_TIME_LIMIT'
                    break
                try:
                    proc.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            try:
                if proc.poll() is None:
                    tree.stop()
            finally:
                tree.close()
    result = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), command=command, pid=proc.pid,
                  status=status if status != 'FINISHED' or proc.returncode == 0 else 'FAILED',
                  exit_code=proc.returncode, elapsed_seconds=time.monotonic()-started,
                  limit_seconds=a.seconds, peak_system_memory_load_percent=peak_load,
                  before=before, after=dict(memory=memory(), disk=shutil.disk_usage('.')._asdict()))
    with (out/'supervisor.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'FINISHED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
