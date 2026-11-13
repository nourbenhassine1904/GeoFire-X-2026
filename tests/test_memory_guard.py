"""Native memory guard checks on tiny owned children, no raster corpus or Torch."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from geofirex.job_control import ProcessTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_memory_guarded import run

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows Job memory accounting')


def test_native_job_rejects_excess_commit(tmp_path):
    source = tmp_path/'probe.py'
    source.write_text('import sys\ntry:\n    value = bytearray(256*1024*1024)\n'
                      'except MemoryError:\n    print("NATIVE_CAP_ENFORCED")\n'
                      'else:\n    sys.exit("Native memory limit was not enforced")\n')
    tree = ProcessTree([sys.executable, str(source)], memory_limit_bytes=128*1024**2,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = tree.proc.communicate(timeout=5)
        sample = tree.memory_sample()
        assert tree.proc.returncode == 0, output
        assert 'NATIVE_CAP_ENFORCED' in output
        # The observed Job peak can include a refused allocation. Enforcement
        # is proved by MemoryError and the configured limit, not peak <= limit.
        assert sample['peak_job_commit_bytes'] > 0
        assert sample['job_commit_limit_bytes'] == 128*1024**2
    finally:
        if tree.proc.poll() is None:
            tree.stop()
        tree.close()


def test_monitor_records_owned_memory(tmp_path):
    result = run([sys.executable, '-c', 'import time; value=bytearray(8*1024*1024); time.sleep(.6)'],
                 tmp_path/'monitor', seconds=5, job_bytes=128*1024**2)
    assert result['status'] == 'FINISHED'
    assert result['peak_tree_rss_bytes'] > 8*1024**2
    assert result['peak_job_commit_bytes'] > 8*1024**2
    assert result['minimum_available_bytes'] >= 2*1024**3
    assert (tmp_path/'monitor/memory.jsonl').stat().st_size > 0


def test_monitor_stops_owned_child_on_rss_limit(tmp_path):
    result = run([sys.executable, '-c', 'import time; time.sleep(10)'], tmp_path/'limited',
                 seconds=5, rss_bytes=1)
    assert result['status'] == 'TREE_RSS_LIMIT'
    assert result['elapsed_seconds'] < 5
    assert result['exit_code'] != 0


def test_monitor_refuses_without_starting_child(tmp_path):
    result = run([sys.executable, '-c', 'raise AssertionError("Must not start")'],
                 tmp_path/'preflight', start_bytes=2**63)
    assert result['status'] == 'PREFLIGHT_LOW_MEMORY'
    assert result['children_started'] == 0


def test_sampling_handles_normal_process_exit():
    for _ in range(8):
        tree = ProcessTree([sys.executable, '-c', 'pass'], memory_limit_bytes=128*1024**2)
        try:
            deadline = time.monotonic()+5
            while tree.proc.poll() is None and time.monotonic() < deadline:
                tree.memory_sample()
            tree.proc.wait(timeout=2)
            assert tree.proc.returncode == 0
            tree.memory_sample()
        finally:
            if tree.proc.poll() is None:
                tree.stop()
            tree.close()
        assert not tree._memory_handles
