"""Standalone supervisor; review by default, never retry an existing attempt."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from kaggle_autonomous import (CEILING_BYTES, DEFAULT_PLAN, INITIAL_BODY_BYTES,
    RESERVE_BYTES, START_BYTES, TREE_BYTES, checkpoint, save_json, validate_autonomous)
from resume_remaining_lots import reserved
from run_bounded_local import memory
from run_memory_guarded import run

ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS = 'C:/Users/nourb/.kaggle/credentials.json'


def status(plan_path):
    plan = json.loads(plan_path.read_text())
    folder = plan_path.parent/'resume-01/autonomous-01'
    result = dict(cumulative_reserved_body_bytes=reserved(plan),
                  remaining_body_bytes=CEILING_BYTES-reserved(plan))
    for key, path in (('checkpoint', folder/'checkpoint.json'),
                      ('supervisor', folder/'monitor/supervisor.json')):
        result[key] = json.loads(path.read_text()) if path.is_file() else None
    timeline = folder/'monitor/memory.jsonl'
    if timeline.is_file():
        with timeline.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-16384))
            for line in reversed(stream.read().splitlines()):
                try:
                    result['last_memory_sample'] = json.loads(line)
                    break
                except ValueError:
                    continue
        result['telemetry_age_seconds'] = max(0, __import__('time').time()-timeline.stat().st_mtime)
    result['note'] = 'Only a finished supervisor confirms termination; stale telemetry requires inspection, never automatic retry'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.status:
        print(json.dumps(status(DEFAULT_PLAN), indent=2))
        return 0
    plan, execution = validate_autonomous(DEFAULT_PLAN)
    missing = [name for name in ('requests', 'kagglesdk') if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing local runtime dependencies: '+', '.join(missing))
    if not Path(CREDENTIALS).is_file() or not Path(plan['raw_root']).is_dir():
        raise FileNotFoundError('Existing credentials or raw corpus unavailable; no network started')
    free = shutil.disk_usage(ROOT).free
    if free < max(s['archive_ceiling_bytes'] for s in plan['lots'][1:])+20_000_000_000:
        raise OSError('Insufficient room for one ZIP and the 20 GB disk reserve')
    sample = memory()
    review = dict(status='OFFLINE_REVIEW_ONLY', same_resume='resume-01', first_lot=3,
                  remaining_lots=list(range(3, 9)), initial_body_bytes=INITIAL_BODY_BYTES,
                  ceiling_body_bytes=CEILING_BYTES, start_bytes=START_BYTES,
                  reserve_bytes=RESERVE_BYTES, tree_limit_bytes=TREE_BYTES,
                  memory=sample, memory_ready=bool(sample and sample['available'] >= START_BYTES),
                  gpu=False, automatic_retries=0, network_requests=0)
    print(json.dumps(review, indent=2), flush=True)
    if not args.execute:
        return 0
    folder = execution.parent/'autonomous-01'
    # Exclusive directory creation is the whole-run lock, including between phases.
    # A stopped/ambiguous attempt is never silently restarted, even after a reboot.
    folder.mkdir(exist_ok=False)
    checkpoint(folder, plan, status='PREFLIGHT', phase='standalone-supervisor', completed=[])
    command = [sys.executable, 'scripts/publish_remaining_lots.py', '--plan', str(DEFAULT_PLAN),
               '--continue-after-indexing', '--autonomous', '--approve-body-bytes', str(CEILING_BYTES),
               '--credentials', CREDENTIALS, '--execute']
    try:
        # Entire descendant tree, including nested phase monitors, remains <=512 MiB.
        outcome = run(command, folder/'monitor', seconds=86_400,
                      start_bytes=START_BYTES, reserve_bytes=RESERVE_BYTES,
                      job_bytes=TREE_BYTES, rss_bytes=TREE_BYTES, verbose=False)
    except BaseException as error:
        old = json.loads((folder/'checkpoint.json').read_text())
        checkpoint(folder, plan, status='SUPERVISOR_INTERRUPTED_NO_RETRY', phase=old['phase'],
                   completed=old['completed_lots'][2:], active_lot=old['active_lot'],
                   error_type=type(error).__name__)
        raise
    old = json.loads((folder/'checkpoint.json').read_text())
    success = outcome['status'] == 'FINISHED' and old['status'] == 'SEVEN_LOTS_PRIVATE_AND_CPU_CERTIFIED_NO_GPU'
    final = 'LOTS003_TO008_PRIVATE_AND_CPU_CERTIFIED_NO_GPU' if success else 'STOPPED_NO_RETRY'
    checkpoint(folder, plan, status=final, phase=old['phase'], completed=old['completed_lots'][2:],
               active_lot=old['active_lot'], supervisor_status=outcome['status'])
    save_json(folder/'result.json', dict(status=final, supervisor_status=outcome['status'],
              cumulative_reserved_body_bytes=reserved(plan), gpu_started=False))
    print(json.dumps(status(DEFAULT_PLAN), indent=2), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never dump SDK response text, credentials, tokens or signed URLs.
        print('STOP_NO_RETRY: '+type(error).__name__+': '+str(error), file=sys.stderr)
        raise SystemExit(1)
