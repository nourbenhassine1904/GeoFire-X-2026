"""Bounded read-only supervisor: status -> retrieve evidence; never submits jobs.

Stop by creating artifacts/orchestrator/monitor-<ID>/stop.request.
Every status/result is append-only. Cloud jobs keep their own hard timeouts.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

from geofirex.cloud_corpus import check_seal
from geofirex.corpus import json_new
from geofirex.orchestration import Orchestrator, timestamp


def supervise(runner, job_ids, *, minutes=65, interval=30):
    monitor = runner.state/('monitor-'+uuid.uuid4().hex[:12])
    json_new(monitor/'started.json', {'pid': os.getpid(), 'started_at': timestamp(),
        'deadline_minutes': minutes, 'job_ids': job_ids, 'scope': 'status and evidence retrieval only'})
    print('Monitor:', monitor, flush=True)
    deadline = time.monotonic()+minutes*60
    pending = set(job_ids)
    while pending and time.monotonic() < deadline and not (monitor/'stop.request').exists():
        cycle = []
        for job_id in sorted(pending):
            try:
                result = runner.refresh(job_id)
                if result['remote_status'] in {'COMPLETE', 'ERROR', 'CANCELACKNOWLEDGED'}:
                    result['retrieval'] = runner.fetch(job_id)
                    retrieval = result['retrieval']
                    if retrieval.get('success') is not None:
                        pending.remove(job_id)
                cycle.append(result)
            except Exception as error:
                cycle.append({'job_id': job_id, 'error_type': type(error).__name__})
        json_new(monitor/(time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.json'),
                 {'time': timestamp(), 'cycle': cycle, 'pending': sorted(pending)})
        print(json.dumps({'time': timestamp(), 'pending': sorted(pending),
            'states': [{k:r[k] for k in ('job_id', 'remote_status', 'error_type') if k in r} for r in cycle]}), flush=True)
        if pending:
            time.sleep(min(interval, max(0, deadline-time.monotonic())))
    json_new(monitor/'finished.json', {'time': timestamp(), 'pending': sorted(pending),
        'reason': 'ALL_TERMINAL_RESULTS_RETRIEVED' if not pending else 'STOP_OR_DEADLINE'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs='+', required=True)
    parser.add_argument('--minutes', type=int, default=65)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 65:
        parser.error('Bounded supervision: 1..65 minutes')
    supervise(Orchestrator(Path(__file__).resolve().parents[1]), args.jobs, minutes=args.minutes)
