"""Bounded autonomous follow-up: acquired lots -> new-session checks -> T4 probe.

Never starts a multi-epoch GPU campaign or another Zenodo transfer. Resume with
the same --chain directory; child job IDs are recorded before submission.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

from geofirex.cloud_corpus import check_seal, seal
from geofirex.corpus import json_new, sha256_file
from geofirex.orchestration import Orchestrator, timestamp


def completed(runner, job_id):
    job, request = runner.load(job_id)
    events = [json.loads(p.read_text()) for p in sorted((job/'events').glob('*.json'))]
    evidence = [e for e in events if e['kind'] == 'EVIDENCE_VERIFIED']
    if evidence:
        if sha256_file(evidence[-1]['path']) != evidence[-1]['sha256']:
            raise ValueError('Archived evidence changed for '+job_id)
        return evidence[-1]['path']
    if any(e['kind'] == 'REMOTE_FAILURE' for e in events):
        raise RuntimeError('Remote failure retained for '+job_id)
    status = runner.refresh(job_id)
    if status['remote_status'] in {'COMPLETE', 'ERROR', 'CANCELACKNOWLEDGED'}:
        result = runner.fetch(job_id)
        if result.get('success') is False:
            raise RuntimeError('Remote failure retained for '+job_id)
        if result.get('success') is True:
            return completed(runner, job_id)
        if status['remote_status'] in {'ERROR', 'CANCELACKNOWLEDGED'}:
            raise RuntimeError('Terminal remote job without a successful result: '+job_id)
    return None


def child(runner, chain, name, kind, owner, **arguments):
    record = chain/(name+'.json')
    if record.exists():
        return json.loads(record.read_text())['job_id']
    job = runner.prepare(kind, owner, **arguments)
    json_new(record, {'job_id': job.name, 'created_at': timestamp()})
    return job.name


def ensure_submitted(runner, job_id, *, authorization=None):
    job, _ = runner.load(job_id)
    events = [json.loads(p.read_text()) for p in (job/'events').glob('*.json')]
    if not any(e['kind'] == 'SUBMISSION_ATTEMPT' for e in events):
        result = runner.submit(job_id, authorization=authorization)
        if result['returncode'] != 0:
            raise RuntimeError('Submission unconfirmed; retained for '+job_id)


def recoverable_publication(runner, parent):
    """Only a complete local acquisition followed by an attempted upload qualifies.

    The next job only reads an existing remote dataset. It never retries create.
    """
    job, _ = runner.load(parent['job_id'])
    for result_path in sorted((job/'retrievals').rglob('result.json')):
        result = json.loads(result_path.read_text())
        publication = result_path.parent/'publication.json'
        acquisition = result_path.parent/'acquisition.json'
        if (result.get('job_id') != parent['job_id'] or result.get('success') is not False
                or not publication.is_file() or not acquisition.is_file()):
            continue
        receipt = json.loads(publication.read_text())
        report = json.loads(acquisition.read_text())
        check_seal(receipt)
        if (receipt.get('upload_called') is True and receipt.get('status') == 'FAILED_FILES_PRESERVED'
                and receipt.get('handle') == parent['dataset_handle']
                and report.get('plan_identity') == parent['plan']['identity']
                and report.get('inventory_identity') == receipt.get('inventory_identity')
                and report.get('integrity', {}).get('complete') is True
                and report['transfer']['reserved_bytes_including_retries'] <= parent['plan']['limits']['transfer_bytes']):
            return publication
    return None


def run(chain, runner, acquisition_ids, scope_path, *, minutes):
    if (chain/'finished.json').exists():
        print((chain/'finished.json').read_text(), flush=True)
        return
    scope = json.loads(Path(scope_path).read_text())
    check_seal(scope)
    parents = [runner.load(job_id)[1] for job_id in acquisition_ids]
    if (any(p['kind'] != 'acquire' for p in parents)
            or set(scope['parent_job_identities']) != {p['identity'] for p in parents}
            or scope.get('purpose') != 'discarded Tesla T4 throughput only'
            or scope.get('safety_cap_seconds') != 900
            or scope.get('long_training_authorized') is not False):
        raise ValueError('Scope does not authorize these follow-up jobs')
    started = chain/'started.json'
    if not started.exists():
        json_new(started, {'pid': os.getpid(), 'time': timestamp(), 'acquisition_jobs': acquisition_ids,
                          'scope_identity': scope['identity'], 'deadline_minutes': minutes})
    else:
        json_new(chain/('resumed-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.json'),
                 {'pid': os.getpid(), 'time': timestamp(), 'deadline_minutes': minutes})
    deadline = time.monotonic()+minutes*60
    while time.monotonic() < deadline and not (chain/'stop.request').exists():
        state = {'time': timestamp(), 'certificates': {}, 'phase': 'WAITING_FOR_ACQUISITION'}
        try:
            for parent in parents:
                try:
                    receipt = completed(runner, parent['job_id'])
                except RuntimeError:
                    attempted = recoverable_publication(runner, parent)
                    if attempted is None:
                        raise
                    plan_path = chain/('plan-'+parent['job_id']+'.json')
                    if not plan_path.exists():
                        json_new(plan_path, parent['plan'])
                    recovery_id = child(runner, chain, 'recover-'+parent['job_id'], 'recover-batch',
                                        parent['owner'], plan=plan_path, proof=attempted)
                    ensure_submitted(runner, recovery_id)
                    receipt = completed(runner, recovery_id)
                if not receipt:
                    continue
                job_id = child(runner, chain, 'certify-'+parent['job_id'], 'verify-batch',
                               parent['owner'], proof=receipt)
                ensure_submitted(runner, job_id)
                certificate = completed(runner, job_id)
                if certificate:
                    state['certificates'][parent['job_id']] = certificate
            if len(state['certificates']) == len(parents):
                state['phase'] = 'T4_THROUGHPUT'
                measure_id = child(runner, chain, 'throughput', 'throughput', parents[0]['owner'],
                                   certificates=list(state['certificates'].values()))
                job, request = runner.load(measure_id)
                permission = chain/'throughput-authorization.json'
                if not permission.exists():
                    json_new(permission, seal({'source_scope_identity': scope['identity'],
                        'authorization_basis': scope['authorization_basis'],
                        'job_identities': [request['identity']], 'safety_cap_seconds': 900,
                        'cost_eur': 0, 'long_training_authorized': False}))
                ensure_submitted(runner, measure_id, authorization=permission)
                evidence = completed(runner, measure_id)
                if evidence:
                    state.update(phase='COMPLETE_AWAITING_LONG_GPU_BUDGET', throughput=evidence)
                    json_new(chain/'finished.json', state)
                    print(json.dumps(state), flush=True)
                    return
            json_new(chain/('cycle-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.json'), state)
            print(json.dumps(state), flush=True)
        except Exception as error:
            state.update(phase='STOPPED_FAILURE_PRESERVED', error_type=type(error).__name__,
                         message=str(error)[:200])
            json_new(chain/'finished.json', state)
            print(json.dumps(state), flush=True)
            return
        time.sleep(min(30, max(0, deadline-time.monotonic())))
    json_new(chain/('stopped-'+uuid.uuid4().hex[:8]+'.json'),
             {'time': timestamp(), 'phase': 'STOP_OR_DEADLINE', 'resume_with_same_chain': str(chain)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisitions', nargs=2, required=True)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--chain', required=True)
    parser.add_argument('--minutes', type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 720:
        parser.error('Bounded orchestration requires 1..720 minutes')
    run(Path(args.chain), Orchestrator(Path(__file__).resolve().parents[1]), args.acquisitions, args.scope,
        minutes=args.minutes)
