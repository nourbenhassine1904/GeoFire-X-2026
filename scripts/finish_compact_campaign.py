"""Wait for successful publication, then single CPU certification submission.

Stops on any publication or remote-check failure. Never uploads dataset bytes,
retries a write, or starts GPU work. A durable marker prevents two finishers.
"""
import datetime as dt
import json
from pathlib import Path
import time
from compact_campaign_remote import BASE,OUT,main as remote


def run():
    with (OUT/'cpu-finisher-started.json').open('x') as f:
        json.dump(dict(time=dt.datetime.now(dt.timezone.utc).isoformat()),f)
    try:
        previous=None
        while True:
            state=json.loads((OUT/'checkpoint.json').read_text())
            scope=(state['status'],len(state['parts']),state.get('current_part'),state.get('file_index'))
            if scope!=previous:
                print(json.dumps(dict(publication=state['status'],confirmed_partitions=len(state['parts']),
                                      current_part=state.get('current_part'),file_index=state.get('file_index'))),flush=True)
                previous=scope
            if state['status'].startswith('STOPPED'):
                raise RuntimeError('Publication stopped; CPU submission refused')
            if state['status'].startswith('ALL_11'):break
            time.sleep(30)
        time.sleep(3) # allow publisher to release the shared budget lock
        remote('submit')
        job=BASE/'compact_cpu_job'
        for _ in range(100):
            time.sleep(30)
            remote('status')
            status=json.loads((job/'status.json').read_text())['status']
            if status in ('COMPLETE','complete'):
                remote('fetch')
                certificate=json.loads((OUT/'remote-certificate.json').read_text())
                assert certificate['status']=='PASS'
                (OUT/'certified.json').write_text(json.dumps(dict(status='CERTIFIED',
                    certificate='remote-certificate.json',gpu_profile_prepared=True,
                    gpu_campaign_started=False,notebook='B1_B2_COMPACT_CERTIFIED_READY.ipynb'),indent=2))
                print('ALL_11_PRIVATE_COMPACT_PARTITIONS_CPU_CERTIFIED',flush=True)
                return 0
            if status not in ('QUEUED','RUNNING','queued','running'):
                raise RuntimeError('CPU certificate failed or unexpected state')
        raise TimeoutError('CPU certificate pending beyond observation window')
    except BaseException as error:
        (OUT/'cpu-finisher-stopped.json').write_text(json.dumps(dict(
            status='STOPPED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__),indent=2))
        print('CPU_FINISHER_STOPPED',type(error).__name__,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(run())
