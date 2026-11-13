"""Sequential, explicitly authorized publication and CPU certification, lots 002..008.

Offline by default. No GPU, version replacement, automatic retry, or lot001 upload.
The CPU automation canary must succeed before any raster is staged or sent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from run_memory_guarded import run as monitored
from kaggle_cpu_job import validate_receipt
from record_lot001_certificate import verify_evidence


def check_plan(plan):
    if [s['number'] for s in plan['lots']]!=list(range(2,9)) or plan['lot001_closed'] is not True:
        raise ValueError('Only lots 002..008 in order; lot001 is closed')
    if (plan['remaining_body_ceiling_bytes']!=31_000_000_000
        or plan['automatic_retries']!=0 or plan['remaining_raw_bytes']!=30_165_878_357):
        raise ValueError('Unexpected phase budget or corpus')
    for spec in plan['lots']:
        path=Path(spec['canonical'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['canonical_sha256']:
            raise ValueError('Frozen canonical manifest changed')
        lot=json.loads(path.read_text())
        if (lot['identity']!=spec['identity'] or lot['handle']!=spec['handle']
            or lot['observations']!=spec['files'] or lot['bytes']!=spec['raw_bytes']):
            raise ValueError('Lot scope differs from approved plan')
    lot001=json.loads((Path(plan['lots'][0]['canonical']).parent/'lot-001.json').read_text())
    verify_evidence(plan['lot001_evidence'],lot001)


def execute(plan_path,a):
    if getattr(a,'autonomous',False) and not getattr(a,'continue_after_indexing',False):
        raise PermissionError('Autonomous mode only continues resume-01 from lot003')
    if getattr(a,'continue_after_indexing',False) and getattr(a,'resume_once',False):
        raise PermissionError('Choose exactly one continuation mode')
    if getattr(a,'continue_after_indexing',False):
        from resume_remaining_lots import continue_after_confirmed_indexing
        return continue_after_confirmed_indexing(plan_path,a)
    if getattr(a,'resume_once',False):
        from resume_remaining_lots import run_resume
        return run_resume(plan_path,a)
    plan=json.loads(Path(plan_path).read_text())
    check_plan(plan)
    if not a.execute:
        print(json.dumps(dict(status='OFFLINE_REVIEW_ONLY',lots=[x['handle'] for x in plan['lots']],
            raw_bytes=plan['remaining_raw_bytes'],body_ceiling=plan['remaining_body_ceiling_bytes'],
            first_gate='Private CPU automation check before any raster transfer; stop on any failure',
            gpu=False,uploads=0),indent=2))
        return
    if a.approve_body_bytes!=plan['remaining_body_ceiling_bytes']:
        raise PermissionError('Explicit --approve-body-bytes 31000000000 required after user authorization')
    out=Path(plan_path).parent
    execution=out/'execution'
    # A failed or ambiguous attempt is never resubmitted by rerunning the command.
    execution.mkdir(exist_ok=False)
    (execution/'started.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        approved_body_bytes=a.approve_body_bytes,gpu=False,lot001_closed=True),indent=2))
    def child(args,name,seconds):
        receipt=monitored([sys.executable,*args],execution/name,seconds=seconds,
            start_bytes=2*1024**3+256*1024**2,reserve_bytes=2*1024**3,
            job_bytes=512*1024**2,rss_bytes=512*1024**2,verbose=False)
        if receipt['status']!='FINISHED': raise RuntimeError('Stopped at '+name+'; inspect logs, no automatic retry')
    summary=dict(status='RUNNING',completed=[])
    try:
        child(['scripts/kaggle_cpu_job.py','--plan',str(plan_path),'--tag','canary','--out',str(execution/'cpu-canary'),
               '--credentials',a.credentials],'canary-monitor',2400)
        canary=json.loads((execution/'cpu-canary/controller-result.json').read_text())
        if canary['status']!='CPU_AUTOMATION_READY': raise RuntimeError('CPU automation unavailable; no raster transfer')
        for spec in plan['lots']:
            tag=spec['tag']
            (execution/tag).mkdir()
            child(['scripts/stage_expanded_lot.py','--lot',spec['canonical'],'--raw-root',plan['raw_root'],
                   '--out',spec['staging'],'--max-bytes',str(spec['archive_ceiling_bytes'])],tag+'/stage-monitor',3600)
            child(['scripts/publish_kaggle_single_attempt.py','--staged',spec['staging'],'--credentials',a.credentials,
                '--journal',plan['journal'],'--attempt',str(execution/tag/'publication'),
                '--max-http-body-bytes',str(plan['remaining_body_ceiling_bytes']),
                '--stop-at-cumulative-body-bytes',str(plan['remaining_body_ceiling_bytes']),
                '--license','CC-BY-4.0','--approved-plan',str(plan_path),'--execute'],tag+'/publish-monitor',7200)
            publication=json.loads((execution/tag/'publication/result.json').read_text())
            if publication['status']!='PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING': raise RuntimeError('Creation not confirmed')
            child(['scripts/kaggle_cpu_job.py','--plan',str(plan_path),'--tag',tag,'--out',str(execution/tag/'cpu'),
                   '--credentials',a.credentials],tag+'/cpu-monitor',2400)
            validate_receipt(execution/tag/'cpu',spec)
            summary['completed'].append(tag)
            (execution/'progress.json').write_text(json.dumps(summary,indent=2))
        # Notebook templates can now embed all eight verified receipts; still no training.
        env_python=Path('.venv/Scripts/python.exe')
        import subprocess
        subprocess.run([str(env_python),'scripts/prepare_remaining_kaggle.py','--out',str(out),'--refresh-training'],check=True,timeout=120)
        summary['status']='SEVEN_LOTS_PRIVATE_AND_CPU_CERTIFIED_NO_GPU'
    except Exception as error:
        summary.update(status='FAILED_STOP_NO_RETRY',error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        (execution/'result.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--credentials',default='C:/Users/nourb/.kaggle/credentials.json')
    p.add_argument('--approve-body-bytes',type=int,default=0)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume-once',action='store_true',help='Single controlled resume; reuse verified ZIP and completed receipts')
    p.add_argument('--continue-after-indexing',action='store_true',help='Same resume only: require lot002 accepted once, private and CPU certified; start at lot003')
    p.add_argument('--autonomous',action='store_true',help='Standalone guarded 3 GiB profile; lots003..008 only, release each ZIP after certification')
    args=p.parse_args()
    execute(Path(args.plan),args)
