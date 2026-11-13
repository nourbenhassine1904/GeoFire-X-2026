"""Single controlled resume of the existing sequential deployment, no automatic retries."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_bounded_local import memory
from run_memory_guarded import run as monitored
from kaggle_cpu_job import validate_receipt


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def reserved(plan):
    used=0
    for line in Path(plan['journal']).read_text().splitlines():
        row=json.loads(line)
        if row['previous']!=used or row['ceiling']!=31_000_000_000:raise ValueError('HTTP budget chain changed')
        used+=row['reserved']
    return used


def run_resume(plan_path,args):
    from publish_remaining_lots import check_plan
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    check_plan(plan)
    if not args.execute or args.approve_body_bytes!=31_000_000_000:
        raise PermissionError('Explicit authorized execution and unchanged 31 GB ceiling required')
    available=memory()
    if not available or available['available']<4*1024**3:
        raise MemoryError('Resume refused: less than 4 GiB available; no network started')
    base=plan_path.parent
    resume=base/'resume-01'
    proof=read(resume/'zip002-verification/verification.json')
    monitor=read(resume/'zip002-monitor/supervisor.json')
    if (not proof or proof['status']!='EXISTING_ZIP_FULLY_VERIFIED_REUSED' or proof['rebuilt'] is not False
        or proof['payload_bytes']!=4_995_525_361 or proof['verified_files']!=1380
        or monitor is None or monitor['status']!='FINISHED'):
        raise ValueError('Existing ZIP 002 verification has not completed successfully')
    if reserved(plan)!=7345:raise ValueError('Unexpected HTTP state before the single authorized resume')
    execution=resume/'execution'
    execution.mkdir(exist_ok=False)
    summary=dict(status='RUNNING',completed=[],reused=[],starting_reserved_body_bytes=7345,
        zip002_rebuilt=False,lot001_unchanged=True,gpu=False)
    (execution/'started.json').write_text(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),approved_body_bytes=31_000_000_000,
        starting_reserved_body_bytes=7345,initial_memory=available,start_threshold_bytes=4*1024**3,
        reserve_bytes=2*1024**3,process_limit_bytes=512*1024**2),indent=2))
    def child(argv,name,seconds):
        outcome=monitored([sys.executable,*argv],execution/name,seconds=seconds,
            start_bytes=4*1024**3,reserve_bytes=2*1024**3,
            job_bytes=512*1024**2,rss_bytes=512*1024**2,verbose=False)
        if outcome['status']!='FINISHED':
            raise RuntimeError('Stopped at '+name+': '+outcome['status']+'; state retained, no automatic retry')
    try:
        # The completed CPU gate is reused, rather than repeating finished work.
        gate=read(base/'execution/cpu-canary/controller-result.json')
        environment=read(base/'execution/cpu-canary/environment.json')
        if not gate or gate['status']!='CPU_AUTOMATION_READY' or not environment or environment['status']!='CPU_AUTOMATION_READY':
            raise ValueError('Existing completed CPU automation gate missing')
        summary['reused_cpu_gate']=str(base/'execution/cpu-canary')
        for spec in plan['lots']:
            tag=spec['tag']
            previous=base/'execution'/tag
            previous_cpu=read(previous/'cpu/controller-result.json')
            if previous_cpu and previous_cpu['status']=='REMOTE_CPU_RECEIPT_VERIFIED':
                validate_receipt(previous/'cpu',spec)
                summary['completed'].append(tag);summary['reused'].append(tag)
                continue
            # No ambiguous upload/create may ever be repeated by this resume.
            if (previous/'publication').exists():
                raise RuntimeError('Previous publication attempt exists for '+tag+'; read-only remote reconciliation required, no upload')
            (execution/tag).mkdir(exist_ok=False)
            stage=Path(spec['staging'])
            receipt=read(stage/'staging-receipt.json')
            if receipt:
                if receipt['lot_identity']!=spec['identity'] or receipt['handle']!=spec['handle']:
                    raise ValueError('Staging receipt differs from pinned lot')
                # Publisher rehashes the entire ZIP immediately before any network.
                summary['reused'].append(tag+'-archive')
            else:
                if stage.exists():
                    raise RuntimeError('Unverified existing staging state for '+tag+'; preserve and inspect instead of rebuilding')
                child(['scripts/stage_expanded_lot.py','--lot',spec['canonical'],'--raw-root',plan['raw_root'],
                       '--out',spec['staging'],'--max-bytes',str(spec['archive_ceiling_bytes'])],tag+'/stage-monitor',3600)
            child(['scripts/publish_kaggle_single_attempt.py','--staged',spec['staging'],'--credentials',args.credentials,
                '--journal',plan['journal'],'--attempt',str(execution/tag/'publication'),
                '--max-http-body-bytes','31000000000','--stop-at-cumulative-body-bytes','31000000000',
                '--license','CC-BY-4.0','--approved-plan',str(plan_path),'--execute'],tag+'/publish-monitor',7200)
            publication=read(execution/tag/'publication/result.json')
            if not publication or publication['status']!='PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING':
                raise RuntimeError('Publication not confirmed for '+tag+'; no retry')
            child(['scripts/kaggle_cpu_job.py','--plan',str(plan_path),'--tag',tag,'--out',str(execution/tag/'cpu'),
                   '--credentials',args.credentials],tag+'/cpu-monitor',2400)
            validate_receipt(execution/tag/'cpu',spec)
            summary['completed'].append(tag)
            summary['cumulative_reserved_body_bytes']=reserved(plan)
            (execution/'progress.json').write_text(json.dumps(summary,indent=2))
        subprocess.run([str(Path('.venv/Scripts/python.exe')),'scripts/prepare_remaining_kaggle.py',
            '--out',str(base),'--refresh-training'],check=True,timeout=120)
        summary['status']='SEVEN_LOTS_PRIVATE_AND_CPU_CERTIFIED_NO_GPU'
    except Exception as error:
        summary.update(status='RESUME_FAILED_STOP_NO_RETRY',error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        summary['cumulative_reserved_body_bytes']=reserved(plan)
        summary['remaining_body_bytes']=31_000_000_000-summary['cumulative_reserved_body_bytes']
        (execution/'result.json').write_text(json.dumps(summary,indent=2))


def validate_continuation(plan_path):
    """Read-only validation shared by review and the existing continuation."""
    from publish_remaining_lots import check_plan
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());check_plan(plan)
    base=plan_path.parent;execution=base/'resume-01/execution'
    previous=read(execution/'result.json')
    publication=read(execution/'lot-002/publication/result.json')
    presence=read(execution/'lot-002/publication/private-presence.json')
    network=[json.loads(line) for line in (execution/'lot-002/publication/network.jsonl').read_text().splitlines()]
    put=[r for r in network if r['operation']=='BlobPUT']
    create=[r for r in network if r['operation']=='CreateDataset']
    if (not previous or previous['status']!='RESUME_FAILED_STOP_NO_RETRY'
        or not publication or publication.get('error_stage')!='confirm_private_remote_presence_from_catalogues'
        or publication.get('create_request_accepted') is not True or publication['blob_sends']!=1
        or publication['confirmed_blob_body_bytes']!=4_995_525_361
        or len(put)!=1 or put[0]['http_status']!=200 or len(create)!=1 or create[0]['http_status']!=200
        or not presence or presence.get('confirmed_after_indexing') is not True or presence['private'] is not True):
        raise RuntimeError('Not an indexing-only stop with a single confirmed upload/create; no continuation')
    lot002=next(s for s in plan['lots'] if s['number']==2)
    controller=read(execution/'lot-002/cpu/controller-result.json')
    if not controller or controller['status']!='REMOTE_CPU_RECEIPT_VERIFIED':raise ValueError('Lot002 CPU certification must finish first')
    validate_receipt(execution/'lot-002/cpu',lot002)
    if any((execution/spec['tag']).exists() for spec in plan['lots'] if spec['number']>=3):
        raise RuntimeError('A later lot attempt exists; never repeat it automatically')
    return plan,execution


def continue_after_confirmed_indexing(plan_path,args):
    """Continue the SAME resume-01; lot002 is never sent or certified again."""
    if not args.execute or args.approve_body_bytes!=31_000_000_000:
        raise PermissionError('Unchanged explicit execution scope required')
    plan_path=Path(plan_path)
    autonomous=getattr(args,'autonomous',False)
    if autonomous:
        from kaggle_autonomous import (START_BYTES,
            checkpoint, release_certified_archive, validate_autonomous)
        plan,execution=validate_autonomous(plan_path)
        state_folder=execution.parent/'autonomous-01'
        if not state_folder.is_dir():raise ValueError('Use the standalone guarded launcher')
    else:
        plan,execution=validate_continuation(plan_path)
    base=plan_path.parent
    start_bytes=START_BYTES if autonomous else 4*1024**3
    current=memory()
    if not current or current['available']<start_bytes:raise MemoryError('Continuation refused: insufficient physical RAM')
    with (execution/'continuation-started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),same_resume='resume-01',
            reason='Accepted upload/create; transient catalogue indexing reconciled; lot002 CPU certified',
            first_new_upload_lot=3,lot002_upload_repetition_forbidden=True,
            current_reserved_body_bytes=reserved(plan),initial_memory=current,
            autonomous=autonomous,start_memory_bytes=start_bytes),stream,indent=2)
    summary=dict(status='RUNNING',completed=['lot-002'],same_resume='resume-01',
        indexing_reconciled=True,lot001_unchanged=True,lot002_reuploaded=False,gpu=False)
    (execution/'continuation-progress.json').write_text(json.dumps(summary,indent=2))
    active_lot=None
    def state(status,phase,**extra):
        if autonomous:
            checkpoint(state_folder,plan,status=status,phase=phase,active_lot=active_lot,
                completed=[int(s.rsplit('-',1)[1]) for s in summary['completed'] if s!='lot-002'],**extra)
    def child(argv,name,seconds,python=None):
        state('RUNNING',name)
        outcome=monitored([python or sys.executable,*argv],execution/name,seconds=seconds,
            start_bytes=start_bytes,reserve_bytes=2*1024**3,job_bytes=512*1024**2,rss_bytes=512*1024**2,verbose=False)
        if outcome['status']!='FINISHED':raise RuntimeError('Stopped at '+name+': '+outcome['status']+'; no retry')
    try:
        for spec in plan['lots']:
            if spec['number']<3:continue
            active_lot=spec['number']
            tag=spec['tag'];(execution/tag).mkdir(exist_ok=False)
            if Path(spec['staging']).exists():raise FileExistsError('Later lot staging already exists; preserve and inspect')
            child(['scripts/stage_expanded_lot.py','--lot',spec['canonical'],'--raw-root',plan['raw_root'],
                '--out',spec['staging'],'--max-bytes',str(spec['archive_ceiling_bytes'])],tag+'/stage-monitor',3600)
            memory_flag=['--min-available-memory-bytes',str(start_bytes)] if autonomous else []
            child(['scripts/publish_kaggle_single_attempt.py','--staged',spec['staging'],'--credentials',args.credentials,
                '--journal',plan['journal'],'--attempt',str(execution/tag/'publication'),
                '--max-http-body-bytes','31000000000','--stop-at-cumulative-body-bytes','31000000000',
                '--license','CC-BY-4.0','--approved-plan',str(plan_path),'--execute',*memory_flag],tag+'/publish-monitor',7200)
            result=read(execution/tag/'publication/result.json')
            if not result or result['status']!='PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING':raise RuntimeError('Publication not confirmed; no retry')
            child(['scripts/kaggle_cpu_job.py','--plan',str(plan_path),'--tag',tag,'--out',str(execution/tag/'cpu'),
                '--credentials',args.credentials,*memory_flag],tag+'/cpu-monitor',2400)
            validate_receipt(execution/tag/'cpu',spec)
            summary['completed'].append(tag);summary['cumulative_reserved_body_bytes']=reserved(plan)
            (execution/'continuation-progress.json').write_text(json.dumps(summary,indent=2))
            if autonomous:
                state('RUNNING','release-certified-archive')
                release_certified_archive(spec,execution/tag)
                state('RUNNING','between-lots')
        active_lot=None
        if autonomous:
            child(['scripts/prepare_remaining_kaggle.py','--out',str(base),'--refresh-training'],
                'training-template-monitor',120,python=str(Path('.venv/Scripts/python.exe')))
        else:
            subprocess.run([str(Path('.venv/Scripts/python.exe')),'scripts/prepare_remaining_kaggle.py',
                '--out',str(base),'--refresh-training'],check=True,timeout=120)
        summary['status']='SEVEN_LOTS_PRIVATE_AND_CPU_CERTIFIED_NO_GPU'
        state(summary['status'],'finished')
    except Exception as error:
        summary.update(status='RESUME_FAILED_STOP_NO_RETRY',error_type=type(error).__name__,reason=str(error))
        state(summary['status'],'stopped',error_type=type(error).__name__)
        raise
    finally:
        summary['cumulative_reserved_body_bytes']=reserved(plan)
        summary['remaining_body_bytes']=31_000_000_000-summary['cumulative_reserved_body_bytes']
        (execution/'continuation-result.json').write_text(json.dumps(summary,indent=2))
