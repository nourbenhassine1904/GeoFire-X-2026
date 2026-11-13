"""Reviewable stage acquisition; offline dry-run by default, never publishes.

--execute must only be used after explicit approval of this session's budget.
Existing rasters across roots are CRC checked and read in place, never copied.
"""
import argparse
import json
import uuid
from pathlib import Path

from geofirex.cloud_corpus import validate_index
from geofirex.corpus import json_new,sha256_file
from geofirex.scientific_scaleup import stage_download_entries
from geofirex.acquisition_session import execute_stage,event_batches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch-config',help='Pinned local launch settings; never authorizes --execute')
    p.add_argument('--plan')
    p.add_argument('--index',default='data/manifests/real/corpus_zip_index_v1.json')
    p.add_argument('--destination')
    p.add_argument('--reuse-root',action='append')
    p.add_argument('--out')
    p.add_argument('--session-dir')
    p.add_argument('--batch-bytes',type=int)
    p.add_argument('--max-mbps',type=float)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.launch_config:
        settings=json.loads(Path(args.launch_config).read_text(encoding='utf-8'))
        allowed={'plan','destination','reuse_root','session_dir','batch_bytes','max_mbps'}
        if set(settings)-allowed: p.error('Unknown launch setting; execute cannot be stored in configuration')
        for key,value in settings.items():
            if getattr(args,key) is None: setattr(args,key,value)
    if not args.plan or not args.destination: p.error('Plan and destination required')
    args.reuse_root=args.reuse_root or []
    args.batch_bytes=256*1024**2 if args.batch_bytes is None else args.batch_bytes
    args.max_mbps=20 if args.max_mbps is None else args.max_mbps
    if args.batch_bytes<1 or not 1<=args.max_mbps<=1000: p.error('Positive batch limit and rate 1..1000 Mbps required')
    if not args.session_dir: p.error('Persistent --session-dir required; no reset of budgets on resume')
    if not args.out: args.out=str(Path(args.session_dir)/f'attempt-{uuid.uuid4().hex}.json')
    if Path(args.out).exists(): p.error('Report exists; choose a new path')
    index=json.loads(Path(args.index).read_text(encoding='utf-8'))
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    validate_index(index,official=True)
    destination=Path(args.destination).resolve()
    missing,reused=stage_download_entries(plan,index,[destination,*map(Path,args.reuse_root)])
    payload=sum(e['compressed_bytes'] for e in missing)
    storage=sum(e['geotiff_bytes'] for e in missing)
    if payload > plan['additional_payload_bytes']:
        p.error('Local reuse decreased since planning: regenerate a new budget before acquisition')
    report=dict(status='DRY_RUN_NOT_APPROVED',plan_identity=plan['identity'],
                execute=args.execute,destination=str(destination),missing_files=len(missing),
                compressed_payload_bytes=payload,additional_extracted_bytes=storage,
                transfer_cap_bytes=plan['proposed_transfer_cap_bytes'],reused=reused,
                upload_bytes=0,publication='none',session_dir=str(Path(args.session_dir).resolve()),
                max_mbps=args.max_mbps,
                batches=[dict(events=sorted({e['event_id'] for e in b}),files=len(b),
                              compressed_payload_bytes=sum(e['compressed_bytes'] for e in b))
                         for b in event_batches(missing,args.batch_bytes)],
                reused_sha256={r['path']:sha256_file(r['locations'][0]) for r in reused})
    if args.execute:
        try:
            execute_stage(plan,index,destination,args.reuse_root,args.session_dir,
                          batch_bytes=args.batch_bytes,max_mbps=args.max_mbps,report=report)
        except BaseException as error:
            report.update(status='FAILED_OR_INTERRUPTED_FILES_AND_BUDGET_PRESERVED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            json_new(args.out,report)
    else:
        json_new(args.out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('reused','reused_sha256','batches')},indent=2))


if __name__=='__main__': main()
