"""Standalone PowerShell operator entry point. No download without --execute.

Acquiring 2021 is storage/QC only; this command never trains or selects models.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import uuid

from geofirex.archive_transfer import download_archive,verify_archive,extract_archive
from geofirex.acquisition_session import execute_stage
from geofirex.cloud_corpus import validate_index,seal,fingerprint,check_seal
from geofirex.corpus import json_new,verify_member
from geofirex.archive_verification import ArchiveVerificationError


def full_entries(plan,index,roots):
    check_seal(plan)
    if plan['index_identity']!=fingerprint(index) or plan['entries']!=index['entries']:
        raise ValueError('Full storage plan differs from pinned index')
    missing,reused=[],[]
    for entry in index['entries']:
        found=[]
        for root in roots:
            path=Path(root)/entry['path']
            if path.exists():
                if not verify_member(path,entry): raise ValueError(f'Existing raster invalid, preserved: {path}')
                found.append(str(path.resolve()))
        if found: reused.append(dict(path=entry['path'],locations=found))
        else: missing.append(entry)
    return missing,reused


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['zip','verify','extract','range'])
    parser.add_argument('--index',default='data/manifests/real/corpus_zip_index_v1.json')
    parser.add_argument('--archive',default='D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip')
    parser.add_argument('--destination',default='D:/GeoFire-X/WildfireSpreadTS/rasters')
    parser.add_argument('--session-dir')
    parser.add_argument('--budget-bytes',type=int,default=55000000000)
    parser.add_argument('--max-mbps',type=float,default=40)
    parser.add_argument('--reuse-root',action='append',default=[])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--verification-dir',help='Persistent CRC journals; same directory on resume and extraction')
    parser.add_argument('--verify-limit',type=int,help='Diagnostic bound: new members per invocation; partial is not success')
    parser.add_argument('--io-mode',choices=['buffered','unbuffered'],default='unbuffered')
    args=parser.parse_args()
    if args.budget_bytes<1 or not 1<=args.max_mbps<=1000: parser.error('Positive budget and rate 1..1000 Mbps required')
    index=json.loads(Path(args.index).read_text(encoding='utf-8'))
    validate_index(index,official=True)
    if args.mode=='verify':
        stopped=threading.Event()
        signal.signal(signal.SIGINT,lambda signum,frame:stopped.set())
        if hasattr(signal,'SIGBREAK'): signal.signal(signal.SIGBREAK,lambda signum,frame:stopped.set())
        try:
            result=verify_archive(args.archive,index,verification_dir=args.verification_dir,
                max_members=args.verify_limit,io_mode=args.io_mode,stop_requested=stopped.is_set)
        except ArchiveVerificationError as error:
            print(json.dumps(error.result,indent=2))
            return 130 if error.result['status']=='INTERRUPTED' else 2
        print(json.dumps(result,indent=2));return 0
    if not args.execute:
        print(json.dumps(dict(status='PREPARED_NO_NETWORK',mode=args.mode,archive=args.archive,
                              destination=args.destination,budget_bytes=args.budget_bytes,
                              archive_bytes=index['archive_bytes'],events=607,observations=13607),indent=2));return
    if args.mode=='extract':
        try:
            result=extract_archive(args.archive,index,args.destination,verification_dir=args.verification_dir)
        except ArchiveVerificationError as error:
            print(json.dumps(error.result,indent=2));return 2
        print(json.dumps(result,indent=2));return 0
    session=Path(args.session_dir or f'D:/GeoFire-X/WildfireSpreadTS/logs/{args.mode}-v1')
    stopped=threading.Event()
    def stop(signum,frame):
        stopped.set()
        print('Stop requested: finishing current chunk/file; progress and partial files retained.',flush=True)
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,'SIGBREAK'): signal.signal(signal.SIGBREAK,stop)
    if args.mode=='zip':
        result=download_archive(index,args.archive,session,args.budget_bytes,
                                stop_requested=stopped.is_set,max_mbps=args.max_mbps)
    else:
        total=sum(e['geotiff_bytes'] for e in index['entries'])
        plan=seal(dict(kind='full_corpus_storage_only',index_identity=fingerprint(index),
            entries=index['entries'],events=sorted({e['event_id'] for e in index['entries']}),
            target_events=607,additional_payload_bytes=sum(e['compressed_bytes'] for e in index['entries']),
            proposed_transfer_cap_bytes=args.budget_bytes,proposed_disk_bytes=2*total+20*10**9,
            test2021='acquisition and byte integrity only; no model use'))
        result={}
        try:
            execute_stage(plan,index,args.destination,args.reuse_root,session,batch_bytes=256*1024**2,
                max_mbps=args.max_mbps,report=result,entry_selector=full_entries,stop_requested=stopped.is_set)
        finally:
            json_new(session/f'attempt-{uuid.uuid4().hex}.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': sys.exit(main())
