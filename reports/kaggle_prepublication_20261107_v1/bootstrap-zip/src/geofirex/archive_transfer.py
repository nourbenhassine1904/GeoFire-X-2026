"""Operator-run full ZIP transfer: resumable prefix, durable body-byte cap."""
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import uuid
import zipfile

from geofirex.acquisition_session import session_lock
from geofirex.corpus import acquire_members,json_new,sha256_file,publication_record
from geofirex.archive_verification import file_digests,verify_zip


class BodyLedger:
    def __init__(self,path,specification):
        self.path=Path(path)
        self.cap=specification['budget_bytes']
        self.reserved=self.confirmed=0
        if self.path.exists():
            rows=[json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
            if not rows or rows[0]!=specification: raise ValueError('Ledger specification changed')
            for row in rows[1:]:
                if not self.reserved<=row['reserved']<=self.cap: raise ValueError('Invalid ledger')
                if not self.confirmed<=row['confirmed']<=row['reserved']: raise ValueError('Invalid body count')
                self.reserved,self.confirmed=row['reserved'],row['confirmed']
        else:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('x',encoding='utf-8') as f:
                f.write(json.dumps(specification)+'\n');f.flush();os.fsync(f.fileno())

    def record(self,kind,**extra):
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(kind=kind,reserved=self.reserved,confirmed=self.confirmed,**extra))+'\n')
            f.flush();os.fsync(f.fileno())

    def reserve(self,n):
        if n<1 or self.reserved+n>self.cap: raise RuntimeError('Cumulative HTTP body budget exhausted')
        self.reserved+=n;self.record('reservation',bytes=n)


def verify_archive(path,index,*,all_crc=True,verification_dir=None,**options):
    return verify_zip(path,index,all_crc=all_crc,verification_dir=verification_dir,**options)


def download_archive(index,output,session,budget,*,stop_requested=lambda:False,attempts=3,
                     max_mbps=40,block_size=4*1024**2):
    output,session=Path(output).resolve(),Path(session).resolve()
    partial=Path(str(output)+'.part')
    with session_lock(session):
        if (session/'STOP').exists(): raise RuntimeError('STOP marker present; no network started')
        if output.exists():
            return {**verify_archive(output,index,all_crc=False), 'status':'ALREADY_PRESENT_VERIFIED'}
        ledger_path=session/'body-transfer.jsonl'
        if partial.exists() and not ledger_path.exists(): raise ValueError('Orphan partial file: no accounted session; preserved')
        output.parent.mkdir(parents=True,exist_ok=True)
        ledger=BodyLedger(ledger_path,dict(url=index['source'],archive_bytes=index['archive_bytes'],
            archive_md5=index['archive_md5'],output=str(output),budget_bytes=budget))
        result=dict(status='STARTED',output=str(output),partial=str(partial))
        try:
            for attempt in range(attempts):
                offset=partial.stat().st_size if partial.exists() else 0
                remaining=index['archive_bytes']-offset
                if remaining<0: raise ValueError('Oversized partial file; preserved')
                if remaining==0: break
                if stop_requested() or (session/'STOP').exists():
                    result['status']='STOPPED_BY_USER';return result
                if shutil.disk_usage(output.parent).free<remaining+20*10**9:
                    raise RuntimeError('Insufficient disk plus 20 GB reserve')
                if ledger.cap-ledger.reserved<remaining:
                    raise RuntimeError('Remaining budget smaller than remaining ZIP; no request started')
                request=urllib.request.Request(index['source'],headers={
                    'Range':f"bytes={offset}-{index['archive_bytes']-1}",'Accept-Encoding':'identity'})
                try:
                    with urllib.request.urlopen(request,timeout=60) as response:
                        expected=f"bytes {offset}-{index['archive_bytes']-1}/{index['archive_bytes']}"
                        if response.status==206:
                            if response.headers.get('Content-Range','').strip()!=expected: raise ValueError('Content-Range mismatch')
                        elif response.status!=200 or offset!=0:
                            raise ValueError('Resume requires HTTP 206; response body refused')
                        length=response.headers.get('Content-Length')
                        if length is not None and int(length)!=remaining: raise ValueError('Unexpected response size')
                        with partial.open('ab') as target:
                            while offset<index['archive_bytes']:
                                if stop_requested() or (session/'STOP').exists():
                                    result['status']='STOPPED_BY_USER';return result
                                if shutil.disk_usage(output.parent).free<block_size+20*10**9:
                                    raise RuntimeError('20 GB disk reserve reached')
                                n=min(block_size,index['archive_bytes']-offset)
                                ledger.reserve(n+1)
                                started=time.monotonic()
                                data=response.read(n)
                                if not data: raise OSError('Premature response end; prefix retained')
                                ledger.confirmed+=len(data)
                                target.write(data);target.flush();os.fsync(target.fileno())
                                offset+=len(data);ledger.record('written',offset=offset)
                                delay=len(data)*8/(max_mbps*10**6)-(time.monotonic()-started)
                                if delay>0: time.sleep(delay)
                                if offset//(64*1024**2)!=(offset-len(data))//(64*1024**2):
                                    print(f"ZIP {offset}/{index['archive_bytes']} bytes; reserved {ledger.reserved}/{ledger.cap}",flush=True)
                except OSError as error:
                    ledger.record('connection_error',error=f'{type(error).__name__}: {error}')
                    if stop_requested(): result['status']='STOPPED_BY_USER';return result
                    if attempt+1==attempts: raise
                    time.sleep(2)
            if partial.stat().st_size!=index['archive_bytes']: raise RuntimeError('ZIP incomplete; resume with same command')
            hashes=file_digests(partial)
            if hashes['md5']!=index['archive_md5']: raise ValueError('MD5 mismatch; partial preserved without overwrite')
            if output.exists(): raise FileExistsError(output)
            partial.rename(output)
            result.update(status='DOWNLOADED_OFFICIAL_MD5_VERIFIED',bytes=output.stat().st_size,**hashes)
            return result
        except BaseException as error:
            result.update(status='INTERRUPTED_PREFIX_PRESERVED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            ledger.record('session_end')
            result.update(reserved_body_bytes=ledger.reserved,confirmed_body_bytes=ledger.confirmed,
                          budget_bytes=ledger.cap,scope='HTTP body reads; protocol overhead excluded')
            json_new(session/f'attempt-{uuid.uuid4().hex}.json',result)


def extract_archive(path,index,destination,*,verification_dir=None):
    # Full proof is required before even creating the raster destination.
    verification=verify_archive(path,index,all_crc=True,verification_dir=verification_dir)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    needed=sum(e['geotiff_bytes'] for e in index['entries'] if not (destination/e['path']).exists())
    if shutil.disk_usage(destination).free<needed+20*10**9: raise RuntimeError('Extraction space plus reserve unavailable')
    files=[];reused_parts=0
    with session_lock(destination/'.extraction-session'),Path(path).open('rb',buffering=0) as raw,zipfile.ZipFile(raw) as zf:
        journal=destination/'.extraction-session'/f'publication-{uuid.uuid4().hex}.jsonl'
        with journal.open('x',encoding='utf-8'):
            pass
        publication_record(journal,'extraction_start',archive=str(Path(path).resolve()),
                           destination=str(destination),preserve_parts=True,method='hardlink')
        try:
            for ordinal,entry in enumerate(index['entries'],1):
                if shutil.disk_usage(destination).free<entry['geotiff_bytes']+20*10**9:
                    raise RuntimeError('20 GB extraction reserve unavailable; all files preserved')
                publication_record(journal,'member_start',ordinal=ordinal,member=entry['path'])
                result=acquire_members(zf,[entry],destination,publication_journal=journal,preserve_parts=True)
                reused_parts+=len(result['reused_parts'])
                target=destination/entry['path']
                proof=dict(path=entry['path'],crc32=entry['crc32'],sha256=sha256_file(target),bytes=target.stat().st_size)
                files.append(proof)
                publication_record(journal,'member_verified',ordinal=ordinal,**proof)
                if ordinal%100==0 or result['reused_parts']:
                    print(f"Extraction {ordinal}/{len(index['entries'])}; reused parts {reused_parts}; journal {journal}",flush=True)
            certificate=dict(archive=verification,files=files,events=verification['events'],
                             observations=len(files),publication_journal=str(journal),reused_parts=reused_parts,
                             test2021='storage only; excluded from model selection/training')
            json_new(destination/'.extraction-session'/f'complete-{uuid.uuid4().hex}.json',certificate)
            publication_record(journal,'extraction_complete',observations=len(files),reused_parts=reused_parts)
        except BaseException as error:
            publication_record(journal,'extraction_stopped',verified_this_attempt=len(files),
                               reused_parts=reused_parts,error_type=type(error).__name__,message=str(error),
                               winerror=getattr(error,'winerror',None),errno=getattr(error,'errno',None))
            raise
    return dict(events=verification['events'],observations=len(files),network_bytes=0,
                publication_journal=str(journal),reused_parts=reused_parts)
