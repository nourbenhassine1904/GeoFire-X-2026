"""Read-only, resumable ZIP integrity verification. No extraction or network."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
import zipfile
import zlib

from geofirex.acquisition_session import session_lock
from geofirex.corpus import json_new


class ArchiveVerificationError(ValueError):
    def __init__(self,result):
        self.result=result
        detail=result.get('failure',{}).get('message',result['status'])
        super().__init__(f"Archive verification {result['status']}: {detail}; {result.get('summary_path')}")


def file_identity(path):
    stat=Path(path).stat()
    return dict(path=str(Path(path).resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,device=stat.st_dev,inode=stat.st_ino)


def file_digests(path,*,stop_requested=lambda:False,progress=None):
    md5,sha=hashlib.md5(),hashlib.sha256()
    read=0
    with Path(path).open('rb',buffering=0) as stream:
        while True:
            if stop_requested(): raise InterruptedError('Verification stopped during global hashing')
            block=stream.read(4*1024**2)
            if not block: break
            md5.update(block);sha.update(block);read+=len(block)
            if progress and read//(1024**3)!=(read-len(block))//(1024**3): progress(read)
    return dict(md5=md5.hexdigest(),sha256=sha.hexdigest())


def validate_structure(zf,index,archive_size):
    members=zf.infolist()
    if len({m.filename for m in members})!=len(members): raise ValueError('Duplicate archive member')
    tiffs={m.filename:m for m in members if m.filename.endswith('.tif')}
    if set(tiffs)!={e['path'] for e in index['entries']}: raise ValueError('Archive/index members differ')
    for member in members:
        if not 0<=member.header_offset<archive_size: raise ValueError(f'Invalid ZIP header offset: {member.filename}')
    for entry in index['entries']:
        member=tiffs[entry['path']]
        if (member.file_size,member.compress_size,member.CRC)!=(entry['geotiff_bytes'],entry['compressed_bytes'],int(entry['crc32'],16)):
            raise ValueError('Archive/index size or CRC metadata differs')
        if 'header_offset' in entry and member.header_offset!=entry['header_offset']:
            raise ValueError('Archive/index header offset differs')
    return members,tiffs


def member_signature(ordinal,member):
    return dict(ordinal=ordinal,member=member.filename,header_offset=member.header_offset,
                expected_bytes=member.file_size,compressed_bytes=member.compress_size,
                expected_crc32=f'{member.CRC:08x}')


def _read_history(directory,members):
    verified=set();errors=[];tails=[]
    for path in sorted(directory.glob('attempt-*.jsonl')):
        with path.open('rb') as stream:
            for position,line in enumerate(stream):
                if not line.endswith(b'\n'):
                    tails.append(dict(log=path.name,error='incomplete final journal record; not accepted as proof'))
                    break
                try: event=json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    raise ValueError(f'Malformed verification journal: {path.name}:{position+1}')
                if event.get('kind')=='member_ok':
                    ordinal=event['ordinal']
                    if not isinstance(ordinal,int) or not 0<=ordinal<len(members): raise ValueError('Invalid journal ordinal')
                    if any(event.get(k)!=v for k,v in member_signature(ordinal,members[ordinal]).items()):
                        raise ValueError('Journal member differs from archive metadata')
                    if event.get('actual_crc32')!=event['expected_crc32'] or event.get('actual_bytes')!=event['expected_bytes']:
                        raise ValueError('Journal contains unverified CRC result')
                    verified.add(ordinal)
                elif event.get('kind')=='member_error': errors.append(event)
    return verified,errors,tails


def verify_zip(path,index,*,all_crc=True,verification_dir=None,block_size=1024**2,
               max_members=None,io_mode='unbuffered',stop_requested=lambda:False,progress_every=100):
    if not 1<=block_size<=16*1024**2: raise ValueError('CRC block must be between 1 byte and 16 MiB')
    if max_members is not None and max_members<1: raise ValueError('Positive member limit required')
    if io_mode not in ('buffered','unbuffered'): raise ValueError('Invalid ZIP I/O mode')
    if progress_every<1: raise ValueError('Positive progress interval required')
    path=Path(path).resolve()
    directory=Path(verification_dir or (str(path)+'.verification-v2')).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    identity=file_identity(path)
    specification=dict(schema_version=2,archive=identity,
        index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]
    summary_path=directory/f'summary-{token}.json'
    result=dict(schema_version=2,status='NOT_VERIFIED',bytes=identity['size'],
                md5_status='NOT_VERIFIED',structure_status='NOT_VERIFIED',all_crc_checked=False,
                events=len({e['event_id'] for e in index['entries']}),observations=len(index['entries']),
                crc_verified_members=0,crc_unverified_members=None,errors_encountered=[],
                verification_dir=str(directory),summary_path=str(summary_path),io_mode=io_mode,
                last_member_started=None,last_member_verified=None,python=sys.version)
    verified=set();members=[];errors=[];phase='identity';cause=None
    with session_lock(directory):
        identity_path=directory/'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8'))!=specification:
                raise ValueError('Archive or index changed; prior CRC proofs not reused; select a new verification directory')
        else: json_new(identity_path,specification)
        with (directory/f'attempt-{token}.jsonl').open('x',encoding='utf-8') as log:
            def record(kind,**fields):
                event=dict(kind=kind,utc=datetime.now(timezone.utc).isoformat(),io_mode=io_mode,**fields)
                log.write(json.dumps(event)+'\n');log.flush();os.fsync(log.fileno())
                return event
            try:
                record('attempt_start',archive=str(path),python=sys.version)
                if identity['size']!=index['archive_bytes']: raise ValueError('Archive size mismatch')
                phase='structure'
                with path.open('rb',buffering=0 if io_mode=='unbuffered' else -1) as raw,zipfile.ZipFile(raw) as zf:
                    members,tiffs=validate_structure(zf,index,identity['size'])
                    result.update(structure_status='VERIFIED',zip_entries=len(members),tiff_members=len(tiffs),
                                  directory_members=sum(m.is_dir() for m in members))
                    verified,errors,tails=_read_history(directory,members)
                    result['incomplete_journal_tails']=tails
                    if verified:
                        result['last_member_verified']=member_signature(max(verified),members[max(verified)])
                    phase='global_md5'
                    digest_path=directory/'global-digest.json'
                    if digest_path.exists():
                        digest=json.loads(digest_path.read_text(encoding='utf-8'))
                        if digest['archive_identity']!=identity: raise ValueError('Global digest identity changed')
                        hashes={k:digest[k] for k in ('md5','sha256')}
                    else:
                        hashes=file_digests(path,stop_requested=stop_requested,
                            progress=lambda n:print(f'MD5/SHA256: {n}/{identity["size"]} bytes',flush=True))
                        if file_identity(path)!=identity: raise ValueError('Archive changed during global hashing')
                        json_new(digest_path,dict(archive_identity=identity,**hashes))
                    result.update(hashes)
                    if hashes['md5']!=index['archive_md5']: raise ValueError('Official archive MD5 mismatch; file preserved')
                    result['md5_status']='VERIFIED'
                    record('global_md5_ok',**hashes)
                    if all_crc:
                        processed=0
                        for ordinal,member in enumerate(members):
                            if ordinal in verified: continue
                            if stop_requested(): raise InterruptedError('Verification stopped at member boundary')
                            if max_members is not None and processed>=max_members: break
                            signature=member_signature(ordinal,member)
                            result['last_member_started']=signature
                            record('member_start',**signature)
                            phase='open_member';size=crc=0
                            try:
                                with zf.open(member,'r') as stream:
                                    phase='read_member'
                                    while True:
                                        if stop_requested(): raise InterruptedError('Verification stopped within member; member will restart')
                                        data=stream.read(block_size)
                                        if not data: break
                                        size+=len(data);crc=zlib.crc32(data,crc)
                                phase='compare_crc'
                                if size!=member.file_size or crc&0xffffffff!=member.CRC:
                                    raise zipfile.BadZipFile('Explicit uncompressed size/CRC mismatch')
                            except InterruptedError: raise
                            except Exception as error:
                                detail=record('member_error',**signature,phase=phase,
                                    classification='IO_ERROR' if isinstance(error,OSError) else 'ZIP_OR_CRC_ERROR',
                                    error_type=type(error).__name__,message=str(error),errno=getattr(error,'errno',None),
                                    winerror=getattr(error,'winerror',None),actual_bytes=size,
                                    traceback=traceback.format_exc())
                                errors.append(detail)
                                raise
                            record('member_ok',**signature,actual_bytes=size,actual_crc32=f'{crc&0xffffffff:08x}')
                            verified.add(ordinal);processed+=1
                            result['last_member_verified']=signature
                            if processed%progress_every==0:
                                print(f'CRC verified {len(verified)}/{len(members)}; last={member.filename}',flush=True)
                    if file_identity(path)!=identity: raise ValueError('Archive identity changed during CRC verification')
                    result['status']='VERIFIED' if all_crc and len(verified)==len(members) else 'PARTIAL' if all_crc else 'MD5_AND_STRUCTURE_VERIFIED'
            except (KeyboardInterrupt,InterruptedError) as error:
                cause=error;result['status']='INTERRUPTED'
                record('interrupted',phase=phase,error_type=type(error).__name__,message=str(error))
            except Exception as error:
                cause=error;result['status']='FAILED'
                record('attempt_error',phase=phase,error_type=type(error).__name__,message=str(error),
                       errno=getattr(error,'errno',None),traceback=traceback.format_exc())
                result['failure']=dict(phase=phase,error_type=type(error).__name__,message=str(error),errno=getattr(error,'errno',None))
            finally:
                result.update(crc_verified_members=len(verified),crc_unverified_members=len(members)-len(verified) if members else None,
                    crc_verified_tiffs=sum(members[i].filename.endswith('.tif') for i in verified),
                    crc_unverified_tiffs=sum(m.filename.endswith('.tif') for i,m in enumerate(members) if i not in verified),
                    errors_encountered=errors,unresolved_member_errors=[e for e in errors if e['ordinal'] not in verified],
                    all_crc_checked=result['status']=='VERIFIED',processed_this_attempt=locals().get('processed',0))
                record('attempt_end',status=result['status'],crc_verified_members=len(verified))
                json_new(summary_path,result)
    if cause is not None or (all_crc and result['status']!='VERIFIED'):
        raise ArchiveVerificationError(result) from cause
    return result
