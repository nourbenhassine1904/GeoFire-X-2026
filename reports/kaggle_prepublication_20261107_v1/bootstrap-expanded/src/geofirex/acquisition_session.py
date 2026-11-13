"""Durable, bounded stage acquisition; no network calls at import or dry-run."""
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

from geofirex.corpus import MemberCachedRangeFile, acquire_members, json_new, sha256_file


@contextmanager
def session_lock(directory):
    """OS lock released on process death; the harmless lock file is preserved."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'session.lock').open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Acquisition session already running') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle,fcntl.LOCK_UN)


class PersistentRangeFile(MemberCachedRangeFile):
    """Resume the SAME hard reservation cap; malformed ledgers fail closed.

    Bounds application HTTP response bodies, including retries and ZIP metadata.
    TCP/TLS overhead cannot be measured or capped by urllib.
    """
    def __init__(self,budget_bytes,*,ledger,plan_identity,max_mbps=20,**kwargs):
        super().__init__(budget_bytes,**kwargs)
        if not 1<=max_mbps<=1000: raise ValueError('max_mbps must be in [1,1000]')
        self.rate=max_mbps*1_000_000/8
        self.next_request=0.
        path=Path(ledger)
        if path.exists():
            lines=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if not lines or lines[0] != dict(plan_identity=plan_identity,budget_bytes=budget_bytes):
                raise ValueError('Ledger budget/plan mismatch')
            previous=0
            for row in lines[1:]:
                value=row['reserved_bytes']
                if not previous<=value<=budget_bytes: raise ValueError('Invalid cumulative ledger')
                previous=value
                self.bytes_reserved=value
                self.requests=row['requests']
                self.bytes_fetched=row.get('confirmed_body_bytes',0)
            self.ledger_path=path
        else:
            self.enable_ledger(path,plan_identity)

    def checkpoint(self,**extra):
        with self.ledger_path.open('a',encoding='utf-8') as output:
            output.write(json.dumps(dict(reserved_bytes=self.bytes_reserved,requests=self.requests,
                                         confirmed_body_bytes=self.bytes_fetched,**extra))+'\n')
            output.flush();os.fsync(output.fileno())

    def _record_reservation(self,start,length):
        self.checkpoint(kind='reservation',start=start,length=length)
        delay=max(0,self.next_request-time.monotonic())
        if delay: time.sleep(delay)
        self.next_request=time.monotonic()+(length+1)/self.rate

    def read(self,n=-1):
        try:
            return super().read(n)
        finally:
            self.checkpoint(kind='checkpoint')


def event_batches(entries,payload_limit):
    """Partition the selected missing files, retaining entire events per lot."""
    if payload_limit<1: raise ValueError('Positive batch limit required')
    groups=defaultdict(list)
    for entry in entries: groups[entry['event_id']].append(entry)
    batches,current,size=[],[],0
    for event,items in sorted(groups.items()):
        cost=sum(e['compressed_bytes'] for e in items)
        if current and size+cost>payload_limit:
            batches.append(current);current=[];size=0
        current.extend(sorted(items,key=lambda e:e['path']));size+=cost
    if current: batches.append(current)
    return batches


def execute_stage(plan,index,destination,roots,session_dir,*,batch_bytes,max_mbps,report,
                  stream_factory=PersistentRangeFile,entry_selector=None,stop_requested=None):
    """No reset of the budget on resume; recheck reuse under an exclusive lock."""
    from geofirex.scientific_scaleup import stage_download_entries
    select_entries=entry_selector or stage_download_entries
    destination,session_dir=Path(destination).resolve(),Path(session_dir).resolve()
    with session_lock(session_dir):
        if (session_dir/'STOP').exists():
            raise RuntimeError('STOP marker present; no network started; explicit operator resume required')
        specification=dict(plan_identity=plan['identity'],index_source=index['source'],
                           destination=str(destination),roots=sorted(str(Path(r).resolve()) for r in roots),
                           budget_bytes=plan['proposed_transfer_cap_bytes'],batch_bytes=batch_bytes,max_mbps=max_mbps)
        pinned=session_dir/'session.json'
        if pinned.exists():
            if json.loads(pinned.read_text(encoding='utf-8'))!=specification:
                raise ValueError('Session configuration changed; existing budget is preserved')
        else: json_new(pinned,specification)
        missing,reused=select_entries(plan,index,[destination,*roots])
        known_hashes={}
        for certificate_path in session_dir.glob('batch-*.json'):
            certificate=json.loads(certificate_path.read_text(encoding='utf-8'))
            if certificate['plan_identity']!=plan['identity']:
                raise ValueError('Batch certificate belongs to a different plan')
            for item in certificate['files']:
                prior=known_hashes.setdefault(item['path'],item['sha256'])
                if prior!=item['sha256']: raise ValueError('Conflicting batch SHA256 certificates')
                path=destination/item['path']
                if path.is_file() and sha256_file(path)!=item['sha256']:
                    raise ValueError('Existing batch SHA256 changed; data preserved')
        if sum(e['compressed_bytes'] for e in missing)>plan['additional_payload_bytes']:
            raise RuntimeError('Previously credited local files disappeared; approval budget invalid')
        destination.mkdir(parents=True,exist_ok=True)
        reserve=20*10**9
        allocation=plan['proposed_disk_bytes']-reserve
        occupied=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
        storage=sum(e['geotiff_bytes'] for e in missing)
        if occupied+storage>allocation or shutil.disk_usage(destination).free<storage+reserve:
            raise RuntimeError('Insufficient disk allocation or 20 GB free reserve; no transfer started')
        stream=stream_factory(plan['proposed_transfer_cap_bytes'],ledger=session_dir/'transfer.jsonl',
                              plan_identity=plan['identity'],max_mbps=max_mbps,
                              url=index['source'],size=index['archive_bytes'])
        report.update(status='RUNNING',reused_files=len(reused),batches=[])
        try:
            if missing:
                if sum(e['compressed_bytes'] for e in missing)>stream.budget_bytes-stream.bytes_reserved:
                    raise RuntimeError('Remaining cumulative budget cannot cover missing payload; new approval required')
                with zipfile.ZipFile(stream) as zf:
                    for batch in event_batches(missing,batch_bytes):
                        batch_id=uuid.uuid4().hex
                        files=[]
                        for entry in batch:
                            if (session_dir/'STOP').exists() or (stop_requested and stop_requested()):
                                if files:
                                    json_new(session_dir/f'batch-{batch_id}.json',dict(
                                        plan_identity=plan['identity'],files=files,partial_batch=True,
                                        reserved_bytes_cumulative=stream.bytes_reserved))
                                report.update(status='STOPPED_BY_USER_AT_FILE_BOUNDARY',
                                              current_batch_verified_files=files)
                                return
                            if (occupied+entry['geotiff_bytes']>allocation or
                                shutil.disk_usage(destination).free<entry['geotiff_bytes']+reserve):
                                raise RuntimeError('Disk reserve reached; prior batches and .part preserved')
                            acquire_members(zf,[entry],destination)
                            path=destination/entry['path']
                            occupied+=path.stat().st_size
                            files.append(dict(path=entry['path'],bytes=path.stat().st_size,
                                              crc32=entry['crc32'],sha256=sha256_file(path)))
                            known_hashes[entry['path']]=files[-1]['sha256']
                        certificate=dict(plan_identity=plan['identity'],files=files,
                                         compressed_payload_bytes=sum(e['compressed_bytes'] for e in batch),
                                         reserved_bytes_cumulative=stream.bytes_reserved)
                        json_new(session_dir/f'batch-{batch_id}.json',certificate)
                        report['batches'].append(dict(certificate=f'batch-{batch_id}.json',files=len(files)))
                        print(f"Batch verified: {len(files)} files; cumulative reserved {stream.bytes_reserved}/{stream.budget_bytes}",flush=True)
            pending,locations=select_entries(plan,index,[destination,*roots])
            if pending: raise RuntimeError('Stage still has missing observations')
            files=[dict(path=r['path'],locations=r['locations'],sha256=sha256_file(r['locations'][0]))
                   for r in locations]
            if any(r['path'] in known_hashes and r['sha256']!=known_hashes[r['path']] for r in files):
                raise ValueError('Final SHA256 verification differs from batch certificates')
            json_new(session_dir/f'complete-{uuid.uuid4().hex}.json',dict(plan_identity=plan['identity'],
                events=plan['events'],observations=len(files),files=files,
                qualification='CRC/SHA verified; pixel QC and DataLoader certification still required'))
            report.update(status='ACQUIRED_CRC_SHA_VERIFIED_REQUIRES_PIXEL_QC',
                          complete_events=len(plan['events']),verified_observations=len(files),
                          downloaded_files=sum(b['files'] for b in report['batches']),
                          files_in_persistent_destination=sum((destination/r['path']).is_file() for r in files),
                          sha256_batch_comparisons=sum(r['path'] in known_hashes for r in files))
        except BaseException as error:
            report.update(status='INTERRUPTED_FILES_AND_BUDGET_PRESERVED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            stream.checkpoint(kind='session_end')
            report['transfer']=dict(confirmed_body_bytes=stream.bytes_fetched,
                                   reserved_bytes_including_retries=stream.bytes_reserved,requests=stream.requests,
                                   cap_bytes=stream.budget_bytes,
                                   note='Reservations conservatively charge interrupted responses; TCP/TLS overhead outside scope')
