"""Single-attempt compact-only campaign. No raw publisher or automatic retry.

Durable conservative HTTP reservations include authentication and control RPCs.
Every blob is streamed exactly once; a failed or ambiguous request stops campaign.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'reports/gpu_acceleration_20261111_v1'
sys.path[:0] = [str(ROOT/'src'), str(BASE/'kaggle_sdk'), str(ROOT/'scripts')]
from geofirex.upload_budget import BodyBudget
from acceleration_kaggle_readonly import client
from kaggle_dataset_preflight import collect
from kaggle_private_presence import confirm_private_presence
from kaggle_blob_session import validate_session

PIN = '87ad4e23ea8b35280b6e589e120a8e40efdbe13f5e78108bdb6d1aa204d502f8'
CEILING = 12_000_000_000
OUT = BASE/'publication-execution'


def save(name, value):
    path = OUT/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_part(part):
    folder = Path(part['folder'])
    expected = {f['alias']:f for f in part['files']}
    controls = ('mount-map.json', 'publication_identity.json')
    actual = {p.name for p in folder.iterdir() if p.is_file()}
    if actual != set(expected)|set(controls)|{'dataset-metadata.json'}:
        raise ValueError('Unexpected staging objects')
    mapping = json.loads((folder/controls[0]).read_text())
    canonical = json.dumps(mapping['files'],sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(canonical).hexdigest()!=part['identity'] or mapping['identity']!=part['identity']:
        raise ValueError('Partition manifest identity differs')
    projected = [{k:f[k] for k in ('path','alias','bytes','sha256')} for f in part['files']]
    if mapping['files']!=projected:
        raise ValueError('Manifest differs from pinned plan')
    ident = json.loads((folder/controls[1]).read_text())
    if ident['identity']!=part['identity'] or ident['files']!=len(expected):
        raise ValueError('Partition identity differs')
    metadata=json.loads((folder/'dataset-metadata.json').read_text())
    if metadata['id']!=part['handle']:
        raise ValueError('Destination metadata differs')
    for name,f in expected.items():
        p=folder/name
        if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:
            raise ValueError('Staged size/SHA mismatch: '+name)
    files=[folder/f['alias'] for f in part['files']]+[folder/n for n in controls]
    if sum(p.stat().st_size for p in files)!=part['upload_file_bytes']:
        raise ValueError('Pinned payload size differs')
    return files,metadata


def remote_files(c, part, version):
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetFilesRequest
    req=ApiListDatasetFilesRequest()
    req.owner_slug,req.dataset_slug=part['handle'].split('/')
    req.dataset_version_number=version
    req.page_size=100
    found={};tokens=set()
    for _ in range(10):
        response=c.datasets.dataset_api_client.list_dataset_files(req)
        if response.error_message:
            raise RuntimeError('Remote file inventory returned error')
        for f in response.dataset_files:
            if f.name in found:raise ValueError('Repeated remote filename')
            found[f.name]=f.total_bytes
        if not response.next_page_token:return found
        if response.next_page_token in tokens:raise ValueError('Repeated file inventory token')
        tokens.add(response.next_page_token);req.page_token=response.next_page_token
    raise RuntimeError('Incomplete remote file inventory')


def certify_presence(part, files):
    owner,slug=part['handle'].split('/')
    for check in range(20):
        with client() as c:
            presence=confirm_private_presence(c,owner,slug)
            if presence['status']=='PRIVATE_REMOTE_PRESENCE_CONFIRMED':
                inventory=remote_files(c,part,presence['version'])
                expected={p.name:p.stat().st_size for p in files}
                if inventory==expected:
                    return dict(**presence,files=[dict(name=n,bytes=b) for n,b in sorted(inventory.items())],
                                all_filenames_and_sizes_match=True)
                if inventory and set(inventory)-set(expected):
                    raise ValueError('Unexpected remote objects')
        print(json.dumps(dict(part=part['number'],phase='remote_indexing',check=check+1)),flush=True)
        time.sleep(15)
    raise TimeoutError('Remote privacy/files not confirmed; no next partition')


def run():
    if '--execute' not in sys.argv:
        raise SystemExit('Explicit --execute required')
    plan_path=BASE/'publication/plan.json'
    if sha(plan_path)!=PIN:raise ValueError('Pinned plan differs')
    plan=json.loads(plan_path.read_text())
    if len(plan['parts'])!=11 or plan['payload_file_bytes']!=9847724207:
        raise ValueError('Authorized scope differs')
    OUT.mkdir(exist_ok=True)
    if (OUT/'checkpoint.json').exists():
        raise RuntimeError('Campaign already started: read-only reconciliation required; no automatic rerun')
    state=dict(status='VERIFYING_EXISTING_BYTES',plan_sha256=PIN,ceiling=CEILING,
               payload_bytes=plan['payload_file_bytes'],parts=[],current_part=None,
               automatic_retries=0,started=dt.datetime.now(dt.timezone.utc).isoformat())
    save('checkpoint.json',state)
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk.blobs.types.blob_api_service import ApiBlobType,ApiStartBlobUploadRequest
    from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest,ApiDatasetNewFile
    original=requests.sessions.Session.send
    allowed={'GenerateAccessToken','IntrospectToken','ListDatasets','ListDatasetFiles',
             'StartBlobUpload','CreateDataset','GetDatasetStatus','GetDataset','ListTreeDatasetFiles'}
    signed_url=None
    confirmed=0
    with BodyBudget(OUT/'http-body.jsonl',CEILING) as budget:
        def guarded(session,request,**kwargs):
            nonlocal confirmed
            p=urlsplit(request.url);op=p.path.rsplit('/',1)[-1]
            blob=request.method=='PUT' and signed_url and request.url==signed_url
            rpc=(request.method=='POST' and p.scheme=='https' and
                 p.hostname in ('api.kaggle.com','www.kaggle.com') and op in allowed)
            if not (blob or rpc):raise RuntimeError('Unexpected campaign network operation')
            n=request.headers.get('Content-Length')
            if n is None and request.body is not None:raise RuntimeError('Unknown body length refused')
            n=int(n or 0)
            for adapter in session.adapters.values():adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(allow_redirects=False,timeout=(15,180) if blob else (15,45))
            budget.reserve(n)
            row=dict(operation='BlobPUT' if blob else op,body_bytes=n,part=state['current_part'])
            try:
                r=original(session,request,**kwargs)
                row['http_status']=r.status_code
                # A received response certifies completion of request transmission.
                confirmed+=n
                return r
            except BaseException as e:
                row['error_type']=type(e).__name__
                raise
            finally:
                row['confirmed_cumulative_body_bytes']=confirmed
                with (OUT/'network.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
        try:
            # Verify all pre-existing archives/manifests before any network write.
            verified={}
            for part in plan['parts']:
                expected='nnnnbhss/geofirex-compact-t3-20261111-p%02d'%part['number']
                if part['handle']!=expected:raise ValueError('Unexpected namespace/order')
                verified[part['number']]=verify_part(part)
                print(json.dumps(dict(part=part['number'],phase='local_sha256_pass')),flush=True)
            state['status']='PUBLISHING';save('checkpoint.json',state)
            requests.sessions.Session.send=guarded
            for part in plan['parts']:
                number=part['number'];state['current_part']=number
                files,metadata=verified[number]
                save('checkpoint.json',state)
                owner,slug=part['handle'].split('/')
                with client() as c:
                    preflight=collect(c,owner,slug,dict(active=True,username=owner),
                                      probe_target=False,catalogue_read_control=True)
                save('p%02d-preflight.json'%number,preflight)
                if not preflight['decision']['allow_create_new']:
                    # Existing target must be reconciled read-only, never uploaded again.
                    raise RuntimeError('Create gate closed: '+preflight['decision']['state'])
                if budget.used+part['upload_file_bytes']+1_000_000>CEILING:
                    raise RuntimeError('Insufficient remaining body budget')
                newfiles=[]
                for index,path in enumerate(files):
                    state.update(phase='upload_blob',current_file=path.name,file_index=index,
                                 reserved_http_body_bytes=budget.used)
                    save('checkpoint.json',state)
                    with client() as c:
                        req=ApiStartBlobUploadRequest();req.type=ApiBlobType.DATASET
                        req.name=path.name;req.content_length=path.stat().st_size
                        req.last_modified_epoch_seconds=int(path.stat().st_mtime)
                        session=validate_session(c.blobs.blob_api_client.start_blob_upload(req))
                    signed_url=session.url
                    with path.open('rb') as stream, requests.Session() as http:
                        response=http.put(signed_url,data=stream,headers={
                            'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)})
                    signed_url=None
                    if response.status_code not in (200,201):
                        raise RuntimeError('Blob PUT failed; no retry')
                    response.close()
                    f=ApiDatasetNewFile();f.token=session.token;newfiles.append(f)
                    print(json.dumps(dict(part=number,file=index+1,total_files=len(files),
                                          phase='blob_confirmed',reserved_body_bytes=budget.used)),flush=True)
                state['phase']='single_private_create';save('checkpoint.json',state)
                req=ApiCreateDatasetRequest();req.owner_slug=owner;req.slug=slug
                req.title=metadata['title'];req.license_name='CC-BY-4.0'
                req.description=metadata['description'];req.is_private=True;req.files=newfiles
                with client() as c:r=c.datasets.dataset_api_client.create_dataset(req)
                if r.error or r.status.lower() in ('error','failed'):
                    raise RuntimeError('Private CreateDataset refused')
                state['phase']='create_accepted_verifying_files';save('checkpoint.json',state)
                presence=certify_presence(part,files)
                save('p%02d-remote.json'%number,presence)
                state['parts'].append(dict(number=number,handle=part['handle'],version=presence['version'],
                    private=True,payload_bytes=part['upload_file_bytes'],files=len(files),
                    filenames_sizes_verified=True,identity=part['identity']))
                state.update(reserved_http_body_bytes=budget.used,confirmed_http_body_bytes=confirmed)
                save('checkpoint.json',state)
                print(json.dumps(dict(part=number,phase='PRIVATE_FILES_CONFIRMED',version=presence['version'])),flush=True)
            state.update(status='ALL_11_PRIVATE_REMOTE_FILES_CONFIRMED_CPU_HASH_CERTIFICATION_PENDING',phase='complete')
        except BaseException as e:
            state.update(status='STOPPED_FIRST_UNRESOLVED_FAILURE',error_type=type(e).__name__,
                         http_status=getattr(getattr(e,'response',None),'status_code',None))
            # No exception repr: signed upload URLs and tokens must never enter logs.
            print(json.dumps(dict(status=state['status'],error_type=type(e).__name__,
                                  part=state['current_part'],phase=state.get('phase'))),flush=True)
        finally:
            requests.sessions.Session.send=original
            state.update(reserved_http_body_bytes=budget.used,confirmed_http_body_bytes=confirmed,
                         remaining_budget_bytes=CEILING-budget.used)
            save('checkpoint.json',state)
    return 0 if state['status'].startswith('ALL_11') else 1


if __name__=='__main__':
    raise SystemExit(run())
