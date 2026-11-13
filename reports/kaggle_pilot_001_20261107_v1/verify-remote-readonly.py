"""Pinned pilot metadata checks only; no upload, dataset change or download."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from geofirex.upload_budget import BodyBudget


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--kernel-status',action='store_true')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest,ApiGetDatasetStatusRequest,ApiListDatasetFilesRequest
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest
    owner='nnnnbhss';slug='geofirex-wsts-fold0-tv-001-v1'
    credential=Path('C:/Users/nourb/.kaggle/credentials.json')
    before=hashlib.sha256(credential.read_bytes()).hexdigest()
    result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),handle=owner+'/'+slug,write_requests=0)
    original=requests.sessions.Session.send
    allowed={'GenerateAccessToken','IntrospectToken','GetDataset','GetDatasetStatus','ListDatasetFiles'}
    if a.kernel_status:allowed|={'GetKernelSessionStatus','ListKernelSessionOutput'}
    events=[];downloads=set()
    with BodyBudget(ROOT/'reports/kaggle_pilot_001_20261107_v1/http-body.jsonl',6_000_000_000) as budget:
        initial=budget.used
        def guard(session,request,**kwargs):
            parsed=urllib.parse.urlsplit(request.url);op=parsed.path.rsplit('/',1)[-1]
            download=request.method=='GET' and request.url in downloads
            if not download and (parsed.scheme!='https' or parsed.hostname not in ('api.kaggle.com','www.kaggle.com')
                or request.method!='POST' or op not in allowed or len(events)>=30):
                raise RuntimeError('READONLY_OPERATION_REFUSED')
            length=request.headers.get('Content-Length')
            if download and request.body is not None:raise RuntimeError('DOWNLOAD_BODY_REFUSED')
            if not download and (length is None or int(length)>16384 or budget.used-initial+int(length)>65536):
                raise RuntimeError('READONLY_BODY_LIMIT')
            for adapter in session.adapters.values():adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,20),allow_redirects=False)
            budget.reserve(int(length or 0))
            row=dict(operation='ReadSmallCertificate' if download else op,body_bytes=int(length or 0));events.append(row)
            try:
                response=original(session,request,**kwargs);row['http_status']=response.status_code
                response.raise_for_status()
                if response.status_code>=300:raise RuntimeError('READONLY_REDIRECT_REFUSED')
                return response
            finally:
                with (out/'network.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        try:
            requests.sessions.Session.send=guard
            creds=KaggleCredentials.load(KaggleClient(),file_path=str(credential))
            if creds is None:raise PermissionError('EXISTING_CREDENTIALS_REQUIRED')
            if not creds._access_token_expiration or creds._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                fresh=creds.generate_access_token();creds._access_token=fresh.token
                creds._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
            if creds.introspect()!=owner:raise PermissionError('OWNER_MISMATCH')
            with KaggleClient(api_token=creds._access_token) as client:
                if a.kernel_status:
                    kernel='geofirex-certify-pilot-001-20261107-v2'
                    req=ApiGetKernelSessionStatusRequest();req.user_name=owner;req.kernel_slug=kernel
                    response=client.kernels.kernels_api_client.get_kernel_session_status(req)
                    result['kernel_status']=response.status.name
                    # Do not copy arbitrary server failure text into logs.
                    result['kernel_failure_message_present']=bool(response.failure_message)
                    if response.status.name=='COMPLETE':
                        req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=kernel;req.page_size=100
                        outputs=client.kernels.kernels_api_client.list_kernel_session_output(req)
                        result['outputs']=[dict(name=x.file_name) for x in outputs.files]
                        expected={
                            'geofirex-pilot-001-v2/pilot-certificate.json':'pilot-certificate.json',
                            'geofirex-pilot-001-v2/lot-001-certificate/verified.jsonl':'verified.jsonl',
                        }
                        downloaded=0
                        from pathlib import PurePosixPath
                        for item in outputs.files:
                            if item.file_name not in expected:continue
                            u=urllib.parse.urlsplit(item.url)
                            # Exact SDK-returned URLs only, with established GCS hosts;
                            # no redirects, credentials, arbitrary destinations or bulk data.
                            host=u.hostname or ''
                            if (u.scheme!='https' or u.port not in (None,443) or u.username or u.password or u.fragment
                                or host!='storage.googleapis.com' and not host.endswith('.storage.googleapis.com')
                                or any(c.isspace() or ord(c)<32 for c in item.url) or '\\' in item.url):
                                raise RuntimeError('CERTIFICATE_DOWNLOAD_URL_REJECTED')
                            downloads.add(item.url)
                            with requests.get(item.url,stream=True,timeout=(10,20),allow_redirects=False) as response:
                                with (out/expected[item.file_name]).open('xb') as sink:
                                    for block in response.iter_content(16384):
                                        downloaded+=len(block)
                                        if downloaded>1_000_000:raise RuntimeError('CERTIFICATE_DOWNLOAD_LIMIT')
                                        sink.write(block)
                        result['certificate_download_bytes']=downloaded
                        if not all((out/name).is_file() for name in expected.values()):raise RuntimeError('CERTIFICATE_OUTPUT_MISSING')
                        certificate=json.loads((out/'pilot-certificate.json').read_text())
                        manifest=json.loads((ROOT/'reports/recovery_20261107_v3/finalized/deployment/bundle/metadata/lot-001.json').read_text())
                        checks=[json.loads(line) for line in (out/'verified.jsonl').read_text().splitlines()]
                        wanted={entry['path']:(entry['sha256'],entry['bytes']) for entry in manifest['files']}
                        actual={entry['path']:(entry['sha256'],entry['bytes']) for entry in checks}
                        if (len(checks)!=1269 or actual!=wanted or certificate.get('identity')!=manifest['identity']
                            or certificate.get('files')!=1269 or certificate.get('events')!=57
                            or certificate.get('bytes')!=manifest['bytes'] or certificate.get('gpu_used') is not False
                            or certificate.get('pytorch_window',{}).get('device')!='cpu'
                            or certificate['pytorch_window'].get('workers')!=0):
                            raise RuntimeError('REMOTE_CPU_CERTIFICATE_MISMATCH')
                        result.update(cpu_certificate=certificate,individual_sha256_verified=1269)
                    result['status']='KERNEL_STATUS_READ'
                else:
                    req=ApiGetDatasetRequest();req.owner_slug=owner;req.dataset_slug=slug
                    metadata=client.datasets.dataset_api_client.get_dataset(req)
                    result.update(remote_ref=metadata.ref,private=metadata.is_private,version=metadata.current_version_number,total_bytes=metadata.total_bytes)
                    if metadata.ref!=owner+'/'+slug or metadata.is_private is not True:raise RuntimeError('DATASET_IDENTITY_OR_PRIVACY_MISMATCH')
                    status_req=ApiGetDatasetStatusRequest();status_req.owner_slug=owner;status_req.dataset_slug=slug
                    state=client.datasets.dataset_api_client.get_dataset_status(status_req).status.name
                    result['dataset_status']=state
                    if state=='READY':
                        req=ApiListDatasetFilesRequest();req.owner_slug=owner;req.dataset_slug=slug
                        req.dataset_version_number=metadata.current_version_number;req.page_size=100
                        files={}
                        for _ in range(20):
                            page=client.datasets.dataset_api_client.list_dataset_files(req)
                            if page.error_message:raise RuntimeError('REMOTE_FILE_LIST_ERROR')
                            for row in page.dataset_files:
                                if row.name in files:raise RuntimeError('REMOTE_FILE_DUPLICATE')
                                files[row.name]=row.total_bytes
                            if not page.next_page_token:break
                            req.page_token=page.next_page_token
                        else:raise RuntimeError('REMOTE_FILE_PAGINATION_LIMIT')
                        canonical=json.loads((ROOT/'reports/recovery_20261107_v3/finalized/deployment/bundle/metadata/lot-001.json').read_text())
                        expected={'rasters/'+row['path']:row['bytes'] for row in canonical['files']}
                        (out/'remote-file-sizes.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
                        result.update(remote_files=len(files),remote_geotiffs=sum(name.lower().endswith(('.tif','.tiff')) for name in files))
                        if set(files)!=set(expected)|{'lot-inventory.json'} or any(files[name]!=size for name,size in expected.items()):
                            raise RuntimeError('REMOTE_FILE_NAMES_OR_SIZES_MISMATCH')
                        result['status']='PRIVATE_READY_INVENTORY_SIZES_VERIFIED_HASHES_PENDING'
                    else:result['status']='PRIVATE_DATASET_PROCESSING'
        except Exception as error:
            result.update(status='READONLY_CHECK_STOPPED',error_type=type(error).__name__,http_status=getattr(getattr(error,'response',None),'status_code',None))
            if isinstance(error,RuntimeError) and str(error).isupper() and len(str(error))<100:result['local_error_code']=str(error)
        finally:
            requests.sessions.Session.send=original
            result.update(diagnostic_body_bytes=budget.used-initial,cumulative_http_body_bytes=budget.used,
                          credentials_unchanged=before==hashlib.sha256(credential.read_bytes()).hexdigest())
            (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 1 if result['status']=='READONLY_CHECK_STOPPED' else 0


if __name__=='__main__':raise SystemExit(main())
