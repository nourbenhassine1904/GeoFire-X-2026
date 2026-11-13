"""One private CPU notebook and bounded receipt retrieval; no dataset writes."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.upload_budget import BodyBudget
from kaggle_blob_session import _allowed_gcs_host


def validate_small_receipt_url(value):
    url=urllib.parse.urlsplit(value)
    if (url.scheme!='https' or url.port not in (None,443) or url.username or url.password or url.fragment
        or not (url.hostname=='www.kaggleusercontent.com' or _allowed_gcs_host(url.hostname or ''))
        or any(c.isspace() or ord(c)<32 for c in value) or '\\' in value):
        raise ValueError('Small receipt download URL refused')
    return value


def validate_receipt(folder,spec):
    folder=Path(folder)
    cert=json.loads((folder/'certificate.json').read_text())
    lot=json.loads(Path(spec['canonical']).read_text())
    rows=[json.loads(line) for line in (folder/'verified.jsonl').read_text().splitlines()]
    if len(rows)!=spec['files'] or {r['path']:r for r in rows}!={r['path']:r for r in lot['files']}:
        raise ValueError('CPU receipt paths/size/SHA256/CRC32 differ')
    if len({r['path'] for r in rows})!=len(rows): raise ValueError('Duplicate CPU receipt file')
    if (cert['status']!='LOT_REMOTE_CPU_CERTIFIED' or cert['identity']!=spec['identity']
        or cert['handle']!=spec['handle'] or cert['role']!=spec['role']
        or cert['files']!=spec['files'] or cert['events']!=spec['events'] or cert['bytes']!=spec['raw_bytes']
        or cert['anomalies']!=[] or cert['gpu_used'] is not False or cert['training'] is not False
        or cert['test_2021_files']!=0 or any(r['bands']!=23 for r in cert['rasterio_reads'])
        or len(cert['loader_reads'])!=3 or any(r['device']!='cpu' or r['X'][2]!=8 for r in cert['loader_reads'])):
        raise ValueError('CPU certificate scope/status/read mismatch')
    return cert


def run_job(plan_path,tag,output,credentials_path,*,recover=False,min_available_memory_bytes=0):
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest,ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest
    plan=json.loads(Path(plan_path).read_text())
    spec=None if tag=='canary' else next(x for x in plan['lots'] if x['tag']==tag)
    notebook_spec=plan['canary'] if spec is None else spec['cpu_notebook']
    folder=Path(notebook_spec['folder'])
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    nb_bytes=(folder/'certify.ipynb').read_bytes()
    if hashlib.sha256(nb_bytes).hexdigest()!=notebook_spec['sha256']: raise ValueError('Pinned notebook changed')
    if (meta['enable_gpu'] is not False or meta['enable_tpu'] is not False or meta['enable_internet'] is not False
        or meta['is_private'] is not True or meta['dataset_sources']!=([] if spec is None else [spec['handle']])
        or meta['id']!=notebook_spec['kernel']):
        raise ValueError('Private CPU notebook metadata mismatch')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original=requests.sessions.Session.send
    downloads=set()
    deadline=time.monotonic()+2100
    cred_path=Path(credentials_path)
    before=hashlib.sha256(cred_path.read_bytes()).hexdigest()
    result=dict(status='CPU_JOB_STOPPED',kernel=meta['id'],private=True,gpu=False)
    allowed={'GenerateAccessToken','IntrospectToken','SaveKernel','GetKernelSessionStatus','ListKernelSessionOutput'}
    sends=0
    with BodyBudget(plan['journal'],plan['remaining_body_ceiling_bytes']) as budget:
        initial=budget.used
        def guard(session,request,**kwargs):
            nonlocal sends
            if sends==0 and min_available_memory_bytes:
                from kaggle_autonomous import require_memory
                require_memory(min_available_memory_bytes)
            url=urllib.parse.urlsplit(request.url)
            op=url.path.rsplit('/',1)[-1]
            download=request.method=='GET' and request.url in downloads
            if time.monotonic()>deadline or sends>=180: raise TimeoutError('CPU controller bound reached')
            if not download and (request.method!='POST' or url.scheme!='https' or url.hostname not in ('api.kaggle.com','www.kaggle.com') or op not in allowed):
                raise RuntimeError('Unexpected operation refused')
            n=int(request.headers.get('Content-Length') or 0)
            if (request.body is not None and 'Content-Length' not in request.headers) or n>2_000_000 or budget.used-initial+n>5_000_000:
                raise RuntimeError('CPU control request body bound reached')
            for adapter in session.adapters.values(): adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,30),allow_redirects=False)
            budget.reserve(n)
            sends+=1
            row=dict(operation='ReadSmallCPUReceipt' if download else op,body_bytes=n)
            try:
                response=original(session,request,**kwargs)
                row['http_status']=response.status_code
                response.raise_for_status()
                if response.status_code>=300: raise RuntimeError('Redirect refused')
                return response
            finally:
                with (output/'network.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
        def refresh(creds):
            if not creds._access_token_expiration or creds._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                token=creds.generate_access_token()
                creds._access_token=token.token
                creds._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=token.expires_in)
        try:
            requests.sessions.Session.send=guard
            creds=KaggleCredentials.load(KaggleClient(),file_path=str(cred_path))
            if creds is None: raise PermissionError('Existing OAuth credentials required')
            refresh(creds)
            if creds.introspect()!='nnnnbhss': raise PermissionError('Owner mismatch')
            body=json.loads(nb_bytes)
            for cell in body['cells']:
                if isinstance(cell.get('source'),list): cell['source']=''.join(cell['source'])
            req=ApiSaveKernelRequest()
            req.slug=meta['id'];req.new_title=meta['title'];req.text=json.dumps(body)
            req.language='python';req.kernel_type='notebook';req.is_private=True
            req.enable_gpu=False;req.enable_tpu=False;req.enable_internet=False
            req.dataset_data_sources=meta['dataset_sources'];req.session_timeout_seconds=1800
            if not recover:
                with KaggleClient(api_token=creds._access_token) as client:
                    response=client.kernels.kernels_api_client.save_kernel(req)
                    if response.error: raise RuntimeError('CPU notebook submission rejected')
                    result['version']=response.version_number
            else:
                result['recovered_existing_job_without_resubmission']=True
            owner,slug=meta['id'].split('/')
            while True:
                refresh(creds)
                with KaggleClient(api_token=creds._access_token) as client:
                    req=ApiGetKernelSessionStatusRequest();req.user_name=owner;req.kernel_slug=slug
                    response=client.kernels.kernels_api_client.get_kernel_session_status(req)
                state=response.status.name
                (output/'remote-status.json').write_text(json.dumps(dict(state=state,kernel=meta['id'])))
                if state=='COMPLETE': break
                if state not in ('QUEUED','RUNNING'): raise RuntimeError('CPU notebook failed/stopped: '+state)
                if time.monotonic()+30>deadline: raise TimeoutError('Remote CPU job deadline')
                time.sleep(30)
            refresh(creds)
            with KaggleClient(api_token=creds._access_token) as client:
                req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=100
                outputs=client.kernels.kernels_api_client.list_kernel_session_output(req)
            wanted={'environment.json'} if spec is None else {'certificate.json','verified.jsonl','progress.json'}
            selected={}
            for item in outputs.files:
                name=Path(item.file_name).name
                if name not in wanted: continue
                # Select the top-level result receipt, not the helper's hashes/certificate.
                if spec is not None and '/results/' not in '/'+item.file_name: continue
                if '/hashes/' in '/'+item.file_name: continue
                if name in selected: raise ValueError('Ambiguous CPU receipt output')
                selected[name]=item
            if set(selected)!=wanted: raise ValueError('CPU receipt outputs incomplete')
            received=0
            for name,item in selected.items():
                validate_small_receipt_url(item.url)
                downloads.add(item.url)
                with requests.get(item.url,stream=True) as response, (output/name).open('xb') as sink:
                    for block in response.iter_content(16384):
                        received+=len(block)
                        if received>2_000_000: raise ValueError('Receipt download cap reached')
                        sink.write(block)
            if spec is None:
                if json.loads((output/'environment.json').read_text())['status']!='CPU_AUTOMATION_READY':
                    raise ValueError('CPU canary failed')
            else: validate_receipt(output,spec)
            result.update(status='CPU_AUTOMATION_READY' if spec is None else 'REMOTE_CPU_RECEIPT_VERIFIED',download_bytes=received)
        except Exception as error:
            result.update(error_type=type(error).__name__,http_status=getattr(getattr(error,'response',None),'status_code',None))
        finally:
            requests.sessions.Session.send=original
            result.update(control_body_bytes=budget.used-initial,cumulative_body_bytes=budget.used,
                          credentials_unchanged=before==hashlib.sha256(cred_path.read_bytes()).hexdigest())
            (output/'controller-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--tag',required=True);p.add_argument('--out',required=True)
    p.add_argument('--credentials',default='C:/Users/nourb/.kaggle/credentials.json')
    p.add_argument('--recover',action='store_true',help='Read an existing job only; never call SaveKernel')
    p.add_argument('--min-available-memory-bytes',type=int,default=0,choices=(0,3*1024**3))
    a=p.parse_args()
    result=run_job(a.plan,a.tag,a.out,a.credentials,recover=a.recover,
        min_available_memory_bytes=a.min_available_memory_bytes)
    raise SystemExit(0 if result['status'] in ('CPU_AUTOMATION_READY','REMOTE_CPU_RECEIPT_VERIFIED') else 1)
