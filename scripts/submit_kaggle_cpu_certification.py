"""Offline by default. Explicitly submit one pinned private CPU certification only."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.parse

from publish_kaggle_single_attempt import PhaseBudget


def submit(a):
    folder=Path(a.folder); metadata=json.loads((folder/'kernel-metadata.json').read_text())
    prepared=json.loads((folder.parent/'prepared.json').read_text())
    pin=next(x for x in prepared['notebooks'] if x['tag']==folder.name)
    notebook=(folder/metadata['code_file']).read_bytes()
    if hashlib.sha256(notebook).hexdigest()!=pin['notebook_sha256']: raise ValueError('Notebook changed')
    if (metadata['enable_gpu'] is not False or metadata['enable_internet'] is not False or metadata['is_private'] is not True
        or metadata.get('enable_tpu',False) or metadata.get('machine_shape') or metadata['id']!=pin['kernel']
        or metadata['dataset_sources']!=pin['sources']): raise ValueError('Pinned CPU-only metadata differs')
    result=dict(status='OFFLINE_CPU_SUBMISSION_REVIEW_ONLY',kernel=metadata['id'],notebook_bytes=len(notebook),
                private=True,gpu=False,internet=False)
    if not a.execute: return result
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    attempt=Path(a.attempt); attempt.mkdir(parents=True,exist_ok=False)
    (attempt/'started.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    before=hashlib.sha256(Path(a.credentials).read_bytes()).hexdigest()
    original=requests.sessions.Session.send; sent=0
    with PhaseBudget(a.journal,a.max_http_body_bytes,a.stop_at_cumulative_body_bytes) as budget:
        def guarded(session,request,**kwargs):
            nonlocal sent
            p=urllib.parse.urlparse(request.url)
            if (p.scheme!='https' or p.hostname not in ('api.kaggle.com','www.kaggle.com') or request.method!='POST'
                or p.path.rsplit('/',1)[-1] not in ('GenerateAccessToken','IntrospectToken','SaveKernel') or sent>=4):
                raise RuntimeError('Unexpected operation refused')
            n=request.headers.get('Content-Length')
            if n is None and request.body is not None: raise RuntimeError('Unknown body size')
            if int(n or 0)>2_000_000: raise RuntimeError('CPU submission body exceeds 2 MB')
            for adapter in session.adapters.values(): adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,30),allow_redirects=False)
            budget.reserve(int(n or 0)); sent+=1
            row=dict(operation=p.path.rsplit('/',1)[-1],body_bytes=int(n or 0))
            try:
                response=original(session,request,**kwargs)
                row['http_status']=response.status_code
                return response
            finally:
                with (attempt/'network.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps(row)+'\n')
        try:
            requests.sessions.Session.send=guarded
            credentials=KaggleCredentials.load(KaggleClient(),file_path=a.credentials)
            if credentials is None: raise PermissionError('Existing OAuth credentials required')
            if not credentials._access_token_expiration or credentials._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                fresh=credentials.generate_access_token(); credentials._access_token=fresh.token
                credentials._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
            if credentials.introspect()!='nnnnbhss': raise PermissionError('Owner differs')
            body=json.loads(notebook)
            for cell in body['cells']:
                if isinstance(cell.get('source'),list): cell['source']=''.join(cell['source'])
            req=ApiSaveKernelRequest(); req.slug=metadata['id']; req.new_title=metadata['title']; req.text=json.dumps(body)
            req.language='python'; req.kernel_type='notebook'; req.is_private=True
            req.enable_gpu=False; req.enable_tpu=False; req.enable_internet=False
            req.dataset_data_sources=metadata['dataset_sources']; req.session_timeout_seconds=1800
            with KaggleClient(api_token=credentials._access_token) as client:
                response=client.kernels.kernels_api_client.save_kernel(req)
            if response.error: raise RuntimeError('CPU notebook submission rejected')
            result.update(status='CPU_SUBMISSION_ACCEPTED_EXECUTION_PENDING',version=response.version_number)
        except Exception as error:
            result.update(status='CPU_SUBMISSION_STOPPED_INSPECT_REMOTE_STATE',error_type=type(error).__name__,
                          http_status=getattr(getattr(error,'response',None),'status_code',None))
        finally:
            requests.sessions.Session.send=original
            result.update(reserved_http_body_bytes=budget.used,requests=sent,
                          credentials_unchanged=before==hashlib.sha256(Path(a.credentials).read_bytes()).hexdigest())
            with (attempt/'result.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','credentials','journal','attempt'): p.add_argument('--'+name,required=True)
    p.add_argument('--max-http-body-bytes',type=int,required=True)
    p.add_argument('--stop-at-cumulative-body-bytes',type=int,required=True); p.add_argument('--execute',action='store_true')
    a=p.parse_args(); result=submit(a); print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status'] in ('OFFLINE_CPU_SUBMISSION_REVIEW_ONLY','CPU_SUBMISSION_ACCEPTED_EXECUTION_PENDING') else 1)
