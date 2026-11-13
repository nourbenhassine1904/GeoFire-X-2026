"""Bounded read-only failure details for the single CPU automation gate."""
import datetime as dt
import json
from pathlib import Path
import re
import sys
import urllib.parse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.upload_budget import BodyBudget

if __name__=='__main__':
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest
    plan=json.loads(Path('reports/kaggle_remaining_20261108_v1/plan.json').read_text())
    original=requests.sessions.Session.send
    result={}
    with BodyBudget(plan['journal'],plan['remaining_body_ceiling_bytes']) as budget:
        initial=budget.used
        def guard(session,request,**kwargs):
            p=urllib.parse.urlsplit(request.url)
            if (p.scheme!='https' or p.hostname not in ('api.kaggle.com','www.kaggle.com') or request.method!='POST'
                or p.path.rsplit('/',1)[-1] not in ('GenerateAccessToken','IntrospectToken','GetKernelSessionStatus','ListKernelSessionOutput')):
                raise RuntimeError('Read-only request refused')
            n=int(request.headers.get('Content-Length') or 0)
            if n>4096 or budget.used-initial+n>16384: raise RuntimeError('Read-only bound reached')
            budget.reserve(n)
            for adapter in session.adapters.values(): adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,20),allow_redirects=False)
            response=original(session,request,**kwargs);response.raise_for_status();return response
        requests.sessions.Session.send=guard
        try:
            creds=KaggleCredentials.load(KaggleClient(),file_path='C:/Users/nourb/.kaggle/credentials.json')
            if not creds._access_token_expiration or creds._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                token=creds.generate_access_token();creds._access_token=token.token
            if creds.introspect()!='nnnnbhss':raise PermissionError('Owner mismatch')
            owner,slug=plan['canary']['kernel'].split('/')
            with KaggleClient(api_token=creds._access_token) as client:
                req=ApiGetKernelSessionStatusRequest();req.user_name=owner;req.kernel_slug=slug
                obj=client.kernels.kernels_api_client.get_kernel_session_status(req)
                result.update(state=obj.status.name,failure_message=re.sub(r'https?://\S+','<redacted-url>',obj.failure_message or '')[:2000])
                req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=100
                files=client.kernels.kernels_api_client.list_kernel_session_output(req)
                result['output_file_names']=[r.file_name for r in files.files]
                result['output_url_metadata']=[dict(name=r.file_name,scheme=urllib.parse.urlsplit(r.url).scheme,
                    host=urllib.parse.urlsplit(r.url).hostname,is_relative=not urllib.parse.urlsplit(r.url).scheme) for r in files.files]
                result['log_tail']=re.sub(r'https?://\S+','<redacted-url>',files.log or '')[-6000:]
        except Exception as error:
            result.update(error_type=type(error).__name__,http_status=getattr(getattr(error,'response',None),'status_code',None))
        finally:
            requests.sessions.Session.send=original
            result['control_body_bytes']=budget.used-initial
    Path('reports/kaggle_remaining_20261108_v1/automation-output-metadata.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
