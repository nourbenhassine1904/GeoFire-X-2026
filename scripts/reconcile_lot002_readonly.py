"""One bounded read-only catalogue reconciliation after accepted lot002 creation."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.upload_budget import BodyBudget
from run_bounded_local import memory


if __name__=='__main__':
    current=memory()
    if not current or current['available']<4*1024**3:raise MemoryError('Read-only reconciliation not started: less than 4 GiB')
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsRequest
    from kagglesdk.datasets.types.dataset_enums import DatasetSelectionGroup
    from kaggle_private_presence import confirm_private_presence
    base=Path('reports/kaggle_remaining_20261108_v1')
    plan=json.loads((base/'plan.json').read_text())
    out=base/'resume-01/reconciliation-lot002'
    out.mkdir(exist_ok=False)
    path=Path('C:/Users/nourb/.kaggle/credentials.json')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    original=requests.sessions.Session.send
    result=dict(status='READONLY_RECONCILIATION_PENDING',handle='nnnnbhss/geofirex-wsts-fold0-tv-002-v1',
        upload_requests=0,dataset_write_requests=0,catalogues={})
    calls=0
    with BodyBudget(plan['journal'],31_000_000_000) as budget:
        initial=budget.used
        def guard(session,request,**kwargs):
            global calls
            url=urllib.parse.urlsplit(request.url)
            operation=url.path.rsplit('/',1)[-1]
            if (request.method!='POST' or url.scheme!='https' or url.hostname not in ('api.kaggle.com','www.kaggle.com')
                or operation not in ('GenerateAccessToken','IntrospectToken','ListDatasets') or calls>=16):
                raise RuntimeError('Only bounded authenticated catalogue reads allowed')
            n=int(request.headers.get('Content-Length') or 0)
            if n>4096 or budget.used-initial+n>16384:raise RuntimeError('Read-only body cap reached')
            budget.reserve(n);calls+=1
            for adapter in session.adapters.values():adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,20),allow_redirects=False)
            row=dict(operation=operation,body_bytes=n)
            try:
                response=original(session,request,**kwargs);row['http_status']=response.status_code
                response.raise_for_status();return response
            finally:
                with (out/'network.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        try:
            requests.sessions.Session.send=guard
            credentials=KaggleCredentials.load(KaggleClient(),file_path=str(path))
            if not credentials._access_token_expiration or credentials._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                token=credentials.generate_access_token();credentials._access_token=token.token
            if credentials.introspect()!='nnnnbhss':raise PermissionError('Owner mismatch')
            with KaggleClient(api_token=credentials._access_token) as client:
                for name,group in (('mine',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY),
                                   ('private',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PRIVATE),
                                   ('public',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PUBLIC)):
                    req=ApiListDatasetsRequest();req.group=group;req.page_size=100
                    handles=[];matches=[]
                    for _ in range(4):
                        page=client.datasets.dataset_api_client.list_datasets(req)
                        handles.extend(r.ref for r in page.datasets)
                        matches.extend(dict(handle=r.ref,private=r.is_private,version=r.current_version_number)
                            for r in page.datasets if r.ref==result['handle'])
                        if not page.next_page_token:break
                        req.page_token=page.next_page_token
                    else:raise ValueError('Catalogue incomplete')
                    result['catalogues'][name]=dict(count=len(handles),complete=True,target=matches)
                # A second create/upload is not possible in this script.
                presence=confirm_private_presence(client,'nnnnbhss','geofirex-wsts-fold0-tv-002-v1')
                result.update(status='PRIVATE_REMOTE_PRESENCE_CONFIRMED',presence=presence)
        except Exception as error:
            result.update(status='READONLY_RECONCILIATION_STOPPED',error_type=type(error).__name__,
                local_reason=str(error) if isinstance(error,ValueError) else None,
                http_status=getattr(getattr(error,'response',None),'status_code',None))
        finally:
            requests.sessions.Session.send=original
            result.update(control_body_bytes=budget.used-initial,cumulative_body_bytes=budget.used,
                credentials_unchanged=before==hashlib.sha256(path.read_bytes()).hexdigest())
            (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
