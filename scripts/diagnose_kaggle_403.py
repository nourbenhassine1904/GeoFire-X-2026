"""Bounded authenticated read-only diagnosis. No uploads, token creation or jobs.

Only an expired existing OAuth access token may be refreshed in memory. The existing
publication ledger is appended for tiny diagnostic HTTP bodies, never reset.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.upload_budget import BodyBudget

OWNER='nnnnbhss'
SLUG='geofirex-wsts-fold0-tv-001-v1'


def diagnose(a):
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest,ApiGetDatasetStatusRequest,ApiListDatasetsRequest
    from kagglesdk.datasets.types.dataset_enums import DatasetSelectionGroup
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    before=Path(a.credentials).read_bytes()
    journal_before=Path(a.journal).read_bytes() if Path(a.journal).exists() else b''
    (out/'publication-ledger-before.jsonl').write_bytes(journal_before)
    secrets=[]
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if isinstance(v,str) and len(v)>16 and any(t in k.lower() for t in ('token','secret','key')):secrets.append(v)
                else:collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    collect(json.loads(before))
    def clean(s):
        s=str(s)
        for secret in secrets:s=s.replace(secret,'[REDACTED]')
        return re.sub(r'KGAT_[A-Za-z0-9_-]+','[REDACTED]',s)[:1000]
    events=[];original=requests.sessions.Session.send
    result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),target=OWNER+'/'+SLUG,
                versions={n:importlib.metadata.version(n) for n in ('kaggle','kagglesdk','kagglehub')},
                write_requests=0,blob_requests=0,notebook_requests=0)
    with BodyBudget(a.journal,6_000_000_000) as budget:
        initial=budget.used
        def guard(session,request,**kwargs):
            p=urllib.parse.urlparse(request.url);op=p.path.rsplit('/',1)[-1]
            if (p.scheme!='https' or p.hostname not in ('www.kaggle.com','api.kaggle.com') or request.method!='POST'
                or op not in ('GenerateAccessToken','IntrospectToken','ListDatasets','GetDataset','GetDatasetStatus')
                or len(events)>=24):raise RuntimeError('Non-read-only or excessive operation blocked')
            n=request.headers.get('Content-Length')
            if n is None or int(n)>16_384 or budget.used-initial+int(n)>65_536:
                raise RuntimeError('Diagnostic body limit reached')
            for adapter in session.adapters.values():adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,20),allow_redirects=False)
            row=dict(method=request.method,endpoint=p.scheme+'://'+p.netloc+p.path,operation=op,body_bytes=int(n),
                     authentication='Bearer OAuth' if request.headers.get('Authorization','').startswith('Bearer ') else 'OAuth token operation')
            if op in ('GetDataset','GetDatasetStatus','ListDatasets'):
                body=json.loads(request.body)
                row['resource']={k:body[k] for k in ('ownerSlug','datasetSlug','group','user','pageToken','pageSize','search') if k in body}
            budget.reserve(int(n));events.append(row)
            try:
                response=original(session,request,**kwargs);row['http_status']=response.status_code
                if response.status_code>=400:
                    try:
                        obj=response.json()
                        if isinstance(obj,dict):row['server_message']=clean(obj.get('message',obj.get('error','')))
                    except (ValueError,TypeError):row['server_message']='Non-JSON error response (body not retained)'
                return response
            except Exception as e:row['error_type']=type(e).__name__;raise
            finally:
                with (out/'requests.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        requests.sessions.Session.send=guard
        try:
            credentials=KaggleCredentials.load(KaggleClient(),file_path=a.credentials)
            if credentials is None:raise PermissionError('Existing OAuth credentials required')
            if not credentials._access_token_expiration or credentials._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                fresh=credentials.generate_access_token();credentials._access_token=fresh.token;secrets.append(fresh.token)
                credentials._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
                result['existing_token_refreshed_in_memory']=True
            introspect=IntrospectTokenRequest();introspect.token=credentials._access_token
            auth=credentials._client.security.oauth_client.introspect_token(introspect)
            result['identity']=dict(username=auth.username,active=auth.active,scope_names=auth.scope.split())
            if not auth.active or auth.username!=OWNER:raise PermissionError('Authenticated identity mismatch')
            with KaggleClient(api_token=credentials._access_token) as client:
                def probe(name,operation):
                    try:
                        value=operation()
                        return dict(ok=True,value=value)
                    except Exception as e:
                        return dict(ok=False,error_type=type(e).__name__,
                                    http_status=getattr(getattr(e,'response',None),'status_code',None))
                inventories={}
                for name,group in [('mine',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY),
                    ('private',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PRIVATE),
                    ('public',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PUBLIC),
                    ('owner_view',DatasetSelectionGroup.DATASET_SELECTION_GROUP_USER)]:
                    req=ApiListDatasetsRequest();req.group=group;req.page_size=100
                    if name=='owner_view':req.user=OWNER
                    inventory=dict(handles=[],complete=False,pages=0)
                    try:
                        seen=set()
                        for _ in range(4):
                            response=client.datasets.dataset_api_client.list_datasets(req);inventory['pages']+=1
                            inventory['handles'].extend(row.ref for row in response.datasets)
                            if not response.next_page_token:inventory['complete']=True;break
                            if response.next_page_token in seen:raise RuntimeError('Repeated pagination token')
                            seen.add(response.next_page_token);req.page_token=response.next_page_token
                    except Exception as e:inventory.update(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None))
                    inventories[name]=inventory
                result['inventories']=inventories
                def get(handle):
                    req=ApiGetDatasetRequest();req.owner_slug,req.dataset_slug=handle.split('/')
                    obj=client.datasets.dataset_api_client.get_dataset(req)
                    return dict(handle=handle,private=obj.is_private,version=obj.current_version_number,total_bytes=obj.total_bytes)
                control=next((h for h in inventories['private']['handles'] if h.startswith(OWNER+'/') and h!=OWNER+'/'+SLUG),None)
                result['private_control']=probe('control',lambda:get(control)) if control else dict(ok=False,reason='No private owner control available')
                result['target_get']=probe('target',lambda:get(OWNER+'/'+SLUG))
                req=ApiGetDatasetStatusRequest();req.owner_slug,req.dataset_slug=OWNER,SLUG
                result['target_status']=probe('status',lambda:client.datasets.dataset_api_client.get_dataset_status(req).status.name)
                # Exercise the actual official CLI methods with the exact same authenticated SDK client.
                saved=os.environ.get('KAGGLE_API_TOKEN');os.environ['KAGGLE_API_TOKEN']=credentials._access_token
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    cli=KaggleApi();cli.build_kaggle_client=lambda:KaggleClient(api_token=credentials._access_token)
                    result['official_cli_mine']=probe('cli_mine',lambda:[r.ref for r in cli.dataset_list_with_response(mine=True,page_size=100).datasets])
                    result['official_cli_status']=probe('cli_status',lambda:cli.dataset_status(OWNER+'/'+SLUG))
                finally:
                    if saved is None:os.environ.pop('KAGGLE_API_TOKEN',None)
                    else:os.environ['KAGGLE_API_TOKEN']=saved
            result['status']='READONLY_DIAGNOSTIC_COMPLETED'
        except Exception as e:result.update(status='READONLY_DIAGNOSTIC_INCOMPLETE',error_type=type(e).__name__)
        finally:
            requests.sessions.Session.send=original
            result.update(requests=events,publication_body_before=initial,diagnostic_http_body_bytes=budget.used-initial,
                          cumulative_http_body_bytes=budget.used,ceiling=budget.ceiling,
                          credentials_unchanged=before==Path(a.credentials).read_bytes())
            after=Path(a.journal).read_bytes()
            result['prior_budget_records_preserved']=after.startswith(journal_before)
            with (out/'diagnosis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('credentials','journal','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=diagnose(a);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='READONLY_DIAGNOSTIC_COMPLETED' else 1)
