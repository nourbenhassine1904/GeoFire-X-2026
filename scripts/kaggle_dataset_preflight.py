"""Read-only create-new gate based on complete owner inventories, not HTTP 403.

ABSENT_VERIFIED means absent from three consistent, unfiltered authenticated owner
catalogues with a successful owner-resource read control. It is not a claim that
403 means missing, nor a guarantee of server-side dataset-create permission.
Always use CreateDataset only after this gate; never create a version as fallback.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse


class Immediate403Stop(BaseException):
    """Fatal publication stop, deliberately not swallowed by read-probe handlers."""
    def __init__(self,response):
        super().__init__('HTTP 403; stop without retry')
        self.response=response


def http_status(error):
    return getattr(getattr(error,'response',None),'status_code',None)


def classify(evidence):
    owner,target=evidence['owner'],evidence['handle']
    result=dict(state='UNKNOWN',allow_create_new=False,owner=owner,handle=target,
                absence_basis=None,direct_lookup_state='UNKNOWN',create_permission_verified=False)
    inventories=evidence.get('inventories',{})
    # Any positive presence evidence blocks creation, even on an incomplete page.
    if any(target in x.get('handles',[]) for x in inventories.values()) or evidence.get('target_status',{}).get('ok'):
        return dict(result,state='DATASET_EXISTS',reason='Positive target presence; creation forbidden')
    auth=evidence.get('identity',{})
    if not auth.get('active') or auth.get('username')!=owner:
        return dict(result,state='ACCESS_FORBIDDEN',reason='Active owner identity not verified')
    direct=evidence.get('target_status',{})
    code=direct.get('http_status')
    skipped=direct.get('skipped') is True and direct.get('basis')=='owner_catalogue_only'
    result['direct_lookup_state']='NOT_QUERIED' if skipped else ('ACCESS_FORBIDDEN' if code in (401,403) else ('NOT_FOUND_RESPONSE' if code==404 else 'UNKNOWN'))
    for name in ('mine','private','public'):
        row=inventories.get(name,{})
        if not row.get('complete'):
            state='ACCESS_FORBIDDEN' if row.get('http_status') in (401,403) else 'UNKNOWN'
            return dict(result,state=state,reason='Owner catalogue incomplete or unavailable: '+name)
        handles=row.get('handles',[])
        if len(handles)!=len(set(handles)):
            return dict(result,reason='Duplicate owner catalogue entries; consistency unknown')
    owned={name:{h for h in inventories[name]['handles'] if h.startswith(owner+'/')} for name in ('mine','private','public')}
    if owned['mine']!=owned['private']|owned['public'] or owned['private']&owned['public']:
        return dict(result,reason='Owner public/private catalogue views disagree')
    control=evidence.get('read_control',{})
    if not control.get('ok'):
        state='ACCESS_FORBIDDEN' if control.get('http_status') in (401,403) else 'UNKNOWN'
        return dict(result,state=state,reason='Independent existing owner-dataset read control not successful')
    if control.get('handle') not in owned['mine'] or control.get('version',0)<1:
        return dict(result,reason='Read control does not bind an existing owner-catalogue dataset')
    if code not in (403,404) and not skipped:
        return dict(result,state='ACCESS_FORBIDDEN' if code==401 else 'UNKNOWN',
                    reason='Target status is inconclusive; no create allowed')
    return dict(result,state='ABSENT_VERIFIED',allow_create_new=True,
                absence_basis='Complete consistent owner catalogues (MY, MY_PRIVATE, MY_PUBLIC) plus successful existing-owner read control',
                reason='Target absent from complete authenticated owner inventory; no HTTP error is used as proof of absence',
                catalogued_owner_datasets=len(owned['mine']),
                limitation='Catalogue visibility/consistency scope only; fresh check before create. CreateDataset must reject any collision; no version/update fallback.')


def collect(client,owner,slug,identity,*,max_pages=4,probe_target=True,catalogue_read_control=False):
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsRequest,ApiGetDatasetRequest,ApiGetDatasetStatusRequest
    from kagglesdk.datasets.types.dataset_enums import DatasetSelectionGroup
    target=owner+'/'+slug
    evidence=dict(owner=owner,handle=target,identity=identity,inventories={})
    catalogue_resources={}
    if not identity.get('active') or identity.get('username')!=owner:
        return dict(evidence,decision=classify(evidence))
    for name,group in [('mine',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY),
                       ('private',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PRIVATE),
                       ('public',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PUBLIC)]:
        req=ApiListDatasetsRequest();req.group=group;req.page_size=100
        row=dict(handles=[],complete=False,pages=0,filters='none')
        evidence['inventories'][name]=row
        try:
            tokens=set()
            for _ in range(max_pages):
                response=client.datasets.dataset_api_client.list_datasets(req);row['pages']+=1
                row['handles'].extend(r.ref for r in response.datasets)
                if catalogue_read_control:
                    for resource in response.datasets:
                        catalogue_resources[resource.ref]=dict(version=resource.current_version_number,private=resource.is_private)
                if target in row['handles']:
                    return dict(evidence,decision=classify(evidence))
                if not response.next_page_token:row['complete']=True;break
                if response.next_page_token in tokens:raise ValueError('Repeated pagination token')
                tokens.add(response.next_page_token);req.page_token=response.next_page_token
        except Exception as error:row.update(http_status=http_status(error),error_type=type(error).__name__)
    candidates=[h for h in evidence['inventories']['private']['handles']+evidence['inventories']['public']['handles'] if h.startswith(owner+'/')]
    if candidates:
        if catalogue_read_control:
            control=catalogue_resources[candidates[0]]
            evidence['read_control']=dict(ok=control['version']>=1,handle=candidates[0],**control,
                basis='Authenticated owner dataset metadata read through complete catalogue views; GetDataset not required')
        else:
            req=ApiGetDatasetRequest();req.owner_slug,req.dataset_slug=candidates[0].split('/')
            try:
                obj=client.datasets.dataset_api_client.get_dataset(req)
                evidence['read_control']=dict(ok=True,handle=candidates[0],version=obj.current_version_number,private=obj.is_private)
            except Exception as error:evidence['read_control']=dict(ok=False,http_status=http_status(error),error_type=type(error).__name__)
    if probe_target:
        req=ApiGetDatasetStatusRequest();req.owner_slug,req.dataset_slug=owner,slug
        try:
            obj=client.datasets.dataset_api_client.get_dataset_status(req)
            evidence['target_status']=dict(ok=True,value=obj.status.name)
        except Exception as error:evidence['target_status']=dict(ok=False,http_status=http_status(error),error_type=type(error).__name__)
    else:
        evidence['target_status']=dict(skipped=True,basis='owner_catalogue_only')
    evidence['decision']=classify(evidence)
    return evidence


def readonly_main(a):
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from geofirex.upload_budget import BodyBudget
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    before=Path(a.credentials).read_bytes();events=[];original=requests.sessions.Session.send
    result=dict(status='READONLY_PREFLIGHT_INCOMPLETE',uploads=0,writes=0)
    with BodyBudget(a.journal,6_000_000_000) as budget:
        initial=budget.used
        def guarded(session,request,**kwargs):
            p=urllib.parse.urlparse(request.url);op=p.path.rsplit('/',1)[-1]
            if (p.scheme!='https' or p.hostname not in ('api.kaggle.com','www.kaggle.com') or request.method!='POST'
                or op not in ('GenerateAccessToken','IntrospectToken','ListDatasets','GetDataset','GetDatasetStatus') or len(events)>=24):
                raise RuntimeError('Only bounded read-only operations allowed')
            n=request.headers.get('Content-Length')
            if n is None or int(n)>16_384 or budget.used-initial+int(n)>65_536:raise RuntimeError('Diagnostic body cap reached')
            for adapter in session.adapters.values():adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,20),allow_redirects=False);budget.reserve(int(n))
            row=dict(operation=op,body_bytes=int(n));events.append(row)
            response=original(session,request,**kwargs);row['http_status']=response.status_code
            return response
        try:
            requests.sessions.Session.send=guarded
            credentials=KaggleCredentials.load(KaggleClient(),file_path=a.credentials)
            if credentials is None:raise PermissionError('Existing credentials required')
            if not credentials._access_token_expiration or credentials._access_token_expiration<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
                fresh=credentials.generate_access_token();credentials._access_token=fresh.token
                credentials._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
            req=IntrospectTokenRequest();req.token=credentials._access_token
            auth=credentials._client.security.oauth_client.introspect_token(req)
            with KaggleClient(api_token=credentials._access_token) as client:
                result=collect(client,'nnnnbhss','geofirex-wsts-fold0-tv-001-v1',dict(active=auth.active,username=auth.username,scope_names=auth.scope.split()))
            result.update(status='READONLY_PREFLIGHT_COMPLETED',uploads=0,writes=0)
        except Exception as error:result.update(error_type=type(error).__name__,http_status=http_status(error))
        finally:
            requests.sessions.Session.send=original
            result.update(credentials_unchanged=before==Path(a.credentials).read_bytes(),requests=events,
                additional_body_bytes=budget.used-initial,cumulative_body_bytes=budget.used,ceiling=budget.ceiling)
            with (out/'preflight.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    return 0 if result.get('decision',{}).get('allow_create_new') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('credentials','journal','out'):p.add_argument('--'+name,required=True)
    raise SystemExit(readonly_main(p.parse_args()))
