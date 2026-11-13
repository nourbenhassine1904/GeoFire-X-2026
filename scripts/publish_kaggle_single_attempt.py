"""Reviewed private-create proposal: one streamed PUT, no retries, durable body budget.

Default is OFFLINE. --execute is required for any network action. Never prints tokens,
signed URLs or response bodies. Does not run notebooks, download data, or replace versions.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.upload_budget import BodyBudget
from kaggle_dataset_preflight import collect as collect_dataset_preflight,Immediate403Stop
from kaggle_blob_session import BlobSessionError,session_metadata,wire_response_metadata,validate_session


class PhaseBudget(BodyBudget):
    def __init__(self, journal, ceiling, stop_at):
        if not 0 < stop_at <= ceiling: raise ValueError('Invalid cumulative phase boundary')
        self.stop_at = stop_at
        super().__init__(journal, ceiling)

    def reserve(self, n):
        if self.used+n > self.stop_at: raise RuntimeError('Cumulative phase boundary reached; STOP')
        super().reserve(n)


def check_staged(staged):
    root = Path(staged)
    receipt = json.loads((root/'staging-receipt.json').read_text())
    payload = root/'payload.zip'
    with payload.open('rb') as f: sha = hashlib.file_digest(f,'sha256').hexdigest()
    if sha != receipt['sha256'] or payload.stat().st_size != receipt['payload_bytes']:
        raise ValueError('Staged bytes differ')
    if receipt['handle'].split('/')[0] != 'nnnnbhss': raise ValueError('Unexpected destination owner')
    return payload, receipt


def stream_once(session, url, payload):
    # No seek/retry loop. A failed/ambiguous response is a hard stop.
    with Path(payload).open('rb') as f:
        r = session.put(url, data=f, headers={'Content-Type':'application/octet-stream',
                        'Content-Length':str(Path(payload).stat().st_size)}, timeout=(10,120), allow_redirects=False)
    try:
        if r.status_code not in (200,201): raise RuntimeError('Blob completion not confirmed')
    finally:
        r.close()


def publish(a):
    payload, receipt = check_staged(a.staged)
    if not a.execute:
        return dict(status='OFFLINE_REVIEW_ONLY_NO_NETWORK', private=True, automatic_retries=0, **receipt)
    if getattr(a,'approved_plan',None):
        plan_path=Path(a.approved_plan)
        plan=json.loads(plan_path.read_text())
        matches=[x for x in plan['lots'] if x['handle']==receipt['handle'] and x['number'] in range(2,9)]
        if len(matches)!=1: raise PermissionError('Only pinned remaining lots 002..008 are allowed')
        spec=matches[0]
        canonical=Path(spec['canonical'])
        if hashlib.sha256(canonical.read_bytes()).hexdigest()!=spec['canonical_sha256']:
            raise ValueError('Pinned canonical manifest changed')
        if (receipt['lot_identity']!=spec['identity'] or receipt['observations']!=spec['files']
            or receipt['payload_bytes']>spec['archive_ceiling_bytes']
            or Path(a.staged).resolve()!=Path(spec['staging']).resolve()
            or Path(a.journal).resolve()!=Path(plan['journal']).resolve()
            or a.max_http_body_bytes!=plan['remaining_body_ceiling_bytes']
            or a.stop_at_cumulative_body_bytes!=plan['remaining_body_ceiling_bytes']):
            raise PermissionError('Remaining-lot publication differs from pinned plan/budget')
    elif (receipt['handle']!='nnnnbhss/geofirex-wsts-fold0-tv-001-v1' or receipt['payload_bytes']!=4829918684
        or receipt['sha256']!='464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890'
        or receipt['lot_identity']!='29ccf1eaa0f4d1fb0ab4333ee307e8af881f4255d1bd5639721ccaea2c540f81'
        or a.max_http_body_bytes!=6_000_000_000 or a.stop_at_cumulative_body_bytes!=6_000_000_000):
        raise PermissionError('Execution authorization is restricted to the pinned lot 001 and strict 6 GB ceiling')
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
    from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiGetDatasetRequest, ApiDatasetNewFile
    attempt = Path(a.attempt)
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt/'started.json').write_text(json.dumps(dict(handle=receipt['handle'], sha256=receipt['sha256'],
        time=dt.datetime.now(dt.timezone.utc).isoformat(), max_http_body_bytes=a.max_http_body_bytes,
        stop_at_cumulative_body_bytes=a.stop_at_cumulative_body_bytes),indent=2),encoding='utf-8')
    credentials_before = hashlib.sha256(Path(a.credentials).read_bytes()).hexdigest()
    original = requests.sessions.Session.send
    allowed_rpc = {'GenerateAccessToken','IntrospectToken','GetDataset','GetDatasetStatus','ListDatasets','StartBlobUpload','CreateDataset'}
    signed_url, sent, puts = None, 0, 0
    error_stage='budget_preflight'
    result = dict(status='STOPPED_REMOTE_STATE_REQUIRES_READONLY_INSPECTION', handle=receipt['handle'],
                  pre_network_zip_sha256_verified=receipt['sha256'],payload_bytes=receipt['payload_bytes'],
                  confirmed_blob_body_bytes=0)
    with PhaseBudget(a.journal,a.max_http_body_bytes,a.stop_at_cumulative_body_bytes) as budget:
        def guarded(session, request, **kwargs):
            nonlocal sent, puts
            parsed = urllib.parse.urlparse(request.url)
            rpc = (parsed.scheme == 'https' and parsed.hostname in ('api.kaggle.com','www.kaggle.com')
                   and parsed.path.rsplit('/',1)[-1] in allowed_rpc and request.method == 'POST')
            blob = signed_url is not None and request.url == signed_url and request.method == 'PUT'
            if not (rpc or blob) or sent >= 40: raise RuntimeError('Unexpected network operation refused')
            if getattr(a,'min_available_memory_bytes',0) and (sent==0 or blob):
                from kaggle_autonomous import require_memory
                require_memory(a.min_available_memory_bytes)
            if blob:
                if puts: raise RuntimeError('Second blob send refused')
                puts += 1
            length = request.headers.get('Content-Length')
            if length is None and request.body is not None: raise RuntimeError('Unknown body size refused')
            for adapter in session.adapters.values(): adapter.max_retries=Retry(total=0,redirect=0)
            kwargs.update(timeout=(10,120) if blob else (10,20), allow_redirects=False)
            budget.reserve(int(length or 0)); sent += 1
            row=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),method=request.method,
                     operation='BlobPUT' if blob else parsed.path.rsplit('/',1)[-1],reserved_body_bytes=int(length or 0))
            try:
                response=original(session,request,**kwargs)
                row['http_status']=response.status_code
                if response.status_code==403:raise Immediate403Stop(response)
                if not blob and parsed.path.endswith('/StartBlobUpload') and response.status_code==200:
                    # Projection only; never persist raw JSON, tokens, URL paths/queries or repr.
                    wire_meta=wire_response_metadata(response.content)
                    with (attempt/'blob-session-wire-metadata.json').open('x',encoding='utf-8') as log:
                        json.dump(wire_meta,log,indent=2)
                return response
            except Exception as error:
                row['error_type']=type(error).__name__
                raise
            finally:
                with (attempt/'network.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(row)+'\n')
        try:
            if budget.used+receipt['payload_bytes']+1_000_000 > budget.stop_at:
                raise RuntimeError('Insufficient remaining phase budget; no network started')
            requests.sessions.Session.send = guarded
            error_stage='existing_credentials_and_in_memory_authentication'
            credentials = KaggleCredentials.load(KaggleClient(),file_path=a.credentials)
            if credentials is None: raise PermissionError('Existing OAuth credentials required')
            if (not credentials._access_token_expiration or credentials._access_token_expiration <=
                    dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5)):
                fresh=credentials.generate_access_token()
                credentials._access_token=fresh.token
                credentials._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
            owner, slug = receipt['handle'].split('/')
            authenticated_owner=credentials.introspect()
            if authenticated_owner != owner: raise PermissionError('Authenticated owner differs')
            result['authenticated_owner_before_upload']=authenticated_owner
            with KaggleClient(api_token=credentials._access_token) as client:
                error_stage='verify_complete_owner_catalogue'
                preflight=collect_dataset_preflight(client,owner,slug,dict(active=True,username=authenticated_owner),
                    probe_target=False,catalogue_read_control=bool(getattr(a,'approved_plan',None)))
                with (attempt/'dataset-preflight.json').open('x',encoding='utf-8') as f:json.dump(preflight,f,indent=2)
                result['dataset_preflight']=preflight['decision']
                if not preflight['decision']['allow_create_new']:
                    raise RuntimeError('Dataset exists, catalogue incomplete or access unavailable; no create allowed')
                error_stage='start_blob_upload'
                req=ApiStartBlobUploadRequest(); req.type=ApiBlobType.DATASET; req.name='payload.zip'
                req.content_length=receipt['payload_bytes']; req.last_modified_epoch_seconds=int(payload.stat().st_mtime)
                blob=client.blobs.blob_api_client.start_blob_upload(req)
                error_stage='validate_blob_session'
                with (attempt/'blob-session-sdk-metadata.json').open('x',encoding='utf-8') as log:
                    json.dump(session_metadata(blob),log,indent=2)
                validated=validate_session(blob)
                signed_url=validated.url
                error_stage='single_blob_put'
                with requests.Session() as session: stream_once(session,signed_url,payload)
                result['confirmed_blob_body_bytes']=receipt['payload_bytes']
                error_stage='create_private_dataset'
                req=ApiCreateDatasetRequest(); req.owner_slug,req.slug,req.title=owner,slug,slug
                req.is_private=True; req.license_name=a.license
                req.description=('GeoFire-X frozen code and metadata; private recovery preparation.' if receipt.get('kind') else
                    'WildfireSpreadTS subset, Gerard, Zhao and Sullivan (2023), DOI 10.5281/zenodo.8006177. '
                    'CC-BY-4.0. Complete train/validation events only; pinned manifest in lot-inventory.json.')
                file=ApiDatasetNewFile(); file.token=validated.token; req.files=[file]
                # A slow multi-GB PUT may outlive the original OAuth access token.
                if (credentials._access_token_expiration <= dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5)):
                    fresh=credentials.generate_access_token()
                    credentials._access_token=fresh.token
                    credentials._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
                with KaggleClient(api_token=credentials._access_token) as create_client:
                    response=create_client.datasets.dataset_api_client.create_dataset(req)
                if response.error or response.status.lower() in ('error','failed'):
                    raise RuntimeError('Dataset creation not accepted')
                result['create_request_accepted']=True
                if getattr(a,'approved_plan',None):
                    from kaggle_private_presence import wait_private_presence
                    error_stage='confirm_private_remote_presence_from_catalogues'
                    with KaggleClient(api_token=credentials._access_token) as presence_client:
                        presence=wait_private_presence(presence_client,owner,slug)
                    (attempt/'private-presence.json').write_text(json.dumps(presence,indent=2),encoding='utf-8')
                    result['private_remote_presence']=presence
            result.update(status='PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING', payload_sha256=receipt['sha256'],
                          private=True, automatic_retries=0)
        except (Exception,Immediate403Stop) as error:
            result.update(error_type=type(error).__name__,error_stage=error_stage,
                          http_status=getattr(getattr(error,'response',None),'status_code',None))
            if isinstance(error,BlobSessionError):
                result.update(local_error_code=error.code,blob_session_metadata=error.metadata)
            elif isinstance(error,Immediate403Stop):result['local_error_code']='HTTP_403_STOP'
            elif error_stage=='start_blob_upload':result['local_error_code']='BLOB_START_RPC_OR_DECODE_FAILED'
            elif error_stage=='validate_blob_session':result['local_error_code']='BLOB_VALIDATION_UNEXPECTED_FAILURE'
            elif error_stage=='single_blob_put':result['local_error_code']='BLOB_PUT_NOT_CONFIRMED'
        finally:
            requests.sessions.Session.send=original
            result.update(reserved_http_body_bytes=budget.used, cumulative_ceiling=budget.ceiling,
                          phase_stop_at=budget.stop_at, blob_sends=puts, requests=sent,
                          credentials_unchanged=credentials_before == hashlib.sha256(Path(a.credentials).read_bytes()).hexdigest())
            with (attempt/'result.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged',required=True); p.add_argument('--credentials',required=True)
    p.add_argument('--journal',required=True); p.add_argument('--attempt',required=True)
    p.add_argument('--max-http-body-bytes',type=int,required=True)
    p.add_argument('--stop-at-cumulative-body-bytes',type=int,required=True)
    p.add_argument('--license',required=True); p.add_argument('--execute',action='store_true')
    p.add_argument('--approved-plan',help='Pinned plan for separately authorized lots 002..008 only')
    p.add_argument('--min-available-memory-bytes',type=int,default=0,choices=(0,3*1024**3))
    a=p.parse_args()
    result=publish(a); print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status'] in ('OFFLINE_REVIEW_ONLY_NO_NETWORK','PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING') else 1)
