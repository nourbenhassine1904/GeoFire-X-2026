"""Bounded personal Kaggle inventory. No dataset writes, sessions or downloads.

Refreshes an existing OAuth token in memory only; never saves credentials.
Run with the existing isolated Kaggle tooling Python, not the ML environment.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.parse


def inspect(credentials_path):
    import requests
    from urllib3.util.retry import Retry
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsRequest, ApiGetDatasetRequest
    from kagglesdk.datasets.types.dataset_enums import DatasetSelectionGroup
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    before = hashlib.sha256(Path(credentials_path).read_bytes()).hexdigest()
    sent = []
    original = requests.sessions.Session.send
    allowed = ('GenerateAccessToken', 'IntrospectToken', 'ListDatasets', 'GetDataset', 'GetAcceleratorQuotaStatistics')
    def bounded(session, request, **kwargs):
        url = urllib.parse.urlparse(request.url)
        if url.hostname not in ('www.kaggle.com', 'api.kaggle.com') or not any(url.path.endswith('/'+name) for name in allowed):
            raise RuntimeError('Non-read-only-account operation refused')
        if len(sent) >= 30:
            raise RuntimeError('Read-only request ceiling reached')
        for adapter in session.adapters.values():
            adapter.max_retries = Retry(total=0)
        kwargs['timeout'] = (10, 20)
        sent.append(url.path.rsplit('/', 1)[-1])
        return original(session, request, **kwargs)
    requests.sessions.Session.send = bounded
    result = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), packages={
        name: importlib.metadata.version(name) for name in ('kaggle', 'kagglehub', 'kagglesdk')},
        dataset_writes=0, sessions_started=0, raw_downloads=0)
    try:
        credentials = KaggleCredentials.load(KaggleClient(), file_path=str(credentials_path))
        if credentials is None:
            raise PermissionError('Existing OAuth credentials unavailable')
        result['saved_token_expiration'] = str(credentials._access_token_expiration)
        if (not credentials._access_token or not credentials._access_token_expiration or
                credentials._access_token_expiration <= dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5)):
            fresh = credentials.generate_access_token()
            credentials._access_token = fresh.token
            credentials._access_token_expiration = dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=fresh.expires_in)
            result['token_refreshed_in_memory'] = True
        result['authenticated_owner'] = credentials.introspect()
        with KaggleClient(api_token=credentials._access_token) as client:
            q = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
            gpu = q.gpu_quota
            used, reserved, allowed_seconds = (gpu.time_used.total_seconds(), gpu.time_reserved.total_seconds(), gpu.total_time_allowed.total_seconds())
            result['gpu'] = dict(allowed_seconds=allowed_seconds, used_seconds=used, reserved_seconds=reserved,
                remaining_seconds=max(0, allowed_seconds-used-reserved), reset_time=str(q.quota_refresh_time),
                pay_to_scale=gpu.is_pay_to_scale_enabled,
                qualification='Authenticated allocation; does not prove GPU capacity or phone-verification eligibility')
            request = ApiListDatasetsRequest()
            request.group = DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PRIVATE
            request.page_size = 100
            datasets = []
            for _ in range(5):
                response = client.datasets.dataset_api_client.list_datasets(request)
                for row in response.datasets:
                    get = ApiGetDatasetRequest()
                    get.owner_slug, get.dataset_slug = row.ref.split('/')
                    metadata = client.datasets.dataset_api_client.get_dataset(get)
                    datasets.append(dict(handle=row.ref, private=metadata.is_private, total_bytes=metadata.total_bytes,
                        current_version=metadata.current_version_number, returned_versions=len(metadata.versions)))
                if not response.next_page_token:
                    break
                request.page_token = response.next_page_token
            else:
                raise RuntimeError('Private dataset pagination exceeded bound')
            result['private_storage'] = dict(datasets=datasets, current_version_api_bytes=sum(d['total_bytes'] for d in datasets),
                account_limit_bytes=None, remaining_bytes=None,
                qualification='Current-version metadata inventory; personal ceiling and historical-version storage require account UI')
        result['status'] = 'AUTHENTICATED_READONLY_API_PASSED'
    except Exception as error:
        result.update(status='READONLY_API_INCOMPLETE', error_type=type(error).__name__,
                      http_status=getattr(getattr(error, 'response', None), 'status_code', None))
    finally:
        requests.sessions.Session.send = original
    result['credentials_unchanged'] = before == hashlib.sha256(Path(credentials_path).read_bytes()).hexdigest()
    result['request_operations'] = sent
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('New output required')
    result = inspect(a.credentials)
    with Path(a.out).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'AUTHENTICATED_READONLY_API_PASSED' else 1)
