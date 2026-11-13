"""Explicit private create only. No kernel push, version replacement or readback download."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geofirex.upload_budget import BodyBudget

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged', required=True)
    p.add_argument('--journal', required=True)
    p.add_argument('--max-http-body-bytes', required=True, type=int)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    root = Path(a.staged)
    receipt = json.loads((root/'staging-receipt.json').read_text())
    payload = root/'payload.zip'
    with payload.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != receipt['sha256'] or payload.stat().st_size != receipt['payload_bytes']:
        raise ValueError('Staged payload changed')
    if not a.execute:
        print(json.dumps(dict(status='DRY_RUN_NO_NETWORK', **receipt), indent=2))
        raise SystemExit(0)
    import kagglehub
    import kagglehub.gcs_upload as uploads
    from kagglehub.auth import get_username
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.blobs.types.blob_api_service import ApiBlobType
    from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiGetDatasetRequest
    from geofirex.upload_budget import BodyBudget
    owner, slug = receipt['handle'].split('/')
    if get_username().casefold() != owner.casefold():
        raise PermissionError('Handle differs from authenticated owner')
    published = root/'publication.json'
    if published.exists():
        raise FileExistsError('Publication receipt already exists; use remote verification, never re-upload automatically')
    # The SDK supports resumable uploads; the durable counter covers every send.
    with BodyBudget(a.journal, a.max_http_body_bytes) as budget:
        restore = budget.install_requests_guard()
        try:
            with build_kaggle_client() as client:
                get = ApiGetDatasetRequest()
                get.owner_slug, get.dataset_slug = owner, slug
                try:
                    client.datasets.dataset_api_client.get_dataset(get)
                except Exception as error:
                    status = getattr(error, 'error_code', None) or getattr(getattr(error, 'response', None), 'status_code', None)
                    if str(status) != '404':
                        raise RuntimeError('Dataset absence not proven; upload refused') from None
                else:
                    raise RuntimeError('Dataset already exists; verify it before attempting anything else')
                uploads.MAX_PARALLEL_UPLOADS = 1
                tokens = uploads.upload_files_and_directories(str(payload), ignore_patterns=[], item_type=ApiBlobType.DATASET).to_proto()
                request = ApiCreateDatasetRequest()
                request.owner_slug, request.slug, request.title = owner, slug, slug
                request.is_private = True
                request.files, request.directories = tokens.files, tokens.directories
                client.datasets.dataset_api_client.create_dataset(request)
            result = dict(status='CREATE_REQUEST_ACCEPTED_REMOTE_INTEGRITY_PENDING', handle=receipt['handle'],
                          lot_identity=receipt['lot_identity'], reserved_http_body_bytes=budget.used,
                          ceiling=a.max_http_body_bytes)
            with published.open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
        finally:
            restore()
