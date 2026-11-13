"""Read-only official SDK account inventory; credentials never enter output."""
import argparse
import datetime as dt
import json
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from kaggle import api
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsRequest, ApiGetDatasetRequest
    from kagglesdk.datasets.types.dataset_enums import DatasetSelectionGroup
    datasets = []
    # An unauthenticated MY_PRIVATE request can return an empty list: never
    # interpret that as an authenticated account inventory or free quota.
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    credentials = KaggleCredentials.load(KaggleClient())
    if credentials is None:
        raise PermissionError('Authenticated OAuth account required for private inventory')
    from datetime import timedelta
    if (credentials._access_token_expiration is None or
            credentials._access_token_expiration <= dt.datetime.now(dt.timezone.utc)+timedelta(minutes=5)):
        credentials.refresh_access_token()
    authenticated_owner = credentials.introspect()
    with api.build_kaggle_client() as client:
        request = ApiListDatasetsRequest()
        request.group = DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PRIVATE
        request.page_size = 100
        seen = set()
        while True:
            response = client.datasets.dataset_api_client.list_datasets(request)
            for row in response.datasets:
                if row.ref in seen:
                    raise ValueError("Repeated dataset in pagination")
                seen.add(row.ref)
                get = ApiGetDatasetRequest()
                get.owner_slug, get.dataset_slug = row.ref.split("/")
                metadata = client.datasets.dataset_api_client.get_dataset(get)
                datasets.append({"handle": row.ref, "private": metadata.is_private,
                    "api_total_bytes": metadata.total_bytes, "version": metadata.current_version_number,
                    "versions_returned": len(metadata.versions)})
            if not response.next_page_token:
                break
            request.page_token = response.next_page_token
    report = {"time": dt.datetime.now(dt.timezone.utc).isoformat(), "authenticated_owner": authenticated_owner, "datasets": datasets,
        "documented_max_dataset_bytes": 200_000_000_000,
        "personal_private_storage_limit_bytes": None,
        "limit_source": "https://www.kaggle.com/docs/datasets#technical-specifications",
        "current_api_bytes": sum(d["api_total_bytes"] for d in datasets),
        "qualification": "Authenticated GetDataset current-version total_bytes per private dataset. The documented per-dataset size limit is not the account's private-storage quota; neither personalized quota nor historical-version storage is measured here."}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
