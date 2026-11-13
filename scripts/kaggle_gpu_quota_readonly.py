"""Read current official accelerator quota; no session creation."""
import argparse
import datetime as dt
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    with api.build_kaggle_client() as client:
        q = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
    gpu = q.gpu_quota
    result = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),
                  used_seconds=gpu.time_used.total_seconds(), reserved_seconds=gpu.time_reserved.total_seconds(),
                  allowed_seconds=gpu.total_time_allowed.total_seconds(), pay_to_scale=gpu.is_pay_to_scale_enabled,
                  reset_time=str(q.quota_refresh_time))
    with Path(a.out).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
