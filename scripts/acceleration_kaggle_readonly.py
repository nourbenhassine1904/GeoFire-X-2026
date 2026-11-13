"""Targeted account quota and supplied kernel metadata. Never prints secrets."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('reports/gpu_acceleration_20261111_v1/kaggle_sdk').resolve()))
from kagglesdk import KaggleClient,KaggleCredentials
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetAcceleratorQuotaStatisticsRequest
from geofirex.mission_streaming import save

def client():
    anonymous=KaggleClient();creds=KaggleCredentials.load(anonymous,'C:/Users/nourb/.kaggle/credentials.json')
    if not creds:raise RuntimeError('Existing credentials unavailable')
    creds.save=lambda *a,**kw:None # refresh in memory; preserve existing credential file
    return KaggleClient(api_token=creds.get_access_token())

if __name__=='__main__':
    result={}
    try:
        with client() as c:
            q=c.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest());g=q.gpu_quota
            result['quota']=dict(used=g.time_used.total_seconds(),reserved=g.time_reserved.total_seconds(),allowed=g.total_time_allowed.total_seconds(),reset=str(q.quota_refresh_time))
            req=ApiGetKernelRequest();req.user_name='nnnnbhss';req.kernel_slug='notebookfb12425f0b'
            r=c.kernels.kernels_api_client.get_kernel(req)
            # Store the source/metadata only; never a client/request/token object.
            result['kernel_response_fields']=list(r.to_dict())
            save('reports/gpu_acceleration_20261111_v1/supplied-kernel.json',r.to_dict())
        result['status']='READ_ONLY_OK'
    except Exception as error:
        result['status']='READ_ONLY_UNAVAILABLE';result['error_type']=type(error).__name__
    save('reports/gpu_acceleration_20261111_v1/remote-readonly.json',result);print(json.dumps(result))
