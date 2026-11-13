"""Read only the new diagnostic job; download JSON results, never corpus data."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('reports/gpu_acceleration_20261111_v1/kaggle_sdk').resolve()))
from acceleration_kaggle_readonly import client
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest

if __name__=='__main__':
    with client() as c:
        r=ApiGetKernelSessionStatusRequest();r.user_name='nnnnbhss';r.kernel_slug='geofirex-loader-profile-20261111-v3'
        status=c.kernels.kernels_api_client.get_kernel_session_status(r)
        print(json.dumps(status.to_dict()))
        req=ApiListKernelSessionOutputRequest();req.user_name='nnnnbhss';req.kernel_slug='geofirex-loader-profile-20261111-v3'
        out=c.kernels.kernels_api_client.list_kernel_session_output(req)
        value=out.to_dict();safe={k:v for k,v in value.items() if k!='files'};safe['files']=[dict(fileName=f.get('fileName')) for f in value.get('files',[])]
        target=Path('reports/gpu_acceleration_20261111_v1/short_gpu_job_v3/status-output.json');target.write_text(json.dumps(safe,indent=2))
        print('output fields',list(safe))
