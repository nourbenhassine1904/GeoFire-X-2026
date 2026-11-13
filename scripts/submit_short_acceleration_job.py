"""Submit only the reviewed <=600 second discarded-model diagnostic job."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('reports/gpu_acceleration_20261111_v1/kaggle_sdk').resolve()))
from acceleration_kaggle_readonly import client
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from geofirex.mission_streaming import save,sha

if __name__=='__main__':
    root=Path('reports/gpu_acceleration_20261111_v1/short_gpu_job_v3');review=json.loads((root/'review.json').read_text());meta=json.loads((root/'kernel-metadata.json').read_text())
    if review['code_sha256']!=sha(root/'job.py') or review['gpu_limit_seconds']!=300:raise ValueError('Reviewed short profile changed')
    if (root/'submission.json').exists():raise FileExistsError('Already submitted: inspect status, never repeat automatically')
    req=ApiSaveKernelRequest();req.slug=meta['id'];req.new_title=meta['title'];req.text=(root/'job.py').read_text();req.language='python';req.kernel_type='script';req.is_private=True;req.enable_gpu=True;req.enable_internet=False;req.machine_shape='NvidiaTeslaT4';req.session_timeout_seconds=300;req.dataset_data_sources=meta['dataset_sources']
    try:
        with client() as c:response=c.kernels.kernels_api_client.save_kernel(req)
    except Exception as error:
        response=getattr(error,'response',None)
        detail=dict(error_type=type(error).__name__,http_status=getattr(response,'status_code',None))
        if response is not None:
            try:
                body=response.json();detail['message']=str(body.get('message') or body.get('error') or body.get('errors'))[:500]
            except ValueError:detail['message']='Non-JSON response'
        save(root/'submission-rejection.json',detail);print(json.dumps(detail));raise SystemExit(1)
    save(root/'submission.json',response.to_dict());print(json.dumps(response.to_dict()))
