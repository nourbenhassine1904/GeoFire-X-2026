"""Read only small JSON outputs of this profile, with no signed-URL logging."""
from pathlib import Path
import json
import sys
import urllib.parse
sys.path.insert(0,str(Path('reports/gpu_acceleration_20261111_v1/kaggle_sdk').resolve()))
from acceleration_kaggle_readonly import client
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
import requests

if __name__=='__main__':
    root=Path('reports/gpu_acceleration_20261111_v1/t4-results-v3');root.mkdir(exist_ok=True)
    req=ApiListKernelSessionOutputRequest();req.user_name='nnnnbhss';req.kernel_slug='geofirex-loader-profile-20261111-v3';req.page_size=100
    with client() as c:value=c.kernels.kernels_api_client.list_kernel_session_output(req).to_dict()
    count=0
    for f in value.get('files',[]):
        name=f.get('fileName','')
        if name.startswith('profile-results/') and name.endswith('.json'):
            url=f['url']
            if urllib.parse.urlparse(url).hostname!='www.kaggleusercontent.com':raise ValueError('Unexpected download host')
            response=requests.get(url,timeout=30)
            if response.status_code!=200 or len(response.content)>200_000:raise RuntimeError('Bounded JSON retrieval failed')
            parsed=response.json();target=root/Path(name).name
            if not target.exists():target.write_text(json.dumps(parsed,indent=2))
            count+=1
    # Do not retain temporary signed output URLs in diagnostic artifacts.
    safe={k:v for k,v in value.items() if k!='files'};safe['files']=[dict(fileName=f.get('fileName')) for f in value.get('files',[])]
    Path('reports/gpu_acceleration_20261111_v1/short_gpu_job_v3/status-output.json').write_text(json.dumps(safe,indent=2))
    print('Small JSON results downloaded',count)
    if (root/'summary.json').exists():
        s=json.loads((root/'summary.json').read_text());print(json.dumps(dict(gpu=s['gpu'],train_tiles=s['train_tiles'],elapsed_seconds=s['elapsed_seconds'],positive_diagnostic=s['diagnostic_positive_pixels'],rows=[{k:r[k] for k in ('loader','workers','model','tiles_per_second','estimated_119_train_epoch_seconds','scaled_250_train_seconds')} for r in s['rows']])))
