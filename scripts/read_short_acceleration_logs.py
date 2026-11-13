"""Bounded read-only stream of the new diagnostic kernel logs."""
import json
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path('reports/gpu_acceleration_20261111_v1/kaggle_sdk').resolve()))
from acceleration_kaggle_readonly import client
import requests
if __name__=='__main__':
    with client() as c:
        with requests.post('https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionLogsStream',json=dict(userName='nnnnbhss',kernelSlug='geofirex-loader-profile-20261111-v2',waitForLogsUrlSeconds=5),headers={'Authorization':'Bearer '+c.api_token},stream=True,timeout=(10,15)) as response:
            print('logs HTTP',response.status_code)
            response.raise_for_status();lines=[];size=0;start=time.monotonic()
            try:
                for line in response.iter_lines():
                    if line:
                        s=line.decode('utf-8','replace');lines.append(s);size+=len(line)
                    if len(lines)>=30 or size>200_000 or time.monotonic()-start>12:break
            except requests.RequestException:pass
            path=Path('reports/gpu_acceleration_20261111_v1/short_gpu_job_v2')/f'logs_{time.time_ns()}.txt';path.write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines)[-13000:])
