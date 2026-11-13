"""CPU certificate submission/status with the SAME independent 12 GB ledger.

All writes are single attempt. A certificate is submitted only after all 11
private file inventories have been confirmed. No GPU or dataset uploads here.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/gpu_acceleration_20261111_v1'
OUT=BASE/'publication-execution'
sys.path[:0]=[str(ROOT/'src'),str(BASE/'kaggle_sdk'),str(ROOT/'scripts')]
from geofirex.upload_budget import BodyBudget
from acceleration_kaggle_readonly import client


@contextlib.contextmanager
def network():
    import requests
    from urllib3.util.retry import Retry
    original=requests.sessions.Session.send
    with BodyBudget(OUT/'http-body.jsonl',12_000_000_000) as budget:
        def guard(session,req,**kwargs):
            p=urlsplit(req.url);op=p.path.rsplit('/',1)[-1]
            if p.scheme!='https':raise RuntimeError('HTTPS required')
            if req.method=='POST':
                if p.hostname not in ('api.kaggle.com','www.kaggle.com') or op not in {
                    'GenerateAccessToken','IntrospectToken','SaveKernel','GetKernelSessionStatus',
                    'ListKernelSessionOutput','GetKernelSessionLog'}:
                    raise RuntimeError('Unexpected certificate RPC')
            elif req.method!='GET' or p.hostname!='www.kaggleusercontent.com':
                raise RuntimeError('Unexpected certificate download')
            n=int(req.headers.get('Content-Length') or 0)
            if req.body is not None and 'Content-Length' not in req.headers:
                raise RuntimeError('Unknown body length')
            for a in session.adapters.values():a.max_retries=Retry(total=0,redirect=0)
            budget.reserve(n);kwargs.update(allow_redirects=False,timeout=(15,60))
            row=dict(operation=op if req.method=='POST' else 'CertificateJSONDownload',body_bytes=n)
            try:
                response=original(session,req,**kwargs);row['http_status']=response.status_code
                row['confirmed_body_bytes']=n
                return response
            finally:
                with (OUT/'cpu-network.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        requests.sessions.Session.send=guard
        try:yield budget
        finally:
            requests.sessions.Session.send=original
            confirmed=0
            for name in ('network.jsonl','cpu-network.jsonl'):
                for line in (OUT/name).read_text().splitlines() if (OUT/name).exists() else []:
                    row=json.loads(line)
                    if 'http_status' in row:confirmed+=row['body_bytes']
            (OUT/'budget-summary.json').write_text(json.dumps(dict(
                ceiling=budget.ceiling,reserved_http_body_bytes=budget.used,
                confirmed_http_body_bytes=confirmed,remaining_bytes=budget.ceiling-budget.used),indent=2))


def main(mode):
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiSaveKernelRequest,ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest)
    job=BASE/'compact_cpu_job';meta=json.loads((job/'metadata.json').read_text())
    owner,slug=meta['id'].split('/')
    with network():
        if mode=='submit':
            state=json.loads((OUT/'checkpoint.json').read_text())
            assert state['status']=='ALL_11_PRIVATE_REMOTE_FILES_CONFIRMED_CPU_HASH_CERTIFICATION_PENDING'
            assert len(state['parts'])==11 and all(p['private'] and p['filenames_sizes_verified'] for p in state['parts'])
            # Write-ahead marker prevents resubmission after any ambiguous response.
            with (job/'submission-attempt.json').open('x') as f:json.dump(dict(private=True,gpu=False),f)
            code=(job/'job.py').read_text(encoding='utf-8')
            assert hashlib.sha256(code.encode()).hexdigest()==meta['code_sha256']
            req=ApiSaveKernelRequest();req.slug=meta['id'];req.new_title=slug
            req.text=code;req.language='python';req.kernel_type='script';req.is_private=True
            req.enable_gpu=False;req.enable_internet=False;req.session_timeout_seconds=1200
            req.dataset_data_sources=meta['dataset_sources']
            with client() as c:r=c.kernels.kernels_api_client.save_kernel(req)
            # Save only non-secret fields.
            info=r.to_dict();safe={k:v for k,v in info.items() if k in ('ref','url','versionNumber','error','kernelSlug','version')}
            (job/'submission.json').write_text(json.dumps(safe,indent=2))
            print('PRIVATE_CPU_CERTIFICATE_SUBMITTED',meta['id'])
        elif mode in ('status','fetch'):
            req=ApiGetKernelSessionStatusRequest();req.user_name=owner;req.kernel_slug=slug
            with client() as c:r=c.kernels.kernels_api_client.get_kernel_session_status(req)
            status=r.to_dict();(job/'status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(status))
            if mode=='fetch':
                req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=100
                with client() as c:r=c.kernels.kernels_api_client.list_kernel_session_output(req)
                files=r.to_dict().get('files',[])
                safe=[dict(fileName=f.get('fileName')) for f in files]
                (job/'output-files.json').write_text(json.dumps(safe,indent=2))
                for f in files:
                    if f.get('fileName')=='compact-certification/certificate.json':
                        import requests
                        response=requests.get(f['url'])
                        assert response.status_code==200 and len(response.content)<200_000
                        cert=response.json();(OUT/'remote-certificate.json').write_text(json.dumps(cert,indent=2))
                        print(json.dumps({k:v for k,v in cert.items() if k not in ('partitions','legacy_fast_tensor_comparisons')}))
                        return
                raise RuntimeError('Certificate JSON not yet available')
        else:raise ValueError('submit/status/fetch only')


if __name__=='__main__':main(sys.argv[1])
