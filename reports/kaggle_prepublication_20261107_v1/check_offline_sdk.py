"""Validate proposed requests and default offline modes with the installed tooling."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'scripts'))
import requests
def refuse(*args,**kwargs): raise AssertionError('No network permitted in offline contract check')
requests.sessions.Session.send=refuse
from publish_kaggle_single_attempt import publish
from submit_kaggle_cpu_certification import submit
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiDatasetNewFile
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest, ApiBlobType

root=Path(__file__).parent
common=dict(execute=False,credentials='NOT_READ_OFFLINE',journal='NOT_CREATED_OFFLINE',attempt='NOT_CREATED_OFFLINE',
            max_http_body_bytes=50_000_000_000,stop_at_cumulative_body_bytes=10_000_000)
result=dict(publisher=publish(argparse.Namespace(staged='reports/recovery_20261107_v3/finalized/code-package',license='other',**common)),
            cpu_submit=submit(argparse.Namespace(folder=str(root/'cpu-certification-v3/code'),**common)))
k=ApiSaveKernelRequest(); k.slug='nnnnbhss/geofirex-certify-code-20261107-v1'; k.text='{"cells":[]}'
k.new_title='GeoFireX Certify Code 20261107 V1'; k.language='python'; k.kernel_type='notebook'
k.enable_gpu=False; k.enable_tpu=False; k.enable_internet=False; k.is_private=True
k.dataset_data_sources=['nnnnbhss/geofirex-expanded-code-recovery-v1']; k.session_timeout_seconds=4200
data=ApiSaveKernelRequest.to_dict(k)
assert data['enableGpu'] is False and data['isPrivate'] is True
d=ApiCreateDatasetRequest(); d.owner_slug='nnnnbhss'; d.slug='offline-fixture'; d.title='offline-fixture'
d.is_private=True; d.license_name='CC-BY-4.0'; file=ApiDatasetNewFile(); file.token='OFFLINE_DUMMY'; d.files=[file]
assert ApiCreateDatasetRequest.to_dict(d)['isPrivate'] is True
b=ApiStartBlobUploadRequest(); b.type=ApiBlobType.DATASET; b.name='payload.zip'; b.content_length=123
assert ApiStartBlobUploadRequest.to_dict(b)['contentLength']==123
result.update(status='OFFLINE_SDK_REQUEST_AND_DRY_MODES_PASSED',network_calls=0,
              caveat='Serialization and offline control only; server acceptance and cloud execution remain untested')
with (root/'offline-sdk-check.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
