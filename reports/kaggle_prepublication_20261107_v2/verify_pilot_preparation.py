"""Offline structural verification; no source raster reads or second whole-ZIP scan."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

root=Path(__file__).parent
sys.path.insert(0,str(Path.cwd()/'scripts'))
import requests
def forbidden(*args,**kwargs): raise AssertionError('No network during pilot preparation verification')
requests.sessions.Session.send=forbidden
from submit_kaggle_cpu_certification import submit
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

plan=json.loads((root/'pilot/pilot-plan.json').read_text())
canonical=Path('reports/recovery_20261107_v3/finalized/deployment/bundle/metadata/lot-001.json')
lot=json.loads(canonical.read_text())
receipt=json.loads(Path(plan['staged_path'],'staging-receipt.json').read_text())
supervisor=json.loads((root/'lot-001-stage-monitor/supervisor.json').read_text())
if supervisor['status']!='FINISHED' or supervisor['exit_code']!=0:raise ValueError('Local staging failed')
archive_path=Path(plan['staged_path'],'payload.zip')
if archive_path.stat().st_size!=receipt['payload_bytes'] or receipt['payload_bytes']!=4829918684:
    raise ValueError('Pilot ZIP size changed')
with zipfile.ZipFile(archive_path) as archive:
    infos=archive.infolist(); expected={'rasters/'+f['path']:f for f in lot['files']}
    if len(infos)!=1270 or set(archive.namelist())!=set(expected)|{'lot-inventory.json'}:
        raise ValueError('Staged ZIP central inventory differs')
    for info in infos:
        if info.filename in expected:
            f=expected[info.filename]
            if info.file_size!=f['bytes'] or f'{info.CRC:08x}'!=f['crc32']:
                raise ValueError('Staged ZIP central size/CRC differs')
    if json.loads(archive.read('lot-inventory.json'))!=lot:raise ValueError('ZIP inventory differs')
folder=root/'pilot/cpu-certification/lot-001'
notebook=json.loads((folder/'certify.ipynb').read_text())
metadata=json.loads((folder/'kernel-metadata.json').read_text())
source=''.join(next(c['source'] for c in notebook['cells'] if c['cell_type']=='code'))
tree=ast.parse(source)
embedded=None
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args:
        arg=node.args[0]
        if isinstance(arg,ast.Constant) and isinstance(arg.value,str) and arg.value.startswith('{"number":1'):
            embedded=json.loads(arg.value)
if embedded!=lot:raise ValueError('Notebook embedded canonical manifest differs')
if metadata['dataset_sources']!=[lot['handle']] or metadata['enable_gpu'] or metadata['enable_tpu'] or metadata['enable_internet'] or not metadata['is_private']:
    raise ValueError('Notebook is not one-private-dataset CPU-only')
offline=submit(argparse.Namespace(folder=str(folder),execute=False))
for cell in notebook['cells']:
    if isinstance(cell.get('source'),list):cell['source']=''.join(cell['source'])
req=ApiSaveKernelRequest();req.slug=metadata['id'];req.new_title=metadata['title'];req.text=json.dumps(notebook)
req.language='python';req.kernel_type='notebook';req.is_private=True;req.enable_gpu=False;req.enable_tpu=False
req.enable_internet=False;req.dataset_data_sources=[lot['handle']];req.session_timeout_seconds=4200
serialized=ApiSaveKernelRequest.to_dict(req)
n=len(json.dumps(serialized).encode())
if n>=2_000_000:raise ValueError('Serialized CPU submission exceeds 2 MB control guard')
frozen=Path('reports/recovery_20261107_v3/finalized/code-package/payload.zip')
if hashlib.sha256(frozen.read_bytes()).hexdigest()!='7f942c6cccebe17096064f3a641f1b110a5048d48a73348bcc1cf6dc0aeeb9fe':
    raise ValueError('Historical frozen code changed')
result=dict(status='OFFLINE_PILOT_PREPARATION_VERIFIED',zip_bytes=receipt['payload_bytes'],zip_sha256=receipt['sha256'],
    source_raster_hashes_verified_during_staging=1269,source_raster_pixels_read=0,
    zip_central_directory_sizes_and_crcs_verified=1269,whole_zip_sha256_streamed_by_completed_staging=True,
    second_whole_zip_sha_scan=False,embedded_canonical_manifest_matches=True,private_cpu_only_one_attached_dataset=True,
    cpu_submission_dry_mode=offline,serialized_cpu_request_bytes=n,uploads=0,cloud_jobs=0,
    original_frozen_code_sha256_unchanged=True,
    source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'scripts/prepare_kaggle_single_lot_pilot.py','scripts/kaggle_certify_inputs.py',
        'scripts/publish_kaggle_single_attempt.py','scripts/submit_kaggle_cpu_certification.py',
        'scripts/run_kaggle_phase_guarded.py')})
with (root/'pilot-verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
