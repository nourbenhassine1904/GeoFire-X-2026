"""Bounded offline assertions for the prepared CPU certification notebook."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import zlib

base=Path('reports/recovery_20261107_v3/finalized/deployment/bundle/metadata')
n=json.loads((base/'normalization.json').read_text())
print('NORMALIZATION_KEYS',list(n), 'NORMALIZER_KEYS',list(n.get('normalization',{})))
for module in ('runner_data','benchmark','corpus','remote_archive','unet_data','windows','real_constants','__init__'):
    tree=ast.parse(Path(f'src/geofirex/{module}.py').read_text(encoding='utf-8'))
    print('IMPORTS',module,[x.module for x in tree.body if isinstance(x,ast.ImportFrom)])
import kagglesdk
print('SDK_AVAILABLE',True)
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
req=ApiSaveKernelRequest()
print('SDK_SESSION_TIMEOUT_SUPPORTED',hasattr(req,'session_timeout_seconds'))
notebook=json.loads(Path('reports/kaggle_certification_20261108_v1/cpu-certification/lot-001/certify.ipynb').read_text())
tree=ast.parse(''.join(notebook['cells'][1]['source']))
blob=None
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in node.targets):
        blob=node.value.args[0].args[0].args[0].value
assert isinstance(blob,str)
payload=json.loads(zlib.decompress(base64.b64decode(blob)))
lot=payload['reference']
assert hashlib.sha256(json.dumps({k:v for k,v in lot.items() if k!='identity'},sort_keys=True,separators=(',',':')).encode()).hexdigest()==lot['identity']
assert {r['relative_path'] for r in payload['observations']}=={f['path'] for f in lot['files']}
index=json.loads((base/'index.json').read_text())
indexed={e['path']:e for e in index['entries']}
for f in lot['files']:
    assert (f['bytes'],f['crc32'])==(indexed[f['path']]['geotiff_bytes'],indexed[f['path']]['crc32'])
assert {e['path'] for e in index['entries'] if e['event_id'] in lot['events']}=={f['path'] for f in lot['files']}
sys.path.insert(0,str(Path('src').resolve()))
from geofirex.benchmark import aligned_windows
groups,rejected=aligned_windows(payload['observations'],payload['split'],'train',contexts=(1,3,5))
assert all(groups[t] for t in (1,3,5))
assert set(n['normalization']) == {'continuous','mean','std','finite_count','fit_paths','include_landcover','clip'}
result=dict(status='OFFLINE_CHECKS_PASSED_REMOTE_UNCERTIFIED',files=1269,events=57,
    canonical_identity=lot['identity'],manifest_json_bytes=(base/'lot-001.json').stat().st_size,
    manifest_top_level_keys=len(lot),manifest_files=len(lot['files']),
    complete_events_against_global_index=True,sizes_and_crc_match_global_index=True,
    aligned_windows={str(k):len(v) for k,v in groups.items()},rejections=rejected)
Path('reports/kaggle_certification_20261108_v1/offline-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
