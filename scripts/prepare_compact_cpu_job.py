"""Prepare the small private CPU certificate kernel; no network or GPU."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

BASE=Path('reports/gpu_acceleration_20261111_v1')
plan=json.loads((BASE/'publication/plan.json').read_text())
expected=[]
for part in plan['parts']:
    folder=Path(part['folder'])
    files={f['alias']:dict(bytes=f['bytes'],sha256=f['sha256']) for f in part['files']}
    for name in ('mount-map.json','publication_identity.json'):
        p=folder/name
        files[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    expected.append(dict(handle=part['handle'],identity=part['identity'],files=files))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in Path('src/geofirex').glob('*.py'):z.write(p,p.as_posix())
    for name in ('compact_publication.py','certify_compact_runtime.py'):
        z.write(Path('scripts')/name,name)
    z.writestr('expected.json',json.dumps(expected))
payload=base64.b64encode(buf.getvalue()).decode()
code='''import base64,io,json,zipfile,sys
from pathlib import Path
if __name__=='__main__':
    workspace=Path('/kaggle/working/compact-certification');workspace.mkdir()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(workspace)
    sys.path[:0]=[str(workspace/'src'),str(workspace)]
    from certify_compact_runtime import run
    run(json.loads((workspace/'expected.json').read_text()),workspace)
'''.replace('PAYLOAD',repr(payload))
compile(code,'cpu-certification','exec')
assert len(code.encode())<1_000_000
out=BASE/'compact_cpu_job';out.mkdir(exist_ok=True)
(out/'job.py').write_text(code,encoding='utf-8')
(out/'metadata.json').write_text(json.dumps(dict(
    id='nnnnbhss/geofirex-compact-cpu-certificate-20261112',
    dataset_sources=[p['handle'] for p in plan['parts']],
    private=True,gpu=False,timeout_seconds=1200,
    code_sha256=hashlib.sha256(code.encode()).hexdigest()),indent=2),encoding='utf-8')
print('CPU certificate kernel prepared',len(code.encode()),'bytes; no network')
