"""Bounded offline contract check with the actual small frozen code archive."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path.cwd()/'scripts'))
from kaggle_certify_inputs import bootstrap_code
from publish_kaggle_single_attempt import check_staged

report=Path(__file__).parent
review=json.loads((report/'frozen-review-final.json').read_text())
frozen=Path('reports/recovery_20261107_v3/finalized/code-package')
pin=review['package']['sha256']; mpin=review['package']['manifest_sha256']
work=bootstrap_code(frozen,report/'bootstrap-zip-final',pin,mpin)
expanded=bootstrap_code(work,report/'bootstrap-expanded-final',pin,mpin)
assert (expanded/'src/geofirex/runner.py').read_bytes() == (work/'src/geofirex/runner.py').read_bytes()
payload,receipt=check_staged(frozen)
assert receipt['sha256']==pin
sources={}
for file in [*Path('scripts').glob('*kaggle*prepublication*.py'), Path('scripts/kaggle_certify_inputs.py'),
             Path('scripts/publish_kaggle_single_attempt.py'), Path('scripts/run_kaggle_phase_guarded.py'),
             Path('scripts/prepare_kaggle_certification_notebooks.py'), Path('scripts/submit_kaggle_cpu_certification.py'),
             Path('tests/test_kaggle_prepublication_small.py')]:
    ast.parse(file.read_text(encoding='utf-8'))
    sources[file.as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
notebooks=[]
for file in sorted((report/'cpu-certification-v3').glob('*/certify.ipynb')):
    nb=json.loads(file.read_text()); metadata=json.loads((file.parent/'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is False and metadata['enable_internet'] is False
    assert len(metadata['dataset_sources']) == (1 if file.parent.name=='code' else (9 if file.parent.name=='all' else 2))
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    notebooks.append(dict(path=file.as_posix(),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),executed=False))
assert len(notebooks)==10
versions={}
for name in ('numpy','rasterio','torch','mlflow','pyyaml','matplotlib','scikit-learn','pyarrow','psutil'):
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
disk={'workspace':shutil.disk_usage('.')._asdict()}
if Path('D:/GeoFire-X/WildfireSpreadTS/rasters').is_dir():
    disk['historical_raw_volume']=shutil.disk_usage('D:/GeoFire-X/WildfireSpreadTS/rasters')._asdict()
result=dict(status='LOCAL_PREPARATION_CONTRACT_PASSED',actual_frozen_zip_bootstrap=True,expanded_bootstrap=True,
    local_distribution_metadata=versions,source_sha256=sources,notebooks=notebooks,disk=disk,
    original_payload_sha256=pin,raw_rasters_read=0,cloud_certifications_executed=0,uploads=0)
with (report/'local-preparation-final.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
