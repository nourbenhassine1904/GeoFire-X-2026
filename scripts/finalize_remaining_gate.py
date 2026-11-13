"""Preserve failed gate and prepare corrected CPU gate; no network or rasters."""
import json
from pathlib import Path
import shutil
import yaml

from prepare_remaining_kaggle import make_notebook,training_notebooks

out=Path('reports/kaggle_remaining_20261108_v1')
plan_path=out/'plan.json'
with plan_path.open('rb') as src,(out/'plan-before-environment-fix.json').open('xb') as dest:
    shutil.copyfileobj(src,dest)
plan=json.loads(plan_path.read_text())
source=("import importlib,importlib.util,json,os\nfrom pathlib import Path\n"
    "os.environ['CUDA_VISIBLE_DEVICES']=''\n"
    "if list(Path('/dev').glob('nvidia[0-9]*')): raise RuntimeError('CPU session required')\n"
    "for name in ('numpy','rasterio','torch','psutil','pyarrow','yaml'): importlib.import_module(name)\n"
    "Path('/kaggle/working/environment.json').write_text(json.dumps({'status':'CPU_AUTOMATION_READY','gpu':False,'training':False,'datasets':[],"
    "'mlflow_available':importlib.util.find_spec('mlflow') is not None,'tracking':'runner_json_csv_checkpoints'}))\n")
plan['canary']=make_notebook(out/'cpu-certification/canary-v2',source,'GeoFireX CPU Automation Gate 20261108 V2',[])
plan['cpu_api_note']='SaveKernel accepted HTTP 200. First CPU gate exposed missing optional mlflow; new run copies use existing mlflow_enabled=False support and retain metrics/checkpoints.'
for path in (out/'training/bundle/configs').glob('*.yaml'):
    config=yaml.safe_load(path.read_text());config['mlflow_enabled']=False
    path.write_text(yaml.safe_dump(config,sort_keys=False))
plan_path.write_text(json.dumps(plan,indent=2))
training_notebooks(out,plan)
print(json.dumps(dict(status='CPU_GATE_FIXED_LOCALLY',canary=plan['canary'],mlflow_enabled=False,training=False),indent=2))
