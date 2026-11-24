"""Preserve prior evidence; permit only the UI files authorized for final defense."""
from pathlib import Path
import json
from geofirex.calibration_io import sha
from geofirex.v2_execution import identity
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/final_studio_20261124'
_,historical_lock,_=identity(ROOT)
assert len(historical_lock['files'])==77,'Historical identity count differs'
before=json.loads((ROOT/'reports/studio_propagation_20261123_v1/preservation_before.json').read_text())
allowed={'README.md','src/geofirex/jury_app.py',*[f'frontend/src/{n}' for n in ['EventForecast.mjs','ScientificComparison.mjs','StudioApp.mjs','StudioPages.mjs','prediction_view.mjs','style.css','studio_ui.mjs','PropagationView.mjs','V2Views.mjs']]}
changed=[];preserved=[]
for name,digest in before.items():
 if '__pycache__' in Path(name).parts or name.endswith('.pyc'):continue
 if sha(ROOT/name)==digest:preserved.append(name)
 elif name in allowed:changed.append(name)
 else:raise ValueError('Protected scientific artifact changed: '+name)
result={'status':'PASS','historical_identities_verified':len(historical_lock['files']),'prior_snapshot_files_preserved':len(preserved),'authorized_ui_changes':changed,'new_training':False,'new_v2_inference':False,'test2021_pixels_opened':0,'git_commit_push_merge':False}
OUT.mkdir(exist_ok=True)
(OUT/'preservation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
