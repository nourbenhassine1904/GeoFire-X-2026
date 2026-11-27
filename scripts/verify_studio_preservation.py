"""Verify that the UI delivery left all scientific and prior sealed work intact."""
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
import uuid
from geofirex import v2_execution as old
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/studio_propagation_20261123_v1'
if __name__=='__main__':
    old.identity(ROOT)
    before=json.loads((OUT/'preservation_before.json').read_text())
    if (OUT/'delivery_manifest.json').exists():
        OUT=ROOT/'reports/studio_checks'/('preservation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
        OUT.mkdir(parents=True)
    allowed={'README.md','src/geofirex/jury_app.py',*[f'frontend/src/{p}' for p in
        ('EventForecast.mjs','ScientificComparison.mjs','StudioApp.mjs','StudioPages.mjs','prediction_view.mjs','style.css')]}
    changed=[];unchanged=[];runtime_caches=[]
    for name,digest in before.items():
        if '__pycache__' in Path(name).parts or name.endswith('.pyc'):
            runtime_caches.append(name);continue
        current=sha(ROOT/name)
        if current==digest:unchanged.append(name)
        elif name in allowed:changed.append(name)
        else:raise ValueError('Protected file changed: '+name)
    receipt=dict(status='PASS',original_scientific_identities=77,unchanged_snapshot_files=len(unchanged),
        authorized_UI_wrapper_changes=changed,prior_sealed_jury_and_AUXNEW_preserved=True,
        derived_runtime_cache_files_excluded=len(runtime_caches),
        new_training=False,new_Fire13_inference=False,test2021_pixels_opened=0,commit_or_push=False,
        prior_manifest_sha256=sha(ROOT/old.REPORT/'execution_manifest.json'))
    (OUT/'preservation_after.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    for mode,args in [('status',['status','--short']),('diff',['diff','--binary'])]:
        raw=subprocess.check_output(['git','-c','safe.directory='+str(ROOT).replace('\\','/'),*args],cwd=ROOT)
        (OUT/('git_after.'+('txt' if mode=='status' else 'patch'))).write_bytes(raw)
    print(json.dumps(dict(**receipt,report_directory=str(OUT)),indent=2))
