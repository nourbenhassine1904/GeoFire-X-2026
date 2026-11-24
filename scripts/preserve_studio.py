"""Additive snapshot before UI work. Does not touch Git configuration or artifacts."""
import json
from pathlib import Path
import shutil
import subprocess
from geofirex.calibration_io import sha
from geofirex import v2_execution as old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/studio_propagation_20261123_v1'
if __name__=='__main__':
    if OUT.exists():raise FileExistsError('Preserve existing snapshot')
    old.identity(ROOT)
    OUT.mkdir()
    paths=set()
    for folder in ('frontend/src','src/geofirex','configs','notebooks','reports/jury_20261202_v1','reports/auxnew_20261123_v1'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    paths.update([ROOT/'README.md',ROOT/'frontend/build.mjs',ROOT/'scripts/serve_jury.py'])
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(paths)}
    (OUT/'preservation_before.json').write_text(json.dumps(files,indent=2),encoding='utf8')
    for p in (ROOT/'frontend/src').rglob('*'):
        if p.is_file():
            dest=OUT/'previous-ui'/p.relative_to(ROOT/'frontend/src');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for mode,args in [('status',['status','--short']),('diff',['diff','--binary'])]:
        raw=subprocess.check_output(['git','-c','safe.directory='+str(ROOT).replace('\\','/'),*args],cwd=ROOT)
        (OUT/('git_before.'+('txt' if mode=='status' else 'patch'))).write_bytes(raw)
    print('Snapshot preserved:',len(files),'files; original 77 identities verified')
