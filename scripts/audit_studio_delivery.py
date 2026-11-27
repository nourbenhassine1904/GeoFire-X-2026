"""Read-only Git/science inventory and a source-only handoff archive. No Git writes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from geofirex.calibration_io import sha
from geofirex.jury_app import ROOT,evidence
from geofirex.studio_v2 import archive_path,QUALIFICATION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    frontend=json.loads((output/'frontend-final-pass/propagation-dom.json').read_text(encoding='utf8'))
    browser=json.loads((output/'browser/browser-audit.json').read_text(encoding='utf8'))
    suites=list(ET.parse(output/'backend-final.xml').iter('testsuite'))
    backend={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    preserved=json.loads((output/'preservation-after.json').read_bytes())
    assert frontend['status']=='PASS' and browser['status'].startswith('PASS')
    assert backend['tests']>0 and backend['failures']==backend['errors']==0
    assert preserved['status']=='PASS_ACCESSIBLE_FILES' and not preserved['changed']
    def git(*args):
        return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),*args],cwd=ROOT,text=True,encoding='utf8').strip()
    tracked=git('diff','--name-only').splitlines();untracked=git('ls-files','--others','--exclude-standard').splitlines()
    sizes=[dict(path=name,bytes=(ROOT/name).stat().st_size) for name in untracked if (ROOT/name).is_file()]
    audit=dict(branch=git('branch','--show-current'),head=git('rev-parse','HEAD'),
        upstream=git('rev-parse','@{u}'),ahead_behind_local_tracking=git('rev-list','--left-right','--count','HEAD...@{u}'),
        tracked_changes=tracked,untracked=untracked,untracked_count=len(untracked),
        largest_untracked=sorted(sizes,key=lambda r:r['bytes'],reverse=True)[:30],git_mutations=False)
    (output/'git-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    q=json.loads(QUALIFICATION.read_text(encoding='utf8'));models=[]
    for source in json.loads(evidence('comparison.json'))['sources']:
        path=archive_path(source)
        if sha(path)!=source['archive_sha256']:raise ValueError('Archive identity differs: '+source['model'])
        with zipfile.ZipFile(path) as z:
            raw=z.read('checkpoint_u0028976.pt')
            if hashlib.sha256(raw).hexdigest()!=source['checkpoint_sha256']:raise ValueError('Checkpoint differs')
            evaluation=json.loads(z.read(source['result_member']))
            if sha_bytes(z.read(source['result_member']))!=source['result_sha256']:raise ValueError('Official result differs')
        models.append(dict(**source,qualification=q['models'][source['model']],
                           official_pooled=evaluation['pooled'],official_new_fire=evaluation['new_fire']))
    (output/'model-inventory.json').write_text(json.dumps(models,indent=2),encoding='utf8')
    legacy=json.loads((ROOT/'configs/prediction_lab_v1.json').read_text())
    index=json.loads((ROOT/'reports/protocol_v2_20261118/index/index.json').read_text())
    before=json.loads((output/'preservation-before.json').read_text(encoding='utf8'))
    external=dict(model_archives=[{k:m[k] for k in ('model','archive','archive_sha256','checkpoint_sha256')} for m in models],
        legacy_pinned_files=legacy['pinned_files'],legacy_compact=legacy['corpus'],
        v2_validation_compact=[dict(logical=logical,**row) for logical,row in index['routes'].items()
            if index['event_roles'][index['events'][logical]]=='val'],
        results_2021_copies={name:row for name,row in before['files'].items() if name.endswith('final-2021-01-RESULTS.zip')},
        environment_variables=['GEOFIREX_V2_ROOT','GEOFIREX_COMPACT_ROOT','GEOFIREX_RESULTS_ARCHIVE'],
        note='Restore exact existing artifacts; never rebuild data, refit moments or rerun 2021 to install the app.')
    (output/'external-dependencies.json').write_text(json.dumps(external,indent=2),encoding='utf8')
    files={ROOT/p for p in ('README.md','pyproject.toml','Start-GeoFireX.ps1','requirements-studio-cpu.txt','feu.png',
        'frontend/package.json','frontend/package-lock.json','frontend/build.mjs','frontend/index.html',
        'frontend/verify-studio-propagation.mjs','frontend/verify-non-regression-browser.mjs')}
    for directory,pattern in [('src/geofirex','*.py'),('frontend/src','*'),('frontend/tests','*'),('configs','*.json')]:
        files.update(p for p in (ROOT/directory).rglob(pattern) if p.is_file())
    files.update(ROOT/p for p in tracked if p.startswith(('scripts/','tests/')) and (ROOT/p).is_file())
    for directory in ('scripts','tests'):
        files.update(p for p in (ROOT/directory).glob('*studio*.py') if p.is_file())
    files.update((ROOT/'reports/jury_20261202_v1').glob('*.json'))
    files.update(p for p in (ROOT/'reports/protocol_v2_20261118/index').glob('*') if p.suffix in ('.json','.npy'))
    files.update(ROOT/p for p in ('reports/fire13_20261122_v1/scene_geometry.json','reports/fire13_20261122_v1/derived_normalization.json','reports/fire13_20261122_v1/execution_manifest.json'))
    files.add(QUALIFICATION)
    files.update(p for p in (ROOT/'docs').glob('*FINALISATION*') if p.is_file())
    files.add(ROOT/'docs/DEPLOYMENT_STUDIO_V2.md')
    manifest={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)}
    (output/'application-files.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    # This is an update for a checkout plus its protected external artifacts.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=output/('GeoFireX-APPLICATION-'+stamp+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('APPLICATION_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('EXTERNAL_DEPENDENCIES.json',json.dumps(external,indent=2))
    with zipfile.ZipFile(archive) as z:
        for name,row in manifest.items():
            if sha_bytes(z.read(name))!=row['sha256']:raise ValueError('Packaged member differs')
    result=dict(status='PASS',archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,source_files=len(files),members=len(files)+2,
                backend=backend,frontend_checks=len(frontend['checks']),browser=browser['status'],protected_files=preserved['verified'],
                kind='source update; existing checkout and external data/results required; not a fresh-machine certification',
                git_mutations=False,training_executed=False,test_2021_pixels_read=False)
    (output/'application-package.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(result,git_untracked=len(untracked))))


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
if __name__=='__main__':main()
