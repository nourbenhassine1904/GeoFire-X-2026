"""Final additive delivery. Never rewrite a previously sealed report."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from geofirex import v2_execution as old
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'reports/studio_propagation_20261123_v1'
OUT=ROOT/'reports/studio_propagation_20261123_v2'
if __name__=='__main__':
    if (OUT/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed final delivery')
    old.identity(ROOT)
    previous=json.loads((PRIOR/'delivery_manifest.json').read_text())
    for name,digest in previous['files'].items():
        assert sha(PRIOR/name)==digest,'Prior sealed evidence changed: '+name
    ui=list((ROOT/'frontend/src').rglob('*'))
    assert all(sha(p)==sha(PRIOR/'kit'/p.relative_to(ROOT)) for p in ui if p.is_file())
    assert sha(ROOT/'frontend/dist/app.js')==sha(PRIOR/'kit/frontend/dist/app.js')
    assert sha(ROOT/'frontend/dist/style.css')==sha(PRIOR/'kit/frontend/dist/style.css')
    # Explicitly attempt a forbidden report target. The runner must reject before writes.
    environment=dict(os.environ,GEOFIREX_TEST_REPORT_DIR=str(PRIOR),GEOFIREX_TEST_URL='http://127.0.0.1:8027')
    rejected=subprocess.run(['node','verify-studio-propagation.mjs'],cwd=ROOT/'frontend',env=environment,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'Refusing to overwrite a sealed delivery' in rejected.stderr
    for name,digest in previous['files'].items():assert sha(PRIOR/name)==digest
    (OUT/'sealed-report-guard.json').write_text(json.dumps(dict(status='PASS',expected_refusal=True,prior_files_unchanged=True,
        exit_code=rejected.returncode,reason='Refusing to overwrite a sealed delivery'),indent=2))
    counts=[]
    for name in ('existing-tests.xml','spatial-tests-pass.xml','report-guard-final.xml'):
        suites=list(ET.parse(OUT/name).getroot().iter('testsuite'))
        assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
        counts.append(sum(int(s.attrib['tests']) for s in suites))
    assert counts[:2]==[13,8]
    dom=json.loads((OUT/'propagation-dom.json').read_text());assert dom['status']=='PASS' and len(dom['checks'])==23
    checks=[]
    for name in ('/health','/','/app.js','/style.css','/studio/overview','/science/v2/comparison','/science/v2/scenes'):
        with urlopen('http://127.0.0.1:8028'+name,timeout=60) as response:
            raw=response.read();assert response.status==200
            checks.append(dict(path=name,status=200,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    assert next(c['sha256'] for c in checks if c['path']=='/app.js')==sha(ROOT/'frontend/dist/app.js')
    (OUT/'launch-http-final.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
    # Reuse unchanged visual evidence, and snapshot only the current whitelisted sources.
    for p in [*PRIOR.glob('*.png'),PRIOR/'browser-audit.json',PRIOR/'visualization_sources.json',PRIOR/'visualization_manifest.json',
              PRIOR/'preservation_before.json',PRIOR/'preservation_after.json',PRIOR/'historical-scene-metrics.json',PRIOR/'V2-demo-scene.json']:
        shutil.copy2(p,OUT/p.name)
    kit=OUT/'kit';kit.mkdir()
    for p in (PRIOR/'kit').rglob('*'):
        if p.is_file():
            relative=p.relative_to(PRIOR/'kit');source=ROOT/relative
            target=kit/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    extra=kit/'scripts/finalize_studio_safe.py';shutil.copy2(Path(__file__),extra)
    delivery=json.loads((ROOT/'reports/studio_propagation_20261123_v1_delivery.json').read_text())
    archive=Path(delivery['archive']);old.verify_zip(archive,delivery['sha256'])
    evidence=OUT/'previous_verified_delivery';evidence.mkdir();shutil.copy2(archive,evidence/archive.name)
    shutil.copy2(archive.with_suffix('.sha256'),evidence/archive.with_suffix('.sha256').name)
    for name,args in [('git_after.txt',['status','--short']),('git_after.patch',['diff','--binary'])]:
        (OUT/name).write_bytes(subprocess.check_output(['git','-c','safe.directory='+str(ROOT).replace('\\','/'),*args],cwd=ROOT))
    manifest=dict(status='STUDIO_FINAL_VERIFIED',unique_CPU_API_tests=21,frontend_controls=23,
        prior_sealed_files_verified=len(previous['files']),prior_delivery_sha256=delivery['sha256'],
        UI_identical_to_native_audited_version=True,sealed_report_refusal_verified=True,original_scientific_identities=77,
        new_training=False,new_V2_inference=False,test2021_pixels_opened=0,commit_or_push=False,
        files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and not any(x.startswith('pytest-') for x in p.relative_to(OUT).parts)})
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    bundle=ROOT/'reports/studio_propagation_20261123_v2_backup';bundle.mkdir()
    for p in OUT.rglob('*'):
        if not p.is_file() or any(x.startswith('pytest-') for x in p.relative_to(OUT).parts):continue
        target=bundle/p.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    final=old.package(bundle,'STUDIO-FINAL-VERIFIED')
    receipt=dict(status='VERIFIED',archive=str(final),sha256=sha(final),CPU_API_tests=21,frontend_controls=23,
        report=str(OUT/'RAPPORT_FINAL.txt'),procedure=str(ROOT/'docs/STUDIO_PROPAGATION_JURY_20261202.txt'))
    (ROOT/'reports/studio_propagation_20261123_v2_delivery.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt,indent=2))
