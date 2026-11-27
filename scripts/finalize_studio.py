"""Seal verified UI evidence and a local overlay backup, without Git mutation."""
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from geofirex import v2_execution as old
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/studio_propagation_20261123_v1'
if __name__=='__main__':
    if (OUT/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed delivery')
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/verify_studio_preservation.py')],check=True)
    counts=[]
    for name in ('existing-tests.xml','spatial-tests.xml'):
        suites=list(ET.parse(OUT/name).getroot().iter('testsuite'))
        assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
        counts.append(sum(int(s.attrib['tests']) for s in suites))
    dom=json.loads((OUT/'propagation-dom.json').read_text());theme=json.loads((OUT/'theme_contract.json').read_text())
    browser=json.loads((OUT/'browser-audit.json').read_text())
    assert dom['status']=='PASS' and len(dom['checks'])==23 and theme['status']=='PASS' and browser['status']=='PASS_OBSERVED'
    checks=[]
    for name in ('/health','/','/app.js','/style.css','/studio/overview','/science/v2/comparison','/science/v2/scenes'):
        with urlopen('http://127.0.0.1:8028'+name,timeout=60) as response:
            raw=response.read();assert response.status==200
            import hashlib
            checks.append(dict(path=name,status=200,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    assert next(c['sha256'] for c in checks if c['path']=='/app.js')==sha(ROOT/'frontend/dist/app.js')
    assert next(c['sha256'] for c in checks if c['path']=='/style.css')==sha(ROOT/'frontend/dist/style.css')
    (OUT/'launch-http-final.json').write_text(json.dumps(dict(status='PASS',command='.\\Start-GeoFireX.ps1 -Port 8028',checks=checks),indent=2))
    # Overlay source snapshot; data and trained weights remain in their original locations.
    kit=OUT/'kit';kit.mkdir()
    paths=[ROOT/'Start-GeoFireX.ps1',ROOT/'README.md',ROOT/'docs/STUDIO_PROPAGATION_JURY_20261202.txt',
        ROOT/'frontend/package.json',ROOT/'frontend/package-lock.json',ROOT/'frontend/build.mjs',ROOT/'frontend/index.html',
        ROOT/'frontend/verify-studio-propagation.mjs',ROOT/'frontend/tests/propagation-dom.mjs',
        ROOT/'frontend/tests/theme-contract.mjs',ROOT/'frontend/tests/layout-v1.json',
        *[ROOT/'src/geofirex'/n for n in ('jury_app.py','studio_spatial.py')],
        *[ROOT/'scripts'/n for n in ('serve_jury.py','preserve_studio.py','prepare_studio_sources.py','verify_studio_preservation.py','finalize_studio.py')],
        ROOT/'tests/test_studio_spatial.py']
    paths += [p for folder in ('frontend/src','frontend/dist') for p in (ROOT/folder).rglob('*') if p.is_file()]
    for p in paths:
        target=kit/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and not any(part.startswith('pytest-') for part in p.relative_to(OUT).parts)}
    manifest=dict(status='STUDIO_VERIFIED',CPU_API_tests=sum(counts),frontend_controls=len(dom['checks']),
        theme_geometry_rules=theme['preserved_geometry_rules'],theme_contrast_pairs=len(theme['contrast_pairs']),
        browser_desktop_mobile='PASS_OBSERVED; Chrome 400px emulation, not physical phone',
        original_identities=77,training_executed=False,new_V2_inference=False,test2021_pixels_opened=0,
        commit_or_push=False,files=files,backup_kind='overlay for existing checkout; corpus/checkpoints excluded')
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    # Exclude test scratch directories through the package helper's supported name.
    # Leave all scratch and prior work in place; package only the copied verified delivery.
    bundle=OUT.parent/'studio_propagation_20261123_v1_backup';bundle.mkdir()
    for p in OUT.rglob('*'):
        if not p.is_file() or any(part.startswith('pytest-') for part in p.relative_to(OUT).parts):continue
        target=bundle/p.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    archive=old.package(bundle,'STUDIO-VERIFIED')
    receipt=dict(status='VERIFIED',archive=str(archive),sha256=sha(archive),CPU_API_tests=sum(counts),frontend_controls=len(dom['checks']),
        report=str(OUT/'RAPPORT_STUDIO.txt'),procedure=str(ROOT/'docs/STUDIO_PROPAGATION_JURY_20261202.txt'))
    (OUT.parent/'studio_propagation_20261123_v1_delivery.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt,indent=2))
