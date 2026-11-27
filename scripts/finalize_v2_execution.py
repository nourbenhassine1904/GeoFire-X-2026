"""CPU-only delivery seal and local PREPARATION backup; no GPU/remote actions."""
from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET
from geofirex import v2_execution as v2

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/v2.REPORT

def main():
    v2.verify_manifest(ROOT,v2.read(OUT/'split_lock.json')['protected_sha256'])
    v2.identity(ROOT)
    suite=ET.parse(OUT/'tests-final.xml').getroot().find('testsuite')
    assert suite is not None and suite.attrib['failures']==suite.attrib['errors']=='0'
    status=dict(technical_decision='READY',launch_decision='BLOCKED',training_authorized=False,
        launch_enabled=False,split='V2_2020-08-23',run_id='B2_C8_V2_seed42',cpu_tests=int(suite.attrib['tests']),
        test_seconds=float(suite.attrib['time']),gpu_checks_executed=False,training_executed=False,
        test_pixels_opened=0,causes=['Explicit authorization for fit/budget remains absent',
         'Current Kaggle versions/T4/quota/RAM/disk/mount preflight not executed'],
        external_volume_observed=dict(path='D:\\GeoFire-X',label='ADATA HV300',serial='4A0E-CF27'),
        new_model_metrics=None,historical_artifacts_modified=False)
    v2.save(OUT/'decision.json',status)
    v2.save(OUT/'gpu_preflight.json',dict(status='BLOCKED_NOT_EXECUTED',gpu_used=False,
        reason='Mission explicitly forbids GPU; notebook preflight must run in a future authorized session',
        cpu_identity_pass=True,budget=v2.read(OUT/'budget.json'),launch_enabled=False))
    files=[p for p in OUT.rglob('*') if p.is_file()]
    files += [ROOT/'notebooks/GEOFIREX_V2_DISABLED.ipynb',ROOT/'tests/test_v2_execution.py']
    files += [ROOT/'scripts'/name for name in ('run_v2_baseline.py','lock_v2_execution.py',
        'package_v2_execution.py','verify_v2_execution_cpu.py','finalize_v2_execution.py','backup_v2_preparation.py')]
    files += [ROOT/'src/geofirex/v2_execution.py',ROOT/'src/geofirex/v2_propagation.py']
    files=sorted(set(files))
    manifest=dict(schema='v2-execution-delivery-1',files={p.relative_to(ROOT).as_posix():v2.sha256(p) for p in files},
                  operation='PREPARATION only; no trained checkpoint or invented score')
    v2.save(OUT/'delivery_manifest.json',manifest)
    (OUT/'delivery_manifest.sha256').write_text(v2.sha256(OUT/'delivery_manifest.json')+'\n',encoding='ascii')
    stage=ROOT/'artifacts/v2_execution/preparation'; stage.mkdir(parents=True,exist_ok=False)
    for p in files+[OUT/'delivery_manifest.json',OUT/'delivery_manifest.sha256']:
        target=stage/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    archive=v2.package(stage,'PREPARATION')
    v2.save(OUT/'preparation_backup_receipt.json',dict(local_archive=str(archive),sha256=v2.sha256(archive),
        all_members_verified=True,external_copy_status='PENDING_COPY',contents='Preparation; no real model checkpoint'))
    print(archive)

if __name__=='__main__': main()
