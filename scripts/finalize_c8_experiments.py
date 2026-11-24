"""Seal additive C8 preparation after CPU tests; no inference, GPU or pixels."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from geofirex import c8_experiments as c

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/c.REPORT


def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),*args],cwd=ROOT,text=True).strip()


def main():
    if (OUT/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed delivery')
    original=c.read(OUT/'preservation_before.json')['preserved_sha256']
    changed=[n for n,h in original.items() if c.sha256(ROOT/n)!=h]
    if changed:raise ValueError('Protected files changed: '+str(changed))
    base,_,_,lock,_=c.identity(ROOT)
    branch=git('branch','--show-current')
    if branch!='codex/c8-propagation-experiments-20261121':raise ValueError('Wrong branch')
    diff=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    archives=[]
    for row in c.read(ROOT/'reports/c36_feasibility_20261121_v1/c8_archive_verification.json')['archives']:
        p=Path(row['path']);s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']) or c.sha256(p)!=row['sha256']:
            raise ValueError('Official C8 archive changed')
        archives.append(dict(path=str(p),sha256=row['sha256'],unchanged=True))
    maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    pre=c.preflight(ROOT,maps)
    c.save(OUT/'cpu_preflight_final.json',pre)
    raw=Path('D:/GeoFire-X/WildfireSpreadTS/rasters');n=0;total=0
    with (ROOT/'reports/c36_cpu_20261121_v1/geometry_headers.csv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f):
            relative=Path(row['relative_path'])
            if row['role'] not in ('train','val') or relative.parts[0]=='2021':raise ValueError('Forbidden raw path')
            p=(raw/relative).resolve()
            if not p.is_relative_to(raw.resolve()) or not p.is_file() or p.stat().st_size<=0:raise ValueError('Missing raw TIFF')
            n+=1;total+=p.stat().st_size
    if n!=6913:raise ValueError('Raw coverage differs')
    testpath=ROOT/'reports/c8_propagation_tests_v3/tests.xml'
    tests=ET.parse(testpath).getroot();cases=tests.findall('.//testcase')
    if len(cases)!=88 or any(tests.findall('.//'+t) for t in ('failure','error','skipped')):
        raise ValueError('88 executed CPU PASS required')
    classes={}
    for case in cases:classes[case.attrib['classname']]=classes.get(case.attrib['classname'],0)+1
    receipt=dict(status='PREPARATION_CPU_PASS_EXECUTION_NO_GO',branch=branch,head=git('rev-parse','HEAD'),
        protected_files_verified=len(original),protected_changed=changed,
        original_scientific_files=len(c.read(ROOT/c.old.REPORT/'execution_manifest.json')['files']),
        external_archives=archives,raw_TIFFs=dict(present=n,bytes=total,check='stat only; no pixel read'),
        disks={drive:dict(shutil.disk_usage(drive)._asdict()) for drive in ('C:/','D:/')},
        host_ram_total_available=c.old.host_memory(),tests=dict(passed=len(cases),failed=0,skipped=0,
        classes=classes,receipt=testpath.relative_to(ROOT).as_posix(),sha256=c.sha256(testpath)),
        git_diff_check='PASS',git_status=git('status','--short'),
        GPU_preflight_executed=False,training_executed=False,real_inference_executed=False,test_pixels_opened=0,
        remote_Kaggle_availability='UNVERIFIED: eleven metadata HTTPError; sanitized p00 probe HTTP403',
        actual_metric_gain_measured=False)
    c.save(OUT/'verification_finale.json',receipt)
    c.save(OUT/'decision.json',dict(A_preparation='PASS_CPU',A_execution='NO_GO',B_preparation='PASS_CPU',
        B_training='NO_GO',C36='NO_GO_UNCHANGED',remaining=['Explicit separate action authorizations',
        'Authenticated/current eleven Kaggle mounts not qualified',
        'Local available RAM below CPU execution gate at preparation measurements',
        'GPU versions/resources/session/quota preflight not executed'],
        baseline_metrics_changed=False,experiments_executed=False))
    names=set(lock['files'])|{p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()}
    names|={'scripts/prepare_c8_experiments.py','scripts/check_c8_kaggle_metadata.py',
        'scripts/profile_c8_cpu.py','scripts/finalize_c8_experiments.py',
        'notebooks/GEOFIREX_C8_THRESHOLDS_DISABLED.ipynb','notebooks/GEOFIREX_C8_NEWPOS4_DISABLED.ipynb',
        testpath.relative_to(ROOT).as_posix()}
    names|={p.relative_to(ROOT).as_posix() for p in (ROOT/'reports/c8_propagation_tests_v3/tmp').rglob('*.png')}
    c.save(OUT/'delivery_manifest.json',dict(schema='c8-propagation-preparation-delivery-1',
        files={n:c.sha256(ROOT/n) for n in sorted(names)},
        execution_manifest_sha256=c.sha256(OUT/'execution_manifest.json'),
        status='CPU_PASS_RUNTIME_NO_GO',published=False))
    digest=c.sha256(OUT/'delivery_manifest.json')
    with (OUT/'delivery_manifest.sha256').open('x',encoding='ascii') as f:f.write(digest+'  delivery_manifest.json\n')
    print(json.dumps(dict(status=receipt['status'],protected_files=receipt['protected_files_verified'],
        original_scientific_files=receipt['original_scientific_files'],tests=receipt['tests'],raw_TIFFs=n,
        disks=receipt['disks'],host_ram=receipt['host_ram_total_available'],delivery_sha256=digest),indent=2))


if __name__=='__main__':main()
