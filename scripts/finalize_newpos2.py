"""Seal this new preparation after targeted CPU checks; never runs GPU or training."""
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from geofirex import newpos2_experiment as c

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/c.REPORT
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Sealed delivery must be preserved')
    protected=c.read(out/'preservation_before.json')['files']
    c.old.verify_manifest(ROOT,protected)
    # Also compare with the first draft snapshot, excluding files created by this mission.
    initial=list((out/'draft_revisions').glob('*/reports/newpos2_20261122_v1/preservation_before.json'))
    for p in initial:
        original=c.read(p)['files']
        c.old.verify_manifest(ROOT,{n:h for n,h in original.items() if n in protected})
    _,config,_,lock,_=c.identity(ROOT)
    c.old.verify_manifest(ROOT,c.read(ROOT/c.old.REPORT/'execution_manifest.json')['files'])
    receipts=[]
    for name,expected in [('reports/newpos2_tests_20261122_v2/tests.xml',37),
                          ('reports/newpos2_regression_20261122_v1/tests.xml',9)]:
        tree=ET.parse(ROOT/name).getroot()
        if len(tree.findall('.//testcase'))!=expected or any(tree.findall('.//'+k) for k in ('failure','error','skipped')):
            raise ValueError('Targeted CPU suite must fully pass: '+name)
        receipts.append(dict(path=name,passed=expected,sha256=c.sha256(ROOT/name)))
    maps=c.read(out/'cpu_mount_preflight.json')
    if maps['loader_lengths']!=[115900,30882] or maps['gpu_checked'] or maps['mounts']['objects']!=459:
        raise ValueError('CPU mounting receipt differs')
    git=['git','-c','safe.directory='+ROOT.as_posix()]
    subprocess.run(git+['diff','--check'],cwd=ROOT,check=True,capture_output=True,text=True)
    branch=subprocess.check_output(git+['branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='codex/newpos2-20261122':raise ValueError('Separate NEWPOS2 branch required')
    kit=out/'kit';kit.mkdir(exist_ok=False)
    for name in ['notebooks/GEOFIREX_C8_NEWPOS2_DISABLED.ipynb','docs/NEWPOS2_LANCEMENT.md',
                 'scripts/run_newpos2.py','scripts/prepare_newpos2.py','scripts/finalize_newpos2.py']:
        target=kit/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as dst,(ROOT/name).open('rb') as src:shutil.copyfileobj(src,dst)
    for r in receipts:
        with (out/Path(r['path']).parent.name).with_suffix('.xml').open('xb') as dst,(ROOT/r['path']).open('rb') as src:
            shutil.copyfileobj(src,dst)
    c.save(out/'decision.json',dict(run_id=c.B_ID,CPU_qualification='GO',
        current_GPU_launch='NO_GO_PENDING_KAGGLE_PREFLIGHT_AND_EXPLICIT_AUTHORIZATION',
        allowed_future_fits=1,conditions=['Eleven actual Kaggle mounts PASS','Historical T4 environment/resources PASS',
            'Current session >=225 minutes and quota/reserve >=7.5 card-hours',
            'Separate explicit NEWPOS2 GPU/action authorization','Fresh output and exclusive single-run lock'],
        loss_factor=2,threshold=.78,fixed_checkpoint_epoch=4,acceptance_policy_sha256=c.sha256(out/'acceptance.json'),
        no_resume=True,training_executed=False,GPU_preflight_executed=False,test_pixels_opened=0,
        automatic_promotion=False,committed=False,pushed=False))
    c.save(out/'verification_finale.json',dict(branch=branch,tests=receipts,total_CPU_tests_passed=46,
        protected_files_verified=len(protected),protected_changed=[],original_scientific_identities_verified=77,
        exact_scientific_change='New-positive BCE weight 2; historical Dice unchanged',
        local_mounts=dict(objects=459,bytes=maps['mounts']['bytes'],check='Pinned identities and sizes; no pixel read'),
        target_mask_pixel_parity='Synthetic CPU fixture; real decoder/index hashes unchanged; full evaluation domain checked at inference',
        GPU_executed=False,training_executed=False,test_pixels_opened=0,git_diff_check='PASS'))
    names=set(lock['files'])|{p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
    names.update(['notebooks/GEOFIREX_C8_NEWPOS2_DISABLED.ipynb','docs/NEWPOS2_LANCEMENT.md','scripts/finalize_newpos2.py'])
    c.save(out/'delivery_manifest.json',dict(schema='newpos2-cpu-delivery-1',
        files={n:c.sha256(ROOT/n) for n in sorted(names)},status='CPU_GO_GPU_LAUNCH_PENDING',
        notebook_sha256=c.sha256(ROOT/'notebooks/GEOFIREX_C8_NEWPOS2_DISABLED.ipynb')))
    with (out/'delivery_manifest.sha256').open('x',encoding='ascii') as f:
        f.write(c.sha256(out/'delivery_manifest.json')+'  delivery_manifest.json\n')
    archive=c.old.package(out,'PREPARATION-CPU-PASS')
    print(json.dumps(dict(CPU='GO',GPU_launch='NO_GO_PENDING_PREFLIGHT_AND_AUTHORIZATION',
         tests_passed=46,protected_files=len(protected),scientific_identities=77,archive=str(archive)),indent=2))


if __name__=='__main__':main()
