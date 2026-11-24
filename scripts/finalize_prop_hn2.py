"""Seal the single PROP-HN2 preparation after CPU PASS; no GPU or publication."""
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from geofirex import prop_hn2_experiment as c

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/c.REPORT
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed delivery')
    protected=c.read(out/'preservation_before.json')['files']
    c.old.verify_manifest(ROOT,protected)
    base,config,_,lock,_=c.identity(ROOT)
    original=c.read(ROOT/c.old.REPORT/'execution_manifest.json')['files']
    c.old.verify_manifest(ROOT,original)
    if len(original)!=77:raise ValueError('77 original identities required')
    tests='reports/prop_hn2_tests_20261122_v1/tests.xml'
    tree=ET.parse(ROOT/tests).getroot();cases=tree.findall('.//testcase')
    if len(cases)!=46 or any(tree.findall('.//'+k) for k in ('failure','error','skipped')):
        raise ValueError('46 fully executed targeted CPU PASS required')
    archives=c.read(out/'newpos2_archive_verification.json')
    for row in archives['archives']:
        p=Path(row['path']);s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']) or c.sha256(p.with_suffix('.sha256'))!=row['sidecar_sha256']:
            raise ValueError('Verified immutable NEWPOS2 archive/sidecar changed')
    qualification=c.read(out/'train_category_qualification.json')
    git=['git','-c','safe.directory='+ROOT.as_posix()]
    subprocess.run(git+['diff','--check'],cwd=ROOT,capture_output=True,text=True,check=True)
    branch=subprocess.check_output(git+['branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='codex/prop-hn2-20261122':raise ValueError('Separate PROP-HN2 branch required')
    c.save(out/'scientific_options.json',dict(
        C8_plus_wind=dict(status='NO_GO_CAUSAL',hypothesis='Directional ranking of spread',
            missing=['Per-asset issue/lead/valid times','Actual availability by cutoff','Version and link to local pixels',
                     'Audited u/v or unambiguous angle provenance'],CPU='Causal timeline + geometry + train-only stats + exhaustive shared-target parity',
            resources='New overlay; input 33/24 of baseline; unqualified T4 budget',feasibility_before_december2='Not established'),
        C8_plus_vegetation=dict(status='NO_GO_CAUSAL',hypothesis='Fuel state discriminates spread',
            missing=['Composite completion and release by cutoff','Exact asset vintage/SHA','Prior published landcover if included'],
            CPU='Asset timeline + geometry + train-only moments + exhaustive parity',resources='New overlay; NDVI/EVI2 + masks gives input 36/24; unqualified fit',
            feasibility_before_december2='Not established'),
        NEWSAMPLE2=dict(status='NOT_SELECTED',hypothesis='More new-positive exposure',
            risk='More positive predictions/calibration shift; no guaranteed ranking improvement'),
        PROP_HN2=dict(status='PRIORITY_CPU_GO',hypothesis='Improve ranking and suppress false propagation without increasing positive exposure',
            risk='Recall, AP or Brier may worsen; no new weather information; single seed',
            CPU_hard_negative_tiles_verified=1839,tests_passed=46,session_minutes=255,card_hours_reserve=8.5,
            feasibility_before_december2='Code and notebook ready; one fit conditional on authorization/runtime/quota')))
    c.save(out/'decision.json',dict(run_id=c.B_ID,CPU_qualification='GO',
        current_GPU_launch='NO_GO_PENDING_NEW_EXPLICIT_AUTHORIZATION_AND_KAGGLE_PREFLIGHT',
        C8_extra_channels='NO_GO_CAUSAL_EVIDENCE_MISSING',priority='PROP_HN2_ONLY',
        future_fits_maximum=1,scientific_factor='Sampling only, baseline C8 loss',
        expected_improvement='Hypothesis only: exact new-fire AP and recall/FP tradeoff; no result claimed',
        acceptance_unchanged=True,acceptance_sha256=c.sha256(out/'acceptance.json'),
        epochs=4,total_updates=28976,checkpoint_epoch=4,threshold=.78,
        no_auto_resume=True,training_executed=False,GPU_executed=False,test_pixels_opened=0,
        promotion_authorized=False,commit=False,push=False))
    c.save(out/'verification_finale.json',dict(branch=branch,tests=dict(passed=46,failed=0,skipped=0,
        receipt=tests,sha256=c.sha256(ROOT/tests)),protected_files_verified=len(protected),protected_changed=[],
        original_scientific_identities_verified=77,NEWPOS2_archives='SHA256 + all members PASS, unchanged',
        hard_negative_train_pixel_checks=1839,validation_pixels_opened=0,test_pixels_opened=0,
        epoch_plans=qualification['epochs'],CPU_qualification_seconds=qualification['seconds'],
        GPU_runtime_qualification='NOT_RUN',scientific_performance='NOT_MEASURED',git_diff_check='PASS'))
    kit=out/'kit';kit.mkdir(exist_ok=False)
    for name in ('notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb','docs/PROP_HN2_PROTOCOLE.md',
                 'scripts/run_prop_hn2.py','scripts/prepare_prop_hn2.py','scripts/finalize_prop_hn2.py'):
        dst=kit/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as target,(ROOT/name).open('rb') as src:shutil.copyfileobj(src,target)
    with (out/'tests_CPU.xml').open('xb') as target,(ROOT/tests).open('rb') as src:shutil.copyfileobj(src,target)
    names=set(lock['files'])|{p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
    names.update(['notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb','docs/PROP_HN2_PROTOCOLE.md','scripts/finalize_prop_hn2.py'])
    c.save(out/'delivery_manifest.json',dict(schema='prop-hn2-delivery-1',
        files={n:c.sha256(ROOT/n) for n in sorted(names)},status='CPU_GO_GPU_LAUNCH_PENDING',
        notebook_sha256=c.sha256(ROOT/'notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb')))
    with (out/'delivery_manifest.sha256').open('x',encoding='ascii') as f:
        f.write(c.sha256(out/'delivery_manifest.json')+'  delivery_manifest.json\n')
    archive=c.old.package(out,'PREPARATION-CPU-PASS')
    print(json.dumps(dict(CPU='GO',GPU_launch='NO_GO_PENDING_NEW_AUTHORIZATION_AND_PREFLIGHT',
         tests_passed=46,protected_files=len(protected),scientific_identities=77,
         real_hard_negative_tiles_verified=1839,archive=str(archive)),indent=2))


if __name__=='__main__':main()
