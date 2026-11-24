"""Seal FIRE13 only after targeted CPU tests and exhaustive real parity."""
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from geofirex import fire13_experiment as c

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/c.REPORT
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed FIRE13 delivery')
    protected=c.read(out/'preservation_before.json')['files'];c.old.verify_manifest(ROOT,protected)
    base,config,_,lock,_=c.identity(ROOT)
    original=c.read(ROOT/c.old.REPORT/'execution_manifest.json')['files']
    c.old.verify_manifest(ROOT,original)
    if len(original)!=77:raise ValueError('Original 77 scientific identities required')
    tests='reports/fire13_tests_20261122_v1/tests.xml'
    save_tests='reports/fire13_saves_20261122_v1/tests.xml'
    cases=[];receipts=[]
    for name,expected in ((tests,38),(save_tests,7)):
        tree=ET.parse(ROOT/name).getroot();current=tree.findall('.//testcase')
        if len(current)!=expected or any(tree.findall('.//'+k) for k in ('failure','error','skipped')):
            raise ValueError('Complete targeted CPU PASS required: '+name)
        cases.extend(current);receipts.append(dict(path=name,passed=expected,sha256=c.sha256(ROOT/name)))
    names={x.attrib['name'] for x in cases}
    for required in ('test_comparable_fp_all_thresholds_locked_before_assessment',
        'test_scientific_contract_real_parity_and_train_stats','test_notebook_disabled_payload_pinned_compilable_and_complete',
        'test_synthetic_checkpoint_restore_and_two_verified_backup_roots'):
        if required not in names:raise ValueError('Required executed CPU check absent: '+required)
    parity=c.read(out/'parity_full.json');stats=c.read(out/'derived_normalization.json')
    if (parity['status']!='PASS' or parity['roles']['val']['evaluation_domain_sha256']!=c.DOMAIN
        or stats['fit_role']!='train' or stats['window_count']!=4803):raise ValueError('Real data qualification incomplete')
    maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    c.save(out/'cpu_preflight_final.json',c.preflight(ROOT,maps))
    git=['git','-c','safe.directory='+ROOT.as_posix()]
    subprocess.run(git+['diff','--check'],cwd=ROOT,check=True,capture_output=True,text=True)
    branch=subprocess.check_output(git+['branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='codex/fire13-20261122':raise ValueError('Separate FIRE13 branch required')
    c.save(out/'decision.json',dict(run_id=c.B_ID,CPU_qualification='GO',
        current_GPU_launch='NO_GO_PENDING_NEW_EXPLICIT_AUTHORIZATION_AND_KAGGLE_PREFLIGHT',
        external_additional_raw_bands='NONE_CERTIFIED',priority='FIRE13_ONLY',
        admissible_derived_channels=['distance capped8 pixels','density7x7','appearance since oldest input',
            'signed observed change','input-only support'],inherited_C8_fire_delivery_assumption=True,
        channels_per_day=13,stacked_channels=39,parameters=125057,first_conv_added_parameters=1080,
        scientific_factor='Input set only; historical C8 loss and sampler unchanged',
        prospective_hypothesis='Improve ranking and recall/FP tradeoff; no performance measured',
        acceptance_unchanged=True,acceptance_sha256=c.sha256(out/'acceptance.json'),
        future_fits_maximum=1,epochs=4,total_updates=28976,checkpoint_epoch=4,threshold=.78,
        no_auto_resume=True,PROP_HN2_preserved=True,training_executed=False,GPU_executed=False,
        test_pixels_opened=0,automatic_promotion=False,commit=False,push=False))
    c.save(out/'verification_finale.json',dict(branch=branch,tests=dict(passed=len(cases),failed=0,skipped=0,
        receipts=receipts),protected_files_verified=len(protected),protected_changed=[],
        original_scientific_identities_verified=77,PROP_HN2_preserved=True,
        raw_TIFF_paths_verified=6913,geometry='Reused header certificate + exact certified metadata equality; two current TIFF window probes',
        train_normalization_windows=4803,normalization_target_reads=0,parity=parity,
        GPU_runtime_qualification='NOT_RUN',scientific_performance='NOT_MEASURED',test_pixels_opened=0,
        git_diff_check='PASS'))
    kit=out/'kit';kit.mkdir(exist_ok=False)
    for name in ('notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb','docs/FIRE13_LANCEMENT.md',
                 'scripts/run_fire13.py','scripts/prepare_fire13.py','scripts/prepare_fire13_data.py','scripts/finalize_fire13.py',
                 'tests/test_fire13.py','tests/test_fire13_saves.py'):
        dst=kit/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as target,(ROOT/name).open('rb') as src:shutil.copyfileobj(src,target)
    with (out/'tests_CPU.xml').open('xb') as target,(ROOT/tests).open('rb') as src:shutil.copyfileobj(src,target)
    with (out/'tests_saves_CPU.xml').open('xb') as target,(ROOT/save_tests).open('rb') as src:shutil.copyfileobj(src,target)
    delivery=set(lock['files'])|{p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
    delivery.update(['notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb','docs/FIRE13_LANCEMENT.md','scripts/finalize_fire13.py','tests/test_fire13_saves.py'])
    c.save(out/'delivery_manifest.json',dict(schema='fire13-cpu-delivery-1',status='CPU_GO_GPU_LAUNCH_PENDING',
        files={n:c.sha256(ROOT/n) for n in sorted(delivery)},
        notebook_sha256=c.sha256(ROOT/'notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb')))
    with (out/'delivery_manifest.sha256').open('x',encoding='ascii') as f:f.write(c.sha256(out/'delivery_manifest.json')+'  delivery_manifest.json\n')
    archive=c.old.package(out,'PREPARATION-CPU-PASS')
    print(json.dumps(dict(CPU='GO',GPU_launch='NO_GO_PENDING_AUTHORIZATION_AND_PREFLIGHT',tests_passed=len(cases),
        real_tiles_compared=146782,protected_files=len(protected),scientific_identities=77,archive=str(archive)),indent=2))


if __name__=='__main__':main()
