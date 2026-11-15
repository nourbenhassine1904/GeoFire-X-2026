"""Offline preparation only. Hash existing objects; never decode/train/submit."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from geofirex.calibration_io import save, sha
from geofirex.experiment_readiness import validate_campaign
from geofirex.focal_experiment import preflight, validate_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/focal_preparation_v1'


def verify_partitions(plan):
    """Full byte hashes, no archive extraction or pixel array access."""
    if len(plan['parts']) != 11:
        raise ValueError('Exactly eleven existing partitions required')
    seen = set(); identities = set(); verified = []; memo = {}; total = 0
    for part in plan['parts']:
        folder = Path(part['folder'])
        mapping = json.loads((folder / 'mount-map.json').read_text(encoding='utf-8'))
        files = [{k: f[k] for k in ('path', 'alias', 'bytes', 'sha256')} for f in part['files']]
        canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
        identity = hashlib.sha256(canonical).hexdigest()
        if (mapping['files'] != files or identity != part['identity']
                or mapping['identity'] != identity or identity in identities):
            raise ValueError('Partition identity differs')
        identities.add(identity)
        for f, pin in zip(part['files'], files):
            logical = Path(pin['path'])
            if logical.is_absolute() or '..' in logical.parts or '2021' in logical.parts or pin['path'] in seen:
                raise ValueError('Unsafe, duplicate or locked compact object')
            seen.add(pin['path'])
            for path in (Path(f['source']), folder / pin['alias']):
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if key not in memo: memo[key] = sha(path)
                if stat.st_size != pin['bytes'] or memo[key] != pin['sha256']:
                    raise ValueError('Object bytes differ: ' + str(path))
            total += pin['bytes']
        verified.append(dict(number=part['number'], handle=part['handle'], identity=identity,
                             objects=len(files), bytes=sum(f['bytes'] for f in files), full_sha256=True))
        print('Verified bytes, partition', part['number'], flush=True)
    if len(seen) != 459:
        raise ValueError('Expected 459 published objects')
    return dict(status='PASS', partitions=verified, objects=len(seen), total_bytes=total,
                unique_file_contents_hashed=len(memo), arrays_decoded=0, downloads=0,
                remote_status='not queried; historical publication certificate only')


def build(out=OUT):
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('New preparation outputs must stay inside reports')
    if out.exists(): raise FileExistsError(out)
    campaign_path = ROOT / 'configs/scientific_improvements_v1.json'
    campaign = json.loads(campaign_path.read_text(encoding='utf-8'))
    validate_campaign(campaign)
    archive = ROOT / 'reports/unet_confirmation_20261112_v1/inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip'
    with zipfile.ZipFile(archive) as z:
        config_bytes = z.read('B2_posweight_10/config.json')
        evaluation_bytes = z.read('B2_posweight_10/calibration32/evaluation.json')
        selection_bytes = z.read('B2_posweight_10/selection32.json')
    reference = json.loads(config_bytes); baseline = json.loads(evaluation_bytes)
    selection = json.loads(selection_bytes)
    if (reference['initial_state_sha256'] != campaign['historical_reference']['initial_state_sha256']
            or selection['threshold'] != .78 or reference['updates_per_run'] != 23404
            or reference['model'] != 'B2' or reference['pos_weight'] != 10
            or reference['torch_version'] != '2.11.0+cu130' or reference['numpy_version'] != '2.1.3'
            or baseline['events'] != reference['calibration_events'] or baseline['tiles'] != 13759
            or reference['index_sha256'] != campaign['historical_reference']['index_sha256']
            or reference['normalization_sha256'] != campaign['historical_reference']['normalization_sha256']):
        raise ValueError('Historical comparison does not match prepared protocol')
    # Validate input identities BEFORE writing a notebook/config.
    plan_path = ROOT / 'reports/gpu_acceleration_20261111_v1/publication/plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    verification = verify_partitions(plan)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'partitions_verified.json', verification)
    for name, data in [('baseline_config.json', config_bytes), ('baseline_evaluation32.json', evaluation_bytes),
                       ('baseline_selection32.json', selection_bytes)]:
        with (out / name).open('xb') as f: f.write(data)
    parts = [dict(handle=p['handle'], identity=p['identity']) for p in plan['parts']]
    c = dict(schema='geofirex-one-focal-v1', experiment='A_focal_gamma2_pos10_seed42_v1',
             training_authorized=False, runs=1, promotion_to_studio=False,
             model='B2', channels=8, context=3, seed=42, epochs=4, batch_size=16,
             train_tiles=93603, updates=23404, training_workers=4, validation_batch_size=32,
             loss=dict(name='binary_focal_logits', gamma=2., positive_weight=10., negative_weight=1.,
                       dice_weight=0., reduction='valid_pixel_mean'),
             optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, gradient_clip=1.),
             events=reference['calibration_events'], thresholds=reference['threshold_candidates'],
             initial_state_sha256=reference['initial_state_sha256'],
             index_sha256=reference['index_sha256'], normalization_sha256=reference['normalization_sha256'],
             torch_version=reference['torch_version'], numpy_version=reference['numpy_version'],
             max_training_seconds=2700, selection_scope='development32',
             confirmation_scope='observed169_descriptive_only', locked_test_years=[2021],
             acceptance=campaign['predeclared_acceptance_on_development32'],
             campaign_sha256=sha(campaign_path), historical_archive_sha256=sha(archive),
             publication_plan_sha256=sha(plan_path), dataset_partitions=parts,
             baseline_evaluation_sha256=sha(out / 'baseline_evaluation32.json'),
             baseline_checkpoint_sha256=selection['checkpoint_sha256'],
             causal_scope='loss family including removal of Dice; focusing alone not isolated',
             checkpoint_policy='fresh seed42; only this experiment may resume with hash/identity, NEW output',
             determinism='same historical seeds/cudnn/sampler; no bitwise GPU guarantee',
             model_selection='fixed epoch4 only; one final evaluation on development32',
             budget='11.1-22.2 min scenario from 887.069 s related T4 run; 45 min training pause cap; evaluation/setup/IO unmeasured and additional')
    validate_config(c)
    metadata = preflight(c, plan['corpus'], plan['index'])
    c['sampler_order_sha256'] = metadata['sampler_order_sha256']
    # Publish ONE frozen configuration, once; the first pass only derives order.
    final_config = out / 'focal_config.json'
    save(final_config, c)
    save(out / 'metadata_preflight.json', preflight(final_config, plan['corpus'], plan['index']))
    names = ('__init__.py', 'calibration_io.py', 'calibration_mounts.py', 'compact_calibration.py',
             'fast_compact.py', 'benchmark_models.py', 'benchmark_metrics.py', 'unet.py',
             'loss_ablation.py', 'focal_loss.py', 'focal_experiment.py')
    hashes = {}
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in names:
                p = ROOT / 'src/geofirex' / name
                source = p.read_text(encoding='utf-8')
                if name == 'fast_compact.py':
                    source = source.replace('from geofirex.mission_streaming import save,sha',
                                            'from geofirex.calibration_io import save,sha')
                compile(source, name, 'exec')
                hashes['geofirex/' + name] = hashlib.sha256(source.encode()).hexdigest()
                z.writestr('geofirex/' + name, source)
            for path, member in [(final_config, 'focal_config.json'),
                                 (out / 'baseline_evaluation32.json', 'baseline_evaluation32.json')]:
                z.writestr(member, path.read_bytes())
        payload = buffer.getvalue()
    code_path = out / 'focal-code.zip'
    with code_path.open('xb') as f: f.write(payload)
    cells = []
    def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s))
    def code(s):
        compile(s, 'focal-notebook', 'exec')
        cells.append(dict(cell_type='code', metadata={}, source=s, execution_count=None, outputs=[]))
    md('''# GeoFire-X - ONE prepared Focal experiment
Execution is disabled. No submission is performed by preparation.
Attach exactly the 11 existing private compact partitions; Internet OFF.
Run metadata checks on CPU first. Do not install/update torch or numpy automatically.
Historical environment required: torch2.11.0+cu130 / numpy2.1.3.
One fresh U-Net B2, T3/eight channels, seed42, four epochs/batch16/23404 updates.
Pure Focal gamma2, positive10/negative1; no Dice. This tests the loss family.
Threshold is selected on the reused development32 only, never observed169 or2021.
Training scenario11.1-22.2 min T4 is unmeasured for Focal; setup, SHA checks,
exact AP evaluation and backup are additional. Training pauses at45min.
No historical checkpoint, public weights or other experiment may initialize/resume this run.
''')
    code('''import base64, hashlib, io, json, os, sys, tempfile, zipfile, subprocess
from pathlib import Path
payload=base64.b64decode(''' + repr(base64.b64encode(payload).decode()) + ''')
assert hashlib.sha256(payload).hexdigest()==''' + repr(hashlib.sha256(payload).hexdigest()) + '''
workspace=Path(tempfile.mkdtemp(prefix="geofirex-focal-code-",dir="/tmp"))
with zipfile.ZipFile(io.BytesIO(payload)) as z: z.extractall(workspace)
code_archive=workspace/"focal-code.zip"
code_archive.write_bytes(payload)
sys.path.insert(0,str(workspace))
os.environ["PYTHONPATH"]=str(workspace)+os.pathsep+os.environ.get("PYTHONPATH", "")
''')
    code('''# Read-only byte verification; NO NPZ decode, model or GPU query.
from geofirex.calibration_mounts import restore_verified
expected=''' + repr([p['identity'] for p in parts]) + '''
maps=[p for p in Path("/kaggle/input").rglob("mount-map.json")
      if json.loads(p.read_text())["identity"] in expected]
corpus=restore_verified(maps,expected,Path(tempfile.mkdtemp(prefix="geofirex-focal-mount-",dir="/tmp"))/"corpus")
command=[sys.executable,"-m","geofirex.focal_experiment","--config",str(workspace/"focal_config.json"),
         "--corpus",str(corpus),"--index",str(corpus/"index")]
subprocess.run(command,check=True,env=os.environ.copy())
''')
    md('''## Explicit authorization required before changing the following cell
No authorization file is created here. It must name this experiment, runs=1,
training_authorized=true, exact config/code SHA256, max_training_seconds=2700,
and an approved checkpoint_backup_destination. Permission remains external to this file.
Keep the original initial-state hash; do not relax environment/identity checks.
The final development32 evaluation occurs once AFTER epoch4. No old model is inferred.
The training checkpoint is written BEFORE evaluation so an AP failure cannot lose it.
''')
    code('''RUN_TRAINING=False
AUTHORIZATION_FILE=None
RESUME=None  # Only a verified checkpoint from this same Focal run, if paused.
RESUME_SHA256=None
OUT=None
if RUN_TRAINING:
    if AUTHORIZATION_FILE is None: raise RuntimeError("Explicit authorization required")
    import uuid
    OUT=Path("/kaggle/working")/("geofirex-focal-"+uuid.uuid4().hex)
    execution=command+["--execute-training","--authorization",str(AUTHORIZATION_FILE),
                       "--code-archive",str(code_archive),"--baseline",str(workspace/"baseline_evaluation32.json"),
                       "--out",str(OUT)]
    if RESUME is not None:
        execution += ["--resume",str(RESUME),"--resume-sha256",str(RESUME_SHA256)]
    subprocess.run(execution,check=True,env=os.environ.copy())
else:
    print("DISABLED: no training/evaluation/GPU call.")
''')
    md('''## Preserve outputs before ending a future authorized session
Checkpoints contain model/optimizer/RNG/cursor/loader state and immutable identities.
Every checkpoint has its SHA256/size sidecar. Save the Kaggle output and copy it to
the approved external destination; verify the archive hash after copying.
Do not replace Studio files. Resuming requires --resume + verified --resume-sha256,
the same frozen config/code/environment and a NEW output directory.
If evaluation fails after training, keep epoch4 and evaluate that checkpoint later;
do not train another model. Never treat observed169 as an unseen test.
''')
    code('''if RUN_TRAINING and OUT is not None and OUT.exists():
    import shutil
    archive=Path(str(OUT)+".zip")
    if archive.exists(): raise FileExistsError(archive)
    shutil.make_archive(str(OUT),"zip",root_dir=OUT)
    digest=hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    sidecar=archive.with_suffix(".sha256")
    with sidecar.open("x") as stream: stream.write(digest.hexdigest()+"  "+archive.name+"\\n")
    print("Preserve and verify:",archive,sidecar)
''')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
                    cells=[dict(id=f'focal-{i}', **cell) for i, cell in enumerate(cells)])
    save(out / 'GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb', notebook)
    save(out / 'dataset-sources.json', [p['handle'] for p in parts])
    save(out / 'build.json', dict(training_executed=False, gpu_queried=False, submitted=False,
         runs=1, config_sha256=sha(final_config), code_sha256=sha(code_path),
         bundled_source_sha256=hashes, baseline_archive_sha256=sha(archive),
         source_adaptation='fast_compact import only: same save/sha helpers, no loader logic changed'))
    print('Prepared DISABLED notebook:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    build(p.parse_args().out)
