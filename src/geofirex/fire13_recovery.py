"""Additive FIRE13 evaluation recovery. Original execution files stay frozen."""
from pathlib import Path
import json
import math
import shutil
import time
import zipfile

from geofirex import fire13_experiment as c

REPORT = Path('reports/fire13_recovery_20261123_v1')
OUT_NAME = c.B_ID + '_EVALUATION_RECOVERY_ONLY'


def identity(root):
    root = Path(root)
    c.identity(root)
    manifest = c.read(root / REPORT / 'recovery_manifest.json')
    c.old.verify_manifest(root, manifest['files'])
    if c.sha256(root / c.REPORT / 'execution_manifest.json') != manifest['original_manifest_sha256']:
        raise ValueError('Original FIRE13 lock changed')
    return manifest


def authorize(root, path, action, parent):
    manifest = identity(root)
    # Keep the original conservative 300 min / 10 card-hour budget unchanged.
    grant = c.authorize(root, path, action, 'cuda', parent)
    expected = dict(recovery_manifest_sha256=c.sha256(Path(root)/REPORT/'recovery_manifest.json'),
                    training_archive_sha256=manifest['training_archive_sha256'],
                    checkpoint_sha256=manifest['checkpoint_sha256'],
                    evaluation_only=True, output_name=OUT_NAME)
    if any(grant.get(k) != v for k, v in expected.items()) or not grant.get('authorization_id'):
        raise ValueError('New checkpoint-bound recovery grant required')
    return grant


def resource_preflight(root, archive, work):
    """Check space before extraction, including copies and archive overhead."""
    manifest = identity(root)
    if c.sha256(archive) != manifest['training_archive_sha256']:
        raise ValueError('Wrong TRAINING-SAFE archive')
    policy = c.read(Path(root)/c.REPORT/'budget.json')
    with zipfile.ZipFile(archive) as z:
        restored_bytes = sum(i.file_size for i in z.infolist())
    # Original 18 GiB scratch reserve PLUS restoration and conservative archive copies.
    required = policy['required_free_bytes'] + 4*restored_bytes
    total, available = c.old.host_memory()
    if total < 16*c.old.GIB or available < 8*c.old.GIB:
        raise RuntimeError('At least 16 GiB total / 8 GiB available RAM required')
    free = c.old.disk_check(work, required)
    return dict(required_free_bytes=required, free_bytes=free,
                archive_expanded_bytes=restored_bytes, host_ram_total_bytes=total,
                host_ram_available_bytes=available, session_minutes=policy['B_session_minutes'],
                card_hours_reserved=policy['B_card_hours'], evaluation_seconds=policy['B_evaluation_seconds'],
                estimate_only=True, quota_receipt_required=True)


def restore(root, archive, sidecar, destination):
    manifest = identity(root)
    if c.sha256(archive) != manifest['training_archive_sha256']:
        raise ValueError('Archive differs from verified recovery source')
    source = c.restore_candidate(root, archive, sidecar, destination)
    _, checkpoint = verify_final(root, source)
    # Original consumed grant and all embedded scientific inputs remain byte-identical.
    original = c.read(Path(root)/c.REPORT/'execution_manifest.json')
    c.old.verify_manifest(source/'inputs', original['files'])
    if c.sha256(source/'inputs'/c.REPORT/'execution_manifest.json') != manifest['original_manifest_sha256']:
        raise ValueError('Archived original execution lock differs')
    return source


def verify_final(root, source):
    manifest = identity(root)
    source = Path(source)
    receipt = c.read(source/'training_complete.json')
    if receipt.get('status') != 'COMPLETE' or receipt.get('epochs') != 4:
        raise ValueError('Four completed epochs required')
    state, checkpoint = c.final_candidate(root, source)
    if c.sha256(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Not the original final FIRE13 checkpoint')
    if (source/'evaluation/complete.json').exists():
        raise ValueError('Already evaluated; preserve existing evaluation')
    return state, checkpoint


def claim(parent, checkpoint_sha256, authorization_id):
    """Exclusive persistent claim; deliberately survives failure/removal of output."""
    lock = Path(parent)/('fire13-evaluate-'+checkpoint_sha256+'.lock.json')
    c.save(lock, dict(action='fire13-evaluate', checkpoint_sha256=checkpoint_sha256,
                      authorization_id=authorization_id, consumed=True, automatic_retry=False))
    return lock


def recover(root, maps, source, out, authorization, gpu_authorization, archive):
    """Only inference, exact metrics, paired comparison and fixed scientific gates."""
    import torch
    root, source, out = Path(root), Path(source), Path(out).resolve()
    if out.name != OUT_NAME or out.exists() or out.parent != Path('/kaggle/working').resolve():
        raise ValueError('Fresh direct Kaggle output required')
    grant = authorize(root, authorization, 'fire13-evaluate', out.parent)
    authorize(root, gpu_authorization, 'gpu-preflight', out.parent)
    resources = resource_preflight(root, archive, out.parent)
    preflight = c.preflight(root, maps, out.parent, 'cuda', gpu=True)
    _, checkpoint = verify_final(root, source)
    digest = c.sha256(checkpoint)
    lock = claim(out.parent, digest, grant['authorization_id'])
    out = c.new_output(out, OUT_NAME)
    started = time.monotonic()
    success = False
    try:
        _, _, _, original, _ = c.identity(root)
        c.preserve(root, out, original)
        manifest = identity(root)
        for name in list(manifest['files']) + [(REPORT/'recovery_manifest.json').as_posix()]:
            dst = out/'recovery_code'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open('xb') as target, (root/name).open('rb') as stream:
                shutil.copyfileobj(stream, target)
        for src, name in [(checkpoint, checkpoint.name),
                          (checkpoint.with_suffix('.json'), checkpoint.with_suffix('.json').name),
                          (source/'training_complete.json', 'training_complete.json'),
                          (source/'authorization.json', 'original_consumed_authorization.json'),
                          (Path(authorization), 'authorization_fire13-evaluate.json'),
                          (Path(gpu_authorization), 'authorization_gpu-preflight.json'),
                          (lock, lock.name)]:
            with (out/name).open('xb') as target, src.open('rb') as stream:
                shutil.copyfileobj(stream, target)
        c.save(out/'preflight.json', dict(resources=resources, gpu=preflight))
        c.configure_runtime(42, 'cuda')
        torch.cuda.reset_peak_memory_stats()
        policy = c.read(root/c.REPORT/'budget.json')
        with c.old.evaluation_deadline(policy['B_evaluation_seconds']):
            c.candidate_evaluation(root, maps, out, out/'evaluation')
        complete = c.read(out/'evaluation/complete.json')
        result = c.read(out/'evaluation/evaluation.json')
        if (complete['checkpoint_sha256'] != digest or result['tiles'] != 30882 or
                len(result['per_event']) != 80 or result['evaluation_domain_sha256'] != c.DOMAIN):
            raise ValueError('Incomplete frozen validation domain')
        c.old.verify_manifest(source/'inputs', original['files'])
        if c.sha256(checkpoint) != digest or c.sha256(out/checkpoint.name) != digest:
            raise ValueError('Checkpoint changed during recovery')
        elapsed = time.monotonic()-started
        c.save(out/'runtime_budget.json', dict(evaluation_seconds=elapsed,
            session_within_budget=elapsed <= policy['B_session_minutes']*60,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_rss_sample_bytes=c.old.host_rss(), training_executed=False, test_pixels_opened=0,
            quota_card_hours='Actual Kaggle deduction must be recorded before integration',
            promotion_authorized=False))
        success = True
    except BaseException as exc:
        c.save(out/'recovery_failure.json', dict(status='PARTIAL', error_type=type(exc).__name__,
            error=str(exc), automatic_retry=False, training_executed=False))
        raise
    finally:
        c.old.package(out, 'EVALUATION-RECOVERY-COMPLETE' if success else 'EVALUATION-RECOVERY-PARTIAL')
    return out
