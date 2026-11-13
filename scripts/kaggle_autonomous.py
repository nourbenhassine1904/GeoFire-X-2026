"""Offline checks and durable state for the standalone lots 003..008 profile."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from run_bounded_local import memory

START_BYTES = 3 * 1024**3
RESERVE_BYTES = 2 * 1024**3
TREE_BYTES = 512 * 1024**2
INITIAL_BODY_BYTES = 4_995_735_202
CEILING_BYTES = 31_000_000_000
DEFAULT_PLAN = Path('reports/kaggle_remaining_20261108_v1/plan.json')


def require_memory(required=START_BYTES):
    sample = memory()
    if sample is None or sample['available'] < required:
        raise MemoryError('Insufficient physical RAM; no new operation started')
    return sample


def save_json(path, value):
    """Atomic checkpoint plus flushed event history, never a historical overwrite."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checkpoint(folder, plan, *, status, phase, completed, active_lot=None, **extra):
    from resume_remaining_lots import reserved
    used = reserved(plan)
    row = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(),
               status=status, phase=phase, active_lot=active_lot,
               same_resume='resume-01', completed_lots=[1, 2, *completed],
               cumulative_reserved_body_bytes=used, remaining_body_bytes=CEILING_BYTES-used,
               start_memory_bytes=START_BYTES, reserve_memory_bytes=RESERVE_BYTES,
               tree_limit_bytes=TREE_BYTES, automatic_retries=0, gpu_started=False, **extra)
    folder = Path(folder)
    with (folder/'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    save_json(folder/'checkpoint.json', row)
    print(json.dumps(row), flush=True)


def validate_autonomous(plan_path):
    """No credentials, network, raster reads, journal writes or attempt creation."""
    from resume_remaining_lots import reserved, validate_continuation
    plan_path = Path(plan_path)
    plan, execution = validate_continuation(plan_path)
    if reserved(plan) != INITIAL_BODY_BYTES:
        raise ValueError('Expected preserved 4,995,735,202-byte HTTP baseline; inspect, never reset')
    certified = json.loads((execution.parent/'checkpoint-certified.json').read_text())
    if (certified['status'] != 'LOT002_PRIVATE_AND_CPU_CERTIFIED_REMAINING_PUBLICATIONS_STOPPED'
            or certified['lot002']['cpu_certified'] is not True
            or certified['cumulative_reserved_body_bytes'] != INITIAL_BODY_BYTES):
        raise ValueError('Official lot002 checkpoint missing or changed')
    if (execution/'continuation-started.json').exists():
        raise FileExistsError('Continuation already attempted; no automatic retry')
    if Path(plan['journal']+'.lock').exists():
        raise FileExistsError('HTTP journal already locked; never remove its lock automatically')
    for spec in plan['lots'][1:]:
        if Path(spec['staging']).exists() or (plan_path.parent/'execution'/spec['tag']).exists():
            raise FileExistsError('Existing later staging/attempt; preserve and inspect before any new upload')
        if spec['number'] not in range(3, 9) or spec['tag'] != f"lot-{spec['number']:03d}":
            raise ValueError('Only lots 003..008 allowed')
        root = Path.cwd().resolve()/'artifacts/kaggle_remaining_20261108_v1'
        if Path(spec['staging']).resolve() != root/spec['tag']:
            raise ValueError('Unexpected staging location')
        lot = json.loads(Path(spec['canonical']).read_text())
        if any(f['path'].split('/')[0] not in ('2018', '2019', '2020') for f in lot['files']):
            raise ValueError('Test 2021 must remain locked')
        notebook = spec['cpu_notebook']
        folder = Path(notebook['folder'])
        if hashlib.sha256((folder/'certify.ipynb').read_bytes()).hexdigest() != notebook['sha256']:
            raise ValueError('Pinned CPU notebook changed')
        meta = json.loads((folder/'kernel-metadata.json').read_text())
        if (meta['is_private'] is not True or meta['enable_gpu'] is not False
                or meta['enable_tpu'] is not False or meta['enable_internet'] is not False
                or meta['dataset_sources'] != [spec['handle']] or meta['id'] != notebook['kernel']):
            raise ValueError('Private CPU certification metadata changed')
    normal = plan_path.parent/'training/bundle/metadata/normalization.json'
    if hashlib.sha256(normal.read_bytes()).hexdigest() != plan['normalizer_sha256']:
        raise ValueError('Train-only normalizer changed')
    # Conservative room for all remaining archives and <=5 MB CPU control per lot.
    projected = INITIAL_BODY_BYTES + sum(s['archive_ceiling_bytes']+6_000_000 for s in plan['lots'][1:])
    if projected > CEILING_BYTES:
        raise ValueError('Insufficient cumulative HTTP envelope')
    return plan, execution


def release_certified_archive(spec, lot_output):
    """Delete only the new certified lot's ZIP; retain manifests and signatures."""
    from kaggle_cpu_job import validate_receipt
    if spec['number'] not in range(3, 9):
        raise ValueError('Never release archives of the closed lots 001/002')
    validate_receipt(Path(lot_output)/'cpu', spec)
    stage = Path(spec['staging']).resolve()
    expected = Path.cwd().resolve()/'artifacts/kaggle_remaining_20261108_v1'/spec['tag']
    payload = stage/'payload.zip'
    if (stage != expected or payload.is_symlink() or payload.resolve().parent != stage
            or (stage/'payload.zip.part').exists()):
        raise ValueError('Unsafe archive release path or partial archive')
    receipt = json.loads((stage/'staging-receipt.json').read_text())
    publication = json.loads((Path(lot_output)/'publication/result.json').read_text())
    presence = json.loads((Path(lot_output)/'publication/private-presence.json').read_text())
    if (receipt['lot_identity'] != spec['identity'] or receipt['handle'] != spec['handle']
            or publication['status'] != 'PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING'
            or publication.get('payload_sha256') != receipt['sha256']
            or publication['confirmed_blob_body_bytes'] != receipt['payload_bytes']
            or presence.get('private') is not True or presence['handle'] != spec['handle']
            or payload.stat().st_size != receipt['payload_bytes']):
        raise ValueError('Archive cannot be released without matching private publication and CPU receipt')
    payload.unlink()  # No recursion; only this exact, newly created certified payload.
    save_json(Path(lot_output)/'archive-released.json', dict(
        status='CERTIFIED_PAYLOAD_RELEASED', payload=str(payload), **receipt,
        staging_receipt_retained=True, cpu_evidence_retained=True))
