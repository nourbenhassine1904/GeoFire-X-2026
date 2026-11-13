"""Offline pilot preparation: one data dataset, embedded manifest, CPU certification.

Never uploads, runs a notebook, installs dependencies or reads source raster pixels.
Requires a completed single-lot local staging receipt.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def prepare(out, staged, canonical):
    out,staged,canonical=Path(out),Path(staged),Path(canonical)
    receipt=json.loads((staged/'staging-receipt.json').read_text())
    lot=json.loads(canonical.read_text())
    unsigned={k:v for k,v in lot.items() if k!='identity'}
    identity=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (lot['number']!=1 or lot['role']!='train' or identity!=lot['identity'] or identity!=receipt['lot_identity']
        or lot['handle']!=receipt['handle'] or receipt['observations']!=lot['observations']):
        raise ValueError('Pilot lot identity differs')
    payload=staged/'payload.zip'
    if payload.stat().st_size!=receipt['payload_bytes']:
        raise ValueError('Staged ZIP size differs')
    # Whole-ZIP SHA was streamed by staging. Publication will recheck it before any network.
    if not all(f['path'].startswith('2018/') for f in lot['files']):
        raise ValueError('Pilot is expected to contain only 2018 train observations')
    out.mkdir(parents=True,exist_ok=False)
    folder=out/'cpu-certification'/'lot-001'; folder.mkdir(parents=True)
    helper=Path('scripts/kaggle_certify_inputs.py').read_text(encoding='utf-8')
    lot_json=json.dumps(lot,separators=(',',':'))
    source=("from pathlib import Path\nimport hashlib, importlib.metadata, json, os, sys\n"
        "os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')\n"
        "namespace={'__name__':'geofirex_pilot_helpers'}\n"
        f"exec(compile({helper!r}, 'kaggle_certify_inputs.py', 'exec'), namespace)\n"
        "work=Path('/kaggle/working/geofirex-pilot-001-v2'); work.mkdir(exist_ok=False)\n"
        f"canonical=work/'lot-001.json'; canonical.write_text({lot_json!r},encoding='utf-8')\n"
        f"expected_identity={identity!r}\n"
        f"inputs=Path('/kaggle/input'); direct=inputs/{lot['slug']!r}\n"
        "if direct.is_dir() and (direct/'lot-inventory.json').is_file():\n    dataset=direct\n"
        "else:\n"
        "    matches=[]\n"
        "    for inventory in inputs.rglob('lot-inventory.json'):\n"
        "        if json.loads(inventory.read_text()).get('identity')==expected_identity:\n"
        "            matches.append(inventory.parent)\n"
        "    if len(matches)!=1: raise RuntimeError('Expected exactly one attached pilot inventory; STOP')\n"
        "    dataset=matches[0]\n"
        "namespace['resource_check'](__import__('time').monotonic()+3600)\n"
        "certificate=namespace['certify_lot'](canonical,dataset,work/'lot-001-certificate',pinned_identity=expected_identity,seconds=3600)\n"
        f"certificate.update(staged_archive_sha256={receipt['sha256']!r},staged_archive_bytes={receipt['payload_bytes']},"
        "python=sys.version,rasterio=importlib.metadata.version('rasterio'),attached_root=str(dataset),gpu_used=False,"
        "archive_sha_remote_claim='Not recomputed: Kaggle extracted the ZIP; every extracted raster SHA256 and CRC32 was checked instead')\n"
        "(work/'pilot-certificate.json').write_text(json.dumps(certificate,indent=2),encoding='utf-8')\n"
        "print(json.dumps(certificate,indent=2))\n")
    ast.parse(source)
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=[
        '# GeoFire-X: standalone CPU certification of private lot 001\n',
        'One attached data dataset. Canonical manifest embedded and pinned. No code dataset, Internet, GPU, training or installation.\n',
        'A missing dependency or checksum mismatch stops certification. Parquet and train-normalizer cloud loading are outside this pilot.\n']),
        dict(cell_type='code',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[])],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    target=folder/'certify.ipynb'; target.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    kernel='nnnnbhss/geofirex-certify-pilot-001-20261107-v2'
    metadata=dict(id=kernel,title='GeoFireX Certify Pilot 001 20261107 V2',code_file='certify.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=[lot['handle']],competition_sources=[],kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    prepared=dict(status='LOCAL_PREPARATION_ONLY_NO_TRANSFER_AUTHORIZED',notebooks=[dict(tag='lot-001',
        notebook_sha256=sha(target),sources=[lot['handle']],kernel=kernel)])
    (folder.parent/'prepared.json').write_text(json.dumps(prepared,indent=2),encoding='utf-8')
    budgets=dict(first_attempt_phase_stop=6_000_000_000,pilot_body_ceiling=11_000_000_000,
        two_full_archive_bodies=2*receipt['payload_bytes'],body_margin_after_two_full_archives=11_000_000_000-2*receipt['payload_bytes'],
        physical_network_planning_envelope=12_000_000_000,automatic_retries=0,
        extra_full_attempts_proposed=1,retry_requires_separate_explicit_approval=True,
        header_tls_tcp_overhead_and_downloads_not_measured_by_body_journal=True)
    # The submission wrapper serializes notebook source into a string. Keep well below its 2 MB body cap.
    for cell in notebook['cells']:
        if isinstance(cell['source'],list): cell['source']=''.join(cell['source'])
    serialized_notebook_bytes=len(json.dumps(notebook).encode())
    if serialized_notebook_bytes>1_000_000: raise ValueError('Pilot notebook exceeds intended small control transfer')
    result=dict(status='ONE_LOT_PILOT_PREPARED_NOT_AUTHORIZED',handle=lot['handle'],private=True,
        events=len(lot['events']),files=lot['observations'],raw_bytes=lot['bytes'],
        zip_bytes_measured=receipt['payload_bytes'],zip_sha256_from_completed_staging=receipt['sha256'],
        manifest_identity=identity,canonical_manifest_sha256=sha(canonical),
        staged_path=staged.as_posix(),notebook_path=target.as_posix(),notebook_sha256=sha(target),
        serialized_notebook_bytes=serialized_notebook_bytes,certification_dataset_sources=[lot['handle']],budgets=budgets,
        code_dataset_upload_required=False,cloud_certification_executed=False,uploads_executed=False,
        parquet_and_normalizer_cloud_validation='Deferred; not claimed by this one-data-lot pilot')
    (out/'pilot-plan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','staged','canonical'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.out,a.staged,a.canonical),indent=2))
