"""Prepare a single self-contained private CPU notebook; no network or rasters."""
import ast
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

BASE = Path('reports/recovery_20261107_v3/finalized/deployment/bundle')
OUT = Path('reports/kaggle_certification_20261108_v2')


def main(out=OUT, *, allow_2017_context=False, prepare_remaining=False):
    import pyarrow.parquet as pq
    from certify_kaggle_lot001_runtime import validate_lot_dates
    out = Path(out)
    if (out/'submission-attempt-01').exists():
        raise RuntimeError('Submitted preparation must remain immutable')
    out.mkdir(parents=True, exist_ok=True)
    lot = json.loads((BASE/'metadata/lot-001.json').read_text())
    normal = json.loads((BASE/'metadata/normalization.json').read_text())
    rows = pq.read_table(BASE/'metadata/observations.parquet').to_pylist()
    selected = [r for r in rows if r['event_id'] in lot['events']]
    selected = [{k:r[k] for k in ('event_id','date','relative_path','height','width')} for r in selected]
    with (BASE/'metadata/fold_00.csv').open() as stream:
        split = {r['event_id']:r['split'] for r in csv.DictReader(stream)}
    if len(selected) != 1269 or {r['relative_path'] for r in selected} != {r['path'] for r in lot['files']}:
        raise ValueError('Selected QC rows do not exactly match canonical lot')
    if any(split[e] != 'train' for e in lot['events']):
        raise ValueError('Non-train event in lot')
    date_policy = validate_lot_dates(lot, split, allow_2017_context=allow_2017_context)
    # Include real repository modules, unchanged, to test the actual loader.
    modules = ('__init__','runner_data','benchmark','corpus','remote_archive','unet_data','windows','real_constants')
    source_manifest = {}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for module in modules:
            name = f'geofirex/{module}.py'
            content = (Path('src')/name).read_bytes()
            source_manifest[name] = hashlib.sha256(content).hexdigest()
            z.writestr(name, content)
    payload = dict(reference=lot, observations=selected, normalization=normal, split=split,
                   allow_2017_context=allow_2017_context)
    blob = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    sources = base64.b64encode(archive.getvalue()).decode()
    runtime = Path('scripts/certify_kaggle_lot001_runtime.py').read_text(encoding='utf-8')
    source = (
        "import os\nos.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')\n"
        "import base64, hashlib, io, json, sys, uuid, zipfile, zlib\nfrom pathlib import Path\n"
        "work=Path('/kaggle/working')/('lot001-certification-datefix-'+uuid.uuid4().hex[:12])\nwork.mkdir(exist_ok=False)\n"
        f"archive=zipfile.ZipFile(io.BytesIO(base64.b64decode({sources!r})))\n"
        f"source_manifest={source_manifest!r}\n"
        "for name, sha in source_manifest.items():\n"
        "    data=archive.read(name)\n"
        "    if hashlib.sha256(data).hexdigest()!=sha: raise ValueError('Repository source hash mismatch')\n"
        "    p=work/'source'/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_bytes(data)\n"
        "sys.path.insert(0,str(work/'source'))\n"
        f"payload=json.loads(zlib.decompress(base64.b64decode({blob!r})))\n"
        "namespace={'__name__':'lot001_cpu_certification'}\n"
        f"exec(compile({runtime!r},'certify_kaggle_lot001_runtime.py','exec'),namespace)\n"
        "certificate=namespace['run'](**payload,out=work/'results')\n"
    )
    ast.parse(source)
    folder = out/'cpu-certification/lot-001'
    folder.mkdir(parents=True, exist_ok=True)
    kernel = 'nnnnbhss/geofirex-certify-lot001-cpu-20261108-v2'
    notebook = dict(cells=[dict(cell_type='markdown',metadata={},source=[
        '# GeoFire-X lot 001: CPU certification\n',
        'Existing private dataset only. Sequential 1 MiB SHA256/CRC32; 57 train events; no 2020/2021 observations; real rasterio and repository PyTorch DataLoader. No upload, GPU, network, or training.\n',
        ('Date policy: only the three pinned 2017-12-29/30/31 context files in 2018/fire_21458836 are explicitly approved. They remain in the 1269-file hash certification and are listed in the certificate.\n'
         if allow_2017_context else 'Date policy: observation dates must remain in 2018/2019; unapproved 2017 context fails closed.\n')]),
        dict(cell_type='code',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[])],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    nb = json.dumps(notebook, indent=2).encode()
    (folder/'certify.ipynb').write_bytes(nb)
    metadata = dict(id=kernel,title='GeoFireX Certify Lot001 CPU 20261108 V2', code_file='certify.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,
        enable_internet=False,dataset_sources=[lot['handle']],competition_sources=[],kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    prepared = dict(status='LOCAL_PREPARATION_ONLY_REMOTE_UNCERTIFIED', notebook_bytes=len(nb), source_manifest=source_manifest,
        date_policy=date_policy,
        canonical_manifest_sha256=hashlib.sha256((BASE/'metadata/lot-001.json').read_bytes()).hexdigest(),
        notebooks=[dict(tag='lot-001', notebook_sha256=hashlib.sha256(nb).hexdigest(),sources=[lot['handle']],kernel=kernel)])
    (folder.parent/'prepared.json').write_text(json.dumps(prepared,indent=2))
    print(json.dumps(dict(folder=str(folder),notebook_bytes=len(nb),sha256=prepared['notebooks'][0]['notebook_sha256'],
        files=len(lot['files']),events=len(lot['events']),sources=[lot['handle']],date_policy=date_policy),indent=2))
    if not prepare_remaining:
        return
    # Read-only plan validation for the seven remaining lots. No archive staging.
    seen = set(lot['events'])
    plans = []
    for i in range(2,9):
        d = json.loads((BASE/f'metadata/lot-{i:03d}.json').read_text())
        unsigned = {k:v for k,v in d.items() if k != 'identity'}
        identity = hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if identity != d['identity'] or seen & set(d['events']) or any(split[e] != d['role'] for e in d['events']):
            raise ValueError('Remaining lot identity/event/split inconsistency')
        if any('2021' in f['path'].split('/') for f in d['files']):
            raise ValueError('2021 in remaining lots')
        if len(d['files']) != d['observations'] or sum(f['bytes'] for f in d['files']) != d['bytes']:
            raise ValueError('Remaining lot count/size mismatch')
        seen.update(d['events'])
        plans.append(dict(number=i,handle=d['handle'],role=d['role'],events=len(d['events']),
            files=len(d['files']),raw_bytes=d['bytes'],identity=identity,transferred=False))
    (out/'remaining-seven-readiness.json').write_text(json.dumps(dict(status='PREPARED_NO_TRANSFER',
        prerequisite='Lot 001 remote CPU certificate must pass before any publication',
        lots=plans,remaining_raw_bytes=sum(x['raw_bytes'] for x in plans),
        all_lots_events=len(seen),test_2021_files=0),indent=2))
    print(json.dumps(dict(folder=str(folder),notebook_bytes=len(nb),sha256=prepared['notebooks'][0]['notebook_sha256'],
        files=len(lot['files']),events=len(lot['events']),sources=[lot['handle']],remaining_lots_prepared=7),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--allow-lot001-2017-context', action='store_true',
                        help='Explicitly approve only the three pinned pre-event December 2017 observations')
    parser.add_argument('--prepare-remaining', action='store_true')
    args = parser.parse_args()
    main(args.out, allow_2017_context=args.allow_lot001_2017_context, prepare_remaining=args.prepare_remaining)
