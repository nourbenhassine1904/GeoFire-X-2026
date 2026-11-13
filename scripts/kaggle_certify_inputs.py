"""CPU-only Kaggle input helpers; bounded streaming, no network or training."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile
import zlib

BLOCK = 1024*1024


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(BLOCK), b''): h.update(block)
    return h.hexdigest()


def safe_child(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe relative path')
    target = Path(root)/name
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes root')
    return target


def bootstrap_code(source, work, payload_sha, manifest_sha):
    source, work = Path(source), Path(work)
    if work.exists(): raise FileExistsError('New working directory required')
    payload = source/'payload.zip'
    archive = None
    try:
        if payload.is_file():
            if sha_file(payload) != payload_sha: raise ValueError('Code ZIP SHA mismatch')
            archive = zipfile.ZipFile(payload)
            names = archive.namelist()
            manifest_bytes = archive.read('code-manifest.json')
            manifest = json.loads(manifest_bytes)
            if len(names) != len(set(names)) or set(names) != set(manifest) | {'code-manifest.json'}:
                raise ValueError('Code ZIP inventory differs')
        else:
            matches = list(source.rglob('code-manifest.json'))
            if len(matches) != 1: raise ValueError('Expected one expanded code manifest')
            source = matches[0].parent
            manifest_bytes = matches[0].read_bytes()
            manifest = json.loads(manifest_bytes)
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
            raise ValueError('Pinned code manifest SHA mismatch')
        work.mkdir(parents=True, exist_ok=False)
        for name, sha in manifest.items():
            dest = safe_child(work, name)
            origin = None if archive else safe_child(source, name)
            if origin and origin.is_symlink(): raise ValueError('Symlink refused')
            dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with (archive.open(name) if archive else origin.open('rb')) as inp, dest.open('xb') as out:
                for block in iter(lambda: inp.read(BLOCK), b''): h.update(block); out.write(block)
            if h.hexdigest() != sha: raise ValueError('Expanded code file SHA mismatch: '+name)
        (work/'code-manifest.json').write_bytes(manifest_bytes)
        return work
    finally:
        if archive: archive.close()


def check_runtime(work, bundle_relative):
    if sys.version_info < (3, 11): raise RuntimeError('Operational preflight requires Python >=3.11')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.path.insert(0, str(Path(work)/'src'))
    versions, missing = {}, []
    for module, distribution in [('numpy','numpy'), ('rasterio','rasterio'), ('torch','torch'), ('mlflow','mlflow'),
                                 ('yaml','pyyaml'), ('matplotlib','matplotlib'), ('sklearn','scikit-learn'),
                                 ('pyarrow','pyarrow'), ('psutil','psutil')]:
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
        except ImportError: missing.append(distribution)
    if missing: raise RuntimeError('Missing dependencies; STOP, no automatic installation: '+','.join(missing))
    psutil_version = tuple(int(v) for v in versions['psutil'].split('.')[:2])
    if not (psutil_version >= (5,9) and psutil_version < (8,0)):
        raise RuntimeError('psutil outside reviewed >=5.9,<8 range')
    import pyarrow.parquet as pq
    bundle = Path(work)/bundle_relative
    parquet_rows = {name: pq.read_table(bundle/f'metadata/{name}.parquet').num_rows for name in ('observations','windows')}
    norm_path = bundle/'metadata/normalization.json'
    norm = json.loads(norm_path.read_text())
    if sha_file(norm_path) != 'aa30df4e3d6d895760f04e35d74bcbbac0d739cea045d236517401f5f9f4c7f6':
        raise ValueError('Train normalizer SHA mismatch')
    provenance = norm['provenance']
    if (provenance['train_events'], provenance['validation_inputs'], provenance['test_inputs']) != (250,0,0):
        raise ValueError('Train-only provenance differs')
    working_free=shutil.disk_usage('/kaggle/working').free
    scratch=Path('/kaggle/temp'); scratch.mkdir(exist_ok=True)
    scratch_free=shutil.disk_usage(scratch).free
    return dict(python=sys.version, versions=versions, parquet_rows=parquet_rows,
                normalizer_sha256=sha_file(norm_path), train_events=250, validation_fit_inputs=0, test_fit_inputs=0,
                working_free_bytes=working_free, scratch_free_bytes=scratch_free,
                disk_gate_for_future_training=working_free >= 20_000_000_000 and scratch_free >= 40_000_000_000)


def resource_check(deadline, min_available=2*1024**3):
    if time.monotonic() > deadline: raise TimeoutError('CPU certification deadline reached')
    mem = Path('/proc/meminfo')
    if mem.exists():
        available = next(int(row.split()[1])*1024 for row in mem.read_text().splitlines() if row.startswith('MemAvailable:'))
        if available < min_available: raise MemoryError('CPU certification memory reserve reached')


def certify_lot(canonical_path, dataset_root, out, *, pinned_identity, seconds=3600, inspect_geotiff=True):
    lot = json.loads(Path(canonical_path).read_text())
    unsigned = {k: v for k, v in lot.items() if k != 'identity'}
    identity = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != lot['identity'] or identity != pinned_identity or lot['role'] not in ('train','val'):
        raise ValueError('Pinned lot identity/role mismatch')
    dataset_root, out = Path(dataset_root), Path(out)
    if json.loads((dataset_root/'lot-inventory.json').read_text()) != lot:
        raise ValueError('Published inventory differs from pinned canonical manifest')
    expected = {f['path'] for f in lot['files']}
    if len(expected) != lot['observations'] or sum(f['bytes'] for f in lot['files']) != lot['bytes']:
        raise ValueError('Manifest counts differ')
    if {str(PurePosixPath(p).parent) for p in expected} != set(lot['events']):
        raise ValueError('Manifest event set differs')
    for p in expected:
        if PurePosixPath(p).parts[0] not in ('2018','2019','2020'): raise ValueError('Held-out year refused')
    rasters = dataset_root/'rasters'
    actual = {p.relative_to(rasters).as_posix() for p in rasters.rglob('*') if p.is_file()}
    if actual != expected: raise ValueError('Published raster inventory differs')
    out.mkdir(parents=True, exist_ok=False)
    deadline, checked_bytes = time.monotonic()+seconds, 0
    with (out/'verified.jsonl').open('x', encoding='utf-8') as log:
        for f in lot['files']:
            resource_check(deadline)
            path = safe_child(rasters, f['path'])
            if path.is_symlink() or path.stat().st_size != f['bytes']: raise ValueError('Published size/path differs')
            sha, crc, n = hashlib.sha256(), 0, 0
            with path.open('rb') as fp:
                for block in iter(lambda: fp.read(BLOCK), b''):
                    resource_check(deadline)
                    sha.update(block); crc=zlib.crc32(block,crc); n+=len(block)
            if n != f['bytes'] or sha.hexdigest() != f['sha256'] or f'{crc & 0xffffffff:08x}' != f['crc32']:
                raise ValueError('Published SHA256/CRC differs: '+f['path'])
            checked_bytes += n
            log.write(json.dumps(dict(path=f['path'], sha256=sha.hexdigest(), bytes=n,
                                      crc32=f'{crc & 0xffffffff:08x}'))+'\n'); log.flush()
    geotiff = None
    if inspect_geotiff:
        import rasterio
        from rasterio.windows import Window
        with rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_NUM_THREADS='1'):
            with rasterio.open(safe_child(rasters,lot['files'][0]['path'])) as src:
                tile = src.read(1, window=Window(0,0,min(64,src.width),min(64,src.height)), masked=True)
                if tile.size == 0: raise ValueError('Empty GeoTIFF window')
                geotiff = dict(bands=src.count, width=src.width, height=src.height, tile_shape=list(tile.shape), crs=str(src.crs))
    result = dict(status='ONE_LOT_REMOTE_BYTES_CERTIFIED', handle=lot['handle'], identity=identity,
                  files=len(expected), events=len(lot['events']), bytes=checked_bytes, role=lot['role'], geotiff=geotiff,
                  claim='This receipt certifies only this attached lot; no training or other lots were validated')
    with (out/'certificate.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    return result
