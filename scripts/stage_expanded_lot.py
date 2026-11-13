"""Stage one immutable certified lot; stream-check while archiving, no network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import zlib


def stage(lot_path, raw_root, output, *, max_bytes=6_000_000_000, reserve=20_000_000_000):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    lot = json.loads(Path(lot_path).read_text(encoding='utf-8'))
    identity = lot.pop('identity')
    if digest(lot) != identity or lot['role'] not in ('train', 'val'):
        raise ValueError('Invalid lot identity/role')
    lot['identity'] = identity
    if sum(f['bytes'] for f in lot['files']) != lot['bytes'] or lot['bytes']+10_000_000 > max_bytes:
        raise ValueError('Lot exceeds staging budget')
    raw_root, output = Path(raw_root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < lot['bytes']+10_000_000+reserve:
        raise RuntimeError('Insufficient staging capacity and 20 GB reserve')
    part = output/'payload.zip.part'
    with zipfile.ZipFile(part, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for f in lot['files']:
            token = f['path']
            if token.split('/')[0] not in ('2018', '2019', '2020') or '..' in token.split('/') or '\\' in token:
                raise ValueError('Unsafe/test raster in lot')
            source = (raw_root/token).resolve()
            if not source.is_relative_to(raw_root) or source.stat().st_size != f['bytes']:
                raise ValueError('Source path/size changed')
            sha, crc, n = hashlib.sha256(), 0, 0
            with source.open('rb') as src, archive.open('rasters/'+token, 'w', force_zip64=True) as dest:
                for block in iter(lambda: src.read(1024*1024), b''):
                    if shutil.disk_usage(output).free < reserve+len(block):
                        raise RuntimeError('Staging reserve reached; partial ZIP retained')
                    dest.write(block)
                    sha.update(block)
                    crc = zlib.crc32(block, crc)
                    n += len(block)
            if n != f['bytes'] or sha.hexdigest() != f['sha256'] or f'{crc & 0xffffffff:08x}' != f['crc32']:
                raise ValueError(f'Source integrity changed: {token}; raw and .part retained')
        archive.writestr('lot-inventory.json', json.dumps(lot, sort_keys=True))
    if part.stat().st_size > max_bytes:
        raise RuntimeError('Archive budget exceeded; .part retained')
    target = output/'payload.zip'
    if target.exists():
        raise FileExistsError(target)
    part.rename(target)
    with target.open('rb') as fp:
        archive_sha = hashlib.file_digest(fp, 'sha256').hexdigest()
    receipt = dict(lot_identity=identity, handle=lot['handle'], payload_bytes=target.stat().st_size,
                   sha256=archive_sha, observations=lot['observations'])
    with (output/'staging-receipt.json').open('x', encoding='utf-8') as fp:
        json.dump(receipt, fp, indent=2)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lot', required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-bytes', type=int, default=6_000_000_000)
    a = p.parse_args()
    print(json.dumps(stage(a.lot, a.raw_root, a.out, max_bytes=a.max_bytes), indent=2))
