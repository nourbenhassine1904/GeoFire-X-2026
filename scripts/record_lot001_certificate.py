"""Preserve downloaded Kaggle evidence; verify small receipts, never reread rasters."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path('reports/recovery_20261107_v3/finalized/deployment/bundle')


def verify_evidence(folder, lot):
    folder = Path(folder)
    certificate = json.loads((folder/'certificate.json').read_text())
    progress = json.loads((folder/'progress.json').read_text())
    rows = [json.loads(line) for line in (folder/'verified.jsonl').read_text().splitlines()]
    expected = {r['path']:r for r in lot['files']}
    actual = {r['path']:r for r in rows}
    if len(rows) != len(expected) or len(actual) != len(rows) or actual != expected:
        raise ValueError('Verified file paths, sizes, SHA256 or CRC32 differ from canonical manifest')
    total = sum(r['bytes'] for r in rows)
    if total != lot['bytes']:
        raise ValueError('Verified byte total differs')
    if (certificate['status'] != 'LOT001_REMOTE_CPU_CERTIFIED'
        or certificate['dataset'] != lot['handle'] or certificate['identity'] != lot['identity']
        or certificate['files_certified'] != len(rows) or certificate['bytes_certified'] != total
        or certificate['events'] != len(lot['events']) or certificate['role'] != lot['role']
        or certificate['anomalies'] != [] or certificate['gpu_used'] is not False
        or certificate['training'] is not False):
        raise ValueError('Certificate scope/status/counts/anomalies differ')
    inventory = certificate['inventory']
    if (inventory['actual_geotiffs'] != len(rows) or inventory['total_attached_files'] != len(rows)+1
        or inventory['heldout_2021_files'] != 0 or inventory['manifest_matches_local'] is not True
        or inventory['forbidden_2020_2021_observations'] != 0):
        raise ValueError('Certificate inventory mismatch')
    if len(certificate['rasterio_reads']) < 3 or any(r['bands'] != 23 for r in certificate['rasterio_reads']):
        raise ValueError('GeoTIFF read evidence missing')
    if certificate['aligned_windows'] != {'1':984,'3':984,'5':984}:
        raise ValueError('Temporal target count mismatch')
    reads = certificate['loader_reads']
    if {r['context_days'] for r in reads} != {1,3,5} or any(
            r['device'] != 'cpu' or r['X'][:3] != [1,r['context_days'],8] or r['finite'] is not True for r in reads):
        raise ValueError('CPU DataLoader evidence mismatch')
    if certificate['path_compatibility']['windows_drive_dependency'] is not False:
        raise ValueError('Unexpected Windows drive dependence')
    for key in ('dataset','identity','files_certified','bytes_certified','anomalies','gpu_used','training'):
        if progress[key] != certificate[key]:
            raise ValueError('Progress snapshot differs: '+key)
    return dict(status='DOWNLOADED_LOT001_EVIDENCE_VERIFIED',dataset=lot['handle'],
        identity=lot['identity'],files=1269,bytes=total,events=57,anomalies=[],
        crc32_and_sha256_rows_match_local_manifest=True,
        progress_status=progress['status'],
        progress_note='RUNNING is the final hash-phase snapshot; certificate.json records completed raster/DataLoader checks',
        remote_certification_rerun=False)


def main(source, destination):
    lot = json.loads((BASE/'metadata/lot-001.json').read_text())
    source, destination = Path(source), Path(destination)
    verify_evidence(source, lot)
    destination.mkdir(parents=True,exist_ok=False)
    receipts = {}
    for name in ('certificate.json','verified.jsonl','progress.json'):
        origin = source/name
        with origin.open('rb') as inp, (destination/name).open('xb') as out:
            shutil.copyfileobj(inp,out,1024**2)
        before = hashlib.sha256(origin.read_bytes()).hexdigest()
        after = hashlib.sha256((destination/name).read_bytes()).hexdigest()
        if before != after:
            raise ValueError('Evidence copy hash mismatch')
        receipts[name] = dict(source=str(origin.resolve()),bytes=(destination/name).stat().st_size,sha256=after)
    result = verify_evidence(destination,lot)
    result['evidence_files'] = receipts
    (destination/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('C:/Users/nourb/Downloads'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    main(a.source,a.out)
