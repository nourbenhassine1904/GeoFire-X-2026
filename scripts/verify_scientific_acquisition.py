"""Post-transfer CRC/SHA/header verification and portable PyTorch manifest.

No network, pixel statistics, training, or writes to historical data.
"""
import argparse
import csv
import json
from pathlib import Path

import rasterio

from geofirex.corpus import json_new,sha256_file
from geofirex.scientific_scaleup import stage_download_entries
from geofirex.unet_data import validate_layout
from geofirex.runner import load_config
from geofirex.runner_data import FoldData


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-config',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    settings=json.loads(Path(args.launch_config).read_text(encoding='utf-8'))
    plan=json.loads(Path(settings['plan']).read_text(encoding='utf-8'))
    index_path=Path('data/manifests/real/corpus_zip_index_v1.json')
    index=json.loads(index_path.read_text(encoding='utf-8'))
    session=Path(settings['session_dir'])
    certificates=list(session.glob('complete-*.json'))
    if not certificates: raise RuntimeError('Acquisition incomplete: no completion certificate')
    complete=json.loads(max(certificates,key=lambda p:p.stat().st_mtime).read_text(encoding='utf-8'))
    if complete['plan_identity']!=plan['identity']: raise ValueError('Completion plan mismatch')
    expected_hashes={r['path']:r['sha256'] for r in complete['files']}
    roots=[Path(settings['destination']),*map(Path,settings['reuse_root'])]
    missing,locations=stage_download_entries(plan,index,roots)
    if missing: raise RuntimeError(f'{len(missing)} observations missing')
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=False)
    rows,geometries,files=[],{},[]
    lookup={e['path']:e for e in plan['entries']}
    for located in locations:
        entry=lookup[located['path']]
        path=Path(located['locations'][0])
        sha=sha256_file(path)
        if sha!=expected_hashes[entry['path']]: raise ValueError(f'SHA256 changed: {path}')
        if len(located['locations'])!=1: raise ValueError('Ambiguous raster across mounts; files preserved')
        with rasterio.open(path) as ds:
            validate_layout(ds)
            if ds.crs is None: raise ValueError(f'Missing CRS: {path}')
            geometry=(ds.height,ds.width,str(ds.crs),tuple(ds.transform))
            if geometries.setdefault(entry['event_id'],geometry)!=geometry:
                raise ValueError(f'Event geometry differs: {entry["event_id"]}')
            rows.append(dict(event_id=entry['event_id'],date=entry['date'],year=entry['year'],
                             relative_path=entry['path'],height=ds.height,width=ds.width))
            files.append(dict(path=entry['path'],sha256=sha,bytes=path.stat().st_size,
                              crc32=entry['crc32'],crs=str(ds.crs),transform=list(ds.transform)))
    with (output/'manifest.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    config=load_config('configs/runner_sample_gpu_pilot.yaml',dict(
        qualification='pilot',manifest=str((output/'manifest.csv').resolve()),
        data_root=str(roots[0].resolve()),data_roots=[str(root.resolve()) for root in roots],
        split='splits/benchmark/annual_v1/fold_00.csv',index=str(index_path.resolve()),device='cpu'))
    data=FoldData(config)  # Metadata contract only: no model, no training, no pixel reads.
    json_new(output/'mounts.json',[str(root.resolve()) for root in roots])
    json_new(output/'verification.json',dict(plan_identity=plan['identity'],events=len(geometries),
        observations=len(rows),crc_sha256_and_headers_verified=True,files=files,
        aligned_windows={role:len(groups[5]) for role,groups in data.groups.items()},
        manifest_sha256=sha256_file(output/'manifest.csv'),test2021_observations=0,
        scope='acquired stage only; raster headers and file integrity; no pixel rarity audit or GPU'))
    print(json.dumps(dict(events=len(geometries),observations=len(rows),
                         aligned_windows={role:len(groups[5]) for role,groups in data.groups.items()}),indent=2))


if __name__=='__main__': main()
