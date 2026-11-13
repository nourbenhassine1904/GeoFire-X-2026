"""Finalize disjoint event partitions without copying compressed raster payloads.

Root manifest is metadata only; archives remain one file per event. File paths
gain partition prefixes. Existing parts and their evidence are never rewritten.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from geofirex.mission_streaming import save,sha,CompactTiles

def finalize(root):
    root=Path(root);parts=sorted(p for p in root.glob('part*') if p.is_dir())
    reports=[];events=[];hashes=set()
    for p in parts:
        report=json.loads((p/'report.json').read_text());proof=json.loads((p/'pytorch-verification.json').read_text());assert proof['status']=='PASS'
        reports.append(report);events+=report['event_ids'];hashes.add(report['normalization_sha256'])
    assert len(events)==len(set(events))==451 and len(hashes)==1
    assert sum(r['train_events'] for r in reports)==250 and sum(r['val_events'] for r in reports)==201
    assert sum(r['observations'] for r in reports)==9490 and sum(r['windows'] for r in reports)==7235
    for name in ('shards.json','samples.json'):
        with (root/name).open('x',encoding='utf-8') as f:
            f.write('[');first=True
            for p in parts:
                rows=json.loads((p/name).read_text())
                for row in rows:
                    if name=='shards.json':row['path']=p.name+'/'+row['path']
                    else:
                        for key in ('context','domain'):row[key]=[p.name+'/'+s for s in row[key]]
                        row['target']=p.name+'/'+row['target']
                    if not first:f.write(',\n')
                    json.dump(row,f,allow_nan=False);first=False
                del rows
            f.write(']\n')
    # Same exact bytes, no normalization refit.
    with (root/'normalization.json').open('xb') as dst,(parts[0]/'normalization.json').open('rb') as src:shutil.copyfileobj(src,dst)
    categories={}
    for role in ('train','val'):
        counts=Counter()
        for r in reports:counts.update(r['categories'][role])
        categories[role]=dict(counts)
    payload=[p for p in root.rglob('*.zip')]+[root/f for f in ('shards.json','samples.json','normalization.json')]
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(payload)]
    archive_bytes=sum(f['bytes'] for f in files if f['path'].endswith('.zip'));total=sum(f['bytes'] for f in files)
    report=dict(status='EXECUTED_AND_PARTITIONS_VERIFIED',format='one ZIP per event containing lossless float32 NPZ day tiles; temporal references',train_events=250,val_events=201,events=451,observations=9490,windows=7235,samples=sum(r['samples'] for r in reports),shards=sum(r['shards'] for r in reports),raw_geotiff_bytes=sum(r['raw_geotiff_bytes'] for r in reports),logical_array_bytes=sum(r['logical_array_bytes'] for r in reports),archive_bytes=archive_bytes,metadata_bytes=total-archive_bytes,transfer_payload_bytes=total,payload_files=len(files),categories=categories,normalization_sha256=hashes.pop(),conversion_worker_seconds=sum(r['elapsed_seconds'] for r in reports),longest_partition_seconds=max(r['elapsed_seconds'] for r in reports),peak_rss_bytes_by_partition=[r['peak_rss_bytes'] for r in reports],roundtrip_days=sum(r['roundtrip_days'] for r in reports),test_pixels_opened=0,network_bytes=0,validation='all official events and every nonoverlapping tile; no rebalancing',source_checks='existing certified original SHA referenced; scene size/CRS/transform checked during conversion; first tile of each day roundtripped; sampled tensors/targets/joint masks compared exactly with original TIFFs in each partition')
    save(root/'report.json',report)
    # Ready to review budget includes this manifest and report, but not receipts
    # from working partitions (they are not required for Kaggle loading).
    manifest=dict(files=files,payload_bytes=total,report_bytes=(root/'report.json').stat().st_size,extra_files=['report.json'],note='Upload only listed files plus report.json and this transfer-manifest.json; protocol overhead/retries are additional; no upload executed')
    save(root/'transfer-manifest.json',manifest)
    save(root/'transfer-budget.json',dict(payload_bytes=total,report_bytes=(root/'report.json').stat().st_size,manifest_bytes=(root/'transfer-manifest.json').stat().st_size,exact_application_file_bytes=total+(root/'report.json').stat().st_size+(root/'transfer-manifest.json').stat().st_size,protocol_overhead='unknown; transport and retry dependent',upload_executed=False))
    from torch.utils.data import DataLoader
    for role in ('train','val'):
        ds=CompactTiles(root,role);batch=next(iter(DataLoader(ds,batch_size=4,num_workers=0)));assert list(batch[0].shape)==[4,3,8,64,64]
        assert {s['event_id'] for s in ds.samples}=={e for e in events if (e.startswith('2020/'))==(role=='val')}
        del ds
    save(root/'pytorch-verification.json',dict(status='PASS',train_events=250,val_events=201,batch_shape=[4,3,8,64,64],parts_verified=len(parts),test_pixels_opened=0))
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();finalize(a.root)
