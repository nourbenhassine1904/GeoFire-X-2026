"""Prepare a CPU-only, read-only integrity job over retained acquisition outputs."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import uuid


DRIVER = r'''
import hashlib, json, signal, time, uuid, zlib
from pathlib import Path
started = time.monotonic()
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600s read-only integrity deadline')))
signal.alarm(600)
out = Path('/kaggle/working/retained-verification')
out.mkdir()
report = {'session': uuid.uuid4().hex, 'kind': 'retained_output_integrity',
    'zenodo_download_bytes': 0, 'payload_upload_bytes': 0, 'jobs': []}
try:
    paths = list(Path('/kaggle/input').rglob('*.tif'))
    if not any('/geofirex-acquisition/payload/' in p.as_posix() for p in paths):
        raise RuntimeError('KAGGLE_REJECTED_CANCELLED_KERNEL_SOURCES: no retained payload attached')
    for spec in SPECS:
        rows = []
        for entry in spec['entries']:
            suffix = '/geofirex-acquisition/payload/'+entry['path']
            matches = [p for p in paths if p.as_posix().endswith(suffix)]
            row = {'path': entry['path'], 'event_id': entry['event_id'], 'valid': False}
            if len(matches) == 1:
                p = matches[0]
                sha, crc, size = hashlib.sha256(), 0, 0
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(4*1024**2), b''):
                        sha.update(block); crc = zlib.crc32(block, crc); size += len(block)
                row.update(bytes=size, sha256=sha.hexdigest(), crc32=f'{crc & 0xffffffff:08x}')
                row['valid'] = size == entry['geotiff_bytes'] and row['crc32'] == entry['crc32']
            else:
                row['error'] = 'MISSING' if not matches else 'AMBIGUOUS'
            rows.append(row)
        complete = [event for event in sorted({e['event_id'] for e in rows})
            if all(e['valid'] for e in rows if e['event_id'] == event)]
        valid = [e for e in rows if e['valid']]
        report['jobs'].append({'job_id': spec['job_id'], 'plan_identity': spec['plan_identity'],
            'source_handle': spec['handle'], 'files': rows, 'valid_files': len(valid),
            'valid_bytes': sum(e['bytes'] for e in valid), 'complete_events': complete,
            'complete_event_count': len(complete), 'planned_files': len(rows),
            'valid_compressed_payload_bytes': sum(e['compressed_bytes'] for e in spec['entries']
                if any(r['path']==e['path'] and r['valid'] for r in rows))})
    report['partial_files'] = [{'path': p.as_posix(), 'bytes': p.stat().st_size}
        for p in Path('/kaggle/input').rglob('*.part')]
    report['success'] = True
except Exception as error:
    report.update(success=False, error_type=type(error).__name__, message=str(error)[:300])
report['seconds'] = time.monotonic()-started
(out/'integrity.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='jobs'}), flush=True)
for row in report['jobs']:
    print(json.dumps({k:v for k,v in row.items() if k!='files'}), flush=True)
'''


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    specs = []
    for job_id in ['20261031-205829-923ffb32', '20261031-205845-9e2ab980']:
        job = json.loads((root/'artifacts/orchestrator/jobs'/job_id/'job.json').read_text())
        specs.append({'job_id': job_id, 'handle': job['handle'],
            'plan_identity': job['plan']['identity'], 'entries': job['plan']['entries']})
    job_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    folder = root/'artifacts/orchestrator/retained-verification'/job_id
    folder.mkdir(parents=True)
    handle = 'nnnnbhss/geofirex-retained-'+job_id
    source = 'SPECS = '+repr(specs)+'\n'+DRIVER
    (folder/'job.py').write_text(source, encoding='utf-8')
    metadata = {'id': handle, 'title': handle.split('/')[1], 'code_file': 'job.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False, 'dataset_sources': [],
        'kernel_sources': [s['handle'] for s in specs], 'competition_sources': [], 'model_sources': []}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (folder/'scope.json').write_text(json.dumps({'handle': handle,
        'purpose': 'Read-only SHA256/CRC32/size verification of retained files; no acquisition',
        'max_seconds': 600, 'gpu': False, 'internet': False, 'zenodo_download_bytes': 0,
        'payload_upload_bytes': 0, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}, indent=2))
    print(folder)
