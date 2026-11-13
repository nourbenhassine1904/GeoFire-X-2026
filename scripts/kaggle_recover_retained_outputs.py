"""Cloud-only recovery from existing Kaggle outputs, never from Zenodo.

Prepared for review; not executed by the acquisition diagnostic. Valid destination
files are reused after CRC/size checks. No dataset publication or acquisition of
missing observations is implemented here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zlib


def digest(path):
    sha, crc, size = hashlib.sha256(), 0, 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024**2), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return {'bytes': size, 'crc32': f'{crc & 0xffffffff:08x}', 'sha256': sha.hexdigest()}


def recover(spec, destination, max_bytes):
    if not Path('/kaggle/working').is_dir():
        raise PermissionError('Recovery must run inside Kaggle; no raster download on the PC')
    destination = destination.resolve()
    if not destination.is_relative_to(Path('/kaggle/working')):
        raise ValueError('Destination must stay in /kaggle/working')
    if spec.get('execution_enabled') is not True:
        raise PermissionError('Prepared recovery remains disabled pending transfer-budget review')
    if max_bytes != spec['recovery_payload_cap_bytes']:
        raise ValueError('An exact reviewed recovery byte cap is required')
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests
    destination.mkdir(parents=True, exist_ok=True)
    ledger = destination/'recovery-transfers.jsonl'
    reserved = sum(json.loads(line)['reserved_bytes'] for line in ledger.read_text().splitlines()) if ledger.exists() else 0
    verified = []
    with api.build_kaggle_client() as client:
        for job in spec['jobs']:
            query = ApiListKernelSessionOutputRequest()
            query.user_name, query.kernel_slug = job['handle'].split('/')
            query.version_label = '1'
            query.page_size = 200
            urls = {}
            while True:
                response = client.kernels.kernels_api_client.list_kernel_session_output(query)
                urls.update((f.file_name, f.url) for f in response.files)
                if not response.next_page_token:
                    break
                query.page_token = response.next_page_token
            for entry in job['retained_entries']:
                relative = Path(entry['path'])
                target = (destination/'payload'/relative).resolve()
                if not target.is_relative_to(destination/'payload'):
                    raise ValueError('Unsafe relative path')
                expected = {'bytes': entry['geotiff_bytes'], 'crc32': entry['crc32']}
                if target.exists():
                    measured = digest(target)
                    if any(measured[k] != value for k,value in expected.items()):
                        raise ValueError('Existing destination differs; preserved without overwrite: '+entry['path'])
                else:
                    name = 'geofirex-acquisition/payload/'+entry['path']
                    if name not in urls:
                        raise ValueError('Retained source missing; no Zenodo fallback: '+entry['path'])
                    if reserved + expected['bytes'] > max_bytes:
                        raise PermissionError('Recovery byte cap reached; all files preserved')
                    with ledger.open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'time': time.time(), 'path': entry['path'],
                            'reserved_bytes': expected['bytes'], 'source': 'existing Kaggle output'})+'\n')
                        f.flush(); os.fsync(f.fileno())
                    reserved += expected['bytes']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')
                    with requests.get(urls[name], stream=True, timeout=(30,60)) as response:
                        response.raise_for_status()
                        with partial.open('xb') as sink:
                            size = 0
                            for block in response.iter_content(4*1024**2):
                                size += len(block)
                                if size > expected['bytes']:
                                    raise ValueError('Retained output exceeds reviewed size')
                                sink.write(block)
                    measured = digest(partial)
                    if any(measured[k] != value for k,value in expected.items()):
                        raise ValueError('CRC/size failure; partial preserved: '+entry['path'])
                    # Exclusive link prevents accidental replacement of any destination.
                    os.link(partial, target)
                    partial.unlink()
                row = {'path': entry['path'], 'event_id': entry['event_id'], **measured}
                verified.append(row)
                with (destination/'verified-files.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
    result = {'verified_files': len(verified), 'verified_bytes': sum(e['bytes'] for e in verified),
        'zenodo_download_bytes': 0, 'dataset_upload_bytes': 0, 'reserved_recovery_bytes': reserved}
    (destination/'recovery.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--max-recovery-bytes', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(recover(json.loads(Path(args.spec).read_text()),
        Path(args.destination), args.max_recovery_bytes), indent=2))
