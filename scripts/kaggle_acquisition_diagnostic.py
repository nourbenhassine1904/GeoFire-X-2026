"""Read-only official Kaggle API inventory; never downloads raster payloads."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sizes', action='store_true', help='One-byte HTTP Range probes; never full raster bodies')
    args = parser.parse_args()
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiListKernelSessionOutputRequest)
    root = Path(__file__).resolve().parents[1]
    report = {'time': dt.datetime.now(dt.timezone.utc).isoformat(),
              'full_raster_files_downloaded_to_pc': 0, 'jobs': []}
    with api.build_kaggle_client() as client:
        for job_id in ['20261031-205829-923ffb32', '20261031-205845-9e2ab980']:
            local = root/'artifacts/orchestrator/jobs'/job_id
            spec = json.loads((local/'job.json').read_text())
            status = api.kernels_status(spec['handle'])
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = spec['handle'].split('/')
            remote = client.kernels.kernels_api_client.get_kernel(request)
            row = {'job_id': job_id, 'handle': spec['handle'],
                   'remote_status': status.status.name, 'failure_message': status.failure_message,
                   'source_sha256': hashlib.sha256(remote.blob.source.encode()).hexdigest(),
                   'version': remote.metadata.current_version_number,
                   'files': [], 'logs': [], 'pages': 0}
            query = ApiListKernelSessionOutputRequest()
            query.user_name, query.kernel_slug = request.user_name, request.kernel_slug
            query.page_size = 200
            tokens = set()
            payload_urls = []
            while True:
                response = client.kernels.kernels_api_client.list_kernel_session_output(query)
                row['pages'] += 1
                row['files'].extend(f.file_name for f in response.files)
                if args.sizes:
                    payload_urls.extend((f.file_name, f.url) for f in response.files
                        if f.file_name.endswith(('.tif', '.part')))
                if response.log and response.log not in row['logs']:
                    row['logs'].append(response.log)
                if not response.next_page_token:
                    break
                if response.next_page_token in tokens:
                    raise ValueError('Repeated pagination token')
                tokens.add(response.next_page_token)
                query.page_token = response.next_page_token
            row['files'] = sorted(set(row['files']))
            row['tiff_paths'] = [p for p in row['files'] if p.lower().endswith('.tif')]
            row['partial_paths'] = [p for p in row['files'] if p.endswith('.part')]
            row['output_file_count'] = len(row['files'])
            row['tiff_file_count'] = len(row['tiff_paths'])
            if args.sizes:
                import concurrent.futures
                import requests
                def head(item):
                    name, url = item
                    try:
                        # The output endpoint rejects HEAD (404), but supports
                        # a one-byte Range response. Never consume a 200 body.
                        with requests.get(url, headers={'Range': 'bytes=0-0'},
                                stream=True, timeout=(10, 20)) as response:
                            content_range = response.headers.get('Content-Range', '')
                            return {'path': name, 'status': response.status_code,
                                'bytes': int(content_range.rsplit('/', 1)[1])
                                    if response.status_code == 206 and content_range.startswith('bytes 0-0/')
                                    else 0 if response.status_code == 416 and content_range == 'bytes */0' else -1,
                                'storage_hash': response.headers.get('x-goog-hash')}
                    except Exception as error:
                        return {'path': name, 'error_type': type(error).__name__}
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                    row['payload_metadata'] = list(pool.map(head, payload_urls))
                row['range_probe_response_body_max_bytes'] = len(payload_urls)
            report['jobs'].append(row)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({**report, 'jobs': [{k:v for k,v in j.items()
        if k not in {'files', 'tiff_paths', 'partial_paths', 'payload_metadata'}} for j in report['jobs']]}, indent=2))


if __name__ == '__main__':
    main()
