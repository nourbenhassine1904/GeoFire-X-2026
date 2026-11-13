"""Authorized local tooling only: <=600 MB cumulative HTTP bodies, no pip index.

Metadata and interrupted/retried reads share a durable conservative reservation
ledger. Completed packages require PyPI SHA-256; partial files are preserved.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid

from packaging.requirements import Requirement

from geofirex.acquisition_session import session_lock
from geofirex.archive_transfer import BodyLedger
from geofirex.corpus import json_new, sha256_file

LIMIT = 600_000_000
VERSIONS = {'pyspark': '4.0.2', 'py4j': '0.10.9.9'}


def metadata(name, version, root, ledger):
    path = root/f'{name}-{version}.metadata.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    bound = 1024**2
    ledger.reserve(bound)
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=60) as response:
        if response.status != 200 or int(response.headers.get('Content-Length', '0')) > bound:
            raise ValueError('Unexpected PyPI metadata response; body refused')
        data = response.read(bound)
    ledger.confirmed += len(data)
    ledger.record('metadata_read', package=name, bytes=len(data))
    result = json.loads(data)
    if result['info']['name'].lower() != name or result['info']['version'] != version:
        raise ValueError('PyPI package identity mismatch')
    json_new(path, result)
    return result


def select_artifact(info):
    wheel = [a for a in info['urls'] if a['filename'].endswith(('py3-none-any.whl', 'py2.py3-none-any.whl'))]
    artifact = min(wheel or [a for a in info['urls'] if a['packagetype'] == 'sdist'], key=lambda a: a['size'])
    if not artifact['url'].startswith('https://files.pythonhosted.org/') or Path(artifact['filename']).name != artifact['filename']:
        raise ValueError('Unexpected package source or filename')
    return dict(filename=artifact['filename'], url=artifact['url'], bytes=artifact['size'], sha256=artifact['digests']['sha256'])


def fetch_artifact(artifact, root, ledger, *, max_mbps=40, block_size=4*1024**2):
    target = root/artifact['filename']
    partial = Path(str(target)+'.part')
    if target.exists():
        if target.stat().st_size != artifact['bytes'] or sha256_file(target) != artifact['sha256']:
            raise ValueError('Existing package invalid; preserved')
        return target
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > artifact['bytes'] or offset > ledger.confirmed:
        raise ValueError('Oversized or unaccounted partial package; preserved')
    remaining = artifact['bytes']-offset
    if remaining:
        if shutil.disk_usage(root).free < remaining + 5*10**9:
            raise RuntimeError('Package plus 5 GB installation reserve unavailable')
        if ledger.cap-ledger.reserved < remaining:
            raise RuntimeError('Remaining authorized body budget cannot cover package; no request started')
        request = urllib.request.Request(artifact['url'], headers={'Accept-Encoding': 'identity',
            **({'Range': f"bytes={offset}-{artifact['bytes']-1}"} if offset else {})})
        first_read = min(block_size, remaining)
        ledger.reserve(first_read)  # Reserve BEFORE opening the network connection.
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset:
                expected = f"bytes {offset}-{artifact['bytes']-1}/{artifact['bytes']}"
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Resume response refused without consuming its body')
            elif response.status != 200:
                raise ValueError('Expected full package response')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != remaining:
                raise ValueError('Unexpected package length; body refused')
            with partial.open('ab') as output:
                first = True
                while offset < artifact['bytes']:
                    n = min(block_size, artifact['bytes']-offset)
                    if not first:
                        ledger.reserve(n)
                    first = False
                    started = time.monotonic()
                    data = response.read(n)
                    ledger.confirmed += len(data)
                    ledger.record('package_body_read', filename=target.name, bytes=len(data))
                    if not data:
                        raise OSError('Incomplete package response; partial retained')
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                    offset += len(data)
                    ledger.record('package_prefix_written', filename=target.name, offset=offset)
                    delay = len(data)*8/(max_mbps*10**6)-(time.monotonic()-started)
                    if delay > 0:
                        time.sleep(delay)
                    if offset//(32*1024**2) != (offset-len(data))//(32*1024**2):
                        print(f"{target.name}: {offset}/{artifact['bytes']} bytes; reserved {ledger.reserved}/{ledger.cap}", flush=True)
    if partial.stat().st_size != artifact['bytes'] or sha256_file(partial) != artifact['sha256']:
        raise ValueError('Package size/SHA-256 mismatch; partial retained')
    os.link(partial, target)  # Exclusive publication, no replacement or .part deletion.
    ledger.record('package_verified', **artifact)
    return target


def install(root, *, metadata_only=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with session_lock(root/'session'):
        ledger = BodyLedger(root/'http-body.jsonl', dict(budget_bytes=LIMIT, packages=VERSIONS,
                                                       scope='PyPI metadata and package HTTP bodies, all attempts'))
        try:
            infos = {name: metadata(name, version, root, ledger) for name, version in VERSIONS.items()}
            requirements = [Requirement(r) for r in infos['pyspark']['info']['requires_dist'] or []]
            py4j = next(r for r in requirements if r.name == 'py4j')
            if VERSIONS['py4j'] not in py4j.specifier:
                raise ValueError('Pinned Py4J does not satisfy PySpark requirements')
            plan = dict(versions=VERSIONS, artifacts=[select_artifact(info) for info in infos.values()])
            if not (root/'plan.json').exists():
                json_new(root/'plan.json', plan)
            elif json.loads((root/'plan.json').read_text()) != plan:
                raise ValueError('Cached package plan differs; preserved')
            if metadata_only:
                return dict(status='METADATA_VERIFIED', **plan, reserved=ledger.reserved, confirmed=ledger.confirmed, cap=ledger.cap)
            paths = [fetch_artifact(a, root, ledger) for a in plan['artifacts']]
            environment = {**os.environ, 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
            for name in ('PYSPARK_HADOOP_VERSION', 'PYSPARK_RELEASE_MIRROR'):
                environment.pop(name, None)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--no-build-isolation', '--no-cache-dir', *map(str, paths)], check=True, env=environment)
            installed = {name: importlib.metadata.version(name) for name in VERSIONS}
            if installed != VERSIONS:
                raise ValueError('Installed versions differ from approved package plan')
            result = dict(status='INSTALLED_OFFLINE_FROM_VERIFIED_PACKAGES', versions=installed,
                          confirmed_body_bytes=ledger.confirmed, reserved_body_bytes=ledger.reserved,
                          cap_bytes=ledger.cap, packages=plan['artifacts'], corpus_downloads=0)
            json_new(root/f'installed-{uuid.uuid4().hex}.json', result)
            return result
        finally:
            ledger.record('attempt_end')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='artifacts/local-spark-tooling-v1')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--metadata-only', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='PREPARED_NO_NETWORK', versions=VERSIONS, cap_bytes=LIMIT)))
        return
    print(json.dumps(install(args.root, metadata_only=args.metadata_only), indent=2))


if __name__ == '__main__':
    main()
