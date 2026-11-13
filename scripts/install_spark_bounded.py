"""Bounded, hash-checked PyPI tooling downloads, only in a cloud working dir."""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


def install_spark(root, limit=600*1024**2):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('This authorized dependency budget is cloud-only')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    remaining = limit-2*1024**2  # metadata and small bootstrap tooling reserve
    packages = []
    def metadata(name, version):
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=60) as response:
            data = response.read(1024**2+1)
        if len(data) > 1024**2:
            raise ValueError('PyPI metadata exceeds reserved budget')
        return json.loads(data)
    spark = metadata('pyspark', '3.5.9')
    from packaging.requirements import Requirement
    requirement = Requirement(next(r for r in spark['info']['requires_dist'] if r.startswith('py4j')))
    py4j_version = '0.10.9.9'
    if py4j_version not in requirement.specifier:
        raise ValueError('Pinned Py4J no longer satisfies the PySpark package metadata')
    py4j = metadata('py4j', py4j_version)
    for info in (spark, py4j):
        candidates = [f for f in info['urls'] if f['filename'].endswith('py2.py3-none-any.whl')
                      or f['filename'].endswith('py3-none-any.whl')]
        artifact = min(candidates or info['urls'], key=lambda f:f['size'])
        if artifact['size'] > remaining:
            raise RuntimeError('Approved dependency download cap would be exceeded')
        target = root/artifact['filename']
        digest, received = hashlib.sha256(), 0
        with urllib.request.urlopen(artifact['url'], timeout=120) as response, target.open('xb') as output:
            while True:
                block = response.read(min(1024**2, remaining+1))
                if not block:
                    break
                received += len(block)
                remaining -= len(block)
                if remaining < 0:
                    raise RuntimeError('Dependency hard download cap exceeded')
                digest.update(block)
                output.write(block)
        if received != artifact['size'] or digest.hexdigest() != artifact['digests']['sha256']:
            raise ValueError('PyPI artifact size/SHA256 mismatch')
        packages.append({'path': str(target), 'bytes': received, 'sha256': digest.hexdigest()})
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
        '--no-build-isolation', *[p['path'] for p in packages]], check=True)
    return {'cap_bytes': limit, 'artifact_bytes': sum(p['bytes'] for p in packages), 'packages': packages}
