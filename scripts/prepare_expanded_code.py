"""Freeze small portable code/metadata package; never includes raw TIFFs or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(deployment, output, handle, *, root=None):
    """Read-only inputs and exclusive output: freezing never edits a deployment."""
    root = Path(root or Path.cwd()).resolve()
    bundle, out = Path(deployment).resolve(), Path(output)
    bundle.relative_to(root)
    import yaml
    configs = sorted((bundle/'configs').glob('*.yaml'))
    if len(configs) != 4:
        raise ValueError('Four reviewed model configurations required')
    for path in configs:
        c = yaml.safe_load(path.read_text(encoding='utf-8'))
        if (c.get('corpus_scope') != 'train_validation'
                or c.get('evaluation_scratch_root') != '/kaggle/temp/geofirex-ap-v1'
                or c.get('evaluation_cleanup_runs') is not True):
            raise ValueError('Prepare a new reviewed train/validation deployment with bounded AP scratch first')
    files = [root/'pyproject.toml']+list((root/'src/geofirex').rglob('*.py'))
    files += [root/'scripts'/name for name in ('run_benchmark.py', 'profile_expanded_gpu.py',
               'run_expanded_benchmark.py', 'mount_expanded_corpus.py')]
    files += [path for path in bundle.rglob('*') if path.is_file() and path.suffix in ('.json', '.parquet', '.csv', '.yaml')]
    manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    out.mkdir(parents=True, exist_ok=False)
    target = out/'payload.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('code-manifest.json', json.dumps(manifest, indent=2))
    with target.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    receipt = dict(handle=handle, payload_bytes=target.stat().st_size, sha256=sha, lot_identity=sha,
                   kind='code_and_metadata_only', files=len(files), raw_tiffs=0)
    with (out/'staging-receipt.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', default='reports/gpu_operational_v1/deployment-v3/bundle')
    p.add_argument('--out', default='artifacts/gpu_operational_v1/code-package')
    p.add_argument('--handle', required=True)
    a = p.parse_args()
    print(json.dumps(package(a.deployment, a.out, a.handle), indent=2))
