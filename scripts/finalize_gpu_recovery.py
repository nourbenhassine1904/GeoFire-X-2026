"""Finish safe local recovery once RAM is available. No upload or GPU execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def verify_validation(directory):
    directory = Path(directory)
    expected = load_script('validate_gpu_isolated').essential_nodes()
    summary = json.loads((directory/'summary.json').read_text())
    if (summary['status'] != 'ESSENTIAL_TESTS_PASSED' or summary['requested'] != len(expected)
            or len(summary['results']) != len(expected) or summary['remaining']):
        raise ValueError('Complete isolated essential validation is required')
    for index, (node, result) in enumerate(zip(expected, summary['results']), 1):
        if result['node'] != node or result['status'] != 'FINISHED' or result['exit_code'] != 0:
            raise ValueError('An essential test did not finish successfully')
        suite = ET.parse(directory/f'test-{index:02d}.xml').find('testsuite')
        if suite is None or any(int(suite.get(key, '0')) != value for key, value in
                                dict(tests=1, errors=0, failures=0, skipped=0).items()):
            raise ValueError('Each isolated test requires one passed JUnit case')
    bindings = json.loads((directory/'source-manifest.json').read_text())
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha for name, sha in bindings.items()):
        raise ValueError('Tested sources changed; package freezing refused')
    return dict(status=summary['status'], tests=len(expected), evidence=str(directory.resolve()),
                source_manifest_sha256=hashlib.sha256((directory/'source-manifest.json').read_bytes()).hexdigest())


def prepare_notebook(out, bundle, receipt, plan):
    """CPU preflight notebook. GPU commands are printed for a later approval."""
    kernel = out/'kaggle-cpu-preflight'
    kernel.mkdir()
    relative = bundle.relative_to(ROOT).as_posix()
    cells = []
    def cell(kind, content):
        value = dict(id=f'cell-{len(cells):02d}', cell_type=kind, metadata={}, source=content.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell('markdown', '# GeoFire-X : contrôle CPU du corpus train/validation\n'
         'Notebook privé, accélérateur désactivé. Attacher le code et les huit lots après autorisation de publication. '
         'Aucun TIFF 2021. Aucun entraînement ou profilage GPU automatique.\n')
    cell('code', f'''from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile
payload = Path('/kaggle/input/{receipt['handle'].split('/')[1]}/payload.zip')
with payload.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == {receipt['sha256']!r}
work = Path('/kaggle/working/geofirex-expanded-recovery')
work.mkdir(exist_ok=False)
with zipfile.ZipFile(payload) as archive:
    manifest = json.loads(archive.read('code-manifest.json'))
    assert len(archive.namelist()) == len(set(archive.namelist()))
    assert set(archive.namelist()) == set(manifest) | {{'code-manifest.json'}}
    for name in archive.namelist():
        target = (work/name).resolve()
        assert target.is_relative_to(work.resolve())
        data = archive.read(name)
        if name in manifest:
            assert hashlib.sha256(data).hexdigest() == manifest[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
os.chdir(work)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.'], check=True)
subprocess.run([sys.executable, '-c', 'import torch,pyarrow,psutil,yaml,mlflow'], check=True)
''')
    cell('code', f'''# CPU : vérification des lots déjà attachés, aucune publication.
subprocess.run([sys.executable, 'scripts/mount_expanded_corpus.py', '--bundle', {relative!r},
                '--out', '/kaggle/working/expanded-mount-v1'], check=True)
''')
    cell('code', '''# Après autorisation GPU et activation manuelle de CUDA : B0/B1 d'abord.
# Ces commandes sont affichées, jamais exécutées par ce notebook CPU.
for model in ('B0', 'B1', 'B2', 'B3'):
    print(f"python scripts/profile_expanded_gpu.py --config /kaggle/working/expanded-mount-v1/{model}_T3_seed42.yaml --out /kaggle/working/profile-{model}-v1 --seconds 900 --steps 50 --warmup 5")
''')
    write_json(kernel/'cpu-preflight.ipynb', dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3')), nbformat=4, nbformat_minor=5))
    owner = receipt['handle'].split('/')[0]
    write_json(kernel/'kernel-metadata.json', dict(id=f'{owner}/geofirex-expanded-cpu-recovery-v1',
        title='GeoFire-X expanded CPU recovery v1', code_file='cpu-preflight.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
        dataset_sources=[receipt['handle']]+[lot['handle'] for lot in plan['lots']], competition_sources=[], kernel_sources=[]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='New report directory inside this checkout')
    p.add_argument('--raw-root', default='D:/GeoFire-X/WildfireSpreadTS/rasters')
    p.add_argument('--handle', default='nnnnbhss/geofirex-expanded-code-recovery-v1')
    p.add_argument('--validation', help='Reuse complete isolated receipts only if all tested source hashes still match')
    a = p.parse_args()
    out = Path(a.out).resolve()
    if not out.is_relative_to(ROOT):
        p.error('Output must stay inside this checkout')
    out.mkdir(parents=True, exist_ok=False)
    bounded = load_script('run_bounded_local')
    resources = bounded.memory()
    if resources and resources['available'] < 3*1024**3:
        write_json(out/'status.json', dict(status='WAITING_FOR_3_GIB_FREE_RAM', memory=resources, children_started=0))
        print('No tests started: free at least 3 GiB RAM, then use a NEW --out directory.')
        return 1
    validation = Path(a.validation) if a.validation else out/'isolated-validation'
    if not a.validation:
        result = subprocess.run([sys.executable, 'scripts/validate_gpu_isolated.py', '--out', str(validation)],
                                cwd=ROOT, timeout=1800)
        if result.returncode:
            write_json(out/'status.json', dict(status='STOPPED_WITHOUT_RETRY', stage='isolated-validation'))
            return 1
    validated = verify_validation(validation)
    write_json(out/'validation-proof.json', validated)
    command = [sys.executable, 'scripts/audit_gpu_recovery.py', '--metadata-only', '--raw-root', a.raw_root,
               '--out', str(out/'metadata-integrity.json')]
    result = subprocess.run([sys.executable, 'scripts/run_memory_guarded.py', '--out', str(out/'metadata-audit'),
                             '--seconds', '120', '--']+command, cwd=ROOT, timeout=150)
    if result.returncode:
        write_json(out/'status.json', dict(status='STOPPED_WITHOUT_RETRY', stage='metadata-audit'))
        return 1
    import yaml
    bundle = out/'deployment/bundle'
    shutil.copytree(ROOT/'reports/gpu_operational_v1/deployment-v3/bundle', bundle)
    prefix = bundle.relative_to(ROOT).as_posix()
    paths = dict(manifest='observations.parquet', window_manifest='windows.parquet', split='fold_00.csv', index='index.json')
    for path in (bundle/'configs').glob('*.yaml'):
        c = yaml.safe_load(path.read_text(encoding='utf-8'))
        c.update({k: f'{prefix}/metadata/{v}' for k, v in paths.items()})
        c.update(normalization_artifact=f'{prefix}/metadata/normalization.json',
                 evaluation_scratch_root='/kaggle/temp/geofirex-ap-v1', evaluation_cleanup_runs=True)
        path.write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    for folder in ('src/geofirex', 'scripts'):
        for path in (ROOT/folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    receipt = load_script('prepare_expanded_code').package(bundle, out/'code-package', a.handle)
    with zipfile.ZipFile(out/'code-package/payload.zip') as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('code-manifest.json'))
        assert set(archive.namelist()) == set(manifest)|{'code-manifest.json'}
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == sha for name, sha in manifest.items())
    plan = json.loads((bundle/'deployment.json').read_text(encoding='utf-8'))
    prepare_notebook(out, bundle, receipt, plan)
    write_json(out/'status.json', dict(status='LOCAL_ESSENTIAL_TESTS_AND_PACKAGE_VALIDATED', code_package=receipt,
               validation=validated, complete_reference_suite_validated=False,
               historical_cpu_integration='Reused after metadata bindings audit; no redundant raster replay',
               uploads=0, gpu_jobs=0, commits=0, private_quota='Refresh authenticated account before any upload or GPU job'))
    print('Local validation finished. No upload or GPU job launched:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
