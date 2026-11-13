"""Small recovery regression checks runnable without Torch, Spark or Kaggle."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from geofirex.job_control import ProcessTree, supervise

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoveryTools(unittest.TestCase):
    def test_deadline_owns_venv_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'child.pid'
            child = f"import os,time,pathlib; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
            with (Path(directory)/'log').open('w') as log:
                result = supervise([sys.executable, '-c', parent], seconds=3, log=log)
            self.assertEqual(result['status'], 'HARD_TIME_LIMIT')
            self.assertLess(result['elapsed_seconds'], 10)
            self.assertTrue(pidfile.exists(), 'Child must have started for this ownership check')
            pid = int(pidfile.read_text())
            if os.name == 'nt':
                import ctypes as ct
                from ctypes import wintypes as wt
                kernel = ct.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
                kernel.OpenProcess.restype = wt.HANDLE
                kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
                kernel.CloseHandle.argtypes = [wt.HANDLE]
                handle = kernel.OpenProcess(0x00100000, False, pid)
                if handle:
                    try:
                        self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0)
                    finally:
                        kernel.CloseHandle(handle)
            else:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    # A terminated orphan may briefly remain a zombie on Linux.
                    self.assertIn('Z', Path(f'/proc/{pid}/stat').read_text().split()[2])

    def test_resource_error_stops_child(self):
        with tempfile.TemporaryDirectory() as directory, (Path(directory)/'log').open('w') as log:
            def fail():
                raise RuntimeError('fixture')
            result = supervise([sys.executable, '-c', 'import time; time.sleep(30)'], seconds=20, log=log, check=fail)
            self.assertEqual(result['status'], 'RESOURCE_LIMIT_OR_INTERRUPTION')
            self.assertEqual(result['resource_error'], 'RuntimeError')
            self.assertLess(result['elapsed_seconds'], 8)

    def test_stored_zip_prediction_matches_streamed_archive(self):
        audit = script('audit_gpu_recovery')
        for limit in (zipfile.ZIP64_LIMIT, 50):
            original = zipfile.ZIP64_LIMIT
            zipfile.ZIP64_LIMIT = limit
            try:
                lot = dict(files=[dict(path='2018/fire_1/2018-01-01.tif', bytes=37),
                                  dict(path='2020/fire_2/2020-02-01.tif', bytes=81)], identity='fixture')
                out = io.BytesIO()
                with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as archive:
                    for member in lot['files']:
                        with archive.open('rasters/'+member['path'], 'w', force_zip64=True) as dest:
                            dest.write(b'x'*member['bytes'])
                    archive.writestr('lot-inventory.json', json.dumps(lot, sort_keys=True))
                self.assertEqual(audit.stored_zip_bytes(lot), len(out.getvalue()))
            finally:
                zipfile.ZIP64_LIMIT = original

    def test_freezer_preserves_inputs_and_refuses_old_configs(self):
        import hashlib
        import yaml
        freezer = script('prepare_expanded_code')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root/'bundle'
            (bundle/'configs').mkdir(parents=True)
            (root/'src/geofirex').mkdir(parents=True)
            (root/'src/geofirex/__init__.py').write_text('')
            (root/'scripts').mkdir()
            for name in ('run_benchmark.py', 'profile_expanded_gpu.py', 'run_expanded_benchmark.py', 'mount_expanded_corpus.py'):
                (root/'scripts'/name).write_text('# fixture')
            (root/'pyproject.toml').write_text('# fixture')
            for model in ('B0', 'B1', 'B2', 'B3'):
                (bundle/'configs'/f'{model}.yaml').write_text(yaml.safe_dump(dict(corpus_scope='train_validation')))
            with self.assertRaisesRegex(ValueError, 'new reviewed'):
                freezer.package(bundle, root/'invalid', 'owner/code', root=root)
            self.assertFalse((root/'invalid').exists())
            for path in (bundle/'configs').glob('*.yaml'):
                path.write_text(yaml.safe_dump(dict(corpus_scope='train_validation',
                                evaluation_scratch_root='/kaggle/temp/geofirex-ap-v1', evaluation_cleanup_runs=True)))
            before = {p: p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
            receipt = freezer.package(bundle, root/'valid', 'owner/code', root=root)
            self.assertEqual(receipt['raw_tiffs'], 0)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with zipfile.ZipFile(root/'valid/payload.zip') as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read('code-manifest.json'))
                self.assertEqual(set(archive.namelist()), set(manifest)|{'code-manifest.json'})
                self.assertTrue(all(hashlib.sha256(archive.read(p)).hexdigest() == digest for p, digest in manifest.items()))
            with self.assertRaises(FileExistsError):
                freezer.package(bundle, root/'valid', 'owner/code', root=root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
