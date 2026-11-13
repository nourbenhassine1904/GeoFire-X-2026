"""Offline standalone profile: budget, fail-closed memory, sequencing and release."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as N
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import kaggle_autonomous as auto
from resume_remaining_lots import continue_after_confirmed_indexing, reserved
from run_memory_guarded import run


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@contextmanager
def fixture():
    before = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            base = Path('reports/phase')
            execution = base/'resume-01/execution'
            execution.mkdir(parents=True)
            (execution.parent/'autonomous-01').mkdir()
            journal = base/'http-body.jsonl'
            write(journal, dict(previous=0, reserved=auto.INITIAL_BODY_BYTES, ceiling=auto.CEILING_BYTES))
            specs = []
            for n in range(2, 9):
                tag = f'lot-{n:03d}'
                entry = dict(path=f'2018/fire_{n}/2018-01-01.tif', bytes=7,
                             sha256=hashlib.sha256(b'fixture').hexdigest(), crc32='00000000')
                canonical = base/f'metadata/{tag}.json'
                lot = dict(files=[entry])
                write(canonical, lot)
                specs.append(dict(number=n, tag=tag, files=1, events=1, raw_bytes=7,
                    role='train', identity=f'identity{n}', handle=f'nnnnbhss/{tag}',
                    canonical=str(canonical), archive_ceiling_bytes=10_000_007,
                    staging=f'artifacts/kaggle_remaining_20261108_v1/{tag}'))
            plan = dict(lots=specs, raw_root='fixture-raw', journal=str(journal))
            path = base/'plan.json'
            write(path, plan)
            yield path, plan, execution
        finally:
            os.chdir(before)


def cpu_receipt(folder, spec):
    write(folder/'certificate.json', dict(status='LOT_REMOTE_CPU_CERTIFIED',
        identity=spec['identity'], handle=spec['handle'], role=spec['role'],
        files=1, events=1, bytes=7, anomalies=[], gpu_used=False, training=False,
        test_2021_files=0, rasterio_reads=[dict(bands=23)]*3,
        loader_reads=[dict(device='cpu', X=[1,t,8,64,64]) for t in (1,3,5)]))
    rows = json.loads(Path(spec['canonical']).read_text())['files']
    (folder/'verified.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


class AutonomousTests(unittest.TestCase):
    def test_three_gib_start_and_two_gib_reserve_are_independent(self):
        self.assertEqual(auto.START_BYTES, 3*1024**3)
        self.assertEqual(auto.RESERVE_BYTES, 2*1024**3)
        self.assertEqual(auto.TREE_BYTES, 512*1024**2)
        for sample in (None, {'available': auto.START_BYTES-1}):
            with patch('kaggle_autonomous.memory', return_value=sample):
                with self.assertRaises(MemoryError): auto.require_memory()
        with patch('kaggle_autonomous.memory', return_value={'available': auto.START_BYTES}):
            self.assertEqual(auto.require_memory()['available'], auto.START_BYTES)

    def test_outer_preflight_never_starts_child_below_three_gib(self):
        with tempfile.TemporaryDirectory() as tmp, patch('run_memory_guarded.memory',
                return_value={'available': auto.START_BYTES-1}), patch('run_memory_guarded.ProcessTree') as child:
            result = run(['must-not-run'], Path(tmp)/'out', start_bytes=auto.START_BYTES,
                         reserve_bytes=auto.RESERVE_BYTES, job_bytes=auto.TREE_BYTES)
            self.assertEqual(result['status'], 'PREFLIGHT_LOW_MEMORY')
            self.assertEqual(result['children_started'], 0)
            child.assert_not_called()

    def test_dynamic_reserve_stops_and_closes_owned_tree(self):
        high = {'available': auto.START_BYTES}
        low = {'available': auto.RESERVE_BYTES-1}
        with tempfile.TemporaryDirectory() as tmp, patch('run_memory_guarded.memory',
                side_effect=[high, low, high]), patch('run_memory_guarded.ProcessTree') as cls:
            tree = cls.return_value
            tree.proc.poll.return_value = None
            tree.proc.pid = 12
            tree.proc.returncode = 1
            tree.memory_sample.return_value = dict(processes=[], rss_bytes=10,
                private_commit_bytes=10, peak_job_commit_bytes=10)
            result = run(['tiny-fixture'], Path(tmp)/'out', start_bytes=auto.START_BYTES,
                reserve_bytes=auto.RESERVE_BYTES, job_bytes=auto.TREE_BYTES, verbose=False)
            self.assertEqual(result['status'], 'LOW_AVAILABLE_MEMORY')
            tree.stop.assert_called_once()
            tree.close.assert_called_once()
            self.assertEqual(cls.call_args.kwargs['memory_limit_bytes'], auto.TREE_BYTES)

    def test_real_saved_state_is_reviewable_without_mutation(self):
        os.chdir(ROOT)
        journal = auto.DEFAULT_PLAN.parent/'http-body.jsonl'
        before = journal.read_bytes()
        plan, execution = auto.validate_autonomous(auto.DEFAULT_PLAN)
        self.assertEqual(reserved(plan), 4_995_735_202)
        self.assertFalse((execution/'lot-003').exists())
        self.assertEqual(journal.read_bytes(), before)

    def test_changed_http_counter_is_not_reset(self):
        with fixture() as (path, plan, execution):
            write(Path(plan['journal']), dict(previous=0, reserved=auto.INITIAL_BODY_BYTES+1,
                ceiling=auto.CEILING_BYTES))
            with patch('resume_remaining_lots.validate_continuation', return_value=(plan, execution)):
                with self.assertRaisesRegex(ValueError, 'baseline'): auto.validate_autonomous(path)
            self.assertEqual(reserved(plan), auto.INITIAL_BODY_BYTES+1)

    def test_release_never_touches_lots001_or002(self):
        for n in (1, 2):
            with self.assertRaisesRegex(ValueError, 'closed lots'): auto.release_certified_archive({'number':n}, 'unused')

    def test_publisher_refuses_network_if_ram_drops_after_zip_check(self):
        import requests
        from publish_kaggle_single_attempt import publish
        with fixture() as (path, plan, execution):
            spec = plan['lots'][1]
            stage = Path(spec['staging'])
            stage.mkdir(parents=True)
            data = b'fixture'
            (stage/'payload.zip').write_bytes(data)
            write(stage/'staging-receipt.json', dict(handle=spec['handle'],
                lot_identity=spec['identity'], observations=1, payload_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest()))
            spec['canonical_sha256'] = hashlib.sha256(Path(spec['canonical']).read_bytes()).hexdigest()
            plan['remaining_body_ceiling_bytes'] = auto.CEILING_BYTES
            write(path, plan)
            Path('credentials-fixture').write_bytes(b'not-an-actual-credential')
            modules = {}
            for name, values in {
                'kagglesdk': dict(KaggleClient=lambda:None),
                'kagglesdk.kaggle_creds': dict(KaggleCredentials=N(load=lambda *a,**k:
                    requests.post('https://api.kaggle.com/GenerateAccessToken', json={}))),
                'kagglesdk.blobs.types.blob_api_service': dict(ApiBlobType=None, ApiStartBlobUploadRequest=None),
                'kagglesdk.datasets.types.dataset_api_service': dict(ApiCreateDatasetRequest=None,
                    ApiGetDatasetRequest=None, ApiDatasetNewFile=None),
            }.items():
                module = ModuleType(name)
                module.__dict__.update(values)
                modules[name] = module
            args = argparse.Namespace(staged=stage, execute=True, approved_plan=path,
                journal=plan['journal'], attempt=execution/'lot-003/publication',
                credentials='credentials-fixture', max_http_body_bytes=auto.CEILING_BYTES,
                stop_at_cumulative_body_bytes=auto.CEILING_BYTES,
                min_available_memory_bytes=auto.START_BYTES, license='CC-BY-4.0')
            with patch.dict(sys.modules, modules), patch('kaggle_autonomous.memory',
                    return_value={'available':auto.START_BYTES-1}), \
                    patch('requests.sessions.Session.send', side_effect=AssertionError('Network forbidden')) as network:
                result = publish(args)
                network.assert_not_called()
            self.assertEqual(result['error_type'], 'MemoryError')
            self.assertEqual(result['blob_sends'], 0)
            self.assertEqual(result['requests'], 0)
            self.assertEqual(reserved(plan), auto.INITIAL_BODY_BYTES)

    def test_existing_attempt_lock_never_launches_second_supervisor(self):
        import run_autonomous_kaggle as launcher
        with fixture() as (path, plan, execution):
            Path('fixture-credentials').write_bytes(b'not-an-actual-credential')
            Path(plan['raw_root']).mkdir()
            with patch.object(launcher, 'ROOT', Path.cwd()), \
                    patch.object(launcher, 'DEFAULT_PLAN', path), \
                    patch.object(launcher, 'CREDENTIALS', 'fixture-credentials'), \
                    patch.object(launcher, 'validate_autonomous', return_value=(plan,execution)), \
                    patch.object(launcher.importlib.util, 'find_spec', return_value=True), \
                    patch.object(launcher.shutil, 'disk_usage', return_value=N(free=100_000_000_000)), \
                    patch.object(launcher, 'run') as start:
                with self.assertRaises(FileExistsError): launcher.main(['--execute'])
                start.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Native nested Windows Jobs')
    def test_nested_native_jobs_support_standalone_phase_supervision(self):
        from geofirex.job_control import ProcessTree
        source = (f'import sys\nsys.path.insert(0,{str(ROOT/"src")!r})\n'
                  'from geofirex.job_control import ProcessTree\n'
                  'tree=ProcessTree([sys.executable,"-c","print(123)"],memory_limit_bytes=128*1024**2)\n'
                  'try:\n    tree.proc.wait(timeout=5)\n    assert tree.proc.returncode==0\n'
                  'finally:\n    tree.close()\n')
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp)/'nested.py'
            script.write_text(source)
            tree = ProcessTree([sys.executable,str(script)], memory_limit_bytes=auto.TREE_BYTES)
            try:
                deadline = time.monotonic()+8
                while tree.proc.poll() is None and time.monotonic()<deadline:
                    tree.memory_sample()
                    time.sleep(.02)
                tree.proc.wait(timeout=2)
                self.assertEqual(tree.proc.returncode, 0)
                self.assertEqual(tree.memory_sample()['job_commit_limit_bytes'], auto.TREE_BYTES)
            finally:
                if tree.proc.poll() is None: tree.stop()
                tree.close()

    def workflow(self, failure=None, corrupt_certificate=False):
        with fixture() as (path, plan, execution):
            calls = []
            def monitored(command, out, **limits):
                self.assertEqual(limits['start_bytes'], auto.START_BYTES)
                self.assertEqual(limits['reserve_bytes'], auto.RESERVE_BYTES)
                self.assertEqual(limits['job_bytes'], auto.TREE_BYTES)
                self.assertEqual(limits['rss_bytes'], auto.TREE_BYTES)
                phase = command[1]
                if phase.endswith('prepare_remaining_kaggle.py'):
                    self.assertIn('--refresh-training', command)
                    calls.append(('templates', None))
                    return dict(status='FINISHED')
                spec = next(s for s in plan['lots'] if s['tag'] in str(out))
                n = spec['number']
                self.assertGreaterEqual(n, 3)
                calls.append((phase, n))
                if failure == (phase, n): return dict(status='FAILED')
                stage = Path(spec['staging'])
                target = execution/spec['tag']
                if phase.endswith('stage_expanded_lot.py'):
                    # Every previous ZIP must already have been released.
                    self.assertEqual(list(Path('artifacts').rglob('payload.zip')), [])
                    stage.mkdir(parents=True)
                    (stage/'payload.zip').write_bytes(b'fixture')
                    write(stage/'staging-receipt.json', dict(lot_identity=spec['identity'],
                        handle=spec['handle'], payload_bytes=7, sha256='zip-sha', observations=1))
                elif phase.endswith('publish_kaggle_single_attempt.py'):
                    self.assertEqual(command[command.index('--min-available-memory-bytes')+1], str(auto.START_BYTES))
                    self.assertEqual(command[command.index('--stop-at-cumulative-body-bytes')+1], str(auto.CEILING_BYTES))
                    write(target/'publication/result.json', dict(status='PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING',
                        confirmed_blob_body_bytes=7, payload_sha256='zip-sha'))
                    write(target/'publication/private-presence.json', dict(private=True, handle=spec['handle']))
                elif phase.endswith('kaggle_cpu_job.py'):
                    cpu_receipt(target/'cpu', spec)
                    if corrupt_certificate:
                        cert = json.loads((target/'cpu/certificate.json').read_text())
                        cert['anomalies'] = ['bad']
                        write(target/'cpu/certificate.json', cert)
                else:
                    self.fail(phase)
                return dict(status='FINISHED')
            args = argparse.Namespace(execute=True, approve_body_bytes=auto.CEILING_BYTES,
                                      credentials='fixture-credentials', autonomous=True)
            with patch('kaggle_autonomous.validate_autonomous', return_value=(plan, execution)), \
                    patch('resume_remaining_lots.memory', return_value={'available':auto.START_BYTES}), \
                    patch('resume_remaining_lots.monitored', side_effect=monitored):
                if failure or corrupt_certificate:
                    with self.assertRaises((RuntimeError, ValueError)):
                        continue_after_confirmed_indexing(path, args)
                    self.assertFalse((execution/'lot-005').exists())
                    self.assertTrue(list(Path('artifacts').rglob('payload.zip')))
                else:
                    continue_after_confirmed_indexing(path, args)
                    self.assertEqual([n for phase,n in calls if phase.endswith('stage_expanded_lot.py')], list(range(3,9)))
                    self.assertEqual(len(calls), 19)
                    self.assertEqual(list(Path('artifacts').rglob('payload.zip')), [])
                    for spec in plan['lots'][1:]:
                        self.assertTrue((Path(spec['staging'])/'staging-receipt.json').exists())
                        self.assertTrue((execution/spec['tag']/'archive-released.json').exists())
            self.assertEqual(reserved(plan), auto.INITIAL_BODY_BYTES)
            return calls

    def test_six_lots_sequential_release_after_verified_cpu_only(self):
        self.workflow()

    def test_publication_failure_stops_without_retry_or_next_lot(self):
        calls = self.workflow(failure=('scripts/publish_kaggle_single_attempt.py', 4))
        self.assertEqual(calls[-1], ('scripts/publish_kaggle_single_attempt.py', 4))
        self.assertEqual(calls.count(calls[-1]), 1)

    def test_invalid_cpu_certificate_retains_archive_and_stops(self):
        calls = self.workflow(corrupt_certificate=True)
        self.assertEqual(calls[-1], ('scripts/kaggle_cpu_job.py', 3))


if __name__ == '__main__':
    unittest.main()
