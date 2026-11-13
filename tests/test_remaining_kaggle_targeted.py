"""Targeted receipt/scope/barrier tests; no network, raw rasters, GPU, or models."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from record_lot001_certificate import verify_evidence
from kaggle_cpu_job import validate_receipt,validate_small_receipt_url
from publish_remaining_lots import check_plan,execute
from publish_kaggle_single_attempt import publish
from kaggle_certify_inputs import certify_lot


class RemainingTargetedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((ROOT/'reports/kaggle_remaining_20261108_v1/plan.json').read_text())
        cls.lot001=json.loads((ROOT/'reports/recovery_20261107_v3/finalized/deployment/bundle/metadata/lot-001.json').read_text())

    def test_saved_lot001_evidence_matches_canonical(self):
        proof=verify_evidence(ROOT/self.plan['lot001_evidence'],self.lot001)
        self.assertEqual((proof['files'],proof['bytes'],proof['events']),(1269,4829474073,57))
        self.assertEqual(proof['anomalies'],[])

    def test_corrupt_crc_evidence_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for name in ('certificate.json','verified.jsonl','progress.json'):
                shutil.copy2(ROOT/self.plan['lot001_evidence']/name,tmp/name)
            rows=(tmp/'verified.jsonl').read_text().splitlines()
            row=json.loads(rows[0]);row['crc32']='00000000';rows[0]=json.dumps(row)
            (tmp/'verified.jsonl').write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError,'CRC32'): verify_evidence(tmp,self.lot001)

    def test_plan_excludes_lot001_and_has_pinned_budget(self):
        check_plan(self.plan)
        for number in (1,9):
            bad=copy.deepcopy(self.plan);bad['lots'][0]['number']=number
            with self.assertRaises(ValueError): check_plan(bad)
        bad=copy.deepcopy(self.plan);bad['remaining_body_ceiling_bytes']+=1
        with self.assertRaises(ValueError): check_plan(bad)

    def test_failed_cpu_canary_blocks_all_raster_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path=Path(tmp)/'plan.json';plan_path.write_text(json.dumps(self.plan))
            args=argparse.Namespace(execute=True,approve_body_bytes=31_000_000_000,credentials='unused')
            with patch('publish_remaining_lots.monitored',return_value={'status':'FAILED'}) as monitor:
                with self.assertRaisesRegex(RuntimeError,'canary-monitor'): execute(plan_path,args)
            self.assertEqual(monitor.call_count,1)
            argv=monitor.call_args.args[0]
            self.assertIn('scripts/kaggle_cpu_job.py',argv)
            self.assertFalse((Path(tmp)/'execution/lot-002').exists())

    def test_old_publisher_cannot_upload_lot001_with_remaining_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=b'fixture only';(root/'payload.zip').write_bytes(data)
            receipt=dict(handle=self.lot001['handle'],payload_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                lot_identity=self.lot001['identity'],observations=1269)
            (root/'staging-receipt.json').write_text(json.dumps(receipt))
            args=argparse.Namespace(staged=root,execute=True,approved_plan=ROOT/'reports/kaggle_remaining_20261108_v1/plan.json')
            with self.assertRaisesRegex(PermissionError,'Only pinned remaining lots'): publish(args)

    def test_validation_receipt_is_role_aware_and_rejects_scope_change(self):
        spec=next(s for s in self.plan['lots'] if s['number']==5)
        lot=json.loads(Path(spec['canonical']).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            (tmp/'verified.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in lot['files']))
            cert=dict(status='LOT_REMOTE_CPU_CERTIFIED',identity=spec['identity'],handle=spec['handle'],role='val',
                files=spec['files'],events=spec['events'],bytes=spec['raw_bytes'],anomalies=[],gpu_used=False,training=False,
                test_2021_files=0,rasterio_reads=[{'bands':23}]*3,
                loader_reads=[dict(device='cpu',X=[1,t,8,64,64]) for t in (1,3,5)])
            (tmp/'certificate.json').write_text(json.dumps(cert))
            self.assertEqual(validate_receipt(tmp,spec)['role'],'val')
            cert['role']='train';(tmp/'certificate.json').write_text(json.dumps(cert))
            with self.assertRaises(ValueError): validate_receipt(tmp,spec)

    def test_streamed_helper_receipt_contains_actual_crc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=b'tiny fixture bytes only';name='2020/fire_fixture/2020-01-01.tif'
            entry=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=f'{zlib.crc32(data)&0xffffffff:08x}')
            lot=dict(number=5,role='val',handle='nnnnbhss/fixture',events=['2020/fire_fixture'],observations=1,bytes=len(data),files=[entry])
            lot['identity']=hashlib.sha256(json.dumps(lot,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            canonical=root/'canonical.json';canonical.write_text(json.dumps(lot))
            dataset=root/'dataset';p=dataset/'rasters'/name;p.parent.mkdir(parents=True);p.write_bytes(data)
            (dataset/'lot-inventory.json').write_text(json.dumps(lot))
            certify_lot(canonical,dataset,root/'receipt',pinned_identity=lot['identity'],inspect_geotiff=False)
            self.assertEqual(json.loads((root/'receipt/verified.jsonl').read_text()),entry)

    def test_receipt_url_supports_observed_official_output_host_only(self):
        for url in ('https://www.kaggleusercontent.com/kf/fixture/environment.json',
                    'https://storage.googleapis.com/fixture/environment.json'):
            self.assertEqual(validate_small_receipt_url(url),url)
        for url in ('https://www.kaggleusercontent.com.evil.test/x','https://evilkaggleusercontent.com/x',
                    'http://www.kaggleusercontent.com/x','https://user@www.kaggleusercontent.com/x',
                    'https://www.kaggleusercontent.com:444/x','https://www.kaggleusercontent.com/x#fragment'):
            with self.assertRaises(ValueError): validate_small_receipt_url(url)


if __name__=='__main__': unittest.main()
