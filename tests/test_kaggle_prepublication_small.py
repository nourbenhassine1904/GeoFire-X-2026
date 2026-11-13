"""Tiny offline fixtures only. No ML imports, real rasters, network or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kaggle_certify_inputs import bootstrap_code, certify_lot, safe_child
from publish_kaggle_single_attempt import PhaseBudget, stream_once
from submit_kaggle_cpu_certification import submit


class SmallPrepublicationTests(unittest.TestCase):
    def fixture_code(self, root):
        source=root/'source'; source.mkdir()
        (source/'hello.py').write_bytes(b'answer = 42\n')
        manifest={'hello.py':hashlib.sha256((source/'hello.py').read_bytes()).hexdigest()}
        m=json.dumps(manifest).encode(); (source/'code-manifest.json').write_bytes(m)
        archive=root/'archive'; archive.mkdir()
        with zipfile.ZipFile(archive/'payload.zip','x') as z:
            for name in ('hello.py','code-manifest.json'): z.write(source/name,name)
        return source,archive,hashlib.sha256((archive/'payload.zip').read_bytes()).hexdigest(),hashlib.sha256(m).hexdigest()

    def fixture_lot(self, root, year='2018'):
        data=b'tiny synthetic bytes, not a GeoTIFF\n'; p=f'{year}/fire_test/2018-01-01.tif'
        lot=dict(number=1,role='train',handle='nnnnbhss/test-fixture',events=[f'{year}/fire_test'],observations=1,
                 bytes=len(data), files=[dict(path=p,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                 crc32=f'{zlib.crc32(data)&0xffffffff:08x}')])
        identity=hashlib.sha256(json.dumps(lot,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        lot['identity']=identity
        canonical=root/'canonical.json'; canonical.write_text(json.dumps(lot))
        dataset=root/'dataset'; (dataset/'rasters'/p).parent.mkdir(parents=True)
        (dataset/'rasters'/p).write_bytes(data); (dataset/'lot-inventory.json').write_text(json.dumps(lot))
        return canonical,dataset,identity,p

    def test_expanded_and_zip_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source,archive,pin,mpin=self.fixture_code(root)
            for i,src in enumerate((source,archive)):
                out=bootstrap_code(src,root/f'work{i}',pin,mpin)
                self.assertEqual((out/'hello.py').read_bytes(),b'answer = 42\n')

    def test_code_tampering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source,archive,pin,mpin=self.fixture_code(root)
            (source/'hello.py').write_bytes(b'tampered')
            with self.assertRaises(ValueError): bootstrap_code(source,root/'work',pin,mpin)
            with self.assertRaises(ValueError): bootstrap_code(archive,root/'work2','0'*64,mpin)

    def test_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ('../escape.py','/absolute','C:/data','foo\\bar'):
                with self.assertRaises(ValueError): safe_child(tmp,bad)

    def test_lot_bytes_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); canonical,dataset,pin,p=self.fixture_lot(root)
            r=certify_lot(canonical,dataset,root/'out',pinned_identity=pin,inspect_geotiff=False)
            self.assertEqual(r['status'],'ONE_LOT_REMOTE_BYTES_CERTIFIED')
            self.assertEqual(r['files'],1)

    def test_lot_corruption_and_extras_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); canonical,dataset,pin,p=self.fixture_lot(root)
            f=dataset/'rasters'/p; f.write_bytes(b'x'*f.stat().st_size)
            with self.assertRaises(ValueError): certify_lot(canonical,dataset,root/'out',pinned_identity=pin,inspect_geotiff=False)
            (dataset/'rasters/extra.tif').write_bytes(b'x')
            with self.assertRaises(ValueError): certify_lot(canonical,dataset,root/'out2',pinned_identity=pin,inspect_geotiff=False)

    def test_2021_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); canonical,dataset,pin,p=self.fixture_lot(root,'2021')
            with self.assertRaises(ValueError): certify_lot(canonical,dataset,root/'out',pinned_identity=pin,inspect_geotiff=False)

    def test_cumulative_budget_persists_and_phase_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=Path(tmp)/'budget.jsonl'
            with PhaseBudget(journal,100,60) as budget:
                budget.reserve(40)
                with self.assertRaises(RuntimeError): budget.reserve(21)
            with PhaseBudget(journal,100,90) as budget:
                self.assertEqual(budget.used,40); budget.reserve(30)
            self.assertEqual(sum(json.loads(x)['reserved'] for x in journal.read_text().splitlines()),70)

    def test_failed_put_is_not_retried(self):
        class Response:
            status_code=503
            def close(self): pass
        class Session:
            calls=0
            def put(self,url,**kwargs):
                self.calls+=1
                self.asserted_body=kwargs['data'].read()
                return Response()
        with tempfile.TemporaryDirectory() as tmp:
            payload=Path(tmp)/'payload.zip'; payload.write_bytes(b'tiny')
            session=Session()
            with self.assertRaises(RuntimeError): stream_once(session,'https://offline.invalid',payload)
            self.assertEqual(session.calls,1); self.assertEqual(session.asserted_body,b'tiny')

    def fixture_cpu(self,root):
        folder=root/'code'; folder.mkdir(); nb=b'{"cells":[]}'
        (folder/'certify.ipynb').write_bytes(nb)
        meta=dict(code_file='certify.ipynb',enable_gpu=False,enable_internet=False,is_private=True,
                  id='nnnnbhss/test-cpu',dataset_sources=['nnnnbhss/test-code'])
        (folder/'kernel-metadata.json').write_text(json.dumps(meta))
        pin=dict(tag='code',notebook_sha256=hashlib.sha256(nb).hexdigest(),kernel=meta['id'],sources=meta['dataset_sources'])
        (root/'prepared.json').write_text(json.dumps(dict(notebooks=[pin])))
        return folder,meta

    def test_cpu_submission_default_is_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,meta=self.fixture_cpu(Path(tmp))
            result=submit(argparse.Namespace(folder=str(folder),execute=False))
            self.assertEqual(result['status'],'OFFLINE_CPU_SUBMISSION_REVIEW_ONLY')

    def test_gpu_metadata_refused_even_in_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,meta=self.fixture_cpu(Path(tmp)); meta['enable_gpu']=True
            (folder/'kernel-metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError): submit(argparse.Namespace(folder=str(folder),execute=False))


if __name__=='__main__': unittest.main()
