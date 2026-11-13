"""Tiny ZIP recovery fixtures only; no real raster reads or network."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_staged_lot_zip import verify


class ExistingZipTests(unittest.TestCase):
    def fixture(self,root,corrupt=False,extra=False):
        value=b'fixture-only, no raster';name='2018/fire_example/2018-01-01.tif'
        entry=dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),crc32=f'{zlib.crc32(value)&0xffffffff:08x}')
        lot=dict(number=2,role='train',handle='nnnnbhss/fixture',observations=1,bytes=len(value),events=['2018/fire_example'],files=[entry])
        lot['identity']=hashlib.sha256(json.dumps(lot,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        canonical=root/'lot.json';canonical.write_text(json.dumps(lot))
        staged=root/'staged';staged.mkdir()
        with zipfile.ZipFile(staged/'payload.zip','w') as archive:
            archive.writestr('rasters/'+name,b'x'*len(value) if corrupt else value)
            archive.writestr('lot-inventory.json',json.dumps(lot))
            if extra:archive.writestr('unexpected',b'x')
        return canonical,staged

    def test_valid_zip_reused_with_matching_final_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);canonical,staged=self.fixture(root)
            original=(staged/'payload.zip').read_bytes()
            proof=verify(canonical,staged,root/'proof')
            self.assertFalse(proof['rebuilt'])
            self.assertEqual(proof['sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual((staged/'payload.zip').read_bytes(),original)

    def test_corrupt_content_refused_without_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);canonical,staged=self.fixture(root,corrupt=True)
            with self.assertRaises(ValueError):verify(canonical,staged,root/'proof')
            self.assertFalse((staged/'staging-receipt.json').exists())

    def test_extra_member_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);canonical,staged=self.fixture(root,extra=True)
            with self.assertRaises(ValueError):verify(canonical,staged,root/'proof')


if __name__=='__main__':unittest.main()
