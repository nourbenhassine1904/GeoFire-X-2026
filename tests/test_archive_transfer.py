import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from unittest.mock import patch

import pytest

from geofirex.archive_transfer import download_archive,verify_archive,extract_archive
from geofirex.corpus import index_zip


def fixture():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('2018/fire_1/2018-01-01.tif',b'fixture'*100)
        z.writestr('2021/fire_2/2021-01-01.tif',b'test storage only'*100)
    data=stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as z: index=index_zip(z,archive_size=len(data))
    index.update(source='https://fixture.invalid/archive.zip',archive_md5=hashlib.md5(data).hexdigest())
    return data,index


def response(data,requests,*,ignore_range=False):
    def open_request(req,**kwargs):
        start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',req.headers['Range']).groups())
        requests.append(start)
        class Response(io.BytesIO):
            status=200 if ignore_range else 206
            headers={'Content-Range':f'bytes {start}-{end}/{len(data)}','Content-Length':str(len(data)-start)}
        return Response(data[start:end+1])
    return open_request


def test_zip_resumes_prefix_and_preserves_budget(tmp_path):
    data,index=fixture();requests=[]
    output,session=tmp_path/'full.zip',tmp_path/'session'
    with patch('urllib.request.urlopen',side_effect=response(data,requests)),patch('geofirex.archive_transfer.time.sleep'):
        stopped=lambda:Path(str(output)+'.part').exists() and Path(str(output)+'.part').stat().st_size>=128
        first=download_archive(index,output,session,100000,stop_requested=stopped,block_size=128)
        assert first['status']=='STOPPED_BY_USER'
        prefix=Path(str(output)+'.part').read_bytes()
        assert len(prefix)==128
        result=download_archive(index,output,session,100000,block_size=128)
        assert output.read_bytes()==data and requests==[0,128]
        assert result['confirmed_body_bytes']==len(data)
        assert result['reserved_body_bytes']>len(data)
    assert verify_archive(output,index)['observations']==2
    assert extract_archive(output,index,tmp_path/'rasters')['observations']==2
    assert extract_archive(output,index,tmp_path/'rasters')['observations']==2


def test_zip_ignored_resume_range_never_appends(tmp_path):
    data,index=fixture();out=tmp_path/'archive.zip';session=tmp_path/'session'
    with patch('urllib.request.urlopen',side_effect=response(data,[])),patch('geofirex.archive_transfer.time.sleep'):
        download_archive(index,out,session,100000,block_size=64,
                         stop_requested=lambda:Path(str(out)+'.part').exists() and Path(str(out)+'.part').stat().st_size>=64)
    original=Path(str(out)+'.part').read_bytes()
    with patch('urllib.request.urlopen',side_effect=response(data,[],ignore_range=True)):
        with pytest.raises(ValueError,match='206'): download_archive(index,out,session,100000,block_size=64)
    assert Path(str(out)+'.part').read_bytes()==original


def test_zip_insufficient_budget_and_md5_mismatch_preserve(tmp_path):
    data,index=fixture()
    with patch('urllib.request.urlopen') as network:
        with pytest.raises(RuntimeError,match='budget'):
            download_archive(index,tmp_path/'a.zip',tmp_path/'session',len(data)-1)
        network.assert_not_called()
    index['archive_md5']='0'*32
    with patch('urllib.request.urlopen',side_effect=response(data,[])),patch('geofirex.archive_transfer.time.sleep'):
        with pytest.raises(ValueError,match='MD5'):
            download_archive(index,tmp_path/'bad.zip',tmp_path/'bad-session',100000)
    assert Path(str(tmp_path/'bad.zip')+'.part').read_bytes()==data


def test_zip_crash_reservation_and_orphan_fail_closed(tmp_path):
    data,index=fixture();out=tmp_path/'archive.zip';session=tmp_path/'session'
    class BrokenResponse(io.BytesIO):
        status=206
        headers={'Content-Range':f'bytes 0-{len(data)-1}/{len(data)}','Content-Length':str(len(data))}
        def read(self,n): raise OSError('interrupted socket')
    with patch('urllib.request.urlopen',return_value=BrokenResponse(data)):
        with pytest.raises(OSError): download_archive(index,out,session,100000,attempts=1,block_size=64)
    rows=[json.loads(line) for line in (session/'body-transfer.jsonl').read_text().splitlines()]
    assert rows[-1]['reserved']==65 and rows[-1]['confirmed']==0
    Path(str(tmp_path/'orphan.zip')+'.part').write_bytes(b'previous unknown transfer')
    with pytest.raises(ValueError,match='Orphan'):
        download_archive(index,tmp_path/'orphan.zip',tmp_path/'orphan-session',100000)
