import io
import json
import re
import zipfile
from collections import namedtuple
from unittest.mock import patch

import pytest

from geofirex.acquisition_session import PersistentRangeFile,event_batches,execute_stage,session_lock
from geofirex.corpus import index_zip
from geofirex.scientific_scaleup import acquisition_stages


def transport(body,requests):
    def response(request,**kwargs):
        start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.headers['Range']).groups())
        requests.append((start,end))
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':f'bytes {start}-{end}/{len(body)}'}
        return Response(body[start:end+1])
    return response


def test_cap_is_cumulative_across_process_restarts(tmp_path):
    ledger=tmp_path/'transfer.jsonl'
    requests=[]
    with patch('urllib.request.urlopen',side_effect=transport(b'x'*100,requests)):
        one=PersistentRangeFile(9,ledger=ledger,plan_identity='p',size=100,retries=1)
        assert one.read(4)==b'xxxx'
        two=PersistentRangeFile(9,ledger=ledger,plan_identity='p',size=100,retries=1)
        with pytest.raises(RuntimeError,match='budget'): two.read(4)
    assert len(requests)==1 and two.bytes_reserved==5
    assert json.loads(ledger.read_text().splitlines()[-1])['reserved_bytes']==5
    with pytest.raises(ValueError,match='mismatch'):
        PersistentRangeFile(99,ledger=ledger,plan_identity='p',size=100)


def test_retry_reserves_before_request_and_cannot_reset(tmp_path):
    ledger=tmp_path/'transfer.jsonl'
    attempts=[]
    success=transport(b'x'*100,attempts)
    def request(req,**kwargs):
        rows=[json.loads(line) for line in ledger.read_text().splitlines()]
        assert rows[-1]['kind']=='reservation'
        if rows[-1]['requests']==1: raise OSError('transient')
        return success(req,**kwargs)
    with patch('urllib.request.urlopen',side_effect=request),patch('geofirex.corpus.time.sleep'):
        stream=PersistentRangeFile(10,ledger=ledger,plan_identity='p',size=100,retries=2)
        assert stream.read(4)==b'xxxx'
        assert stream.bytes_reserved==10 and stream.requests==2
        resumed=PersistentRangeFile(10,ledger=ledger,plan_identity='p',size=100)
        with pytest.raises(RuntimeError,match='budget'): resumed.read(1)


def test_corrupt_ledger_and_concurrent_session_fail_closed(tmp_path):
    ledger=tmp_path/'transfer.jsonl'
    ledger.write_text('{"truncated":',encoding='utf-8')
    with pytest.raises(ValueError): PersistentRangeFile(10,ledger=ledger,plan_identity='p')
    with session_lock(tmp_path/'session'):
        with pytest.raises(RuntimeError,match='already running'):
            with session_lock(tmp_path/'session'): pass
    with session_lock(tmp_path/'session'): pass  # OS lock released, file retained


def fixture_archive():
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('2018/fire_1/2018-01-01.tif',b'fixture data 1'*100)
        z.writestr('2020/fire_2/2020-01-01.tif',b'fixture data 2'*100)
    body=archive.getvalue()
    with zipfile.ZipFile(io.BytesIO(body)) as z: index=index_zip(z,archive_size=len(body))
    index['source']='https://fixture.invalid/archive.zip'
    return body,index,acquisition_stages(index,[],targets=(2,))[0]


def test_sub_batches_interrupt_part_preserved_resume_without_duplicates(tmp_path):
    import shutil
    body,index,plan=fixture_archive()
    destination,session=tmp_path/'rasters',tmp_path/'session'
    requests=[]
    def interrupted(source,target,**kwargs):
        target.write(source.read(3))
        raise OSError('power loss fixture')
    Disk=namedtuple('Disk','total used free')
    with patch('urllib.request.urlopen',side_effect=transport(body,requests)),patch(
            'geofirex.acquisition_session.shutil.disk_usage',return_value=Disk(100*10**9,0,100*10**9)):
        report={}
        with patch('geofirex.corpus.shutil.copyfileobj',side_effect=interrupted):
            with pytest.raises(OSError):
                execute_stage(plan,index,destination,[],session,batch_bytes=1,max_mbps=1000,report=report)
        partials=list(destination.rglob('*.part'))
        assert len(partials)==1 and partials[0].read_bytes()==b'fix'
        reserved=report['transfer']['reserved_bytes_including_retries']
        report={}
        execute_stage(plan,index,destination,[],session,batch_bytes=1,max_mbps=1000,report=report)
        assert report['complete_events']==2 and report['verified_observations']==2
        assert len(report['batches'])==2 and len(list(destination.rglob('*.tif')))==2
        assert report['transfer']['reserved_bytes_including_retries']>reserved
        assert partials[0].read_bytes()==b'fix'
        request_count=len(requests)
        report={}
        execute_stage(plan,index,destination,[],session,batch_bytes=1,max_mbps=1000,report=report)
        assert len(requests)==request_count and report['downloaded_files']==0
        assert report['sha256_batch_comparisons']==2
        certificate=next(session.glob('batch-*.json'))
        contents=json.loads(certificate.read_text())
        contents['files'][0]['sha256']='0'*64
        certificate.write_text(json.dumps(contents))
        with pytest.raises(ValueError,match='SHA256'):
            execute_stage(plan,index,destination,[],session,batch_bytes=1,max_mbps=1000,report={})
        assert len(requests)==request_count
    assert len(list(destination.rglob('*.tif')))==2


def test_disk_guard_precedes_any_network(tmp_path):
    _,index,plan=fixture_archive()
    Disk=namedtuple('Disk','total used free')
    with patch('urllib.request.urlopen') as network,patch('geofirex.acquisition_session.shutil.disk_usage',
                                                        return_value=Disk(100,95,5)):
        with pytest.raises(RuntimeError,match='disk'):
            execute_stage(plan,index,tmp_path/'rasters',[],tmp_path/'session',
                          batch_bytes=1,max_mbps=20,report={})
        network.assert_not_called()


def test_batches_preserve_events():
    _,index,_=fixture_archive()
    entries=index['entries']+[dict(index['entries'][0],path='other')]
    batches=event_batches(entries,1)
    assert len(batches)==2 and len(batches[0])==2


def test_stop_marker_prevents_network_and_is_preserved(tmp_path):
    _,index,plan=fixture_archive()
    session=tmp_path/'session';session.mkdir()
    (session/'STOP').write_text('user requested stop')
    with patch('urllib.request.urlopen') as network:
        with pytest.raises(RuntimeError,match='STOP'):
            execute_stage(plan,index,tmp_path/'rasters',[],session,batch_bytes=1,max_mbps=20,report={})
        network.assert_not_called()
    assert (session/'STOP').read_text()=='user requested stop'


def test_cooperative_stop_finishes_file_then_certifies_partial_batch(tmp_path):
    body,index,plan=fixture_archive()
    destination,session=tmp_path/'rasters',tmp_path/'session'
    stopped=lambda:bool(list(destination.rglob('*.tif')))
    with patch('urllib.request.urlopen',side_effect=transport(body,[])):
        report={}
        execute_stage(plan,index,destination,[],session,batch_bytes=100000,max_mbps=1000,
                      report=report,stop_requested=stopped)
    assert report['status']=='STOPPED_BY_USER_AT_FILE_BOUNDARY'
    assert len(list(destination.rglob('*.tif')))==1
    certificate=json.loads(next(session.glob('batch-*.json')).read_text())
    assert certificate['partial_batch'] is True and len(certificate['files'])==1
    assert json.loads((session/'transfer.jsonl').read_text().splitlines()[-1])['kind']=='session_end'
