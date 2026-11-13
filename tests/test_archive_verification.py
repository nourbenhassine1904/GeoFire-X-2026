import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest

from geofirex.archive_transfer import verify_archive,extract_archive
from geofirex.archive_verification import ArchiveVerificationError
from geofirex.corpus import index_zip


def fixture_archive(tmp_path):
    path=tmp_path/'corpus.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('2018/',b'')
        z.writestr('2018/fire_1/2018-01-01.tif',b'first member data '*100)
        z.writestr('2021/',b'')
        z.writestr('2021/fire_2/2021-01-01.tif',b'storage only '*100)
    with zipfile.ZipFile(path) as z: index=index_zip(z,archive_size=path.stat().st_size)
    index['archive_md5']=hashlib.md5(path.read_bytes()).hexdigest()
    return path,index,tmp_path/'verification'


def test_limit_resume_all_members_and_cached_md5(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    from geofirex.archive_verification import file_digests
    with patch('geofirex.archive_verification.file_digests',wraps=file_digests) as digest:
        with pytest.raises(ArchiveVerificationError) as partial:
            verify_archive(path,index,verification_dir=journal,max_members=2,block_size=64)
        assert partial.value.result['status']=='PARTIAL'
        assert partial.value.result['crc_verified_members']==2
        assert partial.value.result['crc_unverified_members']==2
        assert partial.value.result['md5_status']=='VERIFIED'
        result=verify_archive(path,index,verification_dir=journal,block_size=64)
        assert result['all_crc_checked'] and result['crc_verified_members']==4
        assert result['crc_verified_tiffs']==2 and result['processed_this_attempt']==2
        assert digest.call_count==1
    assert len(list(journal.glob('summary-*.json')))==2


@pytest.mark.parametrize('error_phase',['open','read'])
def test_errno22_retained_never_promoted_to_success_then_resume(tmp_path,error_phase):
    path,index,journal=fixture_archive(tmp_path)
    original=zipfile.ZipFile.open
    def failing_open(z,member,*args,**kwargs):
        name=member.filename if isinstance(member,zipfile.ZipInfo) else member
        if name.endswith('2018-01-01.tif'):
            if error_phase=='open': raise OSError(22,'Invalid argument')
            class Broken(io.BytesIO):
                def read(self,n): raise OSError(22,'Invalid argument')
            return Broken()
        return original(z,member,*args,**kwargs)
    with patch.object(zipfile.ZipFile,'open',failing_open):
        with pytest.raises(ArchiveVerificationError) as failure:
            verify_archive(path,index,verification_dir=journal)
    result=failure.value.result
    assert result['md5_status']=='VERIFIED' and result['structure_status']=='VERIFIED'
    assert not result['all_crc_checked'] and result['crc_verified_members']==1
    assert result['crc_unverified_members']==3
    error=result['errors_encountered'][0]
    assert error['errno']==22 and error['classification']=='IO_ERROR'
    assert error['phase']==('open_member' if error_phase=='open' else 'read_member')
    assert result['last_member_started']['member'].endswith('2018-01-01.tif')
    resumed=verify_archive(path,index,verification_dir=journal)
    assert resumed['all_crc_checked'] and resumed['processed_this_attempt']==3
    assert len(resumed['errors_encountered'])==1 and not resumed['unresolved_member_errors']


def test_crc_computed_explicitly_and_extraction_blocked(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    original=zipfile.ZipFile.open
    def wrong_content(z,member,*args,**kwargs):
        if member.filename.endswith('.tif'): return io.BytesIO(b'X'*member.file_size)
        return original(z,member,*args,**kwargs)
    with patch.object(zipfile.ZipFile,'open',wrong_content):
        with pytest.raises(ArchiveVerificationError) as failure:
            verify_archive(path,index,verification_dir=journal,block_size=32)
        assert failure.value.result['failure']['phase']=='compare_crc'
        destination=tmp_path/'no-extraction'
        with pytest.raises(ArchiveVerificationError):
            extract_archive(path,index,destination,verification_dir=journal)
    assert not destination.exists()


def test_interrupted_member_is_restarted_and_memory_bounded(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    original=zipfile.ZipFile.open
    state={'stop':False}
    def interrupted(z,member,*args,**kwargs):
        stream=original(z,member,*args,**kwargs)
        if member.filename.endswith('.tif'):
            class Reader:
                def __enter__(self): return self
                def __exit__(self,*exc): stream.close()
                def read(self,n):
                    assert 0<n<=64
                    result=stream.read(n);state['stop']=True;return result
            return Reader()
        return stream
    with patch.object(zipfile.ZipFile,'open',interrupted):
        with pytest.raises(ArchiveVerificationError) as stopped:
            verify_archive(path,index,verification_dir=journal,block_size=64,stop_requested=lambda:state['stop'])
    assert stopped.value.result['status']=='INTERRUPTED'
    assert stopped.value.result['crc_verified_members']==1
    resumed=verify_archive(path,index,verification_dir=journal,block_size=64)
    assert resumed['all_crc_checked'] and resumed['processed_this_attempt']==3


def test_changed_archive_or_index_rejects_old_proof(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    verify_archive(path,index,verification_dir=journal)
    changed={**index,'archive_md5':'0'*32}
    with pytest.raises(ValueError,match='changed'): verify_archive(path,changed,verification_dir=journal)
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='changed'): verify_archive(path,index,verification_dir=journal)


def test_incomplete_log_tail_preserved_without_false_crc(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    with pytest.raises(ArchiveVerificationError):
        verify_archive(path,index,verification_dir=journal,max_members=1)
    log=next(journal.glob('attempt-*.jsonl'))
    with log.open('ab') as stream: stream.write(b'{"kind":"member_ok"')
    prior=log.read_bytes()
    result=verify_archive(path,index,verification_dir=journal)
    assert result['all_crc_checked'] and result['processed_this_attempt']==3
    assert result['incomplete_journal_tails'] and log.read_bytes()==prior


def test_metadata_only_is_explicitly_not_full_crc(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    result=verify_archive(path,index,verification_dir=journal,all_crc=False)
    assert result['md5_status']=='VERIFIED'
    assert result['status']=='MD5_AND_STRUCTURE_VERIFIED'
    assert result['crc_unverified_members']==4 and not result['all_crc_checked']


def test_existing_download_does_not_start_network_or_claim_crc(tmp_path):
    from geofirex.archive_transfer import download_archive
    path,index,_=fixture_archive(tmp_path)
    with patch('urllib.request.urlopen') as network:
        result=download_archive(index,path,tmp_path/'download-session',100000)
    network.assert_not_called()
    assert result['status']=='ALREADY_PRESENT_VERIFIED'
    assert result['md5_status']=='VERIFIED' and not result['all_crc_checked']


def test_valid_json_without_committed_newline_is_not_crc_proof(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    result=verify_archive(path,index,verification_dir=journal,all_crc=False)
    log=next(journal.glob('attempt-*.jsonl'))
    # A crash after a complete JSON string but before its committed separator.
    with zipfile.ZipFile(path) as z:
        member=z.infolist()[0]
    from geofirex.archive_verification import member_signature
    with log.open('ab') as stream:
        stream.write(json.dumps(dict(kind='member_ok',**member_signature(0,member),
                                    actual_bytes=0,actual_crc32='00000000')).encode())
    result=verify_archive(path,index,verification_dir=journal)
    assert result['processed_this_attempt']==4 and result['incomplete_journal_tails']


def test_cli_partial_returns_failure_code_and_full_resume_returns_zero(tmp_path,monkeypatch,capsys):
    import scripts.full_corpus_external as cli
    path,index,journal=fixture_archive(tmp_path)
    index_path=tmp_path/'index.json'
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(cli,'validate_index',lambda *a,**kw:None)
    arguments=['tool','verify','--archive',str(path),'--index',str(index_path),
               '--verification-dir',str(journal)]
    monkeypatch.setattr('sys.argv',arguments+['--verify-limit','1'])
    assert cli.main()==2
    assert json.loads(capsys.readouterr().out)['status']=='PARTIAL'
    monkeypatch.setattr('sys.argv',arguments)
    assert cli.main()==0
    assert json.loads(capsys.readouterr().out)['all_crc_checked']


def test_interrupted_global_md5_is_not_certified(tmp_path):
    path,index,journal=fixture_archive(tmp_path)
    with pytest.raises(ArchiveVerificationError) as stopped:
        verify_archive(path,index,verification_dir=journal,stop_requested=lambda:True)
    result=stopped.value.result
    assert result['status']=='INTERRUPTED' and result['md5_status']=='NOT_VERIFIED'
    assert result['crc_verified_members']==0 and result['crc_unverified_members']==4
    assert verify_archive(path,index,verification_dir=journal)['all_crc_checked']
