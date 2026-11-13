import errno
import json
import os
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest

from geofirex.corpus import acquire_members, index_zip, sha256_file


def fixture(tmp_path):
    archive = tmp_path / 'fixture.zip'
    payload = b'verified raster fixture' * 100
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('2018/fire_1/2018-01-01.tif', payload)
    with zipfile.ZipFile(archive) as z:
        entry = index_zip(z)['entries'][0]
    root = tmp_path / 'rasters'
    output = root / entry['path']
    output.parent.mkdir(parents=True)
    part = output.with_name(output.name + '.' + 'a' * 32 + '.part')
    part.write_bytes(payload)
    return archive, entry, root, output, part, payload


def windows_error(code):
    error = PermissionError(errno.EACCES, 'simulated Windows publication refusal')
    error.winerror = code
    return error


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_reuse_valid_part_without_zip_payload_or_new_budget_and_keep_part(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    journal = tmp_path / 'publication.jsonl'
    before = (part.stat().st_mtime_ns, sha256_file(part))
    with zipfile.ZipFile(archive) as z, patch.object(z, 'open', side_effect=AssertionError('No re-extraction')):
        result = acquire_members(z, [entry], root, max_compressed_bytes=0,
                                 preserve_parts=True, publication_journal=journal)
    assert result['reused_parts'] == [dict(path=entry['path'], part=str(part))]
    assert result['planned_compressed_bytes_written'] == 0
    assert part.read_bytes() == output.read_bytes() == payload
    assert os.path.samefile(part, output) and part.stat().st_nlink == 2
    assert before == (part.stat().st_mtime_ns, sha256_file(part))
    assert rows(journal)[-1]['kind'] == 'publication_ok'
    with zipfile.ZipFile(archive) as z, patch.object(z, 'open', side_effect=AssertionError('No ZIP read')):
        assert acquire_members(z, [entry], root, preserve_parts=True)['verified_existing'] == [entry['path']]


@pytest.mark.parametrize('code', [5, 32, 33])
@pytest.mark.parametrize('preserve', [False, True])
def test_transient_windows_errors_retry_publication_only(tmp_path, code, preserve):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    journal = tmp_path / 'retry.jsonl'
    hardlink = preserve or os.name != 'nt'
    operation = os.link if hardlink else Path.rename
    calls = []

    def interrupted(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise windows_error(code)
        return operation(source, target)

    target = 'geofirex.corpus.os.link' if hardlink else 'pathlib.Path.rename'
    with zipfile.ZipFile(archive) as z, patch.object(z, 'open', side_effect=AssertionError('No re-extraction')):
        with patch(target, interrupted), patch('geofirex.corpus.time.sleep') as sleep:
            acquire_members(z, [entry], root, preserve_parts=preserve, publication_journal=journal)
    assert len(calls) == 3 and [c.args[0] for c in sleep.call_args_list] == [.25, .5]
    assert output.read_bytes() == payload
    assert len([r for r in rows(journal) if r['kind'] == 'publication_error']) == 2
    assert part.exists() == hardlink


def test_persistent_refusal_is_bounded_logged_and_resumable(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    journal = tmp_path / 'persistent.jsonl'
    with zipfile.ZipFile(archive) as z, patch('geofirex.corpus.os.link', side_effect=windows_error(5)) as link:
        with patch('geofirex.corpus.time.sleep') as sleep:
            with pytest.raises(PermissionError):
                acquire_members(z, [entry], root, preserve_parts=True, publication_journal=journal)
    assert link.call_count == 5 and sleep.call_count == 4
    assert not output.exists() and part.read_bytes() == payload
    assert rows(journal)[-1]['attempt'] == 5
    old_log = journal.read_bytes()
    with zipfile.ZipFile(archive) as z, patch.object(z, 'open', side_effect=AssertionError('No re-extraction')):
        acquire_members(z, [entry], root, preserve_parts=True, publication_journal=tmp_path / 'resume.jsonl')
    assert journal.read_bytes() == old_log and part.read_bytes() == output.read_bytes() == payload


def test_other_io_error_is_not_retried_or_hidden(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    with zipfile.ZipFile(archive) as z, patch('geofirex.corpus.os.link', side_effect=OSError(22, 'Invalid argument')) as link:
        with patch('geofirex.corpus.time.sleep') as sleep:
            with pytest.raises(OSError):
                acquire_members(z, [entry], root, preserve_parts=True)
    assert link.call_count == 1 and not sleep.called
    assert not output.exists() and part.read_bytes() == payload


@pytest.mark.parametrize('same', [False, True])
def test_concurrent_destination_never_overwritten(tmp_path, same):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    other = payload if same else b'existing historical file'

    def race(source, target):
        Path(target).write_bytes(other)
        raise FileExistsError(errno.EEXIST, 'Concurrent output')

    with zipfile.ZipFile(archive) as z, patch('geofirex.corpus.os.link', race):
        if same:
            acquire_members(z, [entry], root, preserve_parts=True)
        else:
            with pytest.raises(ValueError, match='conflict'):
                acquire_members(z, [entry], root, preserve_parts=True)
    assert output.read_bytes() == other and part.read_bytes() == payload


def test_invalid_and_incomplete_parts_are_preserved_and_not_published(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    part.write_bytes(b'X' * len(payload))  # Correct size, wrong CRC.
    incomplete = output.with_name(output.name + '.' + 'b' * 32 + '.part')
    incomplete.write_bytes(b'prefix')
    with zipfile.ZipFile(archive) as z:
        result = acquire_members(z, [entry], root, preserve_parts=True)
    assert not result['reused_parts'] and result['planned_compressed_bytes_written'] > 0
    assert output.read_bytes() == payload
    assert part.read_bytes() == b'X' * len(payload) and incomplete.read_bytes() == b'prefix'
    assert len(list(output.parent.glob('*.part'))) == 3


def test_directory_conflict_is_preserved_and_rejected(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    output.mkdir()
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match='ordinary file'):
        acquire_members(z, [entry], root, preserve_parts=True)
    assert output.is_dir() and part.read_bytes() == payload


def test_parent_link_escape_is_rejected(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    linked_root = tmp_path / 'linked-root'
    linked_root.mkdir()
    try:
        (linked_root / '2018').symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f'Symlink privilege unavailable: {error}')
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match='escapes'):
        acquire_members(z, [entry], linked_root, preserve_parts=True)
    assert not list(outside.iterdir())


def test_mutated_part_is_not_published(tmp_path):
    archive, entry, root, output, part, payload = fixture(tmp_path)

    def changing_digest(path):
        digest = sha256_file(path)
        Path(path).write_bytes(payload + b'changed')
        return digest

    with zipfile.ZipFile(archive) as z, patch('geofirex.corpus.sha256_file', changing_digest):
        with pytest.raises(RuntimeError, match='changed during validation'):
            acquire_members(z, [entry], root, preserve_parts=True)
    assert not output.exists() and part.read_bytes() == payload + b'changed'


def test_full_extraction_reuses_parts_with_crc_proof_and_durable_journal(tmp_path):
    import hashlib
    from geofirex.archive_transfer import extract_archive, verify_archive
    archive, entry, root, output, part, payload = fixture(tmp_path)
    with zipfile.ZipFile(archive) as z:
        index = index_zip(z, archive_size=archive.stat().st_size)
    index['archive_md5'] = hashlib.md5(archive.read_bytes()).hexdigest()
    verification = tmp_path / 'verification'
    verify_archive(archive, index, verification_dir=verification)
    with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('No archive payload read')):
        result = extract_archive(archive, index, root, verification_dir=verification)
    assert result['observations'] == result['reused_parts'] == 1 and result['network_bytes'] == 0
    log = rows(Path(result['publication_journal']))
    assert log[-1]['kind'] == 'extraction_complete'
    assert any(r['kind'] == 'member_verified' and r['sha256'] == sha256_file(part) for r in log)
    assert part.read_bytes() == output.read_bytes() == payload
    original = Path(result['publication_journal']).read_bytes()
    extract_archive(archive, index, root, verification_dir=verification)
    assert Path(result['publication_journal']).read_bytes() == original


def test_full_extraction_failure_records_stop_and_retains_part(tmp_path):
    import hashlib
    from geofirex.archive_transfer import extract_archive, verify_archive
    archive, entry, root, output, part, payload = fixture(tmp_path)
    with zipfile.ZipFile(archive) as z:
        index = index_zip(z, archive_size=archive.stat().st_size)
    index['archive_md5'] = hashlib.md5(archive.read_bytes()).hexdigest()
    verification = tmp_path / 'verification'
    verify_archive(archive, index, verification_dir=verification)
    with patch('geofirex.corpus.os.link', side_effect=windows_error(5)), patch('geofirex.corpus.time.sleep'):
        with pytest.raises(PermissionError):
            extract_archive(archive, index, root, verification_dir=verification)
    log = rows(next((root / '.extraction-session').glob('publication-*.jsonl')))
    assert log[-1]['kind'] == 'extraction_stopped' and log[-1]['winerror'] == 5
    assert not list((root / '.extraction-session').glob('complete-*.json'))
    assert not output.exists() and part.read_bytes() == payload


@pytest.mark.skipif(os.name != 'nt', reason='Native Windows sharing semantics')
def test_real_windows_handle_blocks_rename_then_bounded_retry_succeeds(tmp_path):
    import ctypes
    from ctypes import wintypes as w
    archive, entry, root, output, part, payload = fixture(tmp_path)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p,
                                  w.DWORD, w.DWORD, w.HANDLE]
    kernel.CreateFileW.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    # Read/write sharing permitted, DELETE sharing deliberately denied.
    handle = kernel.CreateFileW(str(part), 0x80000000, 3, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    journal = tmp_path / 'native-lock.jsonl'
    closed = False

    def release_during_backoff(delay):
        nonlocal closed
        assert not closed
        kernel.CloseHandle(handle)
        closed = True

    try:
        with zipfile.ZipFile(archive) as z, patch('geofirex.corpus.time.sleep', release_during_backoff):
            acquire_members(z, [entry], root, preserve_parts=False, publication_journal=journal)
    finally:
        if not closed:
            kernel.CloseHandle(handle)
    errors = [r for r in rows(journal) if r['kind'] == 'publication_error']
    assert closed and len(errors) == 1 and errors[0]['winerror'] in (5, 32, 33)
    assert output.read_bytes() == payload
