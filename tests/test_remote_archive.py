"""Tests for geofirex.remote_archive.

Entry-parsing/extraction/resume logic is tested against a small LOCAL zip (no
network) so these run fast and offline. The Range-response validation
(_validated_range_get) is tested with a mocked urllib.request.urlopen, since
it is precisely the HTTP-status/Content-Range checking logic that must be
covered without hitting the real Zenodo server. The end-to-end behavior
against the live server was additionally validated manually - see
reports/DATA_AUDIT.md §1.
"""

import http.client
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from geofirex.remote_archive import (
    RangeResponseError,
    _validated_range_get,
    extract_events,
    get_remote_size,
    list_real_events,
    missing_files_for_event,
)


def _build_local_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("2018/fire_111/2018-05-03.tif", b"fake-day-1")
        zf.writestr("2018/fire_111/2018-05-04.tif", b"fake-day-2")
        zf.writestr("2019/fire_222/2019-07-01.tif", b"fake-day-3")
        zf.writestr("2018/fire_111/", b"")  # a folder entry, must be ignored
        zf.writestr("README.txt", b"not an event file")  # unexpected top-level file


def test_list_real_events_parses_year_fire_date_layout(tmp_path: Path):
    zip_path = tmp_path / "fake.zip"
    _build_local_zip(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        events = list_real_events(zf)

    assert set(events.keys()) == {"2018/fire_111", "2019/fire_222"}
    assert events["2018/fire_111"] == ["2018-05-03.tif", "2018-05-04.tif"]
    assert events["2019/fire_222"] == ["2019-07-01.tif"]


def test_extract_events_writes_only_requested_events(tmp_path: Path):
    zip_path = tmp_path / "fake.zip"
    _build_local_zip(zip_path)
    dest = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path) as zf:
        written = extract_events(zf, ["2019/fire_222"], dest)

    assert len(written) == 1
    out_file = dest / "2019" / "fire_222" / "2019-07-01.tif"
    assert out_file in written
    assert out_file.read_bytes() == b"fake-day-3"
    # The other event must not have been extracted.
    assert not (dest / "2018").exists()


# --- Fix 1: an event is only "complete" when every expected file is present ---


def test_missing_files_for_event_empty_dir_reports_all_missing(tmp_path: Path):
    missing = missing_files_for_event(tmp_path, "2018/fire_111", ["2018-05-03.tif", "2018-05-04.tif"])
    assert missing == ["2018-05-03.tif", "2018-05-04.tif"]


def test_missing_files_for_event_single_file_is_not_considered_complete(tmp_path: Path):
    event_dir = tmp_path / "2018" / "fire_111"
    event_dir.mkdir(parents=True)
    (event_dir / "2018-05-03.tif").write_bytes(b"data")

    missing = missing_files_for_event(tmp_path, "2018/fire_111", ["2018-05-03.tif", "2018-05-04.tif"])
    assert missing == ["2018-05-04.tif"]  # still incomplete - not just "some file exists"


def test_missing_files_for_event_ignores_zero_byte_files(tmp_path: Path):
    event_dir = tmp_path / "2018" / "fire_111"
    event_dir.mkdir(parents=True)
    (event_dir / "2018-05-03.tif").write_bytes(b"")  # e.g. left over from an interrupted write

    missing = missing_files_for_event(tmp_path, "2018/fire_111", ["2018-05-03.tif"])
    assert missing == ["2018-05-03.tif"]


def test_missing_files_for_event_all_present_is_complete(tmp_path: Path):
    event_dir = tmp_path / "2018" / "fire_111"
    event_dir.mkdir(parents=True)
    (event_dir / "2018-05-03.tif").write_bytes(b"data")
    (event_dir / "2018-05-04.tif").write_bytes(b"data")

    missing = missing_files_for_event(tmp_path, "2018/fire_111", ["2018-05-03.tif", "2018-05-04.tif"])
    assert missing == []


# --- Fix 1b: extract_events resumes an incomplete extraction at file level ---


def test_extract_events_resumes_incomplete_event(tmp_path: Path):
    zip_path = tmp_path / "fake.zip"
    _build_local_zip(zip_path)
    dest = tmp_path / "extracted"

    # Pre-populate as if a prior run was interrupted: one correct file already
    # extracted, one file present but wrong size (should be re-fetched).
    event_dir = dest / "2018" / "fire_111"
    event_dir.mkdir(parents=True)
    (event_dir / "2018-05-03.tif").write_bytes(b"fake-day-1")  # correct, 10 bytes
    (event_dir / "2018-05-04.tif").write_bytes(b"x")  # wrong size - must be re-extracted

    with zipfile.ZipFile(zip_path) as zf:
        written = extract_events(zf, ["2018/fire_111"], dest)

    written_names = {p.name for p in written}
    assert written_names == {"2018-05-04.tif"}  # only the wrong-sized file was rewritten
    assert (event_dir / "2018-05-04.tif").read_bytes() == b"fake-day-2"
    assert (event_dir / "2018-05-03.tif").read_bytes() == b"fake-day-1"  # untouched


# --- Fix 2: Range responses are validated before their body is ever read ---


class _FakeResponse:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self._body = io.BytesIO(body)

    def getcode(self):
        return self.status

    def read(self):
        return self._body.read()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_validated_range_get_accepts_correct_206_response():
    resp = _FakeResponse(206, {"Content-Range": "bytes 0-3/100"}, b"abcd")
    with patch("urllib.request.urlopen", return_value=resp):
        data = _validated_range_get("http://example.test/f", 0, 3, retries=1)
    assert data == b"abcd"


def test_validated_range_get_rejects_unexpected_200_without_reading_full_body():
    # A 200 response here would normally carry the ENTIRE remote file. We
    # assert the error is raised and that this fake "huge" body was never
    # fully consumed/returned - read() is never called on a non-206 response.
    huge_body = b"X" * 10_000
    resp = _FakeResponse(200, {}, huge_body)
    with patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(RangeResponseError, match="206"):
            _validated_range_get("http://example.test/f", 0, 3, retries=1)
    assert resp._body.tell() == 0  # body was never read


def test_validated_range_get_rejects_content_range_mismatch():
    # Server claims 206 but returns a different range than what was requested.
    resp = _FakeResponse(206, {"Content-Range": "bytes 10-13/100"}, b"abcd")
    with patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(RangeResponseError, match="Content-Range"):
            _validated_range_get("http://example.test/f", 0, 3, retries=1)


def test_validated_range_get_rejects_short_body():
    resp = _FakeResponse(206, {"Content-Range": "bytes 0-3/100"}, b"ab")  # 2 bytes, expected 4
    with patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(RangeResponseError, match="bytes"):
            _validated_range_get("http://example.test/f", 0, 3, retries=1)


def test_get_remote_size_rejects_unexpected_200():
    resp = _FakeResponse(200, {}, b"X" * 100)
    with patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(RangeResponseError, match="206"):
            get_remote_size("http://example.test/f", retries=1)


def test_get_remote_size_parses_total_from_content_range():
    resp = _FakeResponse(206, {"Content-Range": "bytes 0-0/48359369821"}, b"X")
    with patch("urllib.request.urlopen", return_value=resp):
        size = get_remote_size("http://example.test/f", retries=1)
    assert size == 48359369821


def test_validated_range_get_retries_interrupted_response_body():
    interrupted = _FakeResponse(206, {"Content-Range": "bytes 0-3/100"}, b"")
    good = _FakeResponse(206, {"Content-Range": "bytes 0-3/100"}, b"abcd")
    with patch.object(interrupted, "read", side_effect=http.client.IncompleteRead(b"ab", 2)):
        with patch("urllib.request.urlopen", side_effect=[interrupted, good]) as request:
            with patch("geofirex.remote_archive._backoff_sleep"):
                assert _validated_range_get("http://example.test/f", 0, 3, retries=2) == b"abcd"
    assert request.call_count == 2


def test_validated_range_get_stops_after_repeated_body_interruptions():
    response = _FakeResponse(206, {"Content-Range": "bytes 0-3/100"}, b"")
    with patch.object(response, "read", side_effect=http.client.IncompleteRead(b"ab", 2)):
        with patch("urllib.request.urlopen", return_value=response) as request:
            with patch("geofirex.remote_archive._backoff_sleep"):
                with pytest.raises(RangeResponseError, match="Giving up after 2"):
                    _validated_range_get("http://example.test/f", 0, 3, retries=2)
    assert request.call_count == 2


def test_extract_events_preserves_destination_when_read_fails(tmp_path):
    zip_path = tmp_path / "fake.zip"
    _build_local_zip(zip_path)
    dest = tmp_path / "extracted"
    output = dest / "2019/fire_222/2019-07-01.tif"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"original-incomplete")
    with zipfile.ZipFile(zip_path) as zf:
        with patch.object(zf, "open", side_effect=OSError("connection interrupted")):
            with pytest.raises(OSError, match="connection interrupted"):
                extract_events(zf, ["2019/fire_222"], dest)
    assert output.read_bytes() == b"original-incomplete"


def test_extract_events_preserves_backup_of_incomplete_file(tmp_path):
    zip_path = tmp_path / "fake.zip"
    _build_local_zip(zip_path)
    dest = tmp_path / "extracted"
    output = dest / "2019/fire_222/2019-07-01.tif"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"short")
    with zipfile.ZipFile(zip_path) as zf:
        extract_events(zf, ["2019/fire_222"], dest)
    backups = list(output.parent.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"short"
    assert output.read_bytes() == b"fake-day-3"
