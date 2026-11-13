import hashlib
import io
import json
from unittest.mock import patch

import pytest

from geofirex.archive_transfer import BodyLedger
from scripts.install_local_spark_bounded import fetch_artifact


def fixture(tmp_path, payload=b'package fixture bytes'):
    item = dict(filename='fixture.tar.gz', url='https://files.pythonhosted.org/fixture',
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    ledger = BodyLedger(tmp_path/'ledger.jsonl', dict(budget_bytes=1000))
    return item, ledger, payload


class Response(io.BytesIO):
    status = 200
    def __init__(self, data, headers):
        super().__init__(data)
        self.headers = headers


def test_hard_cap_stops_before_request(tmp_path):
    item, ledger, payload = fixture(tmp_path)
    ledger.reserve(999)
    with patch('urllib.request.urlopen') as network:
        with pytest.raises(RuntimeError, match='budget'):
            fetch_artifact(item, tmp_path, ledger)
    network.assert_not_called()


def test_interruption_preserves_prefix_then_range_resume_counts_all_attempts(tmp_path):
    item, ledger, payload = fixture(tmp_path)
    calls = []
    class Interrupted(Response):
        def read(self, n):
            if self.tell():
                raise OSError('Disconnected after first committed block')
            return super().read(n)
    def first(request, **kwargs):
        calls.append(request)
        return Interrupted(payload, {'Content-Length': str(len(payload))})
    with patch('urllib.request.urlopen', first), patch('scripts.install_local_spark_bounded.time.sleep'):
        with pytest.raises(OSError):
            fetch_artifact(item, tmp_path, ledger, block_size=5)
    partial = tmp_path/(item['filename']+'.part')
    assert partial.read_bytes() == payload[:5] and ledger.reserved == 10
    ledger = BodyLedger(tmp_path/'ledger.jsonl', dict(budget_bytes=1000))
    def resumed(request, **kwargs):
        assert request.headers['Range'] == f'bytes=5-{len(payload)-1}'
        r = Response(payload[5:], {'Content-Length': str(len(payload)-5),
                                  'Content-Range': f'bytes 5-{len(payload)-1}/{len(payload)}'})
        r.status = 206
        return r
    with patch('urllib.request.urlopen', resumed), patch('scripts.install_local_spark_bounded.time.sleep'):
        target = fetch_artifact(item, tmp_path, ledger, block_size=5)
    assert target.read_bytes() == partial.read_bytes() == payload
    assert ledger.confirmed == len(payload) and ledger.reserved == len(payload)+5
    with patch('urllib.request.urlopen') as network:
        assert fetch_artifact(item, tmp_path, ledger) == target
    network.assert_not_called()


def test_ignored_resume_never_appends_and_bad_sha_never_installs(tmp_path):
    item, ledger, payload = fixture(tmp_path)
    part = tmp_path/(item['filename']+'.part')
    part.write_bytes(payload[:5]); ledger.confirmed = 5
    with patch('urllib.request.urlopen', return_value=Response(payload, {'Content-Length': str(len(payload))})):
        with pytest.raises(ValueError, match='refused'):
            fetch_artifact(item, tmp_path, ledger)
    assert part.read_bytes() == payload[:5]
    item['sha256'] = '0'*64
    r = Response(payload[5:], {'Content-Length': str(len(payload)-5),
                              'Content-Range': f'bytes 5-{len(payload)-1}/{len(payload)}'})
    r.status = 206
    with patch('urllib.request.urlopen', return_value=r), patch('scripts.install_local_spark_bounded.time.sleep'):
        with pytest.raises(ValueError, match='SHA-256'):
            fetch_artifact(item, tmp_path, ledger)
    assert part.read_bytes() == payload and not (tmp_path/item['filename']).exists()
