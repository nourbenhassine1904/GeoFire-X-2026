"""Synthetic ZIP transport checks, not wildfire observations or cloud metrics."""
import io
import json
import random
import re
import zipfile
from unittest.mock import patch

import pytest

from geofirex.corpus import BudgetRangeFile, MemberCachedRangeFile, acquire_members, index_zip
from geofirex.remote_archive import RangeResponseError


def test_member_cache_reduces_requests_without_reading_unselected_payload(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('2018/fire_1/2018-01-01.tif', random.Random(42).randbytes(2*1024**2))
        z.writestr('2019/fire_2/2019-01-01.tif', random.Random(43).randbytes(1024**2))
    body = archive.getvalue()
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        entries = index_zip(z)['entries']
        first_end = z.getinfo(entries[1]['path']).header_offset
    def fetch(cls, label):
        ranges = []
        def response(request, **kwargs):
            start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.headers['Range']).groups())
            ranges.append((start, end))
            class Response(io.BytesIO):
                status = 206
                headers = {'Content-Range': f'bytes {start}-{end}/{len(body)}'}
            return Response(body[start:end+1])
        stream = cls(10*1024**2, size=len(body), retries=1)
        with patch('urllib.request.urlopen', side_effect=response):
            with zipfile.ZipFile(stream) as z:
                acquire_members(z, entries[:1], tmp_path/label)
        return stream, ranges
    plain, _ = fetch(BudgetRangeFile, 'plain')
    cached, ranges = fetch(MemberCachedRangeFile, 'cached')
    assert cached.requests < plain.requests
    assert cached.bytes_fetched == plain.bytes_fetched
    assert (tmp_path/'cached'/entries[0]['path']).read_bytes() == (tmp_path/'plain'/entries[0]['path']).read_bytes()
    # ZIP central-directory reads occur after the unselected entry's payload.
    assert all(end < first_end or start >= first_end+30+len(entries[1]['path'])+1024**2 for start, end in ranges)


def test_transfer_reservation_is_durable_before_network_failure(tmp_path):
    ledger = tmp_path/'ledger.jsonl'
    stream = BudgetRangeFile(20, size=100, retries=1)
    stream.enable_ledger(ledger, 'plan-id')
    with patch('urllib.request.urlopen', side_effect=OSError('interrupted')):
        with pytest.raises(OSError):
            stream.read(10)
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert records[0]['plan_identity'] == 'plan-id'
    assert records[-1]['reserved_bytes'] == 11
    with pytest.raises(FileExistsError):
        BudgetRangeFile(20, size=100).enable_ledger(ledger, 'another-plan')
