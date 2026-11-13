import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import requests
from sklearn.metrics import average_precision_score

from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.job_control import supervise
from geofirex.runner_data import digest
from geofirex.upload_budget import BodyBudget


def test_exact_ap_scratch_cleanup_preserves_scores_and_unowned_files(tmp_path):
    p = np.random.default_rng(11).random(231)
    p[::3] = .5
    y = (np.arange(len(p)) % 7 == 0).astype(np.uint8)
    ap = ExternalAveragePrecision(tmp_path/'scratch', chunk_pixels=5, fan_in=2, retain_runs=False)
    sentinel = tmp_path/'scratch'/'unowned.bin'
    sentinel.write_bytes(b'preserve')
    for start in range(0, len(p), 11):
        ap.update(p[start:start+11], y[start:start+11])
    assert ap.compute() == pytest.approx(average_precision_score(y, p), abs=1e-14)
    assert list((tmp_path/'scratch').iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'preserve'
    empty = ExternalAveragePrecision(tmp_path/'empty', retain_runs=False)
    empty.update(np.ones(4), np.zeros(4))
    assert empty.compute() is None and not list((tmp_path/'empty').iterdir())


def test_parent_deadline_stops_real_child_and_records_limit(tmp_path):
    with (tmp_path/'log').open('w') as log:
        result = supervise([sys.executable, '-c', 'import time; time.sleep(20)'], seconds=1, log=log)
    assert result['status'] == 'HARD_TIME_LIMIT'
    assert result['elapsed_seconds'] < 8


def test_parent_resource_failure_stops_child(tmp_path):
    def resource():
        raise RuntimeError('resource fixture')
    with (tmp_path/'log').open('w') as log:
        result = supervise([sys.executable, '-c', 'import time; time.sleep(20)'], seconds=20, log=log, check=resource)
    assert result['status'] == 'RESOURCE_LIMIT_OR_INTERRUPTION'
    assert result['resource_error'] == 'RuntimeError'
    assert result['elapsed_seconds'] < 8


def test_http_budget_durable_cumulative_includes_retries(tmp_path):
    journal = tmp_path/'budget.jsonl'
    with BodyBudget(journal, 12) as budget:
        budget.reserve(5)
        budget.reserve(5)  # retry must be charged again
        with pytest.raises(RuntimeError, match='exhausted'):
            budget.reserve(3)
        assert budget.used == 10
        with pytest.raises(FileExistsError):
            BodyBudget(journal, 12)
    with BodyBudget(journal, 12) as resumed:
        assert resumed.used == 10
        resumed.reserve(2)
    with pytest.raises(ValueError, match='mismatch'):
        BodyBudget(journal, 13)


def test_requests_guard_blocks_before_send_and_preserves_partial_journal(tmp_path, monkeypatch):
    sent = []
    def fake_send(self, req, **kwargs):
        sent.append(req)
        return requests.Response()
    monkeypatch.setattr(requests.sessions.Session, 'send', fake_send)
    with BodyBudget(tmp_path/'http.jsonl', 7) as budget:
        restore = budget.install_requests_guard()
        try:
            session = requests.Session()
            req = requests.Request('PUT', 'https://example.invalid', data=b'1234').prepare()
            session.send(req)
            with pytest.raises(RuntimeError, match='exhausted'):
                session.send(req)
            assert len(sent) == 1 and budget.used == 4
            req.headers.pop('Content-Length')
            with pytest.raises(RuntimeError, match='Unknown HTTP body'):
                session.send(req)
        finally:
            restore()
    with (tmp_path/'http.jsonl').open('a') as f:
        f.write('{"partial":')
    with pytest.raises(json.JSONDecodeError):
        BodyBudget(tmp_path/'http.jsonl', 7)


def test_staging_stream_verifies_sha_crc_and_preserves_invalid_part(tmp_path):
    import hashlib
    import zlib
    module_path = Path(__file__).resolve().parents[1]/'scripts/stage_expanded_lot.py'
    spec = importlib.util.spec_from_file_location('stage', module_path)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    root = tmp_path/'raw'
    source = root/'2018/fire_1/2018-01-01.tif'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'fixture-only')
    member = dict(path=source.relative_to(root).as_posix(), bytes=12,
                  sha256=hashlib.sha256(source.read_bytes()).hexdigest(), crc32=f'{zlib.crc32(source.read_bytes()):08x}')
    lot = dict(role='train', handle='owner/lot', bytes=12, observations=1, files=[member])
    lot['identity'] = digest(lot)
    path = tmp_path/'lot.json'
    path.write_text(json.dumps(lot))
    result = stage.stage(path, root, tmp_path/'valid', max_bytes=11_000_000, reserve=0)
    assert result['observations'] == 1 and (tmp_path/'valid/payload.zip').exists()
    source.write_bytes(b'corrupt-only')
    with pytest.raises(ValueError, match='integrity changed'):
        stage.stage(path, root, tmp_path/'invalid', max_bytes=11_000_000, reserve=0)
    assert (tmp_path/'invalid/payload.zip.part').exists() and source.read_bytes() == b'corrupt-only'
