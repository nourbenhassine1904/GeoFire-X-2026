import json
import shutil
from pathlib import Path

import pytest

from geofirex.cloud_corpus import seal
from geofirex.corpus import json_new
from geofirex.orchestration import Orchestrator, redact


@pytest.fixture
def runner(tmp_path):
    root = Path(__file__).resolve().parents[1]
    return Orchestrator(root, state=tmp_path)


def proof(tmp_path):
    value = seal({"kind": "manual_adoption", "upload_called": False,
        "status": "MANUAL_ADOPTED_FRESH_SESSION_PENDING", "authenticated_owner": "owner",
        "private_verified": True, "handle": "owner/batch", "versioned_handle": "owner/batch/versions/1",
        "producer_session": "original-session", "inventory_identity": "inventory", "index_identity": "index",
        "verification": {"files": 14, "bytes": 45, "identity": "inventory"}})
    path = tmp_path/"proof.json"
    json_new(path, value)
    return path


def test_private_cpu_exact_proof_and_tamper_detection(runner, tmp_path):
    original = proof(tmp_path)
    job = runner.prepare("session-b", "owner", proof=original)
    assert (job/"manual-adoption.json").read_bytes() == original.read_bytes()
    metadata = json.loads((job/"kernel-metadata.json").read_text())
    assert metadata["is_private"] is True and metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == ["owner/batch"]
    compile((job/"job.py").read_text(), "job.py", "exec")
    runner.load(job.name)
    (job/"kernel-metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata changed"):
        runner.load(job.name)


def test_ambiguous_submit_cannot_duplicate_remote_job(runner, monkeypatch):
    job = runner.prepare("publication-probe", "owner")
    monkeypatch.setattr(runner, "call", lambda *a, **kw: {"returncode": None})
    runner.submit(job.name)
    with pytest.raises(ValueError, match="already attempted"):
        runner.submit(job.name)
    assert runner.status()["jobs"][0]["latest_local_event"] == "SUBMISSION_UNCONFIRMED"


def test_false_certificate_not_accepted(runner, tmp_path, monkeypatch):
    job = runner.prepare("session-b", "owner", proof=proof(tmp_path))
    def fake_call(job, args, **kwargs):
        folder = Path(args[args.index("-p")+1])/"geofirex-result"
        json_new(folder/"result.json", {"job_id": job.name, "kind": "session-b", "success": True})
        json_new(folder/"bootstrap.json", seal({"status": "PERSISTENCE_VERIFIED_NEW_SESSION"}))
        return {"returncode": 0}
    monkeypatch.setattr(runner, "call", fake_call)
    with pytest.raises(ValueError, match="does not certify"):
        runner.fetch(job.name)
    assert runner.status()["jobs"][0]["evidence"] is None


def test_redaction():
    assert "secret" not in redact('Bearer secret access_token=secret https://a/?code=secret&b=ok')


@pytest.mark.parametrize('remote,expected', [
    ('KernelWorkerStatus.CANCEL_ACKNOWLEDGED', 'CANCELACKNOWLEDGED'),
    ('KernelWorkerStatus.CANCEL_REQUESTED', 'CANCELREQUESTED'),
    ('cancelacknowledged', 'CANCELACKNOWLEDGED'),
    ('KernelWorkerStatus.RUNNING', 'RUNNING'),
])
def test_actual_cli_status_names(runner, monkeypatch, remote, expected):
    job = runner.prepare('publication-probe', 'owner')
    monkeypatch.setattr(runner, 'call', lambda *a, **kw: {
        'returncode': 0, 'stdout': f'owner/job has status "{remote}"', 'stderr': ''})
    assert runner.refresh(job.name)['remote_status'] == expected


def test_reject_unapproved_jobs(runner):
    with pytest.raises(ValueError, match="bounded CPU"):
        runner.prepare("large-gpu-campaign", "owner")
