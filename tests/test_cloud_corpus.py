"""Offline transport/API contracts; these tests do NOT prove live Kaggle persistence."""
import io
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from geofirex import cloud_corpus as cloud
from geofirex.cloud_corpus import (batch_inventory, fingerprint, plan_batch,
                                    prepare_mounted, seal, validate_plan)
from geofirex.corpus import BudgetRangeFile, acquire_members, index_zip, json_new
from geofirex.kaggle_persistence import (package_path, publish_package,
                                         snapshot_training, verify_fresh_session, verify_package)


@pytest.fixture
def batch(tmp_path):
    archive = tmp_path/"source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for year in (2018, 2020):
            for event in (1, 2):
                for day in (1, 2, 3):
                    z.writestr(f"{year}/fire_{event}/{year}-01-{day:02d}.tif",
                               f"fixture {year} {event} {day}".encode()*20)
    with zipfile.ZipFile(archive) as z:
        index = index_zip(z, archive_size=archive.stat().st_size)
    plan = plan_batch(index, years=[2018], max_events=1, transfer_bytes=10**7,
                      storage_bytes=2*1024**3, private_free_bytes=10**7, upload_bytes=10**7)
    root = tmp_path/"producer"
    with zipfile.ZipFile(archive) as z:
        acquire_members(z, plan["entries"], root)
    inventory = batch_inventory(plan, root)
    json_new(root/"inventory.json", inventory)
    return index, plan, root, inventory, archive


class FakeKaggle:
    def __init__(self, store):
        self.store, self.versions, self.uploads = store, {}, 0
        self.private = True
        self.fail_upload = False

    def username(self):
        return "tester"

    def create(self, handle, root, notes):
        if self.versions:
            error = RuntimeError("Dataset already exists")
            error.error_code = 409
            raise error
        return self.upload(handle, root, notes)

    def version(self, handle, root, notes):
        return self.upload(handle, root, notes)

    def metadata(self, handle):
        if not self.versions:
            return None
        return {"private": self.private, "version": len(self.versions)}

    def upload(self, handle, root, notes):
        self.uploads += 1
        if self.fail_upload:
            raise PermissionError("simulated upload permission denied")
        version = len(self.versions)+1
        dest = self.store/str(version)
        shutil.copytree(root, dest, ignore=shutil.ignore_patterns("*.part"))
        self.versions[version] = dest

    def mount(self, handle, *, inventory_only=False):
        version = int(handle.rsplit("/", 1)[1])
        root = self.versions[version]
        return root/"inventory.json" if inventory_only else root


def publish(root, backend, receipt, **kw):
    return publish_package(root, "tester/geofirex-batch", receipt,
                           upload_bytes=10**7, private_free_bytes=10**7,
                           session="session-1", backend=backend, attempts=1, delay=0, **kw)


def test_whole_event_planning_exclusion_quota_and_tamper(batch):
    index, plan, _, _, _ = batch
    assert plan["observations"] == 3
    validate_plan(plan, index)
    with pytest.raises(ValueError, match="identity"):
        validate_plan({**plan, "entries": plan["entries"][:1]}, index)
    partial = seal({k: v for k, v in plan.items() if k != "identity"} | {"entries": plan["entries"][:1]})
    with pytest.raises(ValueError, match="every observation"):
        validate_plan(partial, index)
    other = plan_batch(index, years=[2018], max_events=1, transfer_bytes=10**7,
                       storage_bytes=2*1024**3, private_free_bytes=10**7, upload_bytes=10**7,
                       exclude_events=plan["events"])
    assert not set(other["events"]) & set(plan["events"])
    with pytest.raises(ValueError, match="fits"):
        plan_batch(index, years=[2018], max_events=1, transfer_bytes=100,
                   storage_bytes=2*1024**3, private_free_bytes=10**7, upload_bytes=10**7)


def test_range_zip_stream_batch_resume_and_exact_inventory(batch, tmp_path, monkeypatch):
    index, plan, _, _, archive = batch
    original = cloud.validate_index
    # Tiny archive validates all structural rules, but has no claim to be official.
    monkeypatch.setattr(cloud, "validate_index", lambda i, **kw: original(i))
    body = archive.read_bytes()
    calls = []
    def response(request, **kw):
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        calls.append((start, end))
        class Reply(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes {start}-{end}/{len(body)}"}
        return Reply(body[start:end+1])
    root = tmp_path/"range-destination"
    factory = lambda limit, **kw: BudgetRangeFile(limit, size=len(body), retries=1)
    with patch("urllib.request.urlopen", side_effect=response):
        inv = cloud.acquire_batch(plan, index, root, tmp_path/"first.json", stream_factory=factory)
    assert inv["observations"] == 3 and calls
    with patch("urllib.request.urlopen") as network:
        again = cloud.acquire_batch(plan, index, root, tmp_path/"resume.json", stream_factory=factory)
        network.assert_not_called()
    assert inv == again
    assert json.loads((tmp_path/"resume.json").read_text())["transfer"]["requests"] == 0


def test_private_publish_readback_fresh_session_and_idempotency(batch, tmp_path):
    _, _, root, inventory, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    first = publish(root, backend, tmp_path/"receipt1.json")
    assert first["status"] == "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
    saved = json.loads((tmp_path/"receipt1.json").read_text())
    with pytest.raises(ValueError, match="new Kaggle"):
        verify_fresh_session(saved, session="session-1", backend=backend)
    result = verify_fresh_session(saved, session="session-2", backend=backend)
    assert result["status"] == "PERSISTENCE_VERIFIED_NEW_SESSION"
    assert result["verification"]["files"] == inventory["observations"]
    publish(root, backend, tmp_path/"receipt2.json")
    assert backend.uploads == 1
    backend.private = False
    with pytest.raises(ValueError, match="public"):
        publish(root, backend, tmp_path/"public.json", allow_version=True)
    assert backend.uploads == 1
    assert json.loads((tmp_path/"public.json").read_text())["status"] == "FAILED_FILES_PRESERVED"


def test_upload_failure_hash_mismatch_and_no_data_removal(batch, tmp_path):
    _, _, root, inv, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    backend.fail_upload = True
    with pytest.raises(PermissionError):
        publish(root, backend, tmp_path/"failed.json")
    assert verify_package(root, inv)["files"] == 3
    backend.fail_upload = False
    publish(root, backend, tmp_path/"good.json")
    remote = backend.versions[1]/inv["files"][0]["path"]
    remote.write_bytes(b"X"*remote.stat().st_size)
    with pytest.raises(ValueError, match="SHA256"):
        publish(root, backend, tmp_path/"corrupt.json")
    assert verify_package(root, inv)["files"] == 3


def test_unexpected_files_missing_hashes_and_traversal_refused(batch):
    _, _, root, inv, _ = batch
    (root/"credentials.json").write_text("fixture, not a credential")
    with pytest.raises(ValueError, match="Unexpected"):
        verify_package(root, inv)
    for name in ("../x", "/x", "C:/x", "a\\b", "a/./b"):
        with pytest.raises(ValueError, match="Unsafe"):
            package_path(root, name)


def test_snapshot_preserves_best_last_and_sqlite(tmp_path):
    import sqlite3
    run = tmp_path/"run"
    run.mkdir()
    for epoch, ap in ((1, .5), (2, .4), (3, .3)):
        (run/f"epoch_{epoch:04d}.pt").write_bytes(b"checkpoint fixture")
        json_new(run/f"validation_{epoch:04d}.json", {"micro": {"average_precision": ap}})
    (run/"scratch.bin").write_bytes(b"scratch")
    database = tmp_path/"tracking.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE example (value INTEGER)")
        db.execute("INSERT INTO example VALUES (3)")
    dest = tmp_path/"backup"
    snapshot_training(run, dest, tracking_db=database)
    assert {p.name for p in (dest/"run").glob("*.pt")} == {"epoch_0001.pt", "epoch_0003.pt"}
    assert not (dest/"run/scratch.bin").exists() and (run/"scratch.bin").exists()
    assert verify_package(dest, json.loads((dest/"inventory.json").read_text()))["files"] == 6
    with sqlite3.connect(dest/"tracking.db") as db:
        assert db.execute("SELECT value FROM example").fetchone()[0] == 3


def test_explicit_dataset_version_preserves_prior_version(batch, tmp_path):
    from geofirex.corpus import sha256_file
    _, _, root, inv, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    publish(root, backend, tmp_path/"first.json")
    extra = root/"note.txt"
    extra.write_text("version 2 fixture")
    files = inv["files"] + [{"path": "note.txt", "bytes": extra.stat().st_size, "sha256": sha256_file(extra)}]
    updated = seal({k: v for k, v in inv.items() if k != "identity"} | {"files": files})
    (root/"inventory.json").write_text(json.dumps(updated))
    with pytest.raises(ValueError, match="different content"):
        publish(root, backend, tmp_path/"refused.json")
    result = publish(root, backend, tmp_path/"second.json", allow_version=True)
    assert result["versioned_handle"].endswith("/2") and backend.uploads == 2
    assert verify_package(backend.versions[1], inv)["files"] == 3


def test_upload_quota_refuses_before_any_backend_call(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    with pytest.raises(ValueError, match="quotas"):
        publish_package(root, "tester/geofirex-batch", tmp_path/"quota.json", upload_bytes=1,
                        private_free_bytes=1, session="one", backend=backend)
    assert backend.uploads == 0


def test_full_official_campaign_has_exact_disjoint_coverage(tmp_path):
    from scripts.plan_cloud_campaign import plan_campaign
    index = json.loads((Path(__file__).resolve().parents[1]/"data/manifests/real/corpus_zip_index_v1.json").read_text())
    report = plan_campaign(index, tmp_path/"plans", transfer_bytes=4*1024**3,
                           storage_bytes=6*1024**3, upload_bytes=5*1024**3,
                           private_free_bytes=64*1024**3)
    assert report["events"] == 607 and report["observations"] == 13607
    assert report["by_split"]["train"] == {"events": 250, "observations": 5198}
    entries = []
    for item in report["batches"]:
        plan = json.loads((tmp_path/"plans"/item["plan"]).read_text())
        validate_plan(plan, index)
        entries.extend(e["path"] for e in plan["entries"])
    assert len(entries) == len(set(entries)) == 13607
    assert set(entries) == {e["path"] for e in index["entries"]}


def test_official_backend_refuses_http_download_fallback(monkeypatch):
    import sys
    import types
    from unittest.mock import Mock
    from geofirex.kaggle_persistence import KaggleBackend
    hub = types.ModuleType("kagglehub")
    hub.dataset_download = Mock(return_value="/kaggle/input/shared/inventory.json")
    config = types.ModuleType("kagglehub.config")
    env = types.ModuleType("kagglesdk.kaggle_env")
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    monkeypatch.setitem(sys.modules, "kagglehub.config", config)
    monkeypatch.setitem(sys.modules, "kagglesdk.kaggle_env", env)
    config.is_kaggle_cache_disabled = lambda: True
    env.is_in_kaggle_notebook = lambda: True
    with pytest.raises(RuntimeError, match="shared cache"):
        KaggleBackend().mount("tester/dataset/versions/1")
    hub.dataset_download.assert_not_called()
    config.is_kaggle_cache_disabled = lambda: False
    path = KaggleBackend().mount("tester/dataset/versions/1", inventory_only=True)
    assert path.name == "inventory.json"
    hub.dataset_download.assert_called_once_with("tester/dataset/versions/1", path="inventory.json")
