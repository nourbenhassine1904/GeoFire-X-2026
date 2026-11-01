"""Publication/auth/adoption contract simulations; no live Kaggle API calls."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from test_cloud_corpus import batch, FakeKaggle, publish
from geofirex import cloud_corpus as cloud
from geofirex.cloud_corpus import seal
from geofirex.kaggle_persistence import (KaggleBackend, adopt_manual, inspect_dataset,
    publish_package, validate_bootstrap, verify_fresh_session)


def http_error(status):
    error = RuntimeError(f"HTTP {status} fixture")
    error.response = types.SimpleNamespace(status_code=status)
    return error


def test_create_does_not_lookup_nonexistent_handle_that_returns_403(batch, tmp_path):
    _, _, root, _, _ = batch
    class HiddenMissing(FakeKaggle):
        def metadata(self, handle):
            if not self.versions:
                raise http_error(403)
            return super().metadata(handle)
    backend = HiddenMissing(tmp_path/"remote")
    result = publish(root, backend, tmp_path/"new.json", mode="create")
    assert result["private_verified"] and backend.uploads == 1


def test_403_is_unknown_authorization_and_resume_never_creates(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    backend.metadata = Mock(side_effect=http_error(403))
    result = inspect_dataset("tester/geofirex-batch", backend=backend)
    assert result["existence"] == "UNKNOWN" and result["http_status"] == 403
    with pytest.raises(RuntimeError, match="403"):
        publish(root, backend, tmp_path/"denied.json", mode="resume")
    receipt = json.loads((tmp_path/"denied.json").read_text())
    assert receipt["http_status"] == 403 and not receipt["upload_called"]
    assert backend.uploads == 0


def test_other_user_refused_before_metadata_or_upload(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    backend.username = lambda: "another-user"
    backend.metadata = Mock()
    with pytest.raises(PermissionError, match="authenticated"):
        publish(root, backend, tmp_path/"owner.json")
    backend.metadata.assert_not_called()
    assert backend.uploads == 0


def test_public_create_conflict_has_no_version_fallback(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    publish(root, backend, tmp_path/"created.json")
    backend.private = False
    backend.version = Mock(side_effect=AssertionError("must never version"))
    with pytest.raises(ValueError, match="public"):
        publish(root, backend, tmp_path/"public.json", mode="create")
    backend.version.assert_not_called()
    assert backend.uploads == 1


def test_server_success_then_timeout_recovers_readonly(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    create = backend.create
    def interrupted(*args):
        create(*args)
        raise TimeoutError("response lost after server success")
    backend.create = interrupted
    with pytest.raises(TimeoutError):
        publish(root, backend, tmp_path/"uncertain.json")
    result = publish(root, backend, tmp_path/"recovered.json", mode="resume")
    assert result["status"] == "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
    assert not result["upload_called"] and backend.uploads == 1
    assert result["kind"] == "remote_recovery"
    certificate = verify_fresh_session(json.loads((tmp_path/"recovered.json").read_text()),
                                       session="fresh", backend=backend)
    assert certificate["publication_provenance"] == "recovered_existing"


def test_batch_dynamic_attachment_restriction_fails_fast_without_reupload(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/'remote')
    backend.mount = Mock(side_effect=RuntimeError('New Datasets cannot be attached in non-interactive sessions'))
    with patch('geofirex.kaggle_persistence.time.sleep') as sleep:
        with pytest.raises(RuntimeError, match='requires a new job'):
            publish_package(root, 'tester/geofirex-batch', tmp_path/'blocked.json', backend=backend,
                session='publisher', upload_bytes=10**8, private_free_bytes=10**8, attempts=60)
    receipt = json.loads((tmp_path/'blocked.json').read_text())
    assert receipt['mutation_rpc_returned_success'] is True
    assert receipt['readback_blocker'] == 'KAGGLE_BATCH_DYNAMIC_ATTACHMENT_FORBIDDEN'
    assert backend.uploads == 1 and len(receipt['readback_attempts']) == 1
    sleep.assert_not_called()


def test_resume_allows_already_persisted_mount_without_new_upload_quota(batch, tmp_path):
    _, _, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    publish(root, backend, tmp_path/"first.json")
    result = publish_package(backend.versions[1], "tester/geofirex-batch", tmp_path/"resume.json",
        mode="resume", session="new-session", backend=backend, upload_bytes=0, private_free_bytes=0)
    assert result["private_verified"] and backend.uploads == 1


def test_manual_adoption_is_distinct_and_bootstrap_rechecks_privacy_and_bytes(batch, tmp_path, monkeypatch):
    index, _, root, _, _ = batch
    original = cloud.validate_index
    monkeypatch.setattr(cloud, "validate_index", lambda i, **kw: original(i))
    backend = FakeKaggle(tmp_path/"manually-published")
    # Simulate publication by the user, outside the pipeline under test.
    backend.upload("tester/geofirex-batch", root, "user manual publication fixture")
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    proof = adopt_manual(index, backend.versions[1], "tester/geofirex-batch", session="adoption-session", backend=backend)
    assert proof["kind"] == "manual_adoption" and proof["upload_called"] is False
    assert proof["status"] == "MANUAL_ADOPTED_FRESH_SESSION_PENDING"
    with pytest.raises(ValueError, match="new Kaggle"):
        verify_fresh_session(proof, session="adoption-session", backend=backend)
    certificate = verify_fresh_session(proof, session="independent-session", backend=backend)
    assert certificate["publication_provenance"] == "manual"
    assert validate_bootstrap(certificate, index, backend=backend)["files"] == 3
    assert backend.uploads == 1
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    backend.private = False
    with pytest.raises(ValueError, match="public"):
        validate_bootstrap(certificate, index, backend=backend)
    backend.private = True
    file = backend.versions[1]/next(iter(index["entries"]))["path"]
    file.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="SHA256"):
        validate_bootstrap(certificate, index, backend=backend)


def test_pending_or_forged_manual_certificate_does_not_unlock(batch, tmp_path, monkeypatch):
    index, _, root, _, _ = batch
    original = cloud.validate_index
    monkeypatch.setattr(cloud, "validate_index", lambda i, **kw: original(i))
    backend = FakeKaggle(tmp_path/"remote")
    backend.upload("tester/geofirex-batch", root, "manual fixture")
    proof = adopt_manual(index, root, "tester/geofirex-batch", session="one", backend=backend)
    with pytest.raises(ValueError, match="independent"):
        validate_bootstrap(proof, index, backend=backend)
    forged = seal({k: v for k, v in proof.items() if k != "identity"} |
        {"status": "PERSISTENCE_VERIFIED_NEW_SESSION", "session": "one", "publication_provenance": "manual"})
    with pytest.raises(ValueError, match="provenance/session"):
        validate_bootstrap(forged, index, backend=backend)


def test_official_create_request_is_explicit_private_and_never_gets_or_versions(tmp_path, monkeypatch):
    calls = []
    api = types.SimpleNamespace(create_dataset=lambda request: calls.append(request))
    class Client:
        datasets = types.SimpleNamespace(dataset_api_client=api)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    clients = types.ModuleType("kagglehub.clients")
    clients.build_kaggle_client = Client
    service = types.ModuleType("kagglesdk.datasets.types.dataset_api_service")
    service.ApiCreateDatasetRequest = type("CreateRequest", (), {})
    monkeypatch.setitem(sys.modules, "kagglehub.clients", clients)
    monkeypatch.setitem(sys.modules, "kagglesdk.datasets.types.dataset_api_service", service)
    backend = KaggleBackend()
    monkeypatch.setattr(backend, "_tokens", lambda root: types.SimpleNamespace(files=["file-token"], directories=[]))
    backend.create("tester/new-private", tmp_path, "notes")
    assert len(calls) == 1 and calls[0].is_private is True
    assert calls[0].owner_slug == "tester" and calls[0].slug == "new-private"
    assert calls[0].files == ["file-token"]


def test_official_version_preserves_previous_versions(tmp_path, monkeypatch):
    calls = []
    api = types.SimpleNamespace(create_dataset_version=lambda request: calls.append(request))
    class Client:
        datasets = types.SimpleNamespace(dataset_api_client=api)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    clients = types.ModuleType("kagglehub.clients")
    clients.build_kaggle_client = Client
    service = types.ModuleType("kagglesdk.datasets.types.dataset_api_service")
    service.ApiCreateDatasetVersionRequest = type("VersionRequest", (), {})
    service.ApiCreateDatasetVersionRequestBody = type("VersionBody", (), {})
    monkeypatch.setitem(sys.modules, "kagglehub.clients", clients)
    monkeypatch.setitem(sys.modules, "kagglesdk.datasets.types.dataset_api_service", service)
    backend = KaggleBackend()
    monkeypatch.setattr(backend, "_tokens", lambda root: types.SimpleNamespace(files=["token"], directories=[]))
    backend.version("tester/private", tmp_path, "checked snapshot")
    assert calls[0].body.delete_old_versions is False
    assert calls[0].body.version_notes == "checked snapshot"


def test_probe_exercises_many_files_but_cannot_unlock_corpus_bootstrap(batch, tmp_path):
    from scripts.kaggle_publication_probe import prepare_probe
    index, _, _, _, _ = batch
    root = tmp_path/"probe"
    inventory = prepare_probe(root)
    assert len(inventory["files"]) == 60 and inventory == prepare_probe(root)
    backend = FakeKaggle(tmp_path/"remote")
    publish(root, backend, tmp_path/"probe-receipt.json")
    certificate = verify_fresh_session(json.loads((tmp_path/"probe-receipt.json").read_text()), session="fresh", backend=backend)
    with pytest.raises(ValueError, match="corpus/index"):
        validate_bootstrap(certificate, index, backend=backend)


def test_job_recovery_after_lost_vm_needs_no_local_payload_or_old_quotas(batch, tmp_path):
    from scripts.kaggle_acquire_publish import recover_published
    index, plan, root, _, _ = batch
    backend = FakeKaggle(tmp_path/"remote")
    publish(root, backend, tmp_path/"original.json")
    missing_job = tmp_path/"new-VM-job"
    with patch("urllib.request.urlopen", side_effect=AssertionError("No Zenodo transfer on recovery")):
        recovered = recover_published(index, "tester/geofirex-batch", missing_job/"receipt.json",
            years=[2018], minimum=1, maximum=1, session="new-VM", expected_plan=plan, backend=backend)
    assert not (missing_job/"payload").exists()
    assert recovered["status"] == "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
    assert recovered["kind"] == "remote_recovery" and backend.uploads == 1
