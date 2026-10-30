"""Export integrity and portability; no training or real dataset needed."""
import io
import json
import tarfile

import pytest

from geofirex.corpus import sha256_file
from geofirex.runner_data import digest
from scripts.verify_and_export_benchmark import main, verify_and_export, verify_archive, verify_run


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed(tmp_path):
    directory = tmp_path/"arbitrary-run"
    directory.mkdir()
    (directory/"checkpoint.pt").write_bytes(b"opaque checkpoint, never deserialized")
    put(directory/"config.json", {"mlflow_enabled": False, "data_root": "/private/home/example"})
    put(directory/"summary.json", {"status": "FINISHED", "run_id": "example", "model": "B1",
                                   "qualification": "diagnostic", "fold_id": 4, "context_days": 5,
                                   "seed": 71, "comparison_hash": "same", "best_epoch": 7,
                                   "checkpoint": "checkpoint.pt", "checkpoint_sha256": sha256_file(directory/"checkpoint.pt"),
                                   "threshold": .4, "validation_average_precision": .6})
    put(directory/"data_audit.json", {"windows": {"val": 2}})
    put(directory/"validation_0007.json", {"micro": {"average_precision": .6}, "threshold": .4,
                                          "per_window": [{}, {}], "new_ignitions": {"tp": 3}})
    put(directory/"preflight.json", {"passed": True})
    put(directory/"normalization.json", {"fit_paths": ["/private/home/train-only"]})
    source = directory/"source/src/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# archived source\n")
    hashes = {"src/model.py": sha256_file(source)}
    put(directory/"source_hashes.json", hashes)
    put(directory/"provenance.json", {"source_hash": digest(hashes)})
    (directory/"scratch.bin").write_bytes(b"sorting scratch")
    (directory/"partial.pt.tmp").write_bytes(b"partial")
    return directory


def test_generic_export_verifies_every_file_and_report_omits_private_paths(completed, tmp_path):
    export = tmp_path/"exports"
    report = verify_and_export([completed], export_dir=export)
    assert report["results"][0]["validation_windows"] == 2
    assert report["results"][0]["seed"] == 71
    assert "/private" not in json.dumps(report) and str(tmp_path) not in json.dumps(report)
    archive = export/(completed.name+".tar.gz")
    checked = verify_and_export([completed], archive_dir=export)
    assert report["results"] == checked["results"]
    with tarfile.open(archive) as bundle:
        assert not any(m.name.endswith((".bin", ".tmp")) for m in bundle)


@pytest.mark.parametrize("corruption", ["checksum", "member", "missing"])
def test_archive_rejects_corruption_even_with_new_outer_checksum(completed, tmp_path, corruption):
    export = tmp_path/"exports"
    verify_and_export([completed], export_dir=export)
    archive = export/(completed.name+".tar.gz")
    if corruption == "checksum":
        with archive.open("ab") as f:
            f.write(b"corrupt")
    else:
        with tarfile.open(archive) as bundle:
            members = [(m, bundle.extractfile(m).read()) for m in bundle]
        with tarfile.open(archive, "w:gz") as bundle:
            for member, content in members:
                if member.name.endswith("checkpoint.pt"):
                    if corruption == "missing":
                        continue
                    content = b"x"*len(content)
                bundle.addfile(member, io.BytesIO(content))
        put(archive.with_suffix(archive.suffix+".sha256.json"), {"sha256": sha256_file(archive)})
    with pytest.raises(ValueError, match="mismatch|missing"):
        verify_archive(completed, archive)


def test_export_and_report_collisions_preserve_existing_files(completed, tmp_path):
    export = tmp_path/"exports"
    verify_and_export([completed], export_dir=export)
    original = {p.name: sha256_file(p) for p in export.iterdir()}
    with pytest.raises(FileExistsError):
        verify_and_export([completed], export_dir=export)
    assert original == {p.name: sha256_file(p) for p in export.iterdir()}
    report = tmp_path/"report.json"
    put(report, {"preserve": True})
    with pytest.raises(FileExistsError):
        main(["--runs", str(completed), "--export-dir", str(tmp_path/"unused"), "--report", str(report)])
    assert json.loads(report.read_text()) == {"preserve": True}
    assert not (tmp_path/"unused").exists()


def test_changed_checkpoint_or_archived_source_is_rejected(completed):
    checkpoint = completed/"checkpoint.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint"):
        verify_run(completed)
    checkpoint.write_bytes(original)
    (completed/"source/src/model.py").write_text("modified source")
    with pytest.raises(ValueError, match="Archived source"):
        verify_run(completed)


def test_offline_mlflow_is_explicit_and_never_reported_verified(completed):
    put(completed/"config.json", {"mlflow_enabled": True, "tracking_uri": "sqlite:///missing-store.db"})
    with pytest.raises(ValueError, match="database missing"):
        verify_run(completed)
    assert verify_run(completed, skip_mlflow=True)["mlflow_status"] == "not_checked_offline"


def test_incomparable_runs_fail_before_export(completed, tmp_path):
    import shutil
    second = tmp_path/"different-budget"
    shutil.copytree(completed, second)
    summary = json.loads((second/"summary.json").read_text())
    put(second/"summary.json", {**summary, "comparison_hash": "different"})
    with pytest.raises(ValueError, match="Runs differ"):
        verify_and_export([completed, second], require_comparable=True, export_dir=tmp_path/"unused")
    assert not (tmp_path/"unused").exists()
