"""Private Kaggle publication with version-pinned readback and explicit receipts.

The real backend is optional and lazy. API success alone is never a backup.
There is deliberately no delete/cleanup operation in this module.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path, PurePosixPath

from geofirex.cloud_corpus import check_seal, seal
from geofirex.corpus import json_new, sha256_file, verify_member


def session_id():
    """A marker in the ephemeral VM, outside published payload directories."""
    root = Path(os.environ.get("GEOFIREX_SESSION_DIR", "/kaggle/working"))
    root.mkdir(parents=True, exist_ok=True)
    path = root/".geofirex-session-id"
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(uuid.uuid4().hex)
    except FileExistsError:
        pass
    return path.read_text(encoding="utf-8").strip()


def package_path(root, name):
    p = PurePosixPath(name)
    if (p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name
            or p.as_posix() != name or not p.parts):
        raise ValueError(f"Unsafe package path: {name}")
    root = Path(root).resolve()
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError("Package symlink escapes root")
    return target


def verify_package(root, inventory):
    check_seal(inventory)
    paths = [f["path"] for f in inventory["files"]]
    if not paths or len(paths) != len(set(paths)) or "inventory.json" in paths:
        raise ValueError("Empty or duplicate package inventory")
    for f in inventory["files"]:
        p = package_path(root, f["path"])
        if not p.is_file() or p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            raise ValueError(f"Package size/SHA256 mismatch: {f['path']}")
        if "crc32" in f and not verify_member(p, {"geotiff_bytes": f["bytes"], "crc32": f["crc32"]}):
            raise ValueError(f"Package CRC mismatch: {f['path']}")
    actual = {p.relative_to(root).as_posix() for p in Path(root).rglob("*")
              if p.is_file() and not p.name.endswith(".part")}
    if actual != set(paths) | {"inventory.json"}:
        raise ValueError("Unexpected/missing package files; refusing publication/readback")
    return {"files": len(paths), "bytes": sum(f["bytes"] for f in inventory["files"]),
            "identity": inventory["identity"]}


class KaggleBackend:
    """kagglehub uploader/mounter + official typed SDK for privacy/version metadata.

    Kagglehub creates new datasets private. Existing public datasets are refused
    BEFORE upload. Only an explicit HTTP 404 is treated as nonexistence.
    """
    def metadata(self, handle):
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        owner, slug = handle.split("/")
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = owner, slug
        try:
            with build_kaggle_client() as client:
                result = client.datasets.dataset_api_client.get_dataset(request)
            return {"private": result.is_private, "version": result.current_version_number}
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is None:
                status = getattr(error, "error_code", None)
            if status == 404:
                return None
            raise

    def upload(self, handle, root, notes):
        import kagglehub
        kagglehub.dataset_upload(handle, str(root), version_notes=notes, ignore_patterns=["*.part"])

    def mount(self, versioned_handle, *, inventory_only=False):
        import kagglehub
        from kagglehub.config import is_kaggle_cache_disabled
        from kagglesdk.kaggle_env import is_in_kaggle_notebook
        if not is_in_kaggle_notebook() or is_kaggle_cache_disabled():
            raise RuntimeError("Kaggle shared cache required; refusing a full download to temporary disk")
        # In a Kaggle notebook this attaches shared /kaggle/input resources.
        # No force_download: it can delete a cache and duplicate large payloads.
        mounted = Path(kagglehub.dataset_download(versioned_handle,
                       **({"path": "inventory.json"} if inventory_only else {})))
        if not mounted.resolve().is_relative_to(Path("/kaggle/input").resolve()):
            raise RuntimeError("Expected a shared /kaggle/input mount")
        return mounted


def _private_metadata(backend, handle):
    meta = backend.metadata(handle)
    if meta is not None and meta.get("private") is not True:
        raise ValueError("Dataset is public or privacy unknown; upload refused")
    return meta


def publish_package(root, handle, receipt_path, *, upload_bytes, private_free_bytes,
                    session, backend=None, allow_version=False, attempts=12, delay=5):
    """Validate -> upload -> pin version -> verify ALL hashes from remote mount.

    An existing equal inventory is idempotent. Different content requires an
    explicit version request; a new batch should normally have its own handle.
    Receipt is written even on auth, processing, quota or readback errors.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]+", handle):
        raise ValueError("Expected Kaggle owner/dataset-slug handle")
    if Path(receipt_path).exists():
        raise FileExistsError(receipt_path)
    if attempts < 1 or delay < 0:
        raise ValueError("Invalid publication polling limits")
    root = Path(root).resolve()
    inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
    result = verify_package(root, inventory)
    total = result["bytes"] + (root/"inventory.json").stat().st_size
    if min(upload_bytes, private_free_bytes) < total:
        raise ValueError("Validated upload/private storage quotas insufficient")
    backend = backend or KaggleBackend()
    receipt = {"schema_version": 1, "handle": handle, "inventory_identity": inventory["identity"],
               "producer_session": session, "files": result["files"], "bytes": total,
               "status": "FAILED_FILES_PRESERVED", "upload_called": False}
    try:
        previous = _private_metadata(backend, handle)
        version = None
        if previous:
            pinned = f"{handle}/versions/{previous['version']}"
            remote_file = backend.mount(pinned, inventory_only=True)
            remote = json.loads(remote_file.read_text(encoding="utf-8"))
            check_seal(remote)
            if remote["identity"] == inventory["identity"]:
                version = previous["version"]
            elif not allow_version:
                raise ValueError("Existing dataset has different content; use another handle or explicit --allow-version")
        if version is None:
            receipt["upload_called"] = True
            backend.upload(handle, root, f"GeoFire-X inventory {inventory['identity']}")
            # A successful upload may still be processing or exposing the old version.
            last_error = None
            for attempt in range(attempts):
                try:
                    meta = _private_metadata(backend, handle)
                    if not meta or meta["version"] < 1:
                        raise RuntimeError("Dataset not ready")
                    if previous and meta["version"] <= previous["version"]:
                        raise RuntimeError("New version still processing")
                    candidate = f"{handle}/versions/{meta['version']}"
                    remote_file = backend.mount(candidate, inventory_only=True)
                    remote = json.loads(remote_file.read_text(encoding="utf-8"))
                    if remote != inventory:
                        raise RuntimeError("Published inventory not yet accessible or differs")
                    version = meta["version"]
                    break
                except Exception as error:
                    last_error = error
                    if attempt+1 < attempts:
                        time.sleep(delay)
            if version is None:
                raise RuntimeError("Publication/readback not verified; retain local data") from last_error
        receipt["versioned_handle"] = f"{handle}/versions/{version}"
        mounted = backend.mount(receipt["versioned_handle"])
        if mounted.resolve() == root or mounted.resolve().is_relative_to(root):
            raise ValueError("Readback points to producer files, not a remote dataset")
        actual = json.loads((mounted/"inventory.json").read_text(encoding="utf-8"))
        if actual != inventory:
            raise ValueError("Readback inventory differs")
        receipt["readback"] = verify_package(mounted, inventory)
        receipt["status"] = "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
        return receipt
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        json_new(receipt_path, seal(receipt))


def verify_fresh_session(receipt, *, session, backend=None):
    check_seal(receipt)
    if receipt["status"] != "REMOTE_VERIFIED_FRESH_SESSION_PENDING":
        raise ValueError("Producer publication/readback was not verified")
    if session == receipt["producer_session"]:
        raise ValueError("Verification requires a new Kaggle notebook session")
    backend = backend or KaggleBackend()
    meta = _private_metadata(backend, receipt["handle"])
    if meta is None:
        raise ValueError("Published dataset no longer accessible")
    root = backend.mount(receipt["versioned_handle"])
    inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
    if inventory["identity"] != receipt["inventory_identity"]:
        raise ValueError("Fresh-session inventory identity differs")
    result = verify_package(root, inventory)
    if result["files"] != receipt["files"]:
        raise ValueError("Fresh-session file count differs")
    return seal({"status": "PERSISTENCE_VERIFIED_NEW_SESSION", "session": session,
                 "versioned_handle": receipt["versioned_handle"], "verification": result,
                 "producer_receipt_identity": receipt["identity"]})


def snapshot_training(run, destination, *, tracking_db=None):
    """Small resumable epoch package; excludes AP scratch and old checkpoints.

    Includes portable MLflow run data/metric histories and a consistent SQLite
    backup. Source artifacts remain intact. Call only between optimizer epochs.
    """
    import shutil
    run, destination = Path(run), Path(destination)
    if destination.resolve().is_relative_to(run.resolve()):
        raise ValueError("Training backup must be outside the run directory")
    checkpoints = sorted(run.glob("epoch_*.pt"))
    if not checkpoints:
        raise ValueError("No completed checkpoint to preserve")
    state_epoch = checkpoints[-1].stem
    scores = [(json.loads(p.read_text(encoding="utf-8"))["micro"]["average_precision"],
               -int(p.stem.split("_")[-1])) for p in run.glob("validation_*.json")]
    best_epoch = -max(scores)[1] if scores else int(state_epoch.split("_")[-1])
    keep = {checkpoints[-1].name, f"epoch_{best_epoch:04d}.pt"}
    destination.mkdir(parents=True, exist_ok=False)
    for p in run.rglob("*"):
        if not p.is_file() or p.suffix in {".bin", ".tmp"}:
            continue
        if p.suffix == ".pt" and p.name not in keep:
            continue
        target = destination/"run"/p.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    if tracking_db and Path(tracking_db).is_file():
        with sqlite3.connect(str(tracking_db)) as source, sqlite3.connect(str(destination/"tracking.db")) as target:
            source.backup(target)
    tracking_file = run/"tracking.json"
    if tracking_file.exists():
        info = json.loads(tracking_file.read_text(encoding="utf-8"))
        if info.get("run_id"):
            from mlflow.tracking import MlflowClient
            config = json.loads((run/"config.json").read_text(encoding="utf-8"))
            from geofirex.runner import tracking_uri
            client = MlflowClient(tracking_uri=tracking_uri(config))
            record = client.get_run(info["run_id"])
            json_new(destination/"mlflow_run.json", record.to_dictionary())
            json_new(destination/"mlflow_metric_history.json",
                     {key: [m.to_dictionary() for m in client.get_metric_history(info["run_id"], key)]
                      for key in record.data.metrics})
    files = [{"path": p.relative_to(destination).as_posix(), "bytes": p.stat().st_size,
              "sha256": sha256_file(p)} for p in sorted(destination.rglob("*")) if p.is_file()]
    inventory = seal({"schema_version": 1, "kind": "training", "epoch": state_epoch, "files": files})
    json_new(destination/"inventory.json", inventory)
    return inventory
