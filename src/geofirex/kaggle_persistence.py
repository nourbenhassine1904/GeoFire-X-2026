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
    """Explicit official SDK mutations; NEVER fall back from create to version."""
    def username(self):
        from kagglehub.auth import get_username
        username = get_username()
        if not username:
            raise PermissionError("Kaggle authenticated username could not be verified")
        return username

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
            if http_status(error) == 404:
                return None
            raise

    def _tokens(self, root):
        import shutil
        from kagglehub.gcs_upload import upload_files_and_directories, normalize_patterns, MAX_FILES_TO_UPLOAD
        from kagglehub.datasets import DEFAULT_IGNORE_PATTERNS
        from kagglesdk.blobs.types.blob_api_service import ApiBlobType
        files = [p for p in Path(root).rglob("*") if p.is_file() and not p.name.endswith(".part")]
        # Kagglehub may build an uncompressed temporary ZIP for large file counts.
        if len(files) > MAX_FILES_TO_UPLOAD:
            import tempfile
            if shutil.disk_usage(tempfile.gettempdir()).free < sum(p.stat().st_size for p in files)+1024**3:
                raise RuntimeError("Insufficient temporary ZIP space for official Kaggle uploader")
        return upload_files_and_directories(str(root), item_type=ApiBlobType.DATASET,
            ignore_patterns=normalize_patterns(default=DEFAULT_IGNORE_PATTERNS, additional=["*.part"])).to_proto()

    def create(self, handle, root, notes):
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest
        tokens = self._tokens(root)
        request = ApiCreateDatasetRequest()
        request.owner_slug, request.slug = handle.split("/")
        request.title = request.slug
        request.is_private = True
        request.files, request.directories = tokens.files, tokens.directories
        # No GetDataset prerequisite and no create-or-version helper.
        with build_kaggle_client() as client:
            client.datasets.dataset_api_client.create_dataset(request)

    def version(self, handle, root, notes):
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.datasets.types.dataset_api_service import (ApiCreateDatasetVersionRequest,
                                                                 ApiCreateDatasetVersionRequestBody)
        tokens = self._tokens(root)
        request = ApiCreateDatasetVersionRequest()
        request.owner_slug, request.dataset_slug = handle.split("/")
        request.body = ApiCreateDatasetVersionRequestBody()
        request.body.version_notes = notes
        request.body.delete_old_versions = False
        request.body.files, request.body.directories = tokens.files, tokens.directories
        with build_kaggle_client() as client:
            client.datasets.dataset_api_client.create_dataset_version(request)

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


def http_status(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None:
        status = getattr(error, "error_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def error_evidence(error):
    """Useful failure diagnostics without signed URLs or credential values."""
    message = re.sub(r"https?://[^\s]+", "[URL omitted]", str(error))
    message = re.sub(r"(?i)(Bearer\s+|(?:token|key|authorization)\s*[=:]\s*)[^\s,]+", r"\1[REDACTED]", message)
    return {"type": type(error).__name__, "http_status": http_status(error), "message": message[:800]}


def require_owner(backend, handle):
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]+", handle):
        raise ValueError("Expected Kaggle owner/dataset-slug handle")
    username = backend.username()
    if handle.split("/")[0].casefold() != username.casefold():
        raise PermissionError("Handle must belong to the authenticated Kaggle user")
    return username


def inspect_dataset(handle, *, backend=None):
    backend = backend or KaggleBackend()
    require_owner(backend, handle)
    try:
        meta = backend.metadata(handle)
    except Exception as error:
        if http_status(error) in {401, 403}:
            return {"handle": handle, "state": "AUTHORIZATION_DENIED_OR_EXISTENCE_UNDISCLOSED",
                    "http_status": http_status(error), "existence": "UNKNOWN"}
        raise
    return {"handle": handle, "state": "NOT_FOUND_404" if meta is None else
            ("EXISTING_PRIVATE" if meta.get("private") is True else "EXISTING_PUBLIC_OR_UNKNOWN"),
            "metadata": meta}


def _read_remote_inventory(backend, pinned):
    path = backend.mount(pinned, inventory_only=True)
    inventory = json.loads(path.read_text(encoding="utf-8"))
    check_seal(inventory)
    return inventory


def _safe_version(previous, current):
    if previous.get("kind") != current.get("kind"):
        raise ValueError("Cannot replace a dataset of another package kind")
    if current["kind"] == "corpus":
        files = {f["path"]: f for f in current["files"]}
        if any(files.get(f["path"]) != f for f in previous["files"]):
            raise ValueError("Corpus versions must retain every existing file unchanged")
        if previous.get("index_identity") != current.get("index_identity"):
            raise ValueError("Corpus version belongs to another source index")
    elif current["kind"] == "training":
        if not current.get("run_identity") or previous.get("run_identity") != current["run_identity"]:
            raise ValueError("Checkpoint version belongs to another training identity")
    else:
        raise ValueError("Unsupported package version kind")


def publish_package(root, handle, receipt_path, *, upload_bytes, private_free_bytes,
                    session, backend=None, allow_version=False, mode="create", attempts=12, delay=5):
    """Validate -> upload -> pin version -> verify ALL hashes from remote mount.

    An existing equal inventory is idempotent. Different content requires an
    explicit version request; a new batch should normally have its own handle.
    Receipt is written even on auth, processing, quota or readback errors.
    """
    if mode not in {"create", "resume", "version"}:
        raise ValueError("Publication mode must be create/resume/version")
    if allow_version:
        mode = "version"
    if Path(receipt_path).exists():
        raise FileExistsError(receipt_path)
    if attempts < 1 or delay < 0:
        raise ValueError("Invalid publication polling limits")
    root = Path(root).resolve()
    inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
    result = verify_package(root, inventory)
    total = result["bytes"] + (root/"inventory.json").stat().st_size
    if mode == "create" and min(upload_bytes, private_free_bytes) < total:
        raise ValueError("Validated upload/private storage quotas insufficient")
    backend = backend or KaggleBackend()
    receipt = {"schema_version": 2, "kind": "automatic_publication", "handle": handle,
               "inventory_identity": inventory["identity"], "mode": mode,
               "producer_session": session, "files": result["files"], "bytes": total,
               "status": "FAILED_FILES_PRESERVED", "upload_called": False}
    try:
        receipt["authenticated_owner"] = require_owner(backend, handle)
        previous = None
        version = None
        if mode != "create":
            previous = _private_metadata(backend, handle)
            if previous is None:
                raise ValueError("Dataset not found; use explicit create mode for a new private dataset")
            pinned = f"{handle}/versions/{previous['version']}"
            remote = _read_remote_inventory(backend, pinned)
            if remote["identity"] == inventory["identity"]:
                version = previous["version"]
            elif mode == "resume":
                raise ValueError("Existing dataset has different content; resume never overwrites")
            else:
                _safe_version(remote, inventory)
        if version is None:
            if min(upload_bytes, private_free_bytes) < total:
                raise ValueError("Validated upload/private storage quotas insufficient")
            receipt["upload_called"] = True
            try:
                operation = backend.create if mode == "create" else backend.version
                operation(handle, root, f"GeoFire-X inventory {inventory['identity']}")
                receipt['mutation_rpc_returned_success'] = True
            except Exception as error:
                # A conflict NEVER turns into a version mutation. Only read back
                # the already existing package and accept if it is identical.
                if mode != "create" or http_status(error) != 409:
                    raise
                receipt["creation_conflict"] = True
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
                    remote = _read_remote_inventory(backend, candidate)
                    if remote != inventory:
                        if receipt.get("creation_conflict"):
                            raise ValueError("Existing dataset has different content; create never overwrites")
                        raise RuntimeError("Published inventory not yet accessible or differs")
                    version = meta["version"]
                    break
                except Exception as error:
                    receipt.setdefault("readback_attempts", []).append(error_evidence(error))
                    if 'cannot be attached in non-interactive sessions' in str(error):
                        receipt['readback_blocker'] = 'KAGGLE_BATCH_DYNAMIC_ATTACHMENT_FORBIDDEN'
                        receipt['recovery'] = 'Attach this existing dataset at submission of a new private CPU job; do not upload again'
                        raise RuntimeError('Kaggle batch requires a new job with this dataset attached for readback') from error
                    if isinstance(error, ValueError):
                        raise
                    last_error = error
                    if attempt+1 < attempts:
                        time.sleep(delay)
            if version is None:
                raise RuntimeError("Publication/readback not verified; retain local data") from last_error
        receipt["versioned_handle"] = f"{handle}/versions/{version}"
        mounted = backend.mount(receipt["versioned_handle"])
        if mode != "resume" and (mounted.resolve() == root or mounted.resolve().is_relative_to(root)):
            raise ValueError("Readback points to producer files, not a remote dataset")
        actual = json.loads((mounted/"inventory.json").read_text(encoding="utf-8"))
        if actual != inventory:
            raise ValueError("Readback inventory differs")
        receipt["readback"] = verify_package(mounted, inventory)
        receipt["private_verified"] = True
        if not receipt["upload_called"] or receipt.get("creation_conflict"):
            receipt["kind"] = "remote_recovery"
            receipt["publication_provenance"] = "existing_publication_not_observed"
        else:
            receipt["publication_provenance"] = "automatic_mutation_observed"
        receipt["status"] = "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
        return receipt
    except Exception as error:
        receipt["error"] = error_evidence(error)
        receipt["http_status"] = http_status(error)
        receipt.setdefault("recovery", "Use resume mode to inspect a potentially successful publication; never infer 403 means absent")
        raise
    finally:
        json_new(receipt_path, seal(receipt))


def verify_fresh_session(receipt, *, session, backend=None):
    check_seal(receipt)
    manual = receipt.get("kind") == "manual_adoption"
    recovered = receipt.get("kind") == "remote_recovery" or receipt.get("upload_called") is False
    expected_status = "MANUAL_ADOPTED_FRESH_SESSION_PENDING" if manual else "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
    if receipt["status"] != expected_status:
        raise ValueError("Producer publication/readback was not verified")
    if session == receipt["producer_session"]:
        raise ValueError("Verification requires a new Kaggle notebook session")
    backend = backend or KaggleBackend()
    require_owner(backend, receipt["handle"])
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
    return seal({"schema_version": 2, "status": "PERSISTENCE_VERIFIED_NEW_SESSION", "session": session,
                 "producer_session": receipt["producer_session"], "private_verified": True,
                 "publication_provenance": "manual" if manual else ("recovered_existing" if recovered else "automatic"),
                 "inventory_identity": inventory["identity"], "index_identity": inventory.get("index_identity"),
                 "versioned_handle": receipt["versioned_handle"], "verification": result,
                 "producer_receipt_identity": receipt["identity"]})


def adopt_manual(index, root, handle, *, session, backend=None):
    """Read-only adoption. This is NEVER an automatic-upload receipt."""
    from geofirex.cloud_corpus import validate_index, verify_corpus_package
    validate_index(index, official=True)
    backend = backend or KaggleBackend()
    owner = require_owner(backend, handle)
    meta = _private_metadata(backend, handle)
    if not meta or meta["version"] < 1:
        raise ValueError("Manual dataset must exist, be private and have a published version")
    pinned = f"{handle}/versions/{meta['version']}"
    remote = _read_remote_inventory(backend, pinned)
    inventory = json.loads((Path(root)/"inventory.json").read_text(encoding="utf-8"))
    if remote != inventory:
        raise ValueError("Mounted manual inventory differs from the version-pinned dataset")
    verification = verify_corpus_package(index, root, inventory)
    # Mounting the already attached version does not redownload it.
    published_root = backend.mount(pinned)
    verify_corpus_package(index, published_root, remote)
    return seal({"schema_version": 2, "kind": "manual_adoption",
                 "publication_provenance": "manual_user_published", "upload_called": False,
                 "status": "MANUAL_ADOPTED_FRESH_SESSION_PENDING", "handle": handle,
                 "versioned_handle": pinned, "producer_session": session,
                 "authenticated_owner": owner, "private_verified": True,
                 "inventory_identity": inventory["identity"], "index_identity": inventory["index_identity"],
                 "events": inventory["events"], "files": verification["files"],
                 "verification": verification})


def validate_bootstrap(certificate, index, *, backend=None):
    """Recheck privacy, pinned contents and official completeness at use time."""
    from geofirex.cloud_corpus import verify_corpus_package
    check_seal(certificate)
    if certificate["status"] != "PERSISTENCE_VERIFIED_NEW_SESSION":
        raise ValueError("Bootstrap requires completed independent-session verification")
    if certificate.get("schema_version", 1) >= 2:
        if (certificate.get("publication_provenance") not in {"manual", "automatic", "recovered_existing"}
                or certificate.get("private_verified") is not True
                or certificate.get("session") == certificate.get("producer_session")):
            raise ValueError("Invalid bootstrap provenance/session/privacy evidence")
    pinned = certificate["versioned_handle"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]+/versions/[1-9][0-9]*", pinned):
        raise ValueError("Bootstrap must pin a dataset version")
    backend = backend or KaggleBackend()
    handle = pinned.split("/versions/")[0]
    require_owner(backend, handle)
    if _private_metadata(backend, handle) is None:
        raise ValueError("Bootstrap dataset no longer exists")
    root = backend.mount(pinned)
    inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
    result = verify_corpus_package(index, root, inventory)
    if certificate["verification"] != result:
        raise ValueError("Bootstrap inventory/count/hash evidence differs from current mounted bytes")
    return result


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
    identity_path = run/"identity.json"
    identity = json.loads(identity_path.read_text())["identity"] if identity_path.exists() else None
    inventory = seal({"schema_version": 2, "kind": "training", "epoch": state_epoch,
                      "run_identity": identity, "files": files})
    json_new(destination/"inventory.json", inventory)
    return inventory
