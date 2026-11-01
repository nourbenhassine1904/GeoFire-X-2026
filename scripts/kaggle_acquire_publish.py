"""One Kaggle CPU job: whole-event Range acquisition -> private dataset -> readback.

Dry run unless --execute. Reuse --job across interrupted sessions/files; mount
previous datasets with --acquired-roots to exclude already persisted events.
Multi-event expansion requires the first new-session verification certificate.
"""
import argparse
import json
import uuid
from pathlib import Path

from geofirex.cloud_corpus import (acquire_batch, check_seal, fingerprint,
                                    plan_batch, validate_index, validate_plan)
from geofirex.corpus import json_new
from geofirex.kaggle_persistence import (publish_package, session_id, validate_bootstrap)
from geofirex.cloud_corpus import verify_corpus_package
from scripts.kaggle_cloud import require_kaggle


def recover_published(index, handle, receipt_path, *, years, minimum, maximum,
                      session, expected_plan=None, backend=None):
    """Recover after VM loss, without a local payload or its original quotas."""
    from geofirex.kaggle_persistence import KaggleBackend, require_owner, _private_metadata
    backend = backend or KaggleBackend()
    require_owner(backend, handle)
    meta = _private_metadata(backend, handle)
    if not meta or meta["version"] < 1:
        raise ValueError("No published dataset to recover; explicit create mode required")
    root = backend.mount(f"{handle}/versions/{meta['version']}")
    inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
    verify_corpus_package(index, root, inventory)
    if (not minimum <= len(inventory["events"]) <= maximum
            or any(int(event[:4]) not in years for event in inventory["events"])):
        raise ValueError("Recovered dataset differs from requested event count/years")
    if expected_plan:
        from geofirex.cloud_corpus import validate_plan
        validate_plan(expected_plan, index)
        if set(expected_plan["events"]) != set(inventory["events"]):
            raise ValueError("Recovered dataset differs from requested plan events")
    return publish_package(root, handle, receipt_path, upload_bytes=0, private_free_bytes=0,
                           session=session, mode="resume", backend=backend)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="data/manifests/real/corpus_zip_index_v1.json")
    p.add_argument("--job", required=True)
    p.add_argument("--handle", required=True)
    p.add_argument("--years", nargs="+", type=int, default=[2018])
    p.add_argument("--max-events", type=int, default=1)
    p.add_argument("--min-events", type=int, default=1)
    p.add_argument("--plan", help="Reuse an immutable offline batch plan")
    p.add_argument("--transfer-bytes", type=int, required=True)
    p.add_argument("--storage-bytes", type=int, required=True)
    p.add_argument("--upload-bytes", type=int, required=True)
    p.add_argument("--private-free-bytes", type=int, required=True)
    p.add_argument("--acquired-roots", nargs="*", default=[])
    p.add_argument("--bootstrap-verification")
    p.add_argument("--publication-mode", choices=["create", "resume"], default="create")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    validate_index(index, official=True)
    if args.execute:
        require_kaggle()
        from geofirex.kaggle_persistence import KaggleBackend, require_owner
        require_owner(KaggleBackend(), args.handle)
    if args.max_events > 1 and args.execute:
        if not args.bootstrap_verification:
            p.error("Verify the first private batch from a new notebook before multi-event expansion")
        certificate = json.loads(Path(args.bootstrap_verification).read_text(encoding="utf-8"))
        validate_bootstrap(certificate, index)
    excluded = set()
    for root in args.acquired_roots:
        inventory = json.loads((Path(root)/"inventory.json").read_text(encoding="utf-8"))
        verify_corpus_package(index, root, inventory)
        excluded.update(inventory["events"])
    job = Path(args.job)
    plan_path = job/"plan.json"
    if args.execute and args.publication_mode == "resume":
        expected = Path(args.plan) if args.plan else plan_path
        expected_plan = json.loads(expected.read_text(encoding="utf-8")) if expected.exists() else None
        receipt_path = job/f"publication-{uuid.uuid4().hex[:12]}.json"
        recover_published(index, args.handle, receipt_path, years=args.years,
                          minimum=args.min_events, maximum=args.max_events, session=session_id(), expected_plan=expected_plan)
        print(receipt_path.read_text(encoding="utf-8"), flush=True)
        return
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_plan(plan, index)
        if plan["limits"] != dict(transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                                  private_free_bytes=args.private_free_bytes, upload_bytes=args.upload_bytes):
            p.error("Existing immutable job has different quotas; choose a new --job")
        if len(plan["events"]) > args.max_events or any(int(e[:4]) not in args.years for e in plan["events"]):
            p.error("Existing job differs from requested years/event limit")
    else:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else plan_batch(index, years=args.years, max_events=args.max_events,
                          transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                          upload_bytes=args.upload_bytes, private_free_bytes=args.private_free_bytes,
                          exclude_events=excluded)
        validate_plan(plan, index)
        if set(plan["events"]) & excluded:
            p.error("Plan includes an already acquired event; regenerate the plan with those mounts excluded")
        if plan["limits"] != dict(transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                                  private_free_bytes=args.private_free_bytes, upload_bytes=args.upload_bytes):
            p.error("Explicit plan and command quotas differ")
        if args.execute:
            require_kaggle()
            json_new(plan_path, plan)
    if not args.min_events <= len(plan["events"]) <= args.max_events or any(int(e[:4]) not in args.years for e in plan["events"]):
        p.error("Plan does not meet the requested event count/years")
    print(json.dumps({k: v for k, v in plan.items() if k != "entries"}, indent=2), flush=True)
    if not args.execute:
        print("DRY_RUN: no transfer/publication. Validate quotas and approve this plan before --execute.")
        return
    require_kaggle()
    attempt = uuid.uuid4().hex[:12]
    root = job/"payload"
    acquire_batch(plan, index, root, job/f"acquisition-{attempt}.json")
    receipt_path = job/f"publication-{attempt}.json"
    publish_package(root, args.handle, receipt_path, upload_bytes=args.upload_bytes,
                    private_free_bytes=args.private_free_bytes, session=session_id(), mode=args.publication_mode)
    # Print the small sealed receipt so it can be saved outside the VM/pasted into
    # a new notebook. No transient folder is claimed to be a permanent backup.
    print(receipt_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
