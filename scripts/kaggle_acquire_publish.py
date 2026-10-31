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
from geofirex.kaggle_persistence import (publish_package, session_id, verify_package)
from scripts.kaggle_cloud import require_kaggle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="data/manifests/real/corpus_zip_index_v1.json")
    p.add_argument("--job", required=True)
    p.add_argument("--handle", required=True)
    p.add_argument("--years", nargs="+", type=int, default=[2018])
    p.add_argument("--max-events", type=int, default=1)
    p.add_argument("--transfer-bytes", type=int, required=True)
    p.add_argument("--storage-bytes", type=int, required=True)
    p.add_argument("--upload-bytes", type=int, required=True)
    p.add_argument("--private-free-bytes", type=int, required=True)
    p.add_argument("--acquired-roots", nargs="*", default=[])
    p.add_argument("--bootstrap-verification")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    validate_index(index, official=True)
    if args.max_events > 1 and args.execute:
        if not args.bootstrap_verification:
            p.error("Verify the first private batch from a new notebook before multi-event expansion")
        certificate = json.loads(Path(args.bootstrap_verification).read_text(encoding="utf-8"))
        check_seal(certificate)
        if certificate["status"] != "PERSISTENCE_VERIFIED_NEW_SESSION":
            p.error("Invalid first-publication verification certificate")
    excluded = set()
    for root in args.acquired_roots:
        inventory = json.loads((Path(root)/"inventory.json").read_text(encoding="utf-8"))
        verify_package(root, inventory)
        if inventory.get("kind") != "corpus" or inventory["index_identity"] != fingerprint(index):
            p.error("Acquired dataset/index mismatch")
        excluded.update(inventory["events"])
    job = Path(args.job)
    plan_path = job/"plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_plan(plan, index)
        if plan["limits"] != dict(transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                                  private_free_bytes=args.private_free_bytes, upload_bytes=args.upload_bytes):
            p.error("Existing immutable job has different quotas; choose a new --job")
        if len(plan["events"]) > args.max_events or any(int(e[:4]) not in args.years for e in plan["events"]):
            p.error("Existing job differs from requested years/event limit")
    else:
        plan = plan_batch(index, years=args.years, max_events=args.max_events,
                          transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                          upload_bytes=args.upload_bytes, private_free_bytes=args.private_free_bytes,
                          exclude_events=excluded)
        if args.execute:
            require_kaggle()
            json_new(plan_path, plan)
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
                    private_free_bytes=args.private_free_bytes, session=session_id())
    # Print the small sealed receipt so it can be saved outside the VM/pasted into
    # a new notebook. No transient folder is claimed to be a permanent backup.
    print(receipt_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
