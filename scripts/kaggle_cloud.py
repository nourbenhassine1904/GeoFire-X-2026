"""Kaggle CPU acquisition/persistence CLI. Planning and mounted checks are offline.

Execute only in a Kaggle CPU notebook with Internet enabled. Quotas must be
verified on the account first; no default byte allowance is an authorization.
"""
import argparse
import json
from pathlib import Path

from geofirex.cloud_corpus import (acquire_batch, plan_batch, prepare_mounted,
                                    validate_index)
from geofirex.corpus import json_new
from geofirex.kaggle_persistence import (publish_package, session_id,
                                         snapshot_training, verify_fresh_session, adopt_manual, inspect_dataset)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_kaggle():
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run cloud transfers inside Kaggle; local Windows acquisition is disabled")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--index", required=True)
    plan.add_argument("--years", nargs="+", type=int, required=True)
    plan.add_argument("--max-events", type=int, required=True)
    plan.add_argument("--transfer-bytes", type=int, required=True)
    plan.add_argument("--storage-bytes", type=int, required=True)
    plan.add_argument("--private-free-bytes", type=int, required=True)
    plan.add_argument("--upload-bytes", type=int, required=True)
    plan.add_argument("--acquired-roots", nargs="*", default=[])
    plan.add_argument("--out", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--index", required=True)
    acquire.add_argument("--plan", required=True)
    acquire.add_argument("--destination", required=True)
    acquire.add_argument("--out", required=True)
    acquire.add_argument("--execute", action="store_true")
    publish = commands.add_parser("publish")
    publish.add_argument("--root", required=True)
    publish.add_argument("--handle", required=True)
    publish.add_argument("--upload-bytes", type=int, required=True)
    publish.add_argument("--private-free-bytes", type=int, required=True)
    publish.add_argument("--allow-version", action="store_true")
    publish.add_argument("--publication-mode", choices=["create", "resume", "version"], default="create")
    publish.add_argument("--out", required=True)
    publish.add_argument("--execute", action="store_true")
    verify = commands.add_parser("verify-session")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--out", required=True)
    adopt = commands.add_parser("adopt-manual")
    adopt.add_argument("--index", required=True)
    adopt.add_argument("--root", required=True)
    adopt.add_argument("--handle", required=True)
    adopt.add_argument("--out", required=True)
    inspect = commands.add_parser("inspect-dataset")
    inspect.add_argument("--handle", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--index", required=True)
    prepare.add_argument("--roots", nargs="+", required=True)
    prepare.add_argument("--out", required=True)
    snapshot = commands.add_parser("snapshot-training")
    snapshot.add_argument("--run", required=True)
    snapshot.add_argument("--tracking-db")
    snapshot.add_argument("--out", required=True)
    args = p.parse_args()
    if hasattr(args, "out") and Path(args.out).exists():
        p.error("Output already exists; choose a new path")
    if args.command == "plan":
        from geofirex.kaggle_persistence import verify_package
        index = read(args.index)
        validate_index(index, official=True)
        excluded = set()
        for root in args.acquired_roots:
            inventory = read(Path(root)/"inventory.json")
            verify_package(root, inventory)
            from geofirex.cloud_corpus import fingerprint
            if inventory.get("kind") != "corpus" or inventory["index_identity"] != fingerprint(index):
                raise ValueError("Acquired batch does not belong to this corpus")
            excluded.update(inventory["events"])
        result = plan_batch(index, years=args.years, max_events=args.max_events,
                            transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                            private_free_bytes=args.private_free_bytes, upload_bytes=args.upload_bytes,
                            exclude_events=excluded)
        json_new(args.out, result)
        print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))
    elif args.command == "acquire":
        plan = read(args.plan)
        from geofirex.cloud_corpus import validate_plan
        validate_plan(plan, read(args.index))
        if not args.execute:
            print(json.dumps({"status": "DRY_RUN", "events": plan["events"],
                              "observations": plan["observations"], "limits": plan["limits"]}, indent=2))
            return
        require_kaggle()
        acquire_batch(plan, read(args.index), args.destination, args.out)
    elif args.command == "publish":
        if not args.execute:
            from geofirex.kaggle_persistence import verify_package
            print(json.dumps(verify_package(args.root, read(Path(args.root)/"inventory.json")), indent=2))
            return
        require_kaggle()
        publish_package(args.root, args.handle, args.out, upload_bytes=args.upload_bytes,
                        private_free_bytes=args.private_free_bytes, session=session_id(),
                        allow_version=args.allow_version, mode=args.publication_mode)
    elif args.command == "adopt-manual":
        require_kaggle()
        json_new(args.out, adopt_manual(read(args.index), args.root, args.handle, session=session_id()))
    elif args.command == "inspect-dataset":
        print(json.dumps(inspect_dataset(args.handle), indent=2))
    elif args.command == "verify-session":
        require_kaggle()
        json_new(args.out, verify_fresh_session(read(args.receipt), session=session_id()))
    elif args.command == "prepare":
        print(json.dumps(prepare_mounted(read(args.index), args.roots, args.out), indent=2))
    else:
        snapshot_training(args.run, args.out, tracking_db=args.tracking_db)


if __name__ == "__main__":
    main()
