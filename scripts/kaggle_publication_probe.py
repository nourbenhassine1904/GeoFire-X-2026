"""Tiny real-API probe (>50 files exercises the same ZIP staging as large lots).

No wildfire data or predictions. This probe CANNOT be used as a corpus bootstrap.
Dry run unless --execute; no data transfer from Zenodo.
"""
import argparse
import json
from pathlib import Path

from geofirex.cloud_corpus import seal
from geofirex.corpus import json_new, sha256_file
from geofirex.kaggle_persistence import publish_package, session_id, verify_package


def prepare_probe(root):
    root = Path(root)
    if (root/"inventory.json").exists():
        inventory = json.loads((root/"inventory.json").read_text())
        verify_package(root, inventory)
        return inventory
    root.mkdir(parents=True, exist_ok=False)
    files = []
    for number in range(60):
        path = root/"probe"/f"file_{number:03d}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"GeoFire-X publication transport probe {number}\n", encoding="utf-8")
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    inventory = seal({"schema_version": 2, "kind": "publication_probe", "files": files,
                      "purpose": "Test explicit private creation, ZIP layout and hash readback; no scientific data"})
    json_new(root/"inventory.json", inventory)
    return inventory


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--handle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["create", "resume"], default="create")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        print("DRY_RUN: 60 tiny text files, private create + ZIP/readback; payload cap 1 MiB; no Zenodo transfer")
    else:
        from scripts.kaggle_cloud import require_kaggle
        from geofirex.kaggle_persistence import KaggleBackend, require_owner
        require_kaggle()
        require_owner(KaggleBackend(), args.handle)
        prepare_probe(Path(args.job)/"payload")
        publish_package(Path(args.job)/"payload", args.handle, args.out,
                        upload_bytes=1024**2, private_free_bytes=1024**2,
                        session=session_id(), mode=args.mode)
        print(Path(args.out).read_text())
