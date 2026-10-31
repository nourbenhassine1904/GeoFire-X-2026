"""Whole-event cloud batches, pinned inventories and bounded acquisition.

No downloads at import time. A plan is immutable; interrupted files are kept.
The archive-wide MD5 is provenance, not independently verified by Range reads.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from geofirex.corpus import (ARCHIVE_BYTES, ARCHIVE_MD5, BudgetRangeFile,
                            acquire_members, inventory_local, json_new,
                            safe_entry, sha256_file)
from geofirex.remote_archive import ARCHIVE_URL


def fingerprint(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seal(value):
    return {**value, "identity": fingerprint(value)}


def check_seal(value):
    if value.get("identity") != fingerprint({k: v for k, v in value.items() if k != "identity"}):
        raise ValueError("Manifest/plan identity mismatch")


def validate_index(index, *, official=False):
    entries = index["entries"]
    seen = set()
    for e in entries:
        event, date = safe_entry(e["path"])
        if (event, date, int(event[:4])) != (e["event_id"], e["date"], e["year"]):
            raise ValueError("Index path/event/date/year mismatch")
        if e["path"] in seen or min(e["compressed_bytes"], e["geotiff_bytes"]) < 1:
            raise ValueError("Duplicate or empty indexed file")
        seen.add(e["path"])
    if official and (len(entries) != 13607 or len({e["event_id"] for e in entries}) != 607
                     or index["archive_bytes"] != ARCHIVE_BYTES
                     or index["archive_md5"] != ARCHIVE_MD5 or index["source"] != ARCHIVE_URL):
        raise ValueError("Expected pinned official 607-event / 13607-observation index")


def plan_batch(index, *, years, max_events, transfer_bytes, storage_bytes,
               private_free_bytes, upload_bytes, exclude_events=()):
    """Select deterministic COMPLETE events. Quotas are caller-validated, not guessed.

    Transfer estimate includes directory overhead and 15% headroom. The hard
    Range cap also charges retries, so a pathological server can still stop early.
    """
    validate_index(index)
    limits = dict(transfer_bytes=transfer_bytes, storage_bytes=storage_bytes,
                  private_free_bytes=private_free_bytes, upload_bytes=upload_bytes)
    if max_events < 1 or any(not isinstance(v, int) or v < 1 for v in limits.values()):
        raise ValueError("Positive explicit event and byte quotas required")
    groups = defaultdict(list)
    for e in index["entries"]:
        if e["year"] in years and e["event_id"] not in set(exclude_events):
            groups[e["event_id"]].append(e)
    selected, compressed, extracted = [], 0, 0
    for event, entries in sorted(groups.items()):
        c = sum(e["compressed_bytes"] for e in entries)
        u = sum(e["geotiff_bytes"] for e in entries)
        if (math.ceil((compressed+c)*1.15) + 8*1024**2 > transfer_bytes
                or extracted+u+1024**3 > storage_bytes
                or extracted+u+1024**2 > min(private_free_bytes, upload_bytes)):
            continue  # an oversized event must never be truncated
        selected.extend(entries)
        compressed += c
        extracted += u
        if len({e["event_id"] for e in selected}) == max_events:
            break
    if not selected:
        raise ValueError("No complete event fits the declared quotas")
    selected.sort(key=lambda e: e["path"])
    return seal({"schema_version": 1, "kind": "corpus_plan", "index_identity": fingerprint(index),
                 "source": index["source"], "archive_bytes": index["archive_bytes"],
                 "archive_md5": index["archive_md5"], "entries": selected,
                 "events": sorted({e["event_id"] for e in selected}),
                 "observations": len(selected), "compressed_bytes": compressed,
                 "geotiff_bytes": extracted, "limits": limits,
                 "quota_validation": "caller must verify current Kaggle account/session quotas"})


def validate_plan(plan, index):
    check_seal(plan)
    if plan["index_identity"] != fingerprint(index):
        raise ValueError("Plan belongs to a different index")
    expected = [e for e in index["entries"] if e["event_id"] in plan["events"]]
    if sorted(expected, key=lambda e: e["path"]) != plan["entries"]:
        raise ValueError("Plan must contain every observation of each selected event")


def batch_inventory(plan, root):
    check_seal(plan)
    integrity = inventory_local({"entries": plan["entries"]}, root)
    if not integrity["complete"]:
        raise ValueError(f"Incomplete/corrupt batch: {integrity}")
    files = [{"path": e["path"], "bytes": e["geotiff_bytes"], "crc32": e["crc32"],
              "sha256": sha256_file(Path(root)/e["path"])} for e in plan["entries"]]
    return seal({"schema_version": 1, "kind": "corpus", "plan_identity": plan["identity"],
                 "index_identity": plan["index_identity"], "events": plan["events"],
                 "observations": len(files), "files": files,
                 "complete_batch": True, "complete_corpus": len(files) == 13607 and len(plan["events"]) == 607})


def acquire_batch(plan, index, destination, report_path, *, stream_factory=BudgetRangeFile):
    validate_plan(plan, index)
    validate_index(index, official=True)
    if Path(report_path).exists():
        raise FileExistsError(report_path)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    # The declared allocation includes existing files and abandoned .part files.
    occupied = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    missing = sum(e["geotiff_bytes"] for e in plan["entries"] if not (root/e["path"]).exists())
    required = missing + 1024**3
    if occupied+required > plan["limits"]["storage_bytes"] or shutil.disk_usage(root).free < required:
        raise RuntimeError("Storage quota/free disk insufficient; existing files preserved")
    stream = stream_factory(plan["limits"]["transfer_bytes"], url=plan["source"], size=plan["archive_bytes"])
    report = {"plan_identity": plan["identity"], "status": "FAILED_FILES_PRESERVED"}
    try:
        current = inventory_local({"entries": plan["entries"]}, root)
        if current["corrupt"]:
            raise ValueError("Corrupt existing files preserved; manual repair required")
        if current["missing"]:
            with zipfile.ZipFile(stream) as zf:
                report["acquisition"] = acquire_members(zf, plan["entries"], root)
        inventory = batch_inventory(plan, root)
        target = root/"inventory.json"
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != inventory:
                raise ValueError("Existing inventory differs; preserved")
        else:
            json_new(target, inventory)
        report.update(status="LOCAL_VALIDATED_NOT_BACKED_UP", inventory_identity=inventory["identity"])
        return inventory
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["transfer"] = {"body_bytes": stream.bytes_fetched,
                              "reserved_bytes_including_retries": stream.bytes_reserved,
                              "requests": stream.requests}
        report["integrity"] = inventory_local({"entries": plan["entries"]}, root)
        json_new(report_path, report)


def prepare_mounted(index, roots, destination):
    """Verify mounted batches; write portable manifest + FULL annual split.

    Reads raster headers only after byte/hash checks. Raster pixels are not
    duplicated. Unacquired events remain in the split and completeness report.
    """
    from geofirex.kaggle_persistence import verify_package
    from geofirex.benchmark import event_assignments
    from geofirex.unet_data import validate_layout
    import rasterio

    validate_index(index)
    expected = {e["path"]: e for e in index["entries"]}
    located, events = {}, set()
    for root in map(Path, roots):
        inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
        verify_package(root, inventory)
        if inventory["kind"] != "corpus" or inventory["index_identity"] != fingerprint(index):
            raise ValueError("Mounted corpus/index mismatch")
        if events.intersection(inventory["events"]):
            raise ValueError("Duplicate events across mounted batches")
        batch_paths = {f["path"] for f in inventory["files"]}
        required = {e["path"] for e in index["entries"] if e["event_id"] in inventory["events"]}
        if required != batch_paths:
            raise ValueError("Mounted batch contains incomplete events")
        events.update(inventory["events"])
        for f in inventory["files"]:
            e = expected[f["path"]]
            if (f["bytes"], f["crc32"]) != (e["geotiff_bytes"], e["crc32"]):
                raise ValueError("Mounted entry differs from official index")
            located[f["path"]] = root/f["path"]
    rows = []
    for name, path in sorted(located.items()):
        e = expected[name]
        with rasterio.open(path) as ds:
            validate_layout(ds)
            rows.append(dict(event_id=e["event_id"], date=e["date"], year=e["year"],
                             relative_path=name, height=ds.height, width=ds.width))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/"manifest.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "date", "year", "relative_path", "height", "width"])
        writer.writeheader()
        writer.writerows(rows)
    assignment = event_assignments(sorted({e["event_id"] for e in index["entries"]}), 0)
    with (destination/"split.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "split"])
        writer.writerows(sorted(assignment.items()))
    json_new(destination/"mounts.json", [str(Path(r).resolve()) for r in roots])
    counts = {role: {"events": sum(assignment[e] == role for e in events),
                     "observations": sum(assignment[e["event_id"]] == role for e in rows)}
              for role in ("train", "val", "test")}
    report = {"expected_events": len(assignment), "expected_observations": len(expected),
              "actual_events": len(events), "actual_observations": len(rows), "by_split": counts,
              "missing_events": sorted(set(assignment)-events), "complete_corpus": len(rows) == len(expected)}
    json_new(destination/"completeness.json", report)
    return report
