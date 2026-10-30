"""Dry run by default. --execute requires a caller-approved explicit byte budget.

This CLI never means that the chat user has granted approval: obtain their
agreement to the generated plan before invoking --execute on substantial data.
"""
import argparse
import json
import zipfile
from pathlib import Path

from geofirex.corpus import (BudgetRangeFile, acquire_members, inventory_local,
                            json_new, sha256_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--events", nargs="+")
    p.add_argument("--year", type=int)
    p.add_argument("--max-files", type=int, default=10)
    p.add_argument("--max-transfer-bytes", type=int, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.max_files < 1 or args.max_transfer_bytes < 1:
        p.error("Positive file and byte budgets required")
    if Path(args.out).exists():
        p.error("Output report already exists; choose a new path before any acquisition")
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    entries = [e for e in index["entries"] if (not args.events or e["event_id"] in args.events)
               and (args.year is None or e["year"] == args.year)]
    if not entries:
        p.error("No entries selected")
    selected = {**index, "entries": entries}
    report = {"execute": args.execute, "selected_files": len(entries),
              "compressed_payload_bytes": sum(e["compressed_bytes"] for e in entries),
              "geotiff_bytes": sum(e["geotiff_bytes"] for e in entries),
              "index_sha256": sha256_file(args.index),
              "max_session_transfer_bytes": args.max_transfer_bytes, "max_files": args.max_files}
    if args.verify_only:
        report["integrity"] = inventory_local(selected, args.destination)
    elif args.execute:
        Path(args.destination).mkdir(parents=True, exist_ok=True)
        stream = BudgetRangeFile(args.max_transfer_bytes, size=index["archive_bytes"])
        try:
            with zipfile.ZipFile(stream) as zf:
                report["acquisition"] = acquire_members(zf, entries, args.destination, max_files=args.max_files)
            report["status"] = "COMPLETED_WITHIN_SESSION_LIMITS"
        except Exception as error:
            report["status"] = "FAILED_PARTIAL_FILES_PRESERVED"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["transfer"] = {"successful_body_bytes": stream.bytes_fetched,
                                  "reserved_bytes_including_retries": stream.bytes_reserved,
                                  "requests": stream.requests}
            if report["status"] == "FAILED_PARTIAL_FILES_PRESERVED":
                json_new(args.out, report)
    json_new(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "integrity"}, indent=2))
