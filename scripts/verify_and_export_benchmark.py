"""Verify completed benchmark runs and export/check portable archives.

No raster reads, training, checkpoint deserialization or MLflow mutations.
Paths passed to this CLI are relative to the caller's working directory.
The public JSON report contains run basenames, metrics and hashes, never configs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from geofirex.corpus import sha256_file
from geofirex.runner import export_run, tracking_uri
from geofirex.runner_data import digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_file(directory, relative):
    relative = Path(relative)
    resolved = (directory/relative).resolve()
    if relative.is_absolute() or directory not in resolved.parents:
        raise ValueError("Run metadata references a file outside its directory")
    return resolved


def verify_run(directory, *, skip_mlflow=False, tracking_uri_override=None):
    directory = Path(directory).resolve()
    summary = read_json(directory/"summary.json")
    config = read_json(directory/"config.json")
    audit = read_json(directory/"data_audit.json")
    validation = read_json(directory/f"validation_{summary['best_epoch']:04d}.json")
    checkpoint = run_file(directory, summary["checkpoint"])
    if summary["status"] != "FINISHED":
        raise ValueError("Only completed training/validation runs can be verified")
    if sha256_file(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    if (validation["micro"]["average_precision"] != summary["validation_average_precision"]
            or validation["threshold"] != summary["threshold"]):
        raise ValueError("Summary differs from selected validation metrics")
    if len(validation["per_window"]) != audit["windows"]["val"]:
        raise ValueError("Validation window count differs from audited targets")
    model = summary["model"]
    if model == "B0":
        if summary["parameters"] != 0 or summary["optimizer_updates"] != 0 or validation["new_ignitions"]["tp"] != 0:
            raise ValueError("Persistence must be untrained and cannot predict new ignitions")
    elif model in {"B1", "B2", "B3"}:
        if (directory/"preflight.json").is_file():
            if not read_json(directory/"preflight.json")["passed"]:
                raise ValueError("Run preflight did not pass")
        elif not (directory/"resume.json").is_file():
            raise ValueError("Trainable run has neither preflight nor resume provenance")
    else:
        raise ValueError("Unsupported benchmark model")
    sources = read_json(directory/"source_hashes.json")
    for relative, expected in sources.items():
        if sha256_file(run_file(directory/"source", relative)) != expected:
            raise ValueError("Archived source SHA256 mismatch")
    provenance = read_json(directory/"provenance.json")
    if provenance["source_hash"] != digest(sources):
        raise ValueError("Source inventory differs from provenance")
    mlflow_status = "disabled"
    if config["mlflow_enabled"]:
        mlflow_status = "not_checked_offline" if skip_mlflow else "FINISHED"
        if not skip_mlflow:
            from mlflow.tracking import MlflowClient
            uri = tracking_uri_override or tracking_uri(config)
            if uri.startswith("sqlite:///") and not Path(unquote(uri[len("sqlite:///"):].split("?")[0])).is_file():
                raise ValueError("MLflow database missing; provide --tracking-uri or explicitly --skip-mlflow")
            if MlflowClient(tracking_uri=uri).get_run(summary["run_id"]).info.status != "FINISHED":
                raise ValueError("MLflow run is not FINISHED")
    return {"run": directory.name, "run_id": summary["run_id"], "model": model,
            "qualification": summary["qualification"], "fold": summary["fold_id"],
            "context": summary["context_days"], "seed": summary["seed"],
            "comparison_hash": summary["comparison_hash"], "mlflow_status": mlflow_status,
            "validation_windows": len(validation["per_window"]), "micro": validation["micro"],
            "new_ignitions": validation["new_ignitions"], "threshold": summary["threshold"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "normalization_sha256": sha256_file(directory/"normalization.json"),
            "source_inventory_sha256": sha256_file(directory/"source_hashes.json")}


def verify_archive(directory, archive):
    """Hash every archived file against its run, streaming without extraction."""
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    expected_sha = read_json(archive.with_suffix(archive.suffix+".sha256.json"))["sha256"]
    actual_sha = sha256_file(archive)
    if actual_sha != expected_sha:
        raise ValueError("Archive SHA256 mismatch")
    expected = {f"{directory.name}/{p.relative_to(directory).as_posix()}": p
                for p in directory.rglob("*") if p.is_file() and p.suffix not in {".bin", ".tmp"}}
    seen = set()
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected, duplicate or nonregular archive member")
            source = expected[member.name]
            if member.size != source.stat().st_size:
                raise ValueError("Archive member size mismatch")
            checksum = hashlib.sha256()
            with bundle.extractfile(member) as content:
                for block in iter(lambda: content.read(1024*1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != sha256_file(source):
                raise ValueError("Archive member content mismatch")
            seen.add(member.name)
    if seen != expected.keys():
        raise ValueError("Archive is missing run files")
    return {"archive": archive.name, "sha256": actual_sha, "verified_files": len(seen)}


def verify_and_export(runs, *, export_dir=None, archive_dir=None, skip_mlflow=False,
                      tracking_uri_override=None, require_comparable=False):
    directories = [Path(p).resolve() for p in runs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Provide distinct run directories")
    if export_dir is not None and archive_dir is not None:
        raise ValueError("Choose export creation or existing archive verification")
    results = [verify_run(p, skip_mlflow=skip_mlflow, tracking_uri_override=tracking_uri_override) for p in directories]
    if require_comparable and len({r["comparison_hash"] for r in results}) != 1:
        raise ValueError("Runs differ in protocol, budget, data, normalization or sources")
    archive_root = Path(export_dir if export_dir is not None else archive_dir) if export_dir is not None or archive_dir is not None else None
    if archive_root is not None:
        destinations = [archive_root/(p.name+".tar.gz") for p in directories]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Run basenames collide in the export directory")
        # Refuse collisions for the entire batch before creating any archive.
        if export_dir is not None:
            for destination in destinations:
                if destination.exists() or destination.with_suffix(destination.suffix+".sha256.json").exists():
                    raise FileExistsError("Export or SHA256 sidecar already exists; choose a new directory")
        for directory, destination, result in zip(directories, destinations, results):
            if export_dir is not None:
                export_run(directory, destination)
            result["export"] = verify_archive(directory, destination)
    return {"schema_version": 1, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_required": require_comparable, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-dir", help="Create new archives here; refuse overwrites")
    mode.add_argument("--archive-dir", help="Verify existing <run-name>.tar.gz archives and their SHA256 sidecars")
    parser.add_argument("--skip-mlflow", action="store_true", help="Explicit offline mode, recorded as unchecked")
    parser.add_argument("--tracking-uri", help="Override the original store after relocation; never included in the report")
    parser.add_argument("--require-comparable", action="store_true")
    parser.add_argument("--report", help="New public JSON report (no configs or absolute paths)")
    args = parser.parse_args(argv)
    if args.report and Path(args.report).exists():
        raise FileExistsError("Report already exists; choose a new filename")
    report = verify_and_export(args.runs, export_dir=args.export_dir, archive_dir=args.archive_dir,
                               skip_mlflow=args.skip_mlflow, tracking_uri_override=args.tracking_uri,
                               require_comparable=args.require_comparable)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
