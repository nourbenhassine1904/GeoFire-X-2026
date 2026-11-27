"""Read saved JSON/CSV metadata only; prepare audit artifacts, never train.

No raster/NPZ/model/checkpoint loading, no dataset construction, no API/network,
no Spark, no Kaggle. Existing outputs are refused rather than overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from geofirex.experiment_readiness import (profile_channels, required_bands,
    validate_campaign, validate_saved_counts)
from geofirex.feature_profiles import PROFILES
from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.science_results import ScientificResults

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(rows):
    result = {"events": len(rows), "no_target_positive_events": sum(r["positive_pixels"] == 0 for r in rows),
              "zero_dice_events": sum(r["dice"] == 0 for r in rows)}
    for metric in ("dice", "iou", "precision", "recall", "average_precision", "brier_score"):
        values = sorted(r[metric] for r in rows if r.get(metric) is not None)
        # Linear empirical quantiles, same convention as numpy's default.
        def q(p):
            if not values: return None
            x = p * (len(values) - 1); lo = int(x); hi = min(lo + 1, len(values) - 1)
            return values[lo] + (x - lo) * (values[hi] - values[lo])
        result[metric] = dict(defined=len(values), mean=statistics.fmean(values) if values else None,
                              median=q(.5), q10=q(.1), q90=q(.9))
    return result


def feature_rows():
    rows = []
    for i, name in enumerate(RAW_BAND_NAMES):
        if i in (12, 13, 14):
            source, interval, temporal = "SRTMGL1_003", "static terrain", "existing historical baseline; static source predates2018"
        elif i == 22:
            source, interval, temporal = "VIIRS active fire", "acquisition day J", "existing historical conditional baseline; operational delivery latency not proven"
        elif i < 3:
            source, interval, temporal = "VIIRS VNP09GA", "daily observations", "BLOCKED: asset acquisition and processing/publication times absent"
        elif i in (3, 4):
            source, interval, temporal = "VIIRS VNP13A1", "16-day composite selected by timestamp", "BLOCKED: composite period end and delivery byJ not proven; last is not causal evidence"
        elif i in (5, 6, 7, 8, 9, 10, 11):
            source, interval, temporal = "GRIDMET", "daily consolidated meteorological field", "BLOCKED: early/provisional/permanent revisions; archive vintage/publication byJ unknown"
        elif i == 15:
            source, interval, temporal = "GRIDMET/DROUGHT", "multi-day PDSI validity interval", "BLOCKED: interval coveringJ is not proof of release byJ"
        elif i == 16:
            source, interval, temporal = "MODIS MCD12Q1", "annual land cover for same calendar year", "BLOCKED: whole-year consolidated product; use only a demonstrably prior released vintage"
        else:
            source, interval, temporal = "NOAA GFS0P25", "00UTC cycle, forecast hours1..24 of date on file", "BLOCKED: local TIFF has no issue/valid/release metadata; not automatically a J+1 forecast"
        if i in (7, 19):
            temporal += "; direction excluded; dedicated circular encoding and provenance review required"
        if i == 17:
            temporal += "; sum/last semantics changed2019-11-07 upstream"
        current = {12: "slope_z + slope_missing", 14: "elevation_z + elevation_missing",
                   13: "aspect_sin + aspect_cos + aspect_missing", 22: "active_fire / fire_J"}.get(i, "not served")
        treatment = ("finite train-only z-score, clip+-8; missing -> mean (z0) + missing flag"
                     if i not in (7, 13, 16, 19, 22) else
                     "sin/cos; invalid angle ->0,0 + missing flag" if i == 13 else
                     "17 one-hot classes + unknown; fractional values stay unknown" if i == 16 else
                     "NaN=no detection; finite VIIRS code ->1; not fire intensity" if i == 22 else
                     "no implemented/approved representation in active profiles")
        rows.append(dict(index0=i, band1=i + 1, raw_name=name, source=source,
            observation_period=interval, availability_by_J=temporal, current_output=current,
            missing_and_encoding=treatment,
            candidate_profiles=";".join(p for p in PROFILES if i in required_bands(p))))
    return rows


def build(out):
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / "reports"):
        raise ValueError("Audit outputs must be in a new reports subdirectory")
    names = ["saved_metrics_audit.json", "per_event_saved.csv", "feature_availability_matrix.csv",
             "readiness.json", "sources.json", "gpu_budget_scenarios.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Audit outputs already exist; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    plan = read("configs/scientific_improvements_v1.json")
    validate_campaign(plan)
    # The reader verifies existing saved result locks; it never evaluates a model.
    store = ScientificResults()
    folder = "reports/unet_confirmation_20261112_v1/confirmation_cpu/"
    comparison = read(folder + "comparison.json")["rows"]
    evaluation = read(folder + "B2_w10_holdout169/evaluation.json")
    new_fire = read(folder + "new_fire.json")
    ablation = read("reports/unet_confirmation_20261112_v1/ablation_verified.json")
    principal = read("configs/prediction_lab_v1.json")
    for field in ("checkpoint_sha256", "index_sha256", "normalization_sha256"):
        assert plan["historical_reference"][field] == principal[field], field
    assert plan["historical_reference"]["initial_state_sha256"] == ablation["initial_state_sha256"]
    per_event = {m: [r for r in store.per_event["holdout169", m] if r["threshold"] == .95]
                 for m in ("B1", "B2")}
    per_event["B2_w10"] = evaluation["per_event"]
    reference_domain = {r["event_id"]: (r["valid_pixels"], r["positive_pixels"], r["tiles"], r["invalid_pixels"])
                        for r in per_event["B2_w10"]}
    metrics = []
    for full in comparison:
        validate_saved_counts(full)
        rows = per_event[full["model"]]
        assert len(rows) == 169
        for row in rows:
            validate_saved_counts(row)
            assert (row["valid_pixels"], row["positive_pixels"], row["tiles"], row["invalid_pixels"]) == reference_domain[row["event_id"]]
        for k in ("tp", "fp", "fn", "tn", "valid_pixels", "positive_pixels"):
            assert sum(r[k] for r in rows) == full[k]
        metrics.append(dict(**full, predicted_to_observed_pixel_count_ratio=(full["tp"] + full["fp"]) / full["positive_pixels"],
            false_discovery_fraction=full["fp"] / (full["tp"] + full["fp"]),
            false_positive_rate=full["fp"] / (full["fp"] + full["tn"])))
    for name in ("new_fire_candidates", "observed_fire_candidates"):
        validate_saved_counts(new_fire[name])
    full = comparison[-1]
    assert all(sum(new_fire[name][k] for name in ("new_fire_candidates", "observed_fire_candidates")) == full[k]
               for k in ("tp", "fp", "fn", "tn"))
    paired = {}
    selected = {r["event_id"]: r for r in per_event["B2_w10"]}
    for model in ("B1", "B2"):
        pairs = [(r, selected[r["event_id"]]) for r in per_event[model]]
        paired[model] = {metric: dict(
            better=sum(b[metric] > a[metric] for a, b in pairs if a[metric] is not None and b[metric] is not None),
            worse=sum(b[metric] < a[metric] for a, b in pairs if a[metric] is not None and b[metric] is not None),
            tied=sum(b[metric] == a[metric] for a, b in pairs if a[metric] is not None and b[metric] is not None))
            for metric in ("dice", "average_precision")}
    q = full["positive_pixels"] / full["valid_pixels"]
    p = store.train["prevalence"]
    top_fn = sorted(per_event["B2_w10"], key=lambda r: r["fn"], reverse=True)[:10]
    historical_new = [r for r in store.diagnosis["rows"] if r["phase"] == "holdout169"]
    saved = dict(status="SAVED_OUTPUTS_ALGEBRAIC_AUDIT_ONLY", new_inference=False,
        pooled=metrics, macro_event={m: summarize(rows) for m, rows in per_event.items()},
        paired_event_counts=paired, new_fire_partition=new_fire, historical_new_fire=historical_new,
        train_prevalence=p, panel_prevalence=q,
        brier_constant_train_prevalence=q * (1 - p)**2 + (1 - q) * p**2,
        brier_always_zero=q, new_fire_predicted_to_observed_pixel_count_ratio=new_fire["new_fire_candidates"]["predicted_positive_pixels"] / new_fire["new_fire_candidates"]["positive_pixels"],
        top10_FN_fraction=sum(r["fn"] for r in top_fn) / full["fn"],
        top10_FN_events=[{k: r[k] for k in ("event_id", "fn", "fp", "tp", "dice", "positive_pixels")} for r in top_fn],
        ablation32=[{k: r[k] for k in ("pos_weight", "threshold", "metrics")} for r in ablation["variants"]],
        aggregation_warning="micro pools pixel-days and weights large/long events; macro averages saved per-event metrics with defined counts; macro AP is not pooled AP; no physical area inference")
    compact = read("reports/mission_20261111/compact_full/report.json")
    normalization = read("reports/gpu_acceleration_20261111_v1/index/normalization.json")
    index = read("reports/gpu_acceleration_20261111_v1/index/index.json")
    publication = read("reports/gpu_acceleration_20261111_v1/publication/plan.json")
    assert len(publication["parts"]) == 11
    assert normalization["continuous"] == [12, 14] and not normalization["include_landcover"]
    fit_paths = normalization["fit_paths"]
    assert len(set(fit_paths)) == len(fit_paths)
    assert all("/2018/" in p.replace("\\", "/") or "/2019/" in p.replace("\\", "/") for p in fit_paths)
    readiness = dict(status="PREPARED_NOT_EXECUTABLE", campaign=validate_campaign(plan),
        current_compact=dict(events=compact["events"], train_events=compact["train_events"], val_events=compact["val_events"], roles=index["roles"], partitions=len(publication["parts"])),
        current_normalizer=dict(continuous=normalization["continuous"], mean=normalization["mean"], std=normalization["std"],
            clip=normalization["clip"], fit_paths=len(fit_paths), train_only=True,
            sha256=sha(ROOT / "reports/gpu_acceleration_20261111_v1/index/normalization.json")),
        profile_schemas={p:dict(channels=len(profile_channels(p)), channel_names=profile_channels(p), raw_bands=sorted(required_bands(p)),
            status="EXISTING_HISTORICAL_CONDITIONAL_BASELINE" if p == "conservative_fire_terrain" else "BLOCKED_NO_ASSET_AVAILABILITY_EVIDENCE") for p in PROFILES},
        new_modalities_available_in_current_compact=False, raw_pixels_opened=0, test_2021_pixels_opened=0,
        focal_trainer_implemented=False, experiment_launchers_created=False, studio_model_changed=False)
    training = read("artifacts/propagation-comparison/comparison.json")["training"]
    confirmation = read(folder + "status_complete.json")
    assert training["updates"] == 23404 and training["epochs"] == 4
    seconds = training["seconds"]
    budget = dict(measured_reference=dict(**training, source="artifacts/propagation-comparison/comparison.json",
        qualification="related dual-head U-Net, not a timing of focal/nativeT1/ResNet18/UTAE", updates_per_second=23404 / seconds,
        tile_presentations_per_second=4 * 93603 / seconds),
        scenario_only=dict(A_one_new_run_minutes=[.75 * seconds / 60, 1.5 * seconds / 60], B_one_new_run_minutes=[.75 * seconds / 60, 1.5 * seconds / 60],
            A_plus_B_new_GPU_hours=[2 * .75 * seconds / 3600, 2 * 1.5 * seconds / 3600],
            multipliers=[.75, 1.5], confidence="planning range chosen around one related measured run, not a statistical interval or guaranteed bound",
            excludes=["GPU environment setup", "source/data integrity checks", "selection32 exact AP/scratch", "backup", "additional seeds or mechanism controls"]),
        CPU_confirmation169_seconds=confirmation["elapsed_seconds"],
        C_D_E_GPU_cost="not measured; requires separate future authorization and architecture/loader qualification",
        rejected_extrapolations=["971 tiles/s from four warm-cache raw batches", "5400s wall limit mistaken for measured runtime", "linear GPU time from parameter count"],
        gpu_jobs_started=0)
    inputs = ["configs/prediction_lab_v1.json", "configs/propagation_comparison_v1.json",
        "reports/gpu_acceleration_20261111_v1/index/index.json", "reports/gpu_acceleration_20261111_v1/index/normalization.json",
        "reports/gpu_acceleration_20261111_v1/publication/plan.json", "reports/mission_20261111/compact_full/report.json",
        "reports/unet_confirmation_20261112_v1/ablation_verified.json", folder + "comparison.json", folder + "new_fire.json",
        folder + "B2_w10_holdout169/evaluation.json", "artifacts/propagation-comparison/comparison.json",
        "reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip"]
    sources = dict(local_inputs={p: sha(ROOT / p) for p in inputs}, read_method="JSON/CSV/ZIP JSON members only; no raster/NPZ/checkpoint deserialization",
        external_consulted="2026-11-13", external_sources=[
            "https://github.com/SebastianGer/WildfireSpreadTS",
            "https://github.com/SebastianGer/WildfireSpreadTSCreateDataset/blob/master/DataPreparation/satellites/FirePred.py",
            "https://github.com/SebastianGer/WildfireSpreadTSCreateDataset/blob/master/DataPreparation/DatasetPrepareService.py",
            "https://lpdaac.usgs.gov/documents/1372/VNP13_User_Guide_ATBD_V2.1.2.pdf",
            "https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET",
            "https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT",
            "https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1",
            "https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25",
            "https://arxiv.org/html/2502.12003v3",
            "https://github.com/slahrichi/WildfireSpreadTS",
            "https://huggingface.co/saadlahrichi/WSTSPlus",
            "https://arxiv.org/html/2609.22191v2",
            "https://arxiv.org/abs/1708.02002"],
        warning="Current upstream code is not proof of the exact generation version or historical publication latency of the 2023 archive; public weights were not downloaded")
    for name, data in [(names[0], saved), (names[3], readiness), (names[4], sources), (names[5], budget)]:
        with (out / name).open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False); f.write("\n")
    with (out / names[2]).open("x", encoding="utf-8", newline="") as f:
        rows = feature_rows(); writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fields = ["model", "event_id", "threshold", "valid_pixels", "positive_pixels", "tp", "fp", "fn", "tn",
              "dice", "iou", "precision", "recall", "average_precision", "brier_score"]
    with (out / names[1]).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for model, rows in per_event.items():
            writer.writerows({k: (model if k == "model" else r.get(k)) for k in fields} for r in rows)
    print(json.dumps(dict(status="PREPARED_NOT_EXECUTABLE", saved_events=len(selected),
        raw_bands=23, profile_channels={p: len(profile_channels(p)) for p in PROFILES},
        GPU_reference_seconds=seconds, model_predictions_computed=0, training_runs=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/scientific_improvements_v1")
    build(ROOT / parser.parse_args().out)
