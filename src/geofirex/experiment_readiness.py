"""Offline contracts for proposed experiments, never a training/inference runner.

Only metadata and explicit synthetic fixtures are accepted. Nothing here opens a
raster, instantiates a model, changes a feature policy, or loads a checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import math
import re

from geofirex.feature_profiles import PROFILES
from geofirex.unet_data import Normalizer


def profile_channels(policy):
    """Schema only; these zero/one values are NOT fitted normalization stats."""
    profile = PROFILES[policy]
    k = len(profile["continuous"])
    return Normalizer(profile["continuous"], [0.] * k, [1.] * k,
                      [0] * k, [], profile["landcover"]).channel_names


def required_bands(policy):
    profile = PROFILES[policy]
    return set(profile["continuous"]) | {13, 22} | ({16} if profile["landcover"] else set())


def _utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("An explicit UTC timestamp is required")
    return result


def _event(event_id, day):
    if not re.fullmatch(r"(?:2018|2019|2020)/fire_[0-9]+", event_id):
        raise ValueError("Only historical 2018-2020 event metadata; 2021 locked")
    if str(date.fromisoformat(day).year) != event_id[:4]:
        raise ValueError("Event year and date differ")


def validate_availability(records, expected_keys):
    """Review *every* selected asset, not a self-declared band-level whitelist.

    A key is (event_id, as_of_J, input_date, zero_based_band). Evidence must be
    externally reviewed; this validates its structure/chronology, not its truth.
    Caller must derive expected_keys exhaustively from its immutable manifests.
    This function cannot activate any live feature policy.
    """
    expected = set(expected_keys)
    if not expected or len(expected) != len(expected_keys):
        raise ValueError("Nonempty unique exhaustive expected keys required")
    seen = set()
    for row in records:
        key = (row["event_id"], row["as_of"], row["input_date"], row["band"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate availability asset")
        seen.add(key)
        _event(row["event_id"], row["as_of"])
        _event(row["event_id"], row["input_date"])
        if type(row["band"]) is not int or not 0 <= row["band"] < 23:
            raise ValueError("Raw band index must be in [0,22]")
        if row["band"] in {7, 19}:
            raise ValueError("Wind direction requires a separately audited circular encoding/provenance")
        if row["input_date"] > row["as_of"]:
            raise ValueError("Future input date")
        decision = datetime.combine(date.fromisoformat(row["as_of"]) + timedelta(days=1),
                                    time.min, tzinfo=timezone.utc)
        if _utc(row["decision_at"]) != decision:
            raise ValueError("Decision must be the fixed end-of-day-J UTC cutoff")
        available = _utc(row["available_at"])
        start, end = _utc(row["observation_start"]), _utc(row["observation_end"])
        if start > end or available > decision:
            raise ValueError("Unavailable at J or reversed acquisition interval")
        if row["kind"] == "forecast":
            issued = _utc(row["issue_at"])
            if issued > available or issued > start:
                raise ValueError("Forecast issue/availability/valid-time chronology differs")
        elif row["kind"] in {"observed", "composite", "static"}:
            if end > decision or available < end:
                raise ValueError("Observation/composite not completed and published by J")
        else:
            raise ValueError("Explicit temporal product kind required")
        if (row.get("reviewed") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("evidence_sha256", ""))
                or row["evidence_sha256"] == "0" * 64
                or not all(isinstance(row.get(k), str) and row[k].strip()
                           for k in ("source_product", "product_version", "asset_id", "evidence_uri"))):
            raise ValueError("Reviewed versioned asset evidence is missing")
    if seen != expected:
        raise ValueError("Availability evidence does not cover all selected observations")
    return {"status": "CHRONOLOGY_VALIDATED", "assets": len(seen),
            "external_evidence_review_still_required": True, "policy_activated": False}


def temporal_view(sample, context):
    """Select T=1 or T=3 metadata without changing targets or the T=5 domain."""
    if context not in (1, 3):
        raise ValueError("This preparation compares T=1 and T=3 only")
    _event(sample["event_id"], sample["target_date"])
    expected_role = "val" if sample["event_id"].startswith("2020/") else "train"
    if sample["split"] != expected_role:
        raise ValueError("Annual split differs")
    target = date.fromisoformat(sample["target_date"])
    domain = [(target + timedelta(days=d)).isoformat() for d in (-5, -4, -3, -2, -1)]
    if sample["domain_dates"] != domain or sample["context_dates"] != domain[-3:]:
        raise ValueError("Consecutive canonical T=3 / T=5 dates required")
    if len(sample["context"]) != 3 or len(sample["domain"]) != 5:
        raise ValueError("Temporal references do not match the date contract")
    if sample["context"] != sample["domain"][-3:] or sample["target"] in sample["domain"]:
        raise ValueError("Context/target references differ or contain the future")
    result = deepcopy(sample)
    result["context_dates"] = domain[-context:]
    result["context"] = result["context"][-context:]
    return result


def validate_paired_metadata(t3, t1):
    if not t3 or len(t3) != len(t1):
        raise ValueError("Same nonempty sample count required")
    key = lambda r: (r["event_id"], r["target_date"], r["row"], r["col"])
    left, right = {key(r): r for r in t3}, {key(r): r for r in t1}
    if len(left) != len(t3) or len(right) != len(t1) or left.keys() != right.keys():
        raise ValueError("Target/tile keys differ or are duplicated")
    for k, row in left.items():
        if temporal_view(row, 1) != right[k]:
            raise ValueError("T=1 changed targets, pixels, source identity or maximum-context domain")
    return {"samples": len(left), "same_targets_and_domain_references": True,
            "pixel_reads": 0, "test_2021_pixels_opened": 0}


def validate_saved_counts(row, *, tolerance=1e-10):
    """Algebraic audit of saved counts; not another model evaluation."""
    c = [row[k] for k in ("tp", "fp", "fn", "tn")]
    if any(type(v) is not int or v < 0 for v in c):
        raise ValueError("Nonnegative integer confusion counts required")
    tp, fp, fn, tn = c
    values = {"valid_pixels": sum(c), "positive_pixels": tp + fn,
              "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
              "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
              "precision": tp / (tp + fp) if tp + fp else None,
              "recall": tp / (tp + fn) if tp + fn else None}
    for name, expected in values.items():
        if name not in row:
            continue
        actual = row[name]
        if expected is None:
            if actual is not None:
                raise ValueError("Undefined saved metric: " + name)
        elif actual is None or not math.isfinite(actual) or abs(expected - actual) > tolerance:
            raise ValueError("Saved count/metric mismatch: " + name)
    return values


def validate_campaign(plan):
    if plan["schema"] != "geofirex-scientific-improvements-preparation-v1":
        raise ValueError("Unknown preparation schema")
    if plan["training_authorized"] is not False or plan["training_executed"] is not False:
        raise ValueError("This file is preparation only")
    if plan["seed"] != 42 or plan["epochs"] != 4 or plan["batch_size"] != 16:
        raise ValueError("Historical paired budget differs")
    if plan["train_tiles"] != 93603 or plan["updates_per_new_run"] != 23404:
        raise ValueError("Same exhaustive training budget required")
    if plan["fold"] != {"id": 0, "train_years": [2018, 2019], "validation_years": [2020]}:
        raise ValueError("Annual train/validation fold differs; 2021 cannot enter training")
    reference = plan["historical_reference"]
    if reference["pos_weight"] != 10 or reference["threshold"] != .78:
        raise ValueError("Principal model reference differs")
    if plan["promotion_to_studio"] is not False:
        raise ValueError("Preparation cannot promote a Studio model")
    if plan["locked_test_years"] != [2021] or plan["selection_scope"] != "development32":
        raise ValueError("Only development32 may select model/loss/threshold")
    if plan["confirmation_scope"] != "observed169_descriptive_only":
        raise ValueError("The 169 panel cannot be an unobserved test")
    for experiment in plan["experiments"]:
        if experiment["status"] != "PREPARED_NOT_EXECUTABLE" or experiment["launch_enabled"] is not False:
            raise ValueError("No prepared experiment may enable execution")
        if not experiment["prerequisites"]:
            raise ValueError("Unresolved prerequisites must be explicit")
    return {"status": "PREPARATION_CONTRACT_VALID", "training_enabled": False}
