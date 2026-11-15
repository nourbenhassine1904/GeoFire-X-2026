"""Tiny synthetic/metadata checks only: no optimizer, raster or training call."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from geofirex.experiment_readiness import (
    profile_channels, required_bands, temporal_view, validate_availability,
    validate_campaign, validate_paired_metadata, validate_saved_counts,
)
from geofirex.feature_profiles import feature_profile
from geofirex.fast_compact import DTYPE, FastCompactTiles
from geofirex.unet_data import Normalizer, fire_mask

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("policy,count", [("conservative_fire_terrain", 8),
    ("terrain_vegetation_v1", 36), ("multimodal_observed_j_v1", 50)])
def test_raw_bands_are_not_encoded_channel_counts(policy, count):
    names = profile_channels(policy)
    assert len(names) == len(set(names)) == count
    assert not ({7, 17, 18, 19, 20, 21} & required_bands(policy))
    if count > 8:
        assert names[-1] == "LC_Type1_unknown"


def test_current_eight_channels_missingness_and_circular_aspect():
    n = Normalizer([12, 14], [2., 10.], [2., 5.], [10, 10], [])
    assert n.channel_names == ["slope_z", "elevation_z", "slope_missing",
        "elevation_missing", "aspect_sin", "aspect_cos", "aspect_missing", "fire_J"]
    raw = np.zeros((23, 1, 6), np.float32)
    raw[12] = [2, np.nan, 100, 2, 2, 2]
    raw[14] = 10
    raw[13] = [0, 90, 180, 270, np.nan, -1]
    raw[22] = [np.nan, 918, 2036, 0, np.nan, np.nan]
    x = n.transform(raw)
    assert x[0, 0, 1] == 0 and x[2, 0, 1] == 1
    assert x[0, 0, 2] == 8  # clipping is in z units, not physical units
    np.testing.assert_allclose(x[4, 0, :4], [0, 1, 0, -1], atol=2e-7)
    np.testing.assert_allclose(x[5, 0, :4], [1, 0, -1, 0], atol=2e-7)
    assert np.array_equal(x[4:7, 0, 4:], [[0, 0], [0, 0], [1, 1]])
    assert np.array_equal(fire_mask(raw)[0, 0], [0, 1, 1, 1, 0, 0])


def asset():
    return dict(event_id="2020/fire_1", as_of="2020-01-05", input_date="2020-01-05",
        band=0, kind="observed", observation_start="2020-01-05T00:00:00Z",
        observation_end="2020-01-05T20:00:00Z", available_at="2020-01-05T22:00:00Z",
        decision_at="2020-01-06T00:00:00Z", source_product="SYNTHETIC_FIXTURE",
        product_version="fixture1", asset_id="synthetic-only", evidence_uri="fixture://review",
        evidence_sha256="a" * 64, reviewed=True)


def keys(row):
    return [(row["event_id"], row["as_of"], row["input_date"], row["band"])]


def test_chronology_does_not_claim_truth_or_enable_a_live_profile():
    row = asset()
    result = validate_availability([row], keys(row))
    assert result["policy_activated"] is False
    assert result["external_evidence_review_still_required"] is True


@pytest.mark.parametrize("field,value", [
    ("available_at", "2020-01-06T00:00:01Z"),
    ("observation_end", "2020-01-10T00:00:00Z"),
    ("decision_at", "2020-01-07T00:00:00Z"),
    ("available_at", "2020-01-05T22:00:00"),
    ("available_at", "2020-01-05T22:00:00+01:00"),
    ("reviewed", False), ("evidence_sha256", "0" * 64),
    ("asset_id", ""), ("product_version", ""), ("kind", "unverified"),
])
def test_availability_rejects_unknown_latency_future_composites_and_opaque_claims(field, value):
    row = asset(); row[field] = value
    with pytest.raises(ValueError):
        validate_availability([row], keys(row))


@pytest.mark.parametrize("band", [7, 19])
def test_no_wind_direction_is_enabled_by_a_boolean_declaration(band):
    row = asset(); row["band"] = band
    with pytest.raises(ValueError, match="Wind direction"):
        validate_availability([row], keys(row))


def test_forecast_may_have_future_valid_time_but_must_be_issued_and_available_by_J():
    row = asset(); row.update(band=20, kind="forecast", issue_at="2020-01-05T18:00:00Z",
        observation_start="2020-01-06T00:00:00Z", observation_end="2020-01-07T00:00:00Z")
    validate_availability([row], keys(row))
    row["issue_at"] = "2020-01-06T01:00:00Z"
    with pytest.raises(ValueError):
        validate_availability([row], keys(row))


def test_availability_requires_exhaustive_unique_assets():
    row = asset()
    with pytest.raises(ValueError): validate_availability([], keys(row))
    with pytest.raises(ValueError): validate_availability([row, row], keys(row))
    with pytest.raises(ValueError): validate_availability([row], [])
    row.update(event_id="2021/fire_1", as_of="2021-01-05", input_date="2021-01-05")
    with pytest.raises(ValueError, match="2021 locked"):
        validate_availability([row], keys(row))


def test_legacy_profile_gate_accepts_opaque_evidence_but_new_review_does_not(tmp_path):
    """Document a proven structural gap; do not change the production gate."""
    path = tmp_path / "claim.json"
    path.write_text(json.dumps(dict(policy="terrain_vegetation_v1", decision_time="end_of_day_J",
        available_by_J_bands=list(required_bands("terrain_vegetation_v1")),
        evidence="unverified self-declaration", reviewed=True)), encoding="utf-8")
    assert feature_profile(dict(feature_policy="terrain_vegetation_v1",
        feature_availability=str(path)))["landcover"] is True
    with pytest.raises(ValueError):
        validate_availability([], [("2020/fire_1", "2020-01-05", "2020-01-05", 3)])


def sample():
    dates = [f"2020-01-{d:02d}" for d in range(1, 6)]
    refs = [f"val/2020/fire_1.zip!{d}_0_0.npz" for d in dates]
    return dict(event_id="2020/fire_1", split="val", target_date="2020-01-06",
        row=0, col=0, context_dates=dates[-3:], domain_dates=dates,
        context=refs[-3:], domain=refs, target="val/2020/fire_1.zip!2020-01-06_0_0.npz",
        immutable_source_sha256="b" * 64)


def test_T1_projection_preserves_T5_and_target_without_rebuilding_data():
    row = sample(); original = deepcopy(row)
    one = temporal_view(row, 1)
    assert row == original and one["context"] == row["context"][-1:]
    assert one["domain"] == row["domain"] and one["target"] == row["target"]
    assert validate_paired_metadata([row], [one])["pixel_reads"] == 0


@pytest.mark.parametrize("field", ["domain", "target", "immutable_source_sha256", "context"])
def test_temporal_comparison_rejects_any_changed_domain_or_source(field):
    row = sample(); one = temporal_view(row, 1); one[field] = "changed"
    with pytest.raises(ValueError): validate_paired_metadata([row], [one])


def test_temporal_projection_rejects_gaps_wrong_splits_and_locked_test():
    row = sample(); row["domain_dates"][0] = "2019-12-31"
    with pytest.raises(ValueError): temporal_view(row, 1)
    row = sample(); row["split"] = "train"
    with pytest.raises(ValueError): temporal_view(row, 1)
    row = sample(); row["event_id"] = "2021/fire_1"
    with pytest.raises(ValueError): temporal_view(row, 1)


def test_native_T1_changes_capacity_but_repeating_J_preserves_the_T3_shape():
    from geofirex.benchmark_models import StackedUNet
    t1, t3 = StackedUNet(8, 1), StackedUNet(8, 3)
    assert sum(p.numel() for p in t1.parameters()) == 122825
    assert sum(p.numel() for p in t3.parameters()) == 123977
    synthetic = torch.zeros(1, 3, 8, 2, 3)
    repeated = synthetic[:, -1:].repeat(1, 3, 1, 1, 1)
    assert repeated.shape == synthetic.shape
    # No forward, optimizer, dataset, checkpoint or training is involved.


def test_existing_compact_loader_has_five_day_domain_and_no_future_covariates():
    ds = FastCompactTiles.__new__(FastCompactTiles)
    target = date(2020, 1, 6).toordinal()
    ds.records = np.array([(0, target, 0, 0, 0)], dtype=DTYPE)
    days = {}
    for offset in range(-5, 1):
        x = np.zeros((8, 2, 3), np.float32); x[0] = offset; x[7] = 1
        v = np.ones((1, 2, 3), bool); days[target + offset] = (x, v)
    days[target - 5][1][0, 0, 0] = False
    days[target][1][0, 0, 1] = False
    seen = []
    def fake_day(a, day, r, c):
        seen.append(day); return days[day]
    ds._day = fake_day
    x, y, valid = ds[0]
    assert set(seen) == set(range(target - 5, target + 1))
    assert not valid[0, 0, 0] and not valid[0, 0, 1]
    assert x.shape == (3, 8, 2, 3) and np.array_equal(x[:, 0, 0, 0], [-3, -2, -1])
    days[target][0][0] = 999  # mutate a future covariate, never an input
    assert torch.equal(ds[0][0], x) and torch.equal(ds[0][1], y)


def focal_reference(logits, y, valid, gamma):
    """Definition fixture only; not installed in any trainer or Studio model."""
    p = logits.sigmoid()
    term = 10 * y * (1 - p).pow(gamma) * F.softplus(-logits)
    term += (1 - y) * p.pow(gamma) * F.softplus(logits)
    return (term * valid).sum() / valid.sum().clamp_min(1)


def test_proposed_focal_gamma_zero_is_weighted_BCE_not_an_unstated_alpha_scale():
    z = torch.tensor([[-8., -1., 0., 1., 8.]])
    y = torch.tensor([[0., 1., 0., 1., 1.]])
    v = torch.tensor([[1., 1., 0., 1., 1.]])
    expected = F.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor(10.), reduction="none")
    assert torch.allclose(focal_reference(z, y, v, 0), (expected * v).sum() / v.sum())
    assert torch.isfinite(focal_reference(z * 1000, y, v, 2))
    assert focal_reference(z, y, torch.zeros_like(v), 2) == 0


def test_focal_downweights_easy_examples_without_claiming_an_empirical_gain():
    z = torch.tensor([8., 0.]); y = torch.ones(2); v = torch.ones(2)
    individual = [focal_reference(z[i:i+1], y[i:i+1], v[i:i+1], 2).item() for i in range(2)]
    assert individual[0] < individual[1]


def test_saved_counts_and_new_fire_partition_are_consistent_without_predictions():
    folder = ROOT / "reports/unet_confirmation_20261112_v1/confirmation_cpu"
    comparison = json.loads((folder / "comparison.json").read_text())
    rows = comparison if isinstance(comparison, list) else comparison["rows"]
    for row in rows: validate_saved_counts(row)
    groups = json.loads((folder / "new_fire.json").read_text())
    for row in groups.values():
        if isinstance(row, dict): validate_saved_counts(row)
    full = rows[-1]
    assert all(groups["new_fire_candidates"][k] + groups["observed_fire_candidates"][k] == full[k]
               for k in ("tp", "fp", "fn", "tn"))
    bad = dict(full, dice=0.99)
    with pytest.raises(ValueError): validate_saved_counts(bad)


def test_campaign_is_preparation_only_and_does_not_promote_a_model():
    plan = json.loads((ROOT / "configs/scientific_improvements_v1.json").read_text())
    assert validate_campaign(plan)["training_enabled"] is False
    assert plan["promotion_to_studio"] is False
    plan["training_authorized"] = True
    with pytest.raises(ValueError): validate_campaign(plan)


@pytest.mark.parametrize("field,value", [("promotion_to_studio", True),
    ("fold", {"id": 1, "train_years": [2018, 2021], "validation_years": [2020]})])
def test_campaign_refuses_promotion_or_a_fold_that_exposes_2021(field, value):
    plan = json.loads((ROOT / "configs/scientific_improvements_v1.json").read_text())
    plan[field] = value
    with pytest.raises(ValueError): validate_campaign(plan)


def test_audit_refuses_overwriting_before_reading_any_results(tmp_path, monkeypatch):
    import scripts.audit_scientific_improvements as audit
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    out = tmp_path / "reports" / "audit"; out.mkdir(parents=True)
    protected = out / "saved_metrics_audit.json"; protected.write_text("preserve", encoding="utf-8")
    def forbidden():
        raise AssertionError("Reader must not be initialized before overwrite check")
    monkeypatch.setattr(audit, "ScientificResults", forbidden)
    with pytest.raises(FileExistsError): audit.build(out)
    assert protected.read_text(encoding="utf-8") == "preserve"
