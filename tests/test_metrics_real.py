"""Tests for the Lot 4 additions to geofirex.metrics: validity masks, PR-AUC,
and per-event aggregation. Existing tests/test_metrics.py covers the base
dice/iou/precision/recall behavior and is left untouched.
"""

import torch
from pytest import approx

from geofirex.metrics import (
    aggregate_metrics_by_event,
    dice_score,
    iou_score,
    pr_auc_score,
    precision_score,
    recall_score,
)


def test_mask_excludes_pixels_from_the_score_hand_computable():
    # 4 pixels: pred=[1,1,1,1], target=[1,0,1,0], mask keeps only the first two.
    # Kept: pred=[1,1], target=[1,0] -> TP=1, FP=1, FN=0.
    pred = torch.tensor([1.0, 1.0, 1.0, 1.0]).reshape(1, 1, 2, 2)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, 2, 2)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0]).reshape(1, 1, 2, 2)

    assert precision_score(pred, target, mask=mask) == approx(0.5)
    assert recall_score(pred, target, mask=mask) == approx(1.0)
    assert dice_score(pred, target, mask=mask) == approx(2 / 3)
    assert iou_score(pred, target, mask=mask) == approx(0.5)


def test_no_mask_matches_full_image_behavior():
    pred = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, 2, 2)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 2, 2)
    full_mask = torch.ones_like(pred)

    assert dice_score(pred, target) == approx(dice_score(pred, target, mask=full_mask))


def test_pr_auc_is_perfect_for_perfectly_separable_scores():
    pred = torch.tensor([0.9, 0.8, 0.1, 0.2])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    auc = pr_auc_score(pred, target)
    assert auc == approx(1.0)


def test_pr_auc_is_none_when_no_positive_pixel():
    pred = torch.tensor([0.1, 0.2, 0.3])
    target = torch.zeros(3)
    assert pr_auc_score(pred, target) is None  # undefined, not a fabricated 0.0


def test_aggregate_metrics_by_event_averages_within_event_only():
    per_sample = [
        ("e1", {"dice": 1.0, "pr_auc": None}),
        ("e1", {"dice": 0.5, "pr_auc": 0.8}),
        ("e2", {"dice": 0.0, "pr_auc": 0.2}),
    ]
    agg = aggregate_metrics_by_event(per_sample)

    assert agg["e1"]["dice"] == approx(0.75)
    assert agg["e1"]["pr_auc"] == approx(0.8)  # None excluded, not averaged as 0
    assert agg["e2"]["dice"] == approx(0.0)
    assert agg["e2"]["pr_auc"] == approx(0.2)
