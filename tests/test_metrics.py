import torch

from geofirex.metrics import dice_score, iou_score, precision_score, recall_score


def test_perfect_prediction_gives_perfect_scores():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:5, 2:5] = 1.0
    pred = target.clone()

    assert dice_score(pred, target) > 0.99
    assert iou_score(pred, target) > 0.99
    assert precision_score(pred, target) > 0.99
    assert recall_score(pred, target) > 0.99


def test_disjoint_prediction_gives_zero_scores():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 0:2, 0:2] = 1.0
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, 6:8, 6:8] = 1.0

    assert dice_score(pred, target) < 0.01
    assert iou_score(pred, target) < 0.01
    assert precision_score(pred, target) < 0.01
    assert recall_score(pred, target) < 0.01


def test_empty_prediction_has_zero_recall_but_defined_precision():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 1.0
    pred = torch.zeros(1, 1, 4, 4)

    assert recall_score(pred, target) < 0.01
    # No positive predictions at all: precision is a degenerate 0/0 case,
    # handled by the eps smoothing rather than raising.
    assert 0.0 <= precision_score(pred, target) <= 1.0
