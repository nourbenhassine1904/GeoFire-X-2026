"""Dice, IoU, Precision, Recall, PR-AUC for binary active-fire segmentation.

All functions accept an optional `mask` (same shape as pred/target, 1=valid,
0=excluded) so invalid/no-data pixels can be excluded from a score without
changing the function signatures used by existing callers - `mask=None`
(the default) means every pixel is valid, which is the prior behavior.
"""

from __future__ import annotations

import torch

try:
    from sklearn.metrics import average_precision_score
except ImportError:  # pragma: no cover - sklearn is a declared dependency
    average_precision_score = None


def _binarize(pred: torch.Tensor, threshold: float) -> torch.Tensor:
    return (pred >= threshold).float()


def _apply_mask(
    pred_b: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return pred_b, target
    mask = mask.float()
    return pred_b * mask, target * mask


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> float:
    pred_b = _binarize(pred, threshold)
    pred_b, target = _apply_mask(pred_b, target, mask)
    intersection = (pred_b * target).sum()
    denom = pred_b.sum() + target.sum()
    return float((2 * intersection + eps) / (denom + eps))


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> float:
    pred_b = _binarize(pred, threshold)
    pred_b, target = _apply_mask(pred_b, target, mask)
    intersection = (pred_b * target).sum()
    union = pred_b.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def precision_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> float:
    pred_b = _binarize(pred, threshold)
    pred_b, target = _apply_mask(pred_b, target, mask)
    tp = (pred_b * target).sum()
    fp = (pred_b * (1 - target)).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> float:
    pred_b = _binarize(pred, threshold)
    pred_b, target = _apply_mask(pred_b, target, mask)
    tp = (pred_b * target).sum()
    fn = ((1 - pred_b) * target).sum()
    return float((tp + eps) / (tp + fn + eps))


def pr_auc_score(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float | None:
    """Area under the Precision-Recall curve, using `pred` as a continuous
    score (not thresholded). Returns None (not 0.0 - a real "not applicable",
    never a fabricated score) when there is no positive pixel in `target`,
    since PR-AUC is undefined without at least one positive example."""
    if average_precision_score is None:  # pragma: no cover
        raise ImportError("scikit-learn is required for pr_auc_score")

    if mask is not None:
        keep = mask.flatten().bool()
        pred = pred.flatten()[keep]
        target = target.flatten()[keep]
    else:
        pred = pred.flatten()
        target = target.flatten()

    target_np = target.detach().cpu().numpy()
    if target_np.sum() == 0:
        return None
    pred_np = pred.detach().cpu().numpy()
    return float(average_precision_score(target_np, pred_np))


def all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    mask: torch.Tensor | None = None,
    include_pr_auc: bool = False,
) -> dict[str, float | None]:
    result = {
        "dice": dice_score(pred, target, threshold, mask=mask),
        "iou": iou_score(pred, target, threshold, mask=mask),
        "precision": precision_score(pred, target, threshold, mask=mask),
        "recall": recall_score(pred, target, threshold, mask=mask),
    }
    if include_pr_auc:
        result["pr_auc"] = pr_auc_score(pred, target, mask=mask)
    return result


def aggregate_metrics_by_event(
    per_sample: list[tuple[str, dict[str, float | None]]],
) -> dict[str, dict[str, float]]:
    """Average metrics across samples sharing the same event_id.

    `per_sample` is a list of (event_id, metrics_dict) pairs, e.g. produced by
    scoring every window of a RealGeoFireXDataset. None values (e.g. an
    undefined PR-AUC) are excluded from that metric's average rather than
    treated as 0 or crashing the aggregation.
    """
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for event_id, metrics in per_sample:
        sums.setdefault(event_id, {})
        counts.setdefault(event_id, {})
        for key, value in metrics.items():
            if value is None:
                continue
            sums[event_id][key] = sums[event_id].get(key, 0.0) + value
            counts[event_id][key] = counts[event_id].get(key, 0) + 1

    return {
        event_id: {key: sums[event_id][key] / counts[event_id][key] for key in sums[event_id]}
        for event_id in sums
    }
