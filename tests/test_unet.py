"""Model, numerical stability and padding-aware loss contracts."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from geofirex.unet import LightUNet, SegmentationLoss
from geofirex.unet_evaluation import predict_tiled, score, select_threshold, summarize


@pytest.fixture(autouse=True, scope="module")
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(2, 8, 64, 64), (1, 8, 33, 29)])
def test_unet_forward_backward_reaches_encoder_bottleneck_and_decoder(shape):
    model = LightUNet(8, base_channels=4, depth=3)
    x = torch.randn(shape)
    logits = model(x)
    assert logits.shape == (shape[0], 1, *shape[-2:])
    target = torch.zeros_like(logits)
    target[..., 5:10, 5:10] = 1
    loss = SegmentationLoss()(logits, target)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad.abs().sum() > 0 for name, p in model.named_parameters() if name.endswith("weight"))


def test_bce_mode_matches_torch_and_ignores_padded_pixels():
    logits = torch.tensor([[[[0., 1.], [-2., 100.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 0.], [1., 0.]]]])
    valid = torch.tensor([[[[1., 1.], [1., 0.]]]])
    loss = SegmentationLoss(dice_weight=0)(logits, target, valid)
    expected = F.binary_cross_entropy_with_logits(logits.flatten()[:3], target.flatten()[:3])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[0, 0, 1, 1] == 0


@pytest.mark.parametrize("value,target_value", [(1000., 0.), (-1000., 1.), (-1000., 0.)])
def test_combined_loss_finite_for_extreme_logits_and_empty_targets(value, target_value):
    logits = torch.full((2, 1, 8, 8), value, requires_grad=True)
    target = torch.full_like(logits, target_value)
    loss = SegmentationLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_loss_all_padding_has_zero_gradient():
    x = torch.randn(1, 1, 8, 8, requires_grad=True)
    loss = SegmentationLoss()(x, torch.zeros_like(x), torch.zeros_like(x))
    loss.backward()
    assert loss.item() == 0 and not x.grad.any()


def test_small_pattern_can_be_learned():
    torch.manual_seed(13)
    model = LightUNet(1, base_channels=4, depth=2)
    x = torch.zeros(1, 1, 16, 16)
    x[..., 4:10, 4:10] = 1
    loss_fn = SegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = loss_fn(model(x), x).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        optimizer.step()
    assert loss_fn(model(x), x).item() < initial * 0.5


class IdentityLogits(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


@pytest.mark.parametrize("shape", [(1, 21, 35), (1, 3, 5), (1, 32, 32)])
def test_tiled_inference_covers_edges_without_changing_coordinates(shape):
    image = np.linspace(-3, 3, np.prod(shape), dtype=np.float32).reshape(shape)
    model = IdentityLogits()
    model.train()
    result = predict_tiled(model, image, size=16, batch_size=3)
    assert result.shape == shape
    np.testing.assert_allclose(result, 1 / (1 + np.exp(-image)), atol=1e-7)
    assert model.training


def test_threshold_selection_uses_validation_and_rejects_test():
    records = [{"probability": np.array([.8, .6, .4]), "target": np.array([1, 0, 1])}]
    threshold, curve = select_threshold(records, [.3, .5], split_name="val")
    assert threshold == .3 and curve[0]["dice"] == .8
    with pytest.raises(ValueError, match="validation"):
        select_threshold(records, [.3, .5], split_name="test")
    records[0]["target"][:] = 0
    with pytest.raises(ValueError, match="positive"):
        select_threshold(records, [.3], split_name="val")


def test_metric_empty_conventions_and_hand_computed_counts():
    empty = score(np.zeros(4), np.zeros(4), .5)
    assert empty["dice"] == empty["iou"] == 1
    assert empty["pr_auc"] is empty["precision"] is empty["recall"] is None
    actual = score(np.array([.9, .8, .2, .1]), np.array([1, 0, 1, 0]), .5)
    assert (actual["tp"], actual["fp"], actual["fn"]) == (1, 1, 1)
    assert actual["dice"] == actual["precision"] == actual["recall"] == .5
    assert actual["iou"] == pytest.approx(1/3)
    assert actual["pr_auc"] == pytest.approx((1 + 2/3)/2)


def test_micro_and_positive_window_scores_do_not_hide_empty_windows():
    records = [dict(event_id="a", target_date="1", probability=np.zeros(4),
                    persistence=np.zeros(4), target=np.zeros(4)),
               dict(event_id="a", target_date="2", probability=np.zeros(4),
                    persistence=np.zeros(4), target=np.ones(4))]
    result = summarize(records, .5)
    assert result["macro_window"]["dice"] == .5
    assert result["micro_pixel"]["dice"] == result["macro_positive_window"]["dice"] == 0
    assert result["empty_target_windows"] == 1
    assert result["metric_defined_windows"]["pr_auc"] == 1


def test_invalid_model_and_loss_configuration_fails():
    with pytest.raises(ValueError):
        LightUNet(0)
    with pytest.raises(ValueError):
        SegmentationLoss(dice_weight=2)
    with pytest.raises(ValueError):
        LightUNet(1)(torch.zeros(1, 1, 4, 4))
