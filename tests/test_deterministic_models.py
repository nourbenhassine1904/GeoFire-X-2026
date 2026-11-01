"""Versioned GPU-safe decoders: preserve shapes, gradients and state schema."""
from types import SimpleNamespace

import pytest
import torch

from geofirex.runner import runner_model


@pytest.mark.parametrize("name", ["B2", "B3"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable locally"))])
def test_nearest_decoder_backward_and_checkpoint_schema(name, device):
    config = dict(model=name, context=3, model_kwargs={})
    norm = SimpleNamespace(channel_names=list(range(8)))
    original = runner_model(config, norm)
    config["model_kwargs"] = {"upsampling": "nearest_v1"}
    model = runner_model(config, norm).to(device)
    model.load_state_dict(original.state_dict(), strict=True)
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        x = torch.randn(2, 3, 8, 17, 19, device=device)
        y = model(x)
        assert y.shape == (2, 1, 17, 19)
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)
