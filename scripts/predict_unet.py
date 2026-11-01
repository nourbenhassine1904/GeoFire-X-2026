"""Predict J+1 from a single J GeoTIFF and a trained checkpoint, without labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from geofirex.unet import LightUNet
from geofirex.unet_data import Normalizer, read_raw
from geofirex.unet_evaluation import predict_tiled


def load_predictor(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if checkpoint.get("threshold_selected_on") != "val":
        raise ValueError("Checkpoint has no validation-calibrated threshold")
    model = LightUNet(checkpoint["in_channels"], config["base_channels"], config["depth"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, Normalizer(**checkpoint["normalizer"]), checkpoint


def predict_file(checkpoint_path, input_path, output_dir, device="cpu"):
    model, normalizer, checkpoint = load_predictor(checkpoint_path, device)
    config = checkpoint["config"]
    torch.set_num_threads(config["cpu_threads"])
    probability = predict_tiled(model, normalizer.transform(read_raw(input_path)),
                                config["patch_size"], config["batch_size"], device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with rasterio.open(input_path) as source:
        profile = {"driver": "GTiff", "height": source.height, "width": source.width,
                   "count": 1, "crs": source.crs, "transform": source.transform,
                   "compress": "deflate"}
    for name, arr, dtype in (("probability", probability, "float32"),
                              ("mask", probability >= checkpoint["threshold"], "uint8")):
        with rasterio.open(output / f"{name}.tif", "w", **profile, dtype=dtype) as target:
            target.write(arr.astype(dtype))
            target.set_band_description(1, f"J+1 active fire {name}")
            target.update_tags(threshold=str(checkpoint["threshold"]),
                               source_date=Path(input_path).stem,
                               threshold_selected_on="val", source_checkpoint=str(checkpoint_path))
    return probability


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = p.parse_args()
    predict_file(args.checkpoint, args.input, args.output_dir, args.device)
    print(f"Saved georeferenced probability.tif and mask.tif to {args.output_dir}")
