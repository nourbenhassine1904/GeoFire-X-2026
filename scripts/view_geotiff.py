"""Read-only GeoTIFF metadata and compact plots (band numbers are 1-based).

Example: python scripts/view_geotiff.py FILE.tif --out .demo_scratch/view.png
Without --out, open an interactive Matplotlib window. Existing outputs are refused.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--bands", nargs="+", type=int, default=[23, 4, 6, 15, 17])
    parser.add_argument("--out", type=Path, help="PNG output; omit for interactive display")
    parser.add_argument("--metadata", type=Path, help="Optional JSON for all bands")
    args = parser.parse_args()
    for output in [args.out, args.metadata]:
        if output and (output.exists() or output.resolve() == args.path.resolve()):
            parser.error(f"Refusing to overwrite {output}")
    if args.out and args.metadata and args.out.resolve() == args.metadata.resolve():
        parser.error("PNG and metadata must use different output paths")

    with rasterio.open(args.path) as ds:
        if any(b < 1 or b > ds.count for b in args.bands):
            parser.error(f"--bands must be between 1 and {ds.count}")
        data = ds.read()
        masks = ds.read_masks() > 0
        names = ds.descriptions
        metadata = {
            "path": args.path.as_posix(), "bands": ds.count,
            "height": ds.height, "width": ds.width, "crs": str(ds.crs),
            "transform": list(ds.transform), "dtype": list(ds.dtypes),
            "nodata": [str(v) if v is not None and not math.isfinite(v) else v for v in ds.nodatavals],
            "mask_flags": [[flag.name for flag in flags] for flags in ds.mask_flag_enums],
            "band_statistics": [],
        }
        for index, band in enumerate(data):
            valid = masks[index] & np.isfinite(band)
            values = band[valid]
            metadata["band_statistics"].append({
                "band": index + 1, "description": names[index],
                "gdal_valid_fraction": float(masks[index].mean()),
                "finite_valid_fraction": float(valid.mean()),
                "nan_pixels": int(np.isnan(band).sum()),
                "inf_pixels": int(np.isinf(band).sum()),
                "min": float(values.min()) if values.size else None,
                "max": float(values.max()) if values.size else None,
            })
    print(json.dumps(metadata, indent=2, allow_nan=False))
    print("Active fire: NaN means no detection in WildfireSpreadTS; its GDAL mask is not a label-validity mask.")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, allow_nan=False)

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(args.bands) + 1
    columns = min(count, 3)
    fig, axes = plt.subplots(math.ceil(count / columns), columns,
                             figsize=(5 * columns, 4 * math.ceil(count / columns)), squeeze=False)
    for ax, number in zip(axes.flat, args.bands):
        index = number - 1
        name = names[index] or f"Band {number}"
        if name == "active fire":
            display = (~np.isnan(data[index])).astype("uint8")
            im = ax.imshow(display, cmap="hot", vmin=0, vmax=1)
            name += " (presence, NaN = no detection)"
        else:
            display = np.ma.array(data[index], mask=~(masks[index] & np.isfinite(data[index])))
            im = ax.imshow(display, cmap="tab20" if name == "LC_Type1" else "viridis")
        ax.set_title(f"{number}: {name}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    # Use a non-fire band when available: fire NaNs encode absence, not missing labels.
    validity_index = next((b - 1 for b in args.bands if names[b - 1] != "active fire"), args.bands[0] - 1)
    ax = axes.flat[len(args.bands)]
    ax.imshow(masks[validity_index] & np.isfinite(data[validity_index]), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"GDAL mask AND finite: band {validity_index + 1}", fontsize=9)
    ax.axis("off")
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.suptitle(f"{args.path.name} | {metadata['height']} x {metadata['width']} | {metadata['crs']}")
    fig.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as f:
            fig.savefig(f, format="png", dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
