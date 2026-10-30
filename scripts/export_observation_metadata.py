"""Local raster QC -> Parquet metadata, with explicit local processing scope."""
import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from geofirex.corpus import json_new, sha256_file
from geofirex.unet_data import read_raw

ROOT=Path(__file__).resolve().parents[1]


def observation_metadata(rows,root=ROOT):
    records=[]
    for row in rows:
        path=Path(root)/row["relative_path"]
        raw=read_raw(path)
        with rasterio.open(path) as ds:
            bounds=transform_bounds(ds.crs,"EPSG:4326",*ds.bounds)
            record={"event_id":row["event_id"],"year":int(row["year"]),"date":row["date"],
                    "path":row["relative_path"],"sha256":sha256_file(path),
                    "height":ds.height,"width":ds.width,"crs":str(ds.crs),
                    "geo_partition":str(ds.crs).replace(":","_"),
                    "west":bounds[0],"south":bounds[1],"east":bounds[2],"north":bounds[3],
                    "raster_center_longitude":(bounds[0]+bounds[2])/2,
                    "raster_center_latitude":(bounds[1]+bounds[3])/2,
                    "fire_pixels":int((~np.isnan(raw[22])).sum()),
                    "pixel_count":ds.height*ds.width,
                    "all_band_inf_pixels":int(np.isinf(raw).sum()),
                    "gfs_rain_band_name":ds.descriptions[17]}
        for i,name in ((12,"slope"),(13,"aspect"),(14,"elevation")):
            values=raw[i][np.isfinite(raw[i])].astype(np.float64)
            record.update({name+"_finite_count":len(values),name+"_sum":float(values.sum()),
                           name+"_sum_squares":float(np.square(values).sum())})
        records.append(record)
    return records


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=False)
    with Path(args.manifest).open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    records=observation_metadata(rows)
    pq.write_table(pa.Table.from_pylist(records),output/"observations.parquet",compression="snappy")
    # Round-trip, including all metadata columns and row counts.
    loaded=pq.read_table(output/"observations.parquet").to_pylist()
    if loaded != records:
        raise RuntimeError("Parquet metadata round-trip mismatch")
    json_new(output/"inventory.json",{"scope":"local raster reads; not distributed; not complete corpus",
                                    "observations":len(records),"events":len({r['event_id'] for r in records}),
                                    "manifest_sha256":sha256_file(args.manifest),
                                    "parquet_sha256":sha256_file(output/"observations.parquet")})
    print(f"Exported and round-trip verified {len(records)} local observation metadata rows")
