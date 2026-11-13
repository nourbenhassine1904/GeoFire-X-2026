"""Fold-independent, bounded-memory NumPy memmap cache of selected physical bands.

One event directory: numeric float32 [D,3,H,W], fire uint8 [D,H,W],
LC uint8 [D,H,W] (255 unknown). Normalization is deliberately deferred to folds.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import rasterio

from geofirex.corpus import json_new, sha256_file
from geofirex.unet_data import read_raw


def build_event_cache(rows, root, destination, *, expected_dates):
    if not rows or len({r["event_id"] for r in rows}) != 1:
        raise ValueError("Exactly one nonempty event required")
    rows = sorted(rows,key=lambda r:r["date"])
    if len({r["date"] for r in rows}) != len(rows) or {r["date"] for r in rows} != set(expected_dates):
        raise ValueError("Event manifest is incomplete or duplicated against pinned index")
    destination=Path(destination)
    fingerprints={r["date"]:sha256_file(Path(root)/r["relative_path"]) for r in rows}
    if destination.exists():
        metadata=json.loads((destination/"metadata.json").read_text(encoding="utf-8"))
        if metadata["source_sha256_by_date"] != fingerprints:
            raise ValueError("Existing cache provenance differs; preserved")
        for name,digest in metadata["array_sha256"].items():
            if sha256_file(destination/name) != digest:
                raise ValueError("Existing cache integrity failure; preserved")
        return metadata
    destination.parent.mkdir(parents=True,exist_ok=True)
    work=destination.with_name(destination.name+f".building_{uuid.uuid4().hex}")
    work.mkdir()
    with rasterio.open(Path(root)/rows[0]["relative_path"]) as first:
        shape,crs,transform=first.shape,str(first.crs),tuple(first.transform)
    d,h,w=len(rows),*shape
    numeric=np.lib.format.open_memmap(work/"numeric.npy",mode="w+",dtype="float32",shape=(d,3,h,w))
    fire=np.lib.format.open_memmap(work/"fire.npy",mode="w+",dtype="uint8",shape=(d,h,w))
    lc=np.lib.format.open_memmap(work/"landcover.npy",mode="w+",dtype="uint8",shape=(d,h,w))
    for i,row in enumerate(rows):
        path=Path(root)/row["relative_path"]
        with rasterio.open(path) as ds:
            if (ds.shape,str(ds.crs),tuple(ds.transform)) != (shape,crs,transform):
                raise ValueError("Cache cannot combine misaligned event rasters")
        raw=read_raw(path)
        numeric[i]=raw[[12,13,14]]  # Keep missingness, physical units and angles.
        fire[i]=~np.isnan(raw[22])
        land=raw[16]
        known=np.isfinite(land)&(land>=1)&(land<=17)&(land==np.floor(land))
        lc[i]=np.where(known,land,255).astype(np.uint8)
    numeric.flush(); fire.flush(); lc.flush()
    del numeric,fire,lc
    names=("numeric.npy","fire.npy","landcover.npy")
    metadata={"schema_version":1,"event_id":rows[0]["event_id"],"dates":[r["date"] for r in rows],
              "shape":[d,h,w],"numeric_raw_indices":[12,13,14],"numeric_names":["slope","aspect","elevation"],
              "normalization":"none; fit train inputs of each fold at training time",
              "landcover_policy":"1..17 unchanged, invalid/missing/fractional -> 255; inclusion is a separate feature-policy decision",
              "crs":crs,"transform":list(transform),"source_sha256_by_date":fingerprints,
              "array_sha256":{name:sha256_file(work/name) for name in names},
              "array_bytes":sum((work/name).stat().st_size for name in names)}
    json_new(work/"metadata.json",metadata)
    if destination.exists():
        raise FileExistsError(destination)
    work.rename(destination)
    return metadata


class EventCache:
    def __init__(self,path):
        path=Path(path)
        self.metadata=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        self.numeric=np.load(path/"numeric.npy",mmap_mode="r")
        self.fire=np.load(path/"fire.npy",mmap_mode="r")
        self.landcover=np.load(path/"landcover.npy",mmap_mode="r")

    def raw_patch(self,day,row,col,size):
        """Return the selected raw bands in the original 23-band coordinate system.

        Uncached bands are NaN, preventing their accidental use as fabricated zeros.
        Reader must apply its fold's normalizer and the declared feature policy.
        """
        numeric=np.asarray(self.numeric[day,:,row:row+size,col:col+size])
        raw=np.full((23,*numeric.shape[-2:]),np.nan,dtype=np.float32)
        raw[[12,13,14]]=numeric
        raw[22]=np.where(self.fire[day,row:row+size,col:col+size],0.,np.nan)
        classes=self.landcover[day,row:row+size,col:col+size]
        raw[16]=np.where(classes==255,np.nan,classes)
        return raw

    def close(self):
        # Explicit Windows file handles; evict whole EventCache objects from worker LRU.
        self.numeric._mmap.close()
        self.fire._mmap.close()
        self.landcover._mmap.close()
