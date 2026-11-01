"""Exact disk-backed AP, including global score ties, for large raster streams.

Fixed-memory sorting runs and hierarchical merges avoid concatenating a year's
pixels in RAM. Temporary runs are kept in the caller's new output directory.
This is an exact reference implementation; full-corpus throughput is unmeasured.
"""
from __future__ import annotations

import heapq
import itertools
from pathlib import Path

import numpy as np

DTYPE=np.dtype([("score","<f8"),("count","<u8"),("positive","<u8")])


def _records(path):
    with Path(path).open("rb") as f:
        while True:
            block=np.fromfile(f,dtype=DTYPE,count=4096)
            if not len(block):
                break
            for row in block:
                yield float(row["score"]),int(row["count"]),int(row["positive"])


def _merge(paths):
    iterators=[iter(_records(p)) for p in paths]
    heap=[]
    for i,it in enumerate(iterators):
        row=next(it,None)
        if row is not None:
            heapq.heappush(heap,(-row[0],i,row[1],row[2]))
    while heap:
        score=heap[0][0]
        count=positive=0
        while heap and heap[0][0]==score:
            _,i,n,pos=heapq.heappop(heap)
            count+=n; positive+=pos
            row=next(iterators[i],None)
            if row is not None:
                heapq.heappush(heap,(-row[0],i,row[1],row[2]))
        yield -score,count,positive


class ExternalAveragePrecision:
    def __init__(self,directory,chunk_pixels=250_000,fan_in=32):
        if chunk_pixels<1 or fan_in<2:
            raise ValueError("Positive chunk size and fan_in >=2 required")
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.chunk_pixels,self.fan_in=chunk_pixels,fan_in
        self.paths=[]; self.positives=0; self.pixels=0; self.finalized=False

    def update(self,probability,target):
        if self.finalized:
            raise RuntimeError("AP accumulator already finalized")
        p=np.asarray(probability).ravel()
        y=np.asarray(target).ravel()
        if p.shape!=y.shape or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
            raise ValueError("Finite matching scores and binary labels required")
        for offset in range(0,len(p),self.chunk_pixels):
            scores=p[offset:offset+self.chunk_pixels].astype(np.float64)
            labels=y[offset:offset+self.chunk_pixels].astype(np.uint8)
            order=np.argsort(-scores,kind="stable")
            scores,labels=scores[order],labels[order]
            starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
            counts=np.diff(np.r_[starts,len(scores)])
            positive=np.add.reduceat(labels.astype(np.uint64),starts)
            grouped=np.empty(len(starts),dtype=DTYPE)
            grouped["score"],grouped["count"],grouped["positive"]=scores[starts],counts,positive
            path=self.directory/f"run_{len(self.paths):06d}.bin"
            with path.open("xb") as f:
                grouped.tofile(f)
            self.paths.append(path)
            self.positives+=int(labels.sum()); self.pixels+=len(labels)

    def compute(self):
        if self.finalized:
            raise RuntimeError("AP accumulator already finalized")
        self.finalized=True
        if not self.positives:
            return None
        paths=list(self.paths)
        level=0
        while len(paths)>self.fan_in:
            merged=[]
            for group_id,start in enumerate(range(0,len(paths),self.fan_in)):
                path=self.directory/f"merge_{level}_{group_id}.bin"
                source=_merge(paths[start:start+self.fan_in])
                with path.open("xb") as f:
                    while True:
                        block=list(itertools.islice(source,4096))
                        if not block:
                            break
                        np.array(block,dtype=DTYPE).tofile(f)
                merged.append(path)
            level+=1; paths=merged
        tp=seen=0
        ap=0.
        for _,count,positive in _merge(paths):
            tp+=positive; seen+=count
            ap+=(positive/self.positives)*(tp/seen)
        return float(ap)


def new_fire_metrics(probability,target,observed_fire,threshold):
    """Score ignition candidates only: pixels not detected burning at J.

    At these candidates Persistence is identically zero. Also report the
    number of genuinely new target positives, their TP and FN explicitly.
    """
    from geofirex.unet_evaluation import score
    p=np.asarray(probability); y=np.asarray(target); observed=np.asarray(observed_fire)
    if p.shape!=y.shape or p.shape!=observed.shape:
        raise ValueError("Matching probability, target and J-fire shapes required")
    candidate=~observed.astype(bool)
    if not candidate.any():
        return {"candidate_pixels":0,"new_positive_pixels":0,"metrics":None}
    metrics=score(p[candidate],y[candidate],threshold)
    return {"candidate_pixels":int(candidate.sum()),"new_positive_pixels":metrics["positive_pixels"],
            "new_tp":metrics["tp"],"new_fn":metrics["fn"],"metrics":metrics}


def rank_validation_candidates(candidates):
    """Validation AP only; reject incomparable folds/data/targets/channels.

    This primitive ranks individual completed runs. Aggregate seeds before
    architecture selection as specified in the benchmark protocol.
    """
    if not candidates:
        raise ValueError("No candidates")
    required=("protocol_version","dataset_hash","fold_id","evaluation_target_hash","feature_policy")
    reference=tuple(candidates[0][k] for k in required)
    for c in candidates:
        if tuple(c[k] for k in required)!=reference:
            raise ValueError("Candidates do not share the same evaluation protocol")
        value=c["validation_average_precision"]
        if c["status"]!="FINISHED" or value is None or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError("Only finished runs with defined validation AP can be ranked")
    return sorted(candidates,key=lambda c:(-c["validation_average_precision"],c["run_id"]))


def rank_architectures(candidates,expected_seeds=(42,43,44)):
    """Require the predeclared equal seed set; never choose a lucky single seed."""
    ranked=rank_validation_candidates(candidates)
    groups={}
    for c in ranked:
        key=(c["model"],c["context_days"])
        seeds=groups.setdefault(key,{})
        if c["seed"] in seeds:
            raise ValueError("Duplicate model/context/seed candidate")
        seeds[c["seed"]]=c
    summaries=[]
    for (model,context),seeds in groups.items():
        if set(seeds)!=set(expected_seeds):
            raise ValueError("Each architecture must have exactly the predeclared seeds")
        ap=[seeds[s]["validation_average_precision"] for s in sorted(seeds)]
        summaries.append({"model":model,"context_days":context,
                          "mean_validation_average_precision":float(np.mean(ap)),
                          "std_validation_average_precision":float(np.std(ap,ddof=1)) if len(ap)>1 else 0.,
                          "run_ids":[seeds[s]["run_id"] for s in sorted(seeds)]})
    return sorted(summaries,key=lambda c:(-c["mean_validation_average_precision"],c["model"],c["context_days"]))
