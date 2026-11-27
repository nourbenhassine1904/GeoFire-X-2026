"""Strict C36 geometry, streaming train-only stats and CPU assembly primitives.

No real overlay is built before a complete causal dossier has passed. Geometry
never substitutes for causality; the base C8 target and masks remain authoritative.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from rasterio.crs import CRS
from affine import Affine
from rasterio.transform import array_bounds

COMMON_C36 = (5,6,12,13,14,15,16,17)


def geometry(height, width, crs, transform):
    affine = Affine(*tuple(transform)[:6])
    if (type(height) is not int or type(width) is not int or height < 1 or width < 1
            or not np.isfinite(tuple(affine)).all() or affine.determinant == 0 or not crs):
        raise ValueError('Invalid raster geometry')
    return dict(height=height,width=width,crs=CRS.from_user_input(crs).to_wkt(),
                transform=list(affine),resolution=[math.hypot(affine.a,affine.d),
                math.hypot(affine.b,affine.e)],bounds=list(array_bounds(height,width,affine)))


def require_exact_geometry(source, destination):
    # Compare normalized CRS and every affine coefficient; no visual/shape-only
    # or approximate tolerance. Alignment/reprojection is an explicit prior step.
    a=geometry(source['height'],source['width'],source['crs'],source['transform'])
    b=geometry(destination['height'],destination['width'],destination['crs'],destination['transform'])
    if a != b:
        raise ValueError('Exact dimensions/CRS/affine/resolution/bounds differ')
    return a


def require_coverage(expected, supplied):
    keys=list(supplied)
    if len(keys)!=len(set(keys)) or keys!=list(expected):
        raise ValueError('Overlay keys must match base keys exactly in order')


class TrainMoments:
    """Population moments merged in float64, once per allowlisted train input.

    This accumulator receives finite-pixel blocks and is independent of I/O.
    The production orchestrator must first validate causal evidence and source
    identity. It must call add once per complete unpadded canonical input array.
    """
    def __init__(self, train_paths):
        paths=list(train_paths)
        if not paths or len(paths)!=len(set(paths)) or any('2021' in p.replace('\\','/').split('/') for p in paths):
            raise ValueError('Unique non-test train paths required')
        self.expected=set(paths);self.seen=set();self.count=np.zeros(5,np.int64)
        self.mean=np.zeros(5,np.float64);self.m2=np.zeros(5,np.float64)

    def add(self, path, values):
        self.add_chunks(path, [values])

    def add_chunks(self, path, blocks):
        if path not in self.expected:
            raise ValueError('Validation/test/embargo/unknown input cannot fit statistics')
        if path in self.seen:
            raise ValueError('Repeated input cannot contribute twice')
        had_block=False
        for values in blocks:
            had_block=True;a=np.asarray(values)
            if a.ndim!=3 or a.shape[0]!=5 or min(a.shape[1:])<1:
                raise ValueError('Five unpadded physical feature grids required')
            for i in range(5):
                v=a[i][np.isfinite(a[i])].astype(np.float64)
                n=len(v)
                if not n:continue
                avg=float(v.mean());m2=float(np.square(v-avg).sum());total=int(self.count[i])+n
                delta=avg-self.mean[i]
                self.m2[i]+=m2+delta*delta*int(self.count[i])*n/total
                self.mean[i]+=delta*n/total;self.count[i]=total
        if not had_block:raise ValueError('Empty raster input')
        self.seen.add(path)

    def result(self, *, causal_receipt, base_normalization_sha256, source_sha256):
        if self.seen!=self.expected or (self.count==0).any():
            raise ValueError('Incomplete train population or all-missing feature')
        if (causal_receipt.get('status')!='REVIEWED_EVIDENCE_CONSISTENCY_PASS'
                or set(source_sha256)!=self.expected):
            raise ValueError('Causal receipt and exact train source provenance required')
        std=np.sqrt(self.m2/self.count);std[std<1e-6]=1.
        return dict(schema='geofirex-c36-extra-normalization-1',continuous=[0,1,2,3,4],
            mean=self.mean.tolist(),std=std.tolist(),finite_count=self.count.tolist(),
            fit_paths=sorted(self.expected),source_sha256=source_sha256,
            base_normalization_sha256=base_normalization_sha256,
            causal_evidence_sha256=causal_receipt['evidence_sha256'],
            std_policy='population; std<1e-6 -> 1',clip=8.,padding_included=False)


def assemble_c36(base_x, extra_raw, landcover, normalizer):
    """Names follow the frozen profile. Base is never renormalized or remasked."""
    base=np.asarray(base_x);raw=np.asarray(extra_raw);lc=np.asarray(landcover)
    if (base.dtype!=np.float32 or base.ndim!=4 or base.shape[:2]!=(3,8)
            or raw.shape!=(3,5,*base.shape[-2:]) or lc.shape!=(3,*base.shape[-2:])
            or not np.isfinite(base).all()):
        raise ValueError('Expected T3 base8 and aligned five extra grids plus LC')
    mean=np.asarray(normalizer['mean'],np.float32);std=np.asarray(normalizer['std'],np.float32)
    if (mean.shape!=(5,) or std.shape!=(5,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std<=0) or normalizer['clip']!=8.):
        raise ValueError('Invalid five-feature normalization')
    finite=np.isfinite(raw);mean=mean[None,:,None,None];std=std[None,:,None,None]
    z=np.clip((np.where(finite,raw,mean)-mean)/std,-8.,8.)
    known=np.isfinite(lc)&(lc>=1)&(lc<=17)&(lc==np.floor(lc))
    out=np.zeros((3,36,*base.shape[-2:]),np.float32)
    out[:,:5]=z;out[:,7:12]=~finite
    out[:,COMMON_C36]=base
    for i in range(1,18):out[:,17+i]=(lc==i)&known
    out[:,35]=~known
    if not np.isfinite(out).all():raise ValueError('Nonfinite C36 features')
    return out


def parity_tile(key, base, candidate):
    """Inputs are (x,y,valid,observed_J); checks never execute a model."""
    bx,by,bv,bo=map(np.asarray,base);cx,cy,cv,co=map(np.asarray,candidate)
    if (cx.shape!=(3,36,*bx.shape[-2:]) or bx.shape[:2]!=(3,8)
            or not np.isfinite(cx).all() or cx.dtype!=bx.dtype):
        raise ValueError('C36 feature schema or dtype differs')
    if not all(np.array_equal(a,b) and a.dtype==b.dtype for a,b in ((by,cy),(bv,cv),(bo,co))):
        raise ValueError('Target/valid/observed_J parity failed: '+str(key))
    if not np.array_equal(bx,cx[:,COMMON_C36]):
        raise ValueError('Shared C8 channels changed')
    if (not all(np.isin(a,[0,1]).all() for a in (by,bv,bo))
            or by.shape!=bv.shape or by.shape!=bo.shape or by.shape!=(1,*bx.shape[-2:])):
        raise ValueError('Binary target and common mask shapes required')
    if not np.array_equal(bo,bx[-1,7:8]) or not np.array_equal(co,cx[-1,17:18]):
        raise ValueError('Explicit observed_J differs from last input fire channel')
    # Match V2Metrics exactly: shape here is spatial [H,W], arrays serialized bool.
    identity=hashlib.sha256(json.dumps([key,list(by.shape[-2:])]).encode())
    for a in (by,bv,bo):identity.update(a.astype(bool).tobytes())
    return identity.hexdigest()


def parity_stream(expected_keys, base_tiles, candidate_tiles):
    """Exhaustive paired iterator; O(keys) digests, O(one tile) tensor memory."""
    from itertools import zip_longest
    domains={};sentinel=object()
    for key,b,c in zip_longest(expected_keys,base_tiles,candidate_tiles,fillvalue=sentinel):
        if any(v is sentinel for v in (key,b,c)) or key in domains:
            raise ValueError('Incomplete or duplicate paired coverage')
        if not EVENT_KEY(key):raise ValueError('Only V2 development keys allowed')
        domains[key]=parity_tile(key,b,c)
    if not domains:raise ValueError('Empty paired coverage')
    return dict(status='PASS',tiles=len(domains),
        evaluation_domain_sha256=hashlib.sha256(json.dumps(sorted(domains.items())).encode()).hexdigest())


def EVENT_KEY(key):
    from geofirex.v2_c36_readiness import EVENT
    return EVENT.fullmatch(key[0]) and key[1]<'2021-01-01'


def fit_extra_statistics(contract, raw_root, evidence, evidence_root, base_sha256, overlay_receipt=None):
    """Future callable: guard causal proof BEFORE any raster open; train only.

    Not invoked on real data while causal evidence is missing. Uses bounded
    row windows, checks source bytes and geometry before statistical fitting.
    """
    import rasterio
    from rasterio.windows import Window
    from geofirex.protocol_v2 import sha256
    from geofirex.v2_c36_readiness import validate_causal_evidence, json_digest
    receipt=validate_causal_evidence(contract,evidence,evidence_root)
    if (not overlay_receipt or overlay_receipt.get('status')!='EXACT_CAUSAL_OVERLAY_PASS'
            or overlay_receipt.get('index_sha256')!=contract.index_sha256
            or overlay_receipt.get('causal_evidence_sha256')!=receipt['evidence_sha256']
            or overlay_receipt.get('train_input_paths_sha256')!=json_digest(sorted(contract.inputs['train']))):
        raise ValueError('Exact causal overlay qualification must precede real statistics')
    base_path=contract.index/'normalization.json'
    if sha256(base_path)!=base_sha256:
        raise ValueError('Base normalizer identity differs')
    moments=TrainMoments(contract.inputs['train']);sources={}
    for relative,obs in sorted(contract.inputs['train'].items()):
        path=contract.input_path(raw_root,relative,role='train')
        digest=sha256(path)
        if digest!=obs['sha256']:raise ValueError('Certified source raster bytes changed')
        sources[relative]=digest
        with rasterio.open(path) as ds:
            require_exact_geometry(dict(height=ds.height,width=ds.width,crs=ds.crs,
                                        transform=ds.transform),obs)
            if ds.count!=23:raise ValueError('Certified 23-band source required')
            if any(v!=1 for v in ds.scales[:5]) or any(v!=0 for v in ds.offsets[:5]):
                raise ValueError('Nonidentity scales/offsets require a qualified physical-units conversion')
            def blocks():
                for row in range(0,ds.height,128):
                    a=ds.read([1,2,3,4,5],window=Window(0,row,ds.width,min(128,ds.height-row)),
                              out_dtype='float32',masked=True)
                    yield a.filled(np.nan)
            moments.add_chunks(relative,blocks())
    return moments.result(causal_receipt=receipt,base_normalization_sha256=base_sha256,
                          source_sha256=sources)
