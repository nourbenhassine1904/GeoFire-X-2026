"""C36 development CPU checks, default metadata only. No training command.

Uses existing D: TIFFs in place. An absent causal dossier keeps all real overlay
construction/statistics/parity blocked; header geometry and fixtures are safe.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import rasterio

from geofirex.protocol_v2 import sha256
from geofirex.unet_data import validate_layout
from geofirex.v2_c36_readiness import V2Contract, frozen_inputs, validate_causal_evidence
from geofirex.v2_c36_overlay import require_exact_geometry


def save(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def preserved(root):
    # Reuse the existing inventory, not another general filesystem audit.
    previous=json.loads((root/'reports/c36_feasibility_20261121_v1/preservation_before.json').read_text())
    paths=set(previous['sha256'])
    audit=root/'reports/c36_feasibility_20261121_v1'
    paths.update(p.relative_to(root).as_posix() for p in audit.iterdir() if p.is_file())
    return {name:sha256(root/name) for name in sorted(paths)}


def run(root,raw_root,out,headers=False,evidence_path=None):
    root=root.resolve();raw_root=raw_root.resolve();out=out.resolve()
    if not out.is_relative_to(root/'reports') or out.exists():
        raise ValueError('New repository reports directory required; never overwrite a prior run')
    git=['git','-c','safe.directory='+root.as_posix()]
    branch=subprocess.check_output(git+['branch','--show-current'],cwd=root,text=True).strip()
    if branch in ('main','master',''):
        raise ValueError('Dedicated development branch required')
    frozen=frozen_inputs(root)
    contract=V2Contract.from_repository(root)
    drives={name:dict(shutil.disk_usage(path)._asdict()) for name,path in [('working',root),('raw',raw_root)]}
    # This command writes header metadata only, never a multi-GB overlay.
    if any(d['free']<2*1024**3 for d in drives.values()):
        raise RuntimeError('At least 2 GiB free required for CPU metadata checks')
    counts={}
    for role in ('train','val'):
        missing=[];size_mismatch=[]
        for relative,obs in contract.inputs[role].items():
            p=contract.input_path(raw_root,relative,role)
            if not p.is_file():missing.append(relative)
            elif p.stat().st_size!=obs['bytes']:size_mismatch.append(relative)
        counts[role]=dict(expected=len(contract.inputs[role]),missing=missing,size_mismatch=size_mismatch)
    if any(c['missing'] or c['size_mismatch'] for c in counts.values()):
        raise ValueError('Raw availability/size differs: '+str(counts))
    snapshot=preserved(root)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'preparation_before.json',dict(branch=branch,
        git_status=subprocess.check_output(git+['status','--short'],cwd=root,text=True),
        sha256=snapshot,frozen=frozen,disks=drives,raw_paths=counts,
        metadata_only_required_free_bytes=2*1024**3,
        future_overlay_and_runtime_space_bytes=2*14283644928+16*1024**3))
    save(out/'expected_contract.json',contract.summary())
    if evidence_path is None:
        causal=dict(status='BLOCKED',reason='No reviewed exhaustive per-asset evidence dossier supplied',
            expected_added_band_keys=len(contract.required_keys()),provided_keys=0,
            inherited_C8='Frozen terrain/fire channels; fire delivery remains a conditional historical assumption',
            products={'0,1,2':'VNP09GA: acquisition and actual publication unproven',
                      '3,4':'VNP13A1: composite end, exact vintage and actual publication unproven',
                      '16':'MCD12Q1: prior published year/version and pixel binding unproven'})
    else:
        try:
            doc=json.loads(evidence_path.read_text(encoding='utf8'))
            causal=validate_causal_evidence(contract,doc,evidence_path.parent)
        except (ValueError,KeyError,OSError) as error:
            causal=dict(status='BLOCKED',reason=str(error))
    save(out/'causal_status.json',causal)
    if headers:
        started=time.monotonic();rows=[];errors=[];tag_keys=set()
        for role in ('train','val'):
            for relative,obs in sorted(contract.inputs[role].items()):
                path=contract.input_path(raw_root,relative,role)
                try:
                    with rasterio.open(path) as ds:
                        validate_layout(ds)
                        g=require_exact_geometry(dict(height=ds.height,width=ds.width,
                            crs=ds.crs,transform=ds.transform),obs)
                        if list(ds.res)!=g['resolution'] or list(ds.bounds)!=g['bounds']:
                            raise ValueError('Dataset resolution/bounds disagree with canonical grid')
                        tags=ds.tags();tag_keys.update(tags)
                        rows.append(dict(role=role,relative_path=relative,status='PASS',
                            height=ds.height,width=ds.width,count=ds.count,
                            certified_sha256=obs['sha256'],nodata=repr(ds.nodata),
                            mask_flags=repr(ds.mask_flag_enums),tag_keys=','.join(sorted(tags)),
                            scales=repr(ds.scales),offsets=repr(ds.offsets),units=repr(ds.units)))
                except (ValueError,rasterio.errors.RasterioError) as error:
                    errors.append(dict(relative_path=relative,error=str(error)))
                if (len(rows)+len(errors))%500==0:
                    print('Headers checked',len(rows)+len(errors),flush=True)
        with (out/'geometry_headers.csv').open('x',newline='',encoding='utf8') as f:
            if rows:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(out/'geometry_headers.json',dict(status='PASS' if not errors else 'FAIL',
            checked=len(rows)+len(errors),passed=len(rows),errors=errors,
            elapsed_seconds=time.monotonic()-started,tag_keys=sorted(tag_keys),
            pixel_reads=0,note='Exact original input headers; not a constructed C36 overlay or causal certificate'))
    after=preserved(root)
    changed=[name for name,digest in snapshot.items() if after.get(name)!=digest]
    save(out/'preparation_after.json',dict(status='PASS' if not changed else 'FAIL',
        checked=len(snapshot),changed=changed,frozen=frozen_inputs(root),
        pixel_reads=0,test_pixels_opened=0,gpu_used=False,training_executed=False))
    if changed:raise ValueError('Pre-existing files changed: '+str(changed))
    save(out/'decision.json',dict(status='NO_GO',causal_status=causal['status'],
        real_overlay='BLOCKED_PENDING_CAUSAL_EVIDENCE',real_normalization='BLOCKED_PENDING_CAUSAL_EVIDENCE',
        real_C36_parity='BLOCKED_PENDING_CAUSAL_OVERLAY_AND_NORMALIZATION',
        runner_prepared=False,training_authorized=False,test_2021_reserved=True))
    print(json.dumps(dict(branch=branch,output=str(out),raw_inputs=sum(len(v) for v in contract.inputs.values()),
        causal_status=causal['status'],frozen_files=frozen['files'],preserved_files=len(snapshot)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--raw-root',type=Path,default=Path('D:/GeoFire-X/WildfireSpreadTS/rasters'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--headers',action='store_true',help='Inspect all 6913 allowlisted headers; never read pixels')
    p.add_argument('--evidence',type=Path)
    a=p.parse_args();run(a.root,a.raw_root,a.out,a.headers,a.evidence)
