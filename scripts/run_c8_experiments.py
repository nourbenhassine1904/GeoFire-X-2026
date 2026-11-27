"""Default is metadata-only CPU preflight. All runtime actions need a grant."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',nargs='?',default='preflight',choices=['preflight','gpu-preflight','threshold-infer','threshold-analyze','newpos4-fit','newpos4-evaluate','restore','backup'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--maps',type=Path,nargs='+');p.add_argument('--out',type=Path)
    p.add_argument('--authorization',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--predictions',type=Path)
    p.add_argument('--training-source',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--sidecar',type=Path);p.add_argument('--local-root',type=Path);p.add_argument('--external-root',type=Path)
    a=p.parse_args()
    # Mask GPU before torch import. Metadata default cannot query or use CUDA.
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0' if a.device=='cuda' else ''
    from geofirex import c8_experiments as c
    if a.action=='preflight':result=c.preflight(a.root,a.maps)
    elif a.action=='restore':
        if not all((a.archive,a.sidecar,a.out)) or a.device!='cpu':p.error('restore requires archive, sidecar, fresh out and CPU')
        result=str(c.restore_candidate(a.root,a.archive,a.sidecar,a.out))
    elif a.action=='backup':
        if not all((a.archive,a.sidecar,a.local_root,a.external_root)):p.error('backup requires archive, sidecar and two existing destinations')
        result=c.old.backup(a.archive,a.sidecar,a.local_root,a.external_root)
    else:
        if not a.authorization or not a.out:p.error('runtime action requires authorization and new output path')
        if a.action=='gpu-preflight':
            if a.device!='cuda':p.error('GPU preflight requires explicit --device cuda')
            c.authorize(a.root,a.authorization,'gpu-preflight','cuda',a.out.parent)
            result=c.preflight(a.root,a.maps,a.out.parent,'cuda',gpu=True);c.save(a.out,result)
        elif a.action=='threshold-infer':
            if not a.maps or not a.archive:p.error('inference requires maps and official COMPLETE archive')
            result=str(c.infer_thresholds(a.root,a.maps,a.archive,a.out,a.authorization,a.device))
        elif a.action=='threshold-analyze':
            if not a.predictions or a.device!='cpu':p.error('threshold analysis requires predictions and CPU')
            result=str(c.analyze_thresholds(a.root,a.predictions,a.out,a.authorization))
        elif a.action=='newpos4-fit':
            if not a.maps or a.device!='cuda':p.error('fit requires maps and --device cuda')
            result=str(c.fit(a.root,a.maps,a.out,a.authorization))
        else:
            if not a.maps or not a.training_source or a.device!='cuda':p.error('recovery requires maps, training-source and CUDA')
            result=str(c.evaluate_candidate(a.root,a.maps,a.training_source,a.out,a.authorization))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
