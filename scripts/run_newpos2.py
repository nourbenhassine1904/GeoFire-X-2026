"""Metadata-only CPU by default; all GPU actions require an external exact grant."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',nargs='?',default='preflight',choices=['preflight','gpu-preflight','fit','evaluate','restore','backup'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--maps',nargs='+',type=Path);p.add_argument('--authorization',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--archive',type=Path);p.add_argument('--sidecar',type=Path)
    p.add_argument('--local-root',type=Path);p.add_argument('--external-root',type=Path)
    a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0' if a.action in ('gpu-preflight','fit','evaluate') else ''
    from geofirex import newpos2_experiment as c
    if a.action=='preflight':result=c.preflight(a.root,a.maps)
    elif a.action=='restore':result=str(c.restore_candidate(a.root,a.archive,a.sidecar,a.out))
    elif a.action=='backup':result=c.old.backup(a.archive,a.sidecar,a.local_root,a.external_root)
    else:
        if not all((a.authorization,a.out,a.maps)):p.error('GPU action requires authorization, new out and eleven maps')
        if a.action=='gpu-preflight':
            c.authorize(a.root,a.authorization,'gpu-preflight','cuda',a.out.parent)
            result=c.preflight(a.root,a.maps,a.out.parent,'cuda',gpu=True);c.save(a.out,result)
        elif a.action=='fit':result=str(c.fit(a.root,a.maps,a.out,a.authorization))
        else:result=str(c.evaluate_candidate(a.root,a.maps,a.source,a.out,a.authorization))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
