"""Independent saved-score CPU analysis. Default: resource check only."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--archive',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args()
    from geofirex.newpos4_analysis import ResourceGuard,analysis
    parent=a.out.parent if a.out else a.root/'reports'
    guard=ResourceGuard(parent)
    print(json.dumps(guard.receipt(),indent=2),flush=True)
    if a.execute:
        if not a.archive or not a.out:p.error('execute requires existing archive and new out')
        print(analysis(a.root,a.archive,a.out))
    else:print('RESOURCE_CHECK_ONLY: no score calibration, GPU, inference or training')


if __name__=='__main__':main()
