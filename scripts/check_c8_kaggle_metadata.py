"""Read-only metadata requests for the eleven existing datasets. No download."""
import contextlib
import io
import json
from pathlib import Path
import socket
from concurrent.futures import ThreadPoolExecutor,as_completed
from geofirex.kaggle_persistence import KaggleBackend

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    socket.setdefaulttimeout(20)
    root=Path(__file__).resolve().parents[1]
    handles=json.loads((root/'reports/protocol_v2_execution/dataset-sources.json').read_text())
    def check(handle):
        try:
            result=KaggleBackend().metadata(handle)
            return dict(dataset=handle,status='AVAILABLE' if result is not None else 'NOT_FOUND',metadata=result)
        except Exception as error:
            # Never serialize auth exceptions/URLs/tokens. Existing login only.
            return dict(dataset=handle,status='UNVERIFIED',error_type=type(error).__name__)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        with ThreadPoolExecutor(max_workers=3) as executor:
            rows=[future.result() for future in as_completed([executor.submit(check,h) for h in handles])]
    result=dict(read_only=True,downloads=0,mutations=0,datasets=sorted(rows,key=lambda r:r['dataset']),
        all_available=all(r['status']=='AVAILABLE' for r in rows))
    target=args.out or root/'reports/c8_propagation_20261121_v1/kaggle_metadata_current.json'
    if target.exists():raise FileExistsError('Preserve earlier metadata receipt')
    with target.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
