"""Small code/metadata job only. No upload of compact or raw corpus."""
import base64
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from geofirex.fast_compact import FastCompactTiles
from geofirex.mission_streaming import save,sha

def main():
    root=Path('reports/gpu_acceleration_20261111_v1');job=root/'short_gpu_job_v3';job.mkdir(exist_ok=False)
    lots=Path('reports/kaggle_remaining_20261108_v1/training/bundle/metadata');train=sorted(set(json.loads((lots/'lot-001.json').read_text())['events'])|set(json.loads((lots/'lot-002.json').read_text())['events']))
    val=['2020/fire_23654679','2020/fire_24104674'];ds=FastCompactTiles('reports/mission_20261111/compact_full',root/'index','val')
    diagnostic=[]
    for e in val:
        candidates=[i for i in np.flatnonzero(ds.records['category']==1) if ds.event(int(i))==e]
        for i in candidates[:6]:
            from datetime import date
            r=ds.records[i];diagnostic.append([e,date.fromordinal(int(r['day'])).isoformat(),int(r['row']),int(r['col'])])
    scope=dict(train=train,val=val,diagnostic=diagnostic)
    if not (root/'scope119.json').exists():save(root/'scope119.json',scope)
    buf=io.BytesIO();prepared=Path('reports/full_corpus_v1/prepared_spark_v1')
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in Path('src/geofirex').glob('*.py'):z.write(p,p.as_posix())
        z.write('scripts/profile_acceleration_gpu.py','profile_acceleration_gpu.py')
        for name in ('observations.parquet','windows.parquet'):
            table=pq.read_table(prepared/name);rows=[r for r in table.to_pylist() if r['split'] in ('train','val')]
            if name=='observations.parquet':
                for row in rows:row['sha256']=None
            else:rows=[r for r in rows if r['context'] in (3,5)]
            b=io.BytesIO();pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),b,compression='gzip');z.writestr('metadata/'+name,b.getvalue())
        z.write(prepared/'normalization.json','metadata/normalization.json');z.writestr('scope.json',json.dumps(scope))
    payload=base64.b64encode(buf.getvalue()).decode()
    code='''import base64,io,zipfile,json,sys,time\nfrom pathlib import Path\nif __name__=="__main__":\n    workspace=Path("/kaggle/working/acceleration-code");workspace.mkdir(exist_ok=False)\n    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(workspace)\n    sys.path[:0]=[str(workspace/"src"),str(workspace)]\n    from profile_acceleration_gpu import run\n    roots=sorted({p.parent for p in Path("/kaggle/input").rglob("2018") if p.is_dir()}|{p.parent for p in Path("/kaggle/input").rglob("2020") if p.is_dir()})\n    scope=json.loads((workspace/"scope.json").read_text())\n    result=run(workspace/"metadata",scope,roots,"/kaggle/working/profile-results",max_seconds=180,steps=4)\n    print(json.dumps(result))\n'''.replace('PAYLOAD',repr(payload))
    (job/'job.py').write_text(code,encoding='utf-8');compile(code,'job.py','exec')
    if (job/'job.py').stat().st_size>=1_000_000:raise ValueError('Kaggle source limit')
    save(job/'kernel-metadata.json',dict(id='nnnnbhss/geofirex-loader-profile-20261111-v3',title='geofirex-loader-profile-20261111-v3',code_file='job.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=['nnnnbhss/geofirex-wsts-fold0-tv-001-v1','nnnnbhss/geofirex-wsts-fold0-tv-002-v1','nnnnbhss/geofire-x-wildfirespreadts-sample'],kernel_sources=[],competition_sources=[]))
    save(job/'review.json',dict(code_bytes=(job/'job.py').stat().st_size,code_sha256=sha(job/'job.py'),gpu_limit_seconds=300,script_budget_seconds=180,train_events=len(train),excluded_event='2019/fire_23300839',data_upload_bytes=0,raw_or_compact_rebuilt=False))
    print(job)

if __name__=='__main__':main()
