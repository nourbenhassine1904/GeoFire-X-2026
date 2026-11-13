"""Reuse the prepared standalone notebook; add one bounded CPU PyTorch window."""
import ast
import hashlib
import json
from pathlib import Path

original=Path('reports/kaggle_prepublication_20261107_v2/pilot/cpu-certification/lot-001')
target=Path('reports/kaggle_pilot_001_20261107_v1/prepared-cpu/cpu-certification/lot-001')
target.mkdir(parents=True,exist_ok=False)
notebook=json.loads((original/'certify.ipynb').read_text())
cell=next(cell for cell in notebook['cells'] if cell['cell_type']=='code')
source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
addition='''
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
class SingleWindow(Dataset):
    def __len__(self): return 1
    def __getitem__(self, index):
        if index != 0: raise IndexError(index)
        import numpy as np
        import rasterio
        from rasterio.windows import Window
        lot=json.loads(canonical.read_text())
        path=namespace['safe_child'](dataset/'rasters',lot['files'][0]['path'])
        with rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_NUM_THREADS='1'):
            with rasterio.open(path) as raster:
                window=raster.read(1,window=Window(0,0,min(64,raster.width),min(64,raster.height)),masked=True)
                values=np.asarray(window.filled(0),dtype=np.float32).copy()
        return torch.from_numpy(values)
namespace['resource_check'](__import__('time').monotonic()+60)
batch=next(iter(DataLoader(SingleWindow(),batch_size=1,num_workers=0,pin_memory=False)))
if batch.device.type!='cpu' or batch.numel()==0 or batch.ndim!=3: raise RuntimeError('CPU PyTorch window invalid')
certificate['pytorch_window']=dict(device=str(batch.device),shape=list(batch.shape),dtype=str(batch.dtype),
    workers=0,pin_memory=False,torch_version=torch.__version__,training=False)
'''
marker="(work/'pilot-certificate.json').write_text"
if source.count(marker)!=1:raise ValueError('Prepared notebook marker differs')
source=source.replace(marker,addition+'\n'+marker)
ast.parse(source)
cell['source']=source.splitlines(keepends=True)
payload=json.dumps(notebook,indent=2).encode()
(target/'certify.ipynb').write_bytes(payload)
metadata=json.loads((original/'kernel-metadata.json').read_text())
if metadata['enable_gpu'] or metadata['enable_internet'] or not metadata['is_private'] or metadata['dataset_sources']!=['nnnnbhss/geofirex-wsts-fold0-tv-001-v1']:
    raise ValueError('CPU-only standalone pin differs')
(target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
receipt=dict(status='LOCAL_PREPARATION_ONLY',notebooks=[dict(tag='lot-001',notebook_sha256=hashlib.sha256(payload).hexdigest(),
    sources=metadata['dataset_sources'],kernel=metadata['id'])])
(target.parent/'prepared.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(notebook_bytes=len(payload),sha256=receipt['notebooks'][0]['notebook_sha256'],gpu=False,workers=0,network_calls=0)))
