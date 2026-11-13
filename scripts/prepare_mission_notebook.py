"""Self-contained notebook: reuse published raw lots + historical val sample.

Embeds code and small metadata, never TIFFs, checkpoints or credentials.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from geofirex.mission_streaming import save

def main():
    out=Path('reports/mission_20261111');prepared=Path('reports/full_corpus_v1/prepared_spark_v1')
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path('src/geofirex').glob('*.py')):z.write(p,p.as_posix())
        for name in ('observations.parquet','windows.parquet'):
            table=pq.read_table(prepared/name);rows=[r for r in table.to_pylist() if r['split'] in ('train','val')];b=io.BytesIO();pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),b,compression='snappy');z.writestr('metadata/'+name,b.getvalue())
        z.write(prepared/'normalization.json','metadata/normalization.json')
    payload=buf.getvalue();digest=hashlib.sha256(payload).hexdigest()
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))
    md('# GeoFire-X — B1 CNN, données réelles, T=3\nNotebook autonome : code + métadonnées intégrés, aucun GeoTIFF intégré. Attacher les datasets déjà publiés `nnnnbhss/geofirex-wsts-fold0-tv-001-v1`, `nnnnbhss/geofirex-wsts-fold0-tv-002-v1` et `nnnnbhss/geofire-x-wildfirespreadts-sample` (validation 2020 historique). Aucun re-upload. Les 119 événements train des deux lots sont un inventaire historique certifié, vérifié à nouveau au montage. La validation disponible dans le sample reste limitée à deux événements : aucune revendication sur les 201. Alternative : attacher le corpus compact après autorisation de transfert.\n\nLe profil effectue de vraies mises à jour puis évalue. Les époques complètes sont dans une cellule séparée, à exécuter après accord sur le budget estimé. Aucun job ne se lance à l’ouverture. Garder les sorties Kaggle avant arrêt de session. Aucun test 2021.')
    code('import base64, hashlib, io, zipfile, sys, json, time, subprocess\nfrom pathlib import Path\npayload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(digest)+'\nworkspace=Path("/kaggle/working/geofirex-mission-code")\nworkspace.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(payload)) as z: z.extractall(workspace)\nsys.path.insert(0,str(workspace/"src"))\nprepared=workspace/"metadata"\nprint("Code+metadata verified:",len(payload),"bytes")')
    code('import torch, numpy as np, rasterio, pyarrow\nassert torch.cuda.is_available(), "Activer le GPU Kaggle; ne pas remplacer torch"\nprint(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))\nprint(subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout)\nprint("Dataset mounts:",[str(p) for p in Path("/kaggle/input").iterdir()])\n# Le quota restant doit être lu dans l’interface Kaggle : il n’est pas déduit du GPU alloué.\n# Aucun téléchargement, transfert ou installation automatique.')
    code('from geofirex.mission_streaming import CompactTiles,save\nfrom geofirex.mission_training import attached_datasets,train\ncompact_roots=[p.parent for p in Path("/kaggle/input").rglob("samples.json") if (p.parent/"normalization.json").is_file()]\nassert len(compact_roots)<=1, "Choisir explicitement un corpus compact si plusieurs sont attachés"\nif compact_roots:\n    corpus=compact_roots[0]\n    train_ds,val_ds=CompactTiles(corpus,"train"),CompactTiles(corpus,"val")\n    provenance={"source":"compact", "root":str(corpus)}\nelse:\n    roots=sorted({p.parent for p in Path("/kaggle/input").rglob("2018") if p.is_dir()} | {p.parent for p in Path("/kaggle/input").rglob("2020") if p.is_dir()})\n    assert roots,"Aucun montage raster identifié"\n    train_ds,val_ds,provenance=attached_datasets(prepared,roots)\nprint("train tiles",len(train_ds),"val tiles",len(val_ds))\nprint("train events",len({s["event_id"] for s in train_ds.samples}),"val events",len({s["event_id"] for s in val_ds.samples}))\nfrom torch.utils.data import DataLoader\nx,y,v=next(iter(DataLoader(train_ds,batch_size=4,num_workers=0)))\nassert x.shape[1:]==(3,8,64,64) and torch.isfinite(x).all()\nprint(x.shape, y.shape, v.shape)')
    code('stamp=time.strftime("%Y%m%d_%H%M%S")\nprofile=train(train_ds,val_ds,Path("/kaggle/working")/("B1-profile-"+stamp),device="cuda",epochs=1,profile_steps=25,wall_seconds=180,provenance=provenance)\nprint(json.dumps(profile,indent=2))\n# Includes full validation time, so conservative for a training-only estimate.\nestimated_epoch_seconds=profile["elapsed_seconds"]*len(train_ds)/max(1,profile["train_samples_seen"])\nprint("Conservative seconds/epoch",estimated_epoch_seconds)')
    md('Exécuter la cellule suivante uniquement après autorisation d’une campagne GPU prolongée et vérification du quota courant. Elle démarre un nouveau B1 avec la même seed : les mises à jour du profil ne sont pas mélangées aux résultats. Chaque époque complète visite toutes les tuiles train une fois ; la validation est exhaustive et non rééquilibrée. Le délai maximum coupe le run avec un statut explicite et conserve le dernier checkpoint produit. Les checkpoints incluent modèle, optimiseur, RNG, normalisation liée et comptes réellement vus.')
    code('RUN_FULL_EPOCHS=False  # Passer à True après autorisation du budget GPU\nif RUN_FULL_EPOCHS:\n    result=train(train_ds,val_ds,Path("/kaggle/working")/("B1-full-"+time.strftime("%Y%m%d_%H%M%S")),device="cuda",epochs=3,wall_seconds=1800,provenance=provenance)\n    print(json.dumps(result,indent=2))\n    from IPython.display import Image,display\n    display(Image(filename=str(sorted(Path("/kaggle/working").glob("B1-full-*/loss.png"))[-1])))')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(id=f'mission-{i:02d}',**c) for i,c in enumerate(cells)])
    path=out/'B1_GPU_FINAL_20261111.ipynb';save(path,notebook)
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],'notebook-cell','exec')
    save(out/'notebook-build-final.json',dict(status='BUILT_AND_ALL_CODE_CELLS_COMPILED',notebook=str(path),embedded_zip_bytes=len(payload),embedded_zip_sha256=digest,raw_data_embedded=False,gpu_executed=False,dataset_sources=['nnnnbhss/geofirex-wsts-fold0-tv-001-v1','nnnnbhss/geofirex-wsts-fold0-tv-002-v1','nnnnbhss/geofire-x-wildfirespreadts-sample'],quota_status='UNKNOWN: Browser host unavailable; native UI stopped because current URL could not be verified'))
    print(path)

if __name__=='__main__':main()
