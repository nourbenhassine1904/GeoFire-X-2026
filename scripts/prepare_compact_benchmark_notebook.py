"""Ready notebook for the reviewed 11 private compact partitions, no submission."""
import base64,io,json,zipfile
from pathlib import Path
from geofirex.mission_streaming import save,sha

def main():
    root=Path('reports/gpu_acceleration_20261111_v1');plan=json.loads((root/'publication/plan.json').read_text());buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in Path('src/geofirex').glob('*.py'):z.write(p,p.as_posix())
        for name in ('train_compact_benchmark.py','compact_publication.py','profile_compact_loader.py','profile_compact_180.py'):z.write(Path('scripts')/name,name)
        z.write('reports/mission_20261111/b0_v2/results.json','b0_existing.json')
    payload=base64.b64encode(buf.getvalue()).decode();cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))
    md('# GeoFire-X — B1/B2 sur le corpus compact intégral\nAprès publication privée autorisée et vérifiée : attacher les 11 datasets listés dans dataset-sources.json. Aucune reconstruction, aucun B0 relancé. T=3, seed42, 250 train, validation 2020 par événements complets. Les scores sont comparés aux B0 existants sur le même domaine. Les campagnes sont désactivées tant que leur budget GPU n’est pas autorisé.')
    code('import base64,io,zipfile,sys,json,time,tempfile\nfrom pathlib import Path\nworkspace=Path("/kaggle/working/compact-benchmark-"+time.strftime("%Y%m%d_%H%M%S"));workspace.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode('+repr(payload)+'))) as z:z.extractall(workspace)\nsys.path[:0]=[str(workspace/"src"),str(workspace)]\nfrom compact_publication import restore\nexpected='+repr({p['identity'] for p in plan['parts']})+'\nmaps=list(Path("/kaggle/input").rglob("mount-map.json"))\nassert {json.loads(p.read_text())["identity"] for p in maps}==expected, "Partitions manquantes ou différentes"\ncorpus=restore([p.parent for p in maps],Path(tempfile.mkdtemp(prefix="geofirex-compact-",dir="/tmp"))/"corpus",verify=True)\nindex=corpus/"index"\nfrom geofirex.fast_compact import FastCompactTiles\ntrain,val=FastCompactTiles(corpus,index,"train"),FastCompactTiles(corpus,index,"val")\nassert len(train)==93603 and len(val)==78771\nprint("Verified",len(train),len(val),"tiles; archives linked, not extracted")')
    md('Avant une campagne : profiler le chargeur compact sur cette T4 avec les partitions effectivement montées. Les débits mesurés sur GeoTIFF et sur disque local ne sont pas présentés comme un débit du compact Kaggle. Commencer par une époque B1, puis une époque B2, mêmes 32 événements val sélectionnés par métadonnées, toutes leurs tuiles sans rééquilibrage. Passer à 201 événements après mesure du temps et du scratch AP.')
    code('import torch\nfrom profile_compact_180 import run as profile_compact\nRUN_PROFILE=False # Activer seulement pour le profil court, apres certificat CPU PASS.\nif RUN_PROFILE:\n    print(json.dumps(profile_compact(corpus,index,Path("/kaggle/working")/("compact-profile-"+time.strftime("%Y%m%d_%H%M%S")),max_seconds=180),indent=2))')
    code('import argparse,torch\nfrom train_compact_benchmark import run\nassert torch.cuda.is_available()\nprint(torch.cuda.get_device_name(0))\nRUN_B1=False\nRUN_B2=False\nfor model,enabled in (("B1",RUN_B1),("B2",RUN_B2)):\n    if enabled:\n        config=argparse.Namespace(corpus=str(corpus),index=str(index),out=str(Path("/kaggle/working")/(model+"-compact-"+time.strftime("%Y%m%d_%H%M%S"))),model=model,device="cuda",workers=4,batch_size=16,epochs=1,validation_events=32,seconds=5400,resume=None,max_updates=0,b0=str(workspace/"b0_existing.json"))\n        result=run(config)\n        print(json.dumps(result,indent=2))')
    md('Reprise : conserver le dernier checkpoint (toutes les 100 mises à jour et avant validation), puis définir `config.resume` sur ce fichier et une nouvelle sortie. Le curseur est celui des échantillons effectivement entraînés, pas celui des batches préchargés. Modèle, optimiseur, RNG et accumulateurs de loss sont restaurés ; changement de modèle, seed, cibles, normalisation, batch ou périmètre val refusé. Les sorties incluent loss, AP exacte, Dice/IoU/précision/rappel, PNG/NPZ réels et B0 agrégé depuis les résultats historiques. Aucun score officiel n’est publié si la validation est interrompue.')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(id=f'compact-{i}',**c) for i,c in enumerate(cells)])
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],'notebook','exec')
    (root/'B1_B2_COMPACT_CERTIFIED_READY.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print('B1/B2 notebook prepared, all code cells compile')

if __name__=='__main__':main()
