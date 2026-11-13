"""Offline standalone Kaggle preparation. All training disabled by default."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/loss_ablation_20261112_v1'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip'
    with zipfile.ZipFile(archive) as z:protocol=json.loads(z.read('protocol.json'))
    plan=json.loads((ROOT/'reports/gpu_acceleration_20261111_v1/publication/plan.json').read_text())
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('__init__.py','calibration_io.py','calibration_mounts.py','compact_calibration.py','fast_compact.py','benchmark_models.py','benchmark_metrics.py','unet.py','loss_ablation.py'):
                source=(ROOT/'src/geofirex'/name).read_text(encoding='utf-8')
                if name=='fast_compact.py':source=source.replace('from geofirex.mission_streaming import save,sha','from geofirex.calibration_io import save,sha')
                compile(source,name,'exec');z.writestr('geofirex/'+name,source)
            z.writestr('protocol.json',json.dumps(protocol,indent=2))
            z.writestr('select_loss_ablation.py',(ROOT/'scripts/select_loss_ablation.py').read_text(encoding='utf-8'))
        payload=buffer.getvalue()
    (OUT/'loss-ablation-code.zip').write_bytes(payload)
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s):
        compile(s,'loss-ablation-notebook','exec');cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))
    md('''# GeoFire-X — Ablation indépendante de pos_weight
Poids préenregistrés : **100 (contrôle), 30, 10**, pour CNN B1 et U-Net B2.
Chaque variante initialise un nouveau modèle, seed 42, sans charger les checkpoints
historiques. Budget fixe : 4 époques / 23 404 updates, batch train 16, AdamW identique,
Dice weight 0,5, corpus et normalisation inchangés. Les données déjà calibrées ne
sont pas recalibrées par ce notebook : seuls les nouveaux modèles seront évalués.

Attacher les **11 partitions compactes existantes**. Aucun dataset ni checkpoint
supplémentaire à transférer. GPU T4, Internet désactivé. L'entraînement reste désactivé
par défaut ; ne l'activer qu'après autorisation du budget. Une seule variante par
lancement. La limite de temps sauvegarde un checkpoint de reprise dans son nouveau run.
''')
    code('''import base64, hashlib, io, json, os, sys, tempfile, time, zipfile, subprocess
from pathlib import Path
workspace=Path(tempfile.mkdtemp(prefix="geofirex-loss-code-",dir="/tmp"))
payload=base64.b64decode('''+repr(base64.b64encode(payload).decode())+''')
assert hashlib.sha256(payload).hexdigest()=='''+repr(hashlib.sha256(payload).hexdigest())+'''
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    z.extractall(workspace)
sys.path.insert(0,str(workspace))
os.environ["PYTHONPATH"]=str(workspace)+os.pathsep+os.environ.get("PYTHONPATH", "")
from geofirex.calibration_mounts import restore_verified
expected='''+repr([p['identity'] for p in plan['parts']])+'''
maps=[p for p in Path("/kaggle/input").rglob("mount-map.json") if json.loads(p.read_text())["identity"] in expected]
corpus=restore_verified(maps,expected,Path(tempfile.mkdtemp(prefix="geofirex-loss-mount-",dir="/tmp"))/"corpus")
index=corpus/"index"
''')
    code('''# Précontrôle uniquement : aucune tuile n'est entraînée ou inférée ici.
MODEL, POS_WEIGHT = "B1", 100
OUT=Path("/kaggle/working")/(f"{MODEL}-w{POS_WEIGHT}-independent-"+time.strftime("%Y%m%d_%H%M%S"))
command=[sys.executable,"-m","geofirex.loss_ablation","--corpus",str(corpus),"--index",str(index),
         "--protocol",str(workspace/"protocol.json"),"--model",MODEL,"--pos-weight",str(POS_WEIGHT),
         "--out",str(OUT),"--device","cuda","--workers","4","--seconds","5400"]
subprocess.run(command,check=True,env=os.environ.copy())
''')
    md('''## Lancement après autorisation seulement
Six exécutions indépendantes à prévoir : (B1,100), (B1,30), (B1,10), (B2,100),
(B2,30), (B2,10). La commande ci-dessous ne lance qu'un couple. Même état initial
par architecture : contrôler `initial_state_sha256` dans les six `config.json`.
La durée de 5 400 s est une limite par session, pas une estimation de quatre époques.
En cas de pause, sauvegarder les outputs, définir RESUME sur le **nouveau run de
cette même variante**, garder modèle/poids identiques et choisir un nouveau OUT.
Ne jamais utiliser les checkpoints B1/B2 historiques comme RESUME.
''')
    code('''RUN_TRAINING=False  # Activer seulement après autorisation de la campagne/budget.
RESUME=None
if RUN_TRAINING:
    training_command=command+["--execute-training"]
    if RESUME: training_command += ["--resume",str(RESUME)]
    subprocess.run(training_command,check=True,env=os.environ.copy())
else:
    print("Entraînement désactivé. Précontrôle terminé.")
''')
    md('''## Sélection finale sur les 32 seulement
Le checkpoint comparé est fixé à l'époque 4 pour toutes les variantes. La grille
de seuils est préenregistrée : 0,10–0,95 par 0,005, puis 0,96 / 0,97 / 0,98 / 0,99 /
0,995 / 0,999. AP et Brier utilisent les probabilités. Aucun pixel des 169 événements
ou du test 2021 n'est inféré. Après les six runs, utiliser `select_loss_ablation.py`
avec leurs six dossiers : Dice micro, égalité AP puis Brier. Figer le gagnant et son
seuil avant une éventuelle confirmation unique ; les 169 ne serviront jamais à
relancer la recherche. Aucun entraînement ou résultat nouveau n'est revendiqué ici.
''')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
            cells=[dict(id=f'loss-{i}',**c) for i,c in enumerate(cells)])
    name='GEOFIREX_POS_WEIGHT_INDEPENDENT_20261112.ipynb'
    (OUT/name).write_text(json.dumps(nb,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'protocol32.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (OUT/'dataset-sources.json').write_text(json.dumps([p['handle'] for p in plan['parts']],indent=2),encoding='utf-8')
    (OUT/'build.json').write_text(json.dumps(dict(source_results_zip_sha256=sha(archive),code_sha256=sha(OUT/'loss-ablation-code.zip'),training_executed=False,weights=[100,30,10],runs=6),indent=2),encoding='utf-8')
    print(OUT/name)


if __name__=='__main__':build()
