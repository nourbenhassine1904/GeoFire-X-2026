"""Build one offline, disabled-by-default experiment; no Kaggle submission."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from geofirex.calibration_io import save,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/propagation_20261112_v1'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ROOT/'reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip') as z:
        protocol=z.read('protocol.json')
    with zipfile.ZipFile(ROOT/'reports/unet_confirmation_20261112_v1/inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip') as z:
        baseline=z.read('B2_posweight_10/calibration32/evaluation.json')
    plan=json.loads((ROOT/'reports/gpu_acceleration_20261111_v1/publication/plan.json').read_text())
    names=('__init__.py','calibration_io.py','calibration_mounts.py','compact_calibration.py','fast_compact.py',
           'benchmark_models.py','benchmark_metrics.py','unet.py','propagation_experiment.py')
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for name in names:
                source=(ROOT/'src/geofirex'/name).read_text(encoding='utf-8')
                if name=='fast_compact.py':source=source.replace('from geofirex.mission_streaming import save,sha','from geofirex.calibration_io import save,sha')
                compile(source,name,'exec');z.writestr('geofirex/'+name,source)
            z.writestr('protocol.json',protocol);z.writestr('baseline32.json',baseline)
        payload=buffer.getvalue()
    (OUT/'propagation-code.zip').write_bytes(payload)
    (OUT/'protocol.json').write_bytes(protocol);(OUT/'baseline32.json').write_bytes(baseline)
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s):
        compile(s,'propagation-notebook','exec');cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))
    md('''# GeoFire-X — UNE expérience : tête spécialisée nouveaux feux
U-Net base8/depth3, T=3, nouvelle tête 1×1 au décodeur pleine résolution.
L = L principal historique (poids 10) + 0,5 L nouveaux pixels (poids 10,
BCE + Dice groupé par batch). Aucun nouveau balayage de poids.
250 événements train / 93 603 tuiles, seed42, 4 époques / 23 404 updates.
La tête spécialisée est utilisée si feu_J=0 ; sinon la tête principale.
Un seul seuil global choisi par Dice micro global sur les 32 historiques.
Comparer aussi Dice et rappel des nouveaux feux à la référence sauvegardée.

**Aucun lancement ni transfert effectué par la préparation.** Attacher les 11
partitions compactes déjà publiées. Internet OFF. Ne pas attacher de test 2021.
Précontrôle possible sans GPU. Après autorisation seulement : activer un seul
T4 et RUN_TRAINING. Plafond 7 200 s de processus GPU, arrêt anticipé avec checkpoint
ou interruption externe de secours. Si incomplet : aucun score comparable à époque4.
Sélection32 séparée sur CPU, limite 3 600 s ; pas de nouvelle confirmation169 automatique.
''')
    code('''import base64, hashlib, io, json, os, sys, tempfile, time, subprocess, zipfile
from pathlib import Path
workspace=Path(tempfile.mkdtemp(prefix="geofirex-propagation-code-",dir="/tmp"))
payload=base64.b64decode('''+repr(base64.b64encode(payload).decode())+''')
assert hashlib.sha256(payload).hexdigest()=='''+repr(hashlib.sha256(payload).hexdigest())+'''
with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(workspace)
sys.path.insert(0,str(workspace))
os.environ["PYTHONPATH"]=str(workspace)+os.pathsep+os.environ.get("PYTHONPATH", "")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # Un seul GPU même si le runtime en expose deux.
import torch, numpy, psutil, matplotlib
print({"torch":torch.__version__,"numpy":numpy.__version__,"psutil":psutil.__version__,"matplotlib":matplotlib.__version__})
from geofirex.calibration_mounts import restore_verified
expected='''+repr([p['identity'] for p in plan['parts']])+'''
maps=[p for p in Path("/kaggle/input").rglob("mount-map.json") if json.loads(p.read_text())["identity"] in expected]
corpus=restore_verified(maps,expected,Path(tempfile.mkdtemp(prefix="geofirex-propagation-mount-",dir="/tmp"))/"corpus")
index=corpus/"index"
OUT=Path("/kaggle/working")/("UNET_NEW_FIRE_HEAD_V1-"+time.strftime("%Y%m%d_%H%M%S"))
common=[sys.executable,"-m","geofirex.propagation_experiment","--corpus",str(corpus),"--index",str(index),
        "--protocol",str(workspace/"protocol.json"),"--baseline32",str(workspace/"baseline32.json"),"--workers","4"]
subprocess.run(common+["--out",str(OUT)],check=True,env=os.environ.copy(),timeout=120)
''')
    md('''## Entraînement borné — uniquement après autorisation
Un seul nouveau modèle indépendant. Aucun checkpoint historique en initialisation.
Les checkpoints sauvegardent optimiseur, RNG, curseur et temps consommé ; reprise
explicite de ce même modèle seulement, sans extension du plafond cumulé 7 200 s.
Conserver les sorties Kaggle avant de fermer la session. Ne jamais relancer
un modèle déjà complet. Quatre époques ne sont pas garanties dans le plafond.
''')
    code('''RUN_TRAINING=False
RESUME=None  # Seulement un checkpoint de CETTE expérience interrompue.
if RUN_TRAINING:
    assert torch.__version__.split("+")[0]=="2.11.0", "Conserver l'environnement des entraînements indépendants précédents"
    assert numpy.__version__=="2.1.3"
    # Contrôle CPU avant tout entraînement : identique au vrai backbone initial historique.
    from geofirex.propagation_experiment import PropagationUNet, state_hash, EXPECTED_MAIN_INITIAL_SHA
    torch.manual_seed(42)
    probe=PropagationUNet()
    assert state_hash(probe,True)==EXPECTED_MAIN_INITIAL_SHA, "Initialisation différente : arrêter, aucun update autorisé par ce protocole"
    del probe
    command=common+["--out",str(OUT),"--mode","train","--execute-training"]
    consumed=0.
    if RESUME:
        state=torch.load(RESUME,map_location="cpu",weights_only=False)
        assert state["config"]["experiment"]=="UNET_NEW_FIRE_HEAD_V1"
        consumed=state["gpu_wall_seconds"]
        command += ["--resume",str(RESUME)]
    remaining=7200-int(consumed)
    assert remaining>30, "Budget épuisé : aucun nouveau lancement"
    # Timeout dur même si le chargement d'un batch se bloque. Dernier checkpoint conservé.
    try:subprocess.run(command,check=True,env=os.environ.copy(),timeout=remaining)
    except subprocess.TimeoutExpired:print("Plafond atteint ; conserver le dernier checkpoint. Aucun relancement automatique.")
else:print("GPU et entraînement désactivés.")
''')
    md('''## Sélection32 sur CPU — nouveau modèle complet uniquement
Grille figée : 0,10–0,95 par 0,005 puis 0,96 / 0,97 / 0,98 / 0,99 / 0,995 / 0,999.
Même règle de seuil historique (Dice micro global, égalité précision puis seuil élevé).
Acceptation de développement : Dice global et Dice nouveaux feux au moins égaux
à U-Net poids10/0,78 ; rappel nouveaux feux strictement supérieur. Sinon résultat négatif,
pas de recherche supplémentaire automatique. AP et Brier restent des métriques de probabilités.
Les 32 servent au développement ; les 169 déjà vus ne sont plus un test aveugle.
''')
    code('''RUN_SELECTION32=False
CHECKPOINT=OUT/"checkpoint_latest.pt"
if RUN_SELECTION32:
    assert CHECKPOINT.is_file()
    subprocess.run(common+["--out",str(OUT)+"-selection32","--mode","select32","--checkpoint",str(CHECKPOINT)],
                   check=True,env=os.environ.copy(),timeout=3600)
else:print("Nouvelle évaluation désactivée. Aucun calcul historique répété.")
''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
        cells=[dict(id=f'propagation-{i}',**c) for i,c in enumerate(cells)])
    path=OUT/'GEOFIREX_PROPAGATION_NEW_FIRE_HEAD_V1.ipynb'
    path.write_text(json.dumps(notebook,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'dataset-sources.json').write_text(json.dumps([p['handle'] for p in plan['parts']],indent=2),encoding='utf-8')
    (OUT/'requirements-kaggle.txt').write_text('torch==2.11.0\nnumpy==2.1.3\npsutil>=5.9,<8\nmatplotlib>=3.6\n',encoding='utf-8')
    (OUT/'notebook_build.json').write_text(json.dumps(dict(notebook_sha256=sha(path),code_sha256=sha(OUT/'propagation-code.zip'),
        protocol_sha256=sha(OUT/'protocol.json'),baseline32_sha256=sha(OUT/'baseline32.json'),runs=1,
        training_executed=False,gpu_executed=False,kaggle_submitted=False),indent=2),encoding='utf-8')
    print(path)


if __name__=='__main__':build()
