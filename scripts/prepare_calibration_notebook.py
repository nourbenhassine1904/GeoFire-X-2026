"""Build standalone inference-only notebook, source bundle and operational files.

No API, publication, training, commit or push is performed by this builder.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/calibration_20261112_v1'


def build(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'reports/gpu_acceleration_20261111_v1/publication/plan.json').read_text())
    buffer=io.BytesIO();manifest={}
    modules=('__init__.py','calibration_io.py','calibration_mounts.py','compact_calibration.py',
             'fast_compact.py','benchmark_models.py','benchmark_metrics.py','unet.py')
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name in modules:
            source=(ROOT/'src/geofirex'/name).read_text(encoding='utf-8')
            original=hashlib.sha256(source.encode()).hexdigest()
            if name=='fast_compact.py':
                old='from geofirex.mission_streaming import save,sha'
                assert source.count(old)==1
                source=source.replace(old,'from geofirex.calibration_io import save,sha')
            compile(source,name,'exec')
            z.writestr('geofirex/'+name,source)
            manifest[name]=dict(source_sha256=original,bundle_sha256=hashlib.sha256(source.encode()).hexdigest())
        z.writestr('calibrate_compact.py',"from geofirex.compact_calibration import main\nif __name__ == '__main__': main()\n")
    payload=buffer.getvalue()
    (out/'calibration-code.zip').write_bytes(payload)
    digest=hashlib.sha256(payload).hexdigest()
    (out/'source-manifest.json').write_text(json.dumps(dict(zip_sha256=digest,modules=manifest,
        adapter='fast_compact: only save/sha utility import replaced; loader, masks, normalization and models byte-identical'),indent=2),encoding='utf-8')
    expected=[p['identity'] for p in plan['parts']]
    handles=[p['handle'] for p in plan['parts']]
    (out/'dataset-sources.json').write_text(json.dumps(handles,indent=2),encoding='utf-8')
    (out/'requirements-calibration.txt').write_text('numpy>=1.26,<3\ntorch>=2.6,<3\nmatplotlib>=3.8,<4\n',encoding='utf-8')
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s):
        compile(s,'calibration-notebook','exec')
        cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))
    md('''# GeoFire-X — Calibration scientifique B1/B2 (12 novembre 2026)
Inférence uniquement depuis les checkpoints achevés à l'époque 4. Aucun entraînement.

**Entrées :** les 11 partitions existantes de `dataset-sources.json`, et les sorties
enregistrées des notebooks contenant les deux checkpoints (Add Input → Notebook Output).
Choisir GPU T4 pour l'inférence et Internet désactivé. Aucun nouveau dataset à publier.

**Protocole figé :** 32 événements historiques explicitement lus dans les checkpoints,
13 759 tuiles ; seuil grossier 0,10–0,95 par 0,05, affiné ±0,05 par 0,005 ; sélection
Dice micro, égalité départagée par précision puis seuil supérieur. Les deux seuils sont
enregistrés avant toute inférence sur les 169 autres événements 2020. Tous leurs pixels
valides sont conservés. AP exacte et Brier restent calculés sur les probabilités originales.
Les 32 événements ayant déjà servi à la validation de l'entraînement, ils ne constituent
pas un test indépendant. Aucun pixel 2021 n'est ouvert.

Le code embarqué ne dépend ni du dépôt, ni d'Internet, ni de Spark/rasterio/pyarrow.
L'audit des pixels train est une lecture CPU sans prédiction ni apprentissage.
''')
    code('''import base64, hashlib, importlib.util, io, json, os, sys, tempfile, time, zipfile
from pathlib import Path
missing = [name for name in ("numpy", "torch", "matplotlib") if importlib.util.find_spec(name) is None]
assert not missing, f"Dépendances absentes : {missing}. Voir requirements-calibration.txt."
import numpy as np, torch, matplotlib
print({"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__})
DEVICE = "cuda"  # mettre "cpu" explicitement pour une exécution sans accélérateur
BATCH_SIZE, WORKERS, CACHE_MB = 32, 4, 64
CHECKPOINTS = {
    "B1": Path("/kaggle/working/B1-continued-20261112_135150/checkpoint_e005_u0023404_epoch_complete.pt"),
    "B2": Path("/kaggle/working/B2-continued-20261112_141330/checkpoint_e005_u0023404_epoch_complete.pt"),
}
# Si nouvelle session : définir ces chemins sous /kaggle/input/... après avoir
# attaché les sorties sauvegardées des notebooks B1/B2. Aucun upload de dataset.
OUT = Path("/kaggle/working") / ("calibration-B1-B2-" + time.strftime("%Y%m%d_%H%M%S"))
''')
    code('''# Extraction du code seulement (jamais des archives du corpus).
payload = base64.b64decode('''+repr(base64.b64encode(payload).decode())+''')
assert hashlib.sha256(payload).hexdigest() == '''+repr(digest)+'''
workspace = Path(tempfile.mkdtemp(prefix="geofirex-calibration-code-", dir="/tmp"))
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    bundle.extractall(workspace)
sys.path.insert(0, str(workspace))
os.environ["PYTHONPATH"] = str(workspace) + os.pathsep + os.environ.get("PYTHONPATH", "")
print("Code autonome vérifié :", workspace)
''')
    md('''## Vérification des entrées
Les deux fichiers portent le même nom : l'identité B1/B2 est vérifiée à partir du
champ `config.model`, jamais déduite du nom seul. Le chemin explicite est prioritaire.
En nouvelle session, une recherche dans les sorties montées est possible si le
fichier demandé n'existe plus dans `/kaggle/working`. Une ambiguïté provoque un arrêt.
''')
    code('''filename = "checkpoint_e005_u0023404_epoch_complete.pt"
for name, path in list(CHECKPOINTS.items()):
    if not path.is_file():
        candidates = list(Path("/kaggle/input").rglob(filename))
        matching = []
        for candidate in candidates:
            # Uniquement vos checkpoints de confiance, attachés comme sorties.
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if state.get("config", {}).get("model") == name:
                matching.append(candidate)
            del state
        assert len(matching) == 1, f"{name}: spécifier le chemin exact CHECKPOINTS ; candidats={matching}"
        CHECKPOINTS[name] = matching[0]
assert CHECKPOINTS["B1"].resolve() != CHECKPOINTS["B2"].resolve()
assert DEVICE == "cpu" or torch.cuda.is_available(), "Sélectionner un accélérateur GPU ou DEVICE=cpu"
print("Checkpoints :", CHECKPOINTS)
''')
    code('''from geofirex.calibration_mounts import restore_verified
expected = '''+repr(expected)+'''
maps = list(Path("/kaggle/input").rglob("mount-map.json"))
# Filtrer seulement les 11 identités attendues ; les sorties notebook peuvent
# aussi contenir des anciennes copies de fichiers de contrôle.
maps = [p for p in maps if json.loads(p.read_text())["identity"] in expected]
corpus = restore_verified(maps, expected,
    Path(tempfile.mkdtemp(prefix="geofirex-calibration-mount-", dir="/tmp")) / "corpus")
index = corpus / "index"
print("Corpus relié sans reconstruction :", corpus)
''')
    md('''## Calcul complet
Exécuter cette cellule une seule fois. Elle réalise l'audit train CPU, l'inférence
des 32 événements pour B1/B2, le verrouillage des seuils, puis les 169 événements
aux seuls seuils 0,5 et fixés. Les tris/fusions AP exacte utilisent le CPU et le disque,
et peuvent prendre plus de temps que l'inférence. Aucun débit GPU ou délai n'est promis.
Scratch : prévoir environ 12–15 Go libres pour la phase de validation la plus grande,
hors sorties déjà présentes ; les runs AP temporaires sont supprimés après calcul.
Les journaux indiquent l'événement terminé. Une interruption ne produit pas de
`status_complete.json` et ne permet pas de présenter la validation comme complète.
''')
    code('''import subprocess
command = [sys.executable, str(workspace / "calibrate_compact.py"),
    "--corpus", str(corpus), "--index", str(index),
    "--b1", str(CHECKPOINTS["B1"]), "--b2", str(CHECKPOINTS["B2"]),
    "--out", str(OUT), "--device", DEVICE, "--batch-size", str(BATCH_SIZE),
    "--workers", str(WORKERS), "--cache-mb", str(CACHE_MB)]
print("Commande :", command, flush=True)
subprocess.run(command, check=True, env=os.environ.copy())
assert json.loads((OUT / "status_complete.json").read_text())["status"] == "COMPLETE"
''')
    md('''## Comparaison et visualisations
Les lignes baseline/calibré ont rigoureusement les mêmes AP et Brier. Les CSV
contiennent TP/FP/FN/TN, Dice/IoU/précision/rappel, couverture et prévalence. Les
distributions macro donnent le même poids à chaque événement ; les métriques micro
agrègent les pixels. Les sous-domaines feu nouveau / déjà observé décomposent les erreurs.
''')
    code('''from IPython.display import display, Image
comparison = json.loads((OUT / "comparison.json").read_text())
try:
    import pandas as pd
    display(pd.DataFrame(comparison["rows"]))
except ImportError:
    print(json.dumps(comparison["rows"], indent=2))
print("Seuils fixés :", json.loads((OUT / "thresholds_locked.json").read_text())["thresholds"])
for name in ("calibration_curves.png", "calibration32_diagnostics.png", "holdout169_diagnostics.png"):
    display(Image(filename=str(OUT / name)))
manifest = json.loads((OUT / "app_manifest.json").read_text())
for case in manifest["maps"]:
    display(Image(filename=str(OUT / case["directory"] / case["preview"])))
''')
    md('''## Export pour React/FastAPI et conservation
Télécharger le dossier `calibration-B1-B2-*` depuis les outputs du notebook après
Save Version. `app_manifest.json` décrit les seuils, scores, distributions, fichiers
NPZ et couches PNG transparentes à résolution native. Erreurs : TN=0, TP=1 (vert),
FP=2 (rouge), FN=3 (bleu), invalide=255. `valid` est toujours fourni séparément.
Ces mosaïques sont des grilles de pixels : pas de géoréférencement inventé, pas de
GeoTIFF sans CRS/affine certifiés. L'application de production doit vérifier le SHA
du checkpoint et le normaliseur associés au seuil avant de l'utiliser. Le serveur
historique ne charge pas directement ce format de checkpoint compact.

Les 169 événements ne doivent pas servir à réajuster le seuil. Voir `DIAGNOSTIC.md`
pour les deux expériences futures proposées, qui nécessitent une autorisation GPU.
''')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.11')),
            cells=[dict(id=f'calibration-{i:02d}',**c) for i,c in enumerate(cells)])
    notebook='GEOFIREX_B1_B2_CALIBRATION_20261112.ipynb'
    (out/notebook).write_text(json.dumps(nb,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(dict(id='nnnnbhss/geofirex-b1-b2-calibration-20261112',
        title='GeoFire-X B1 B2 Calibration 20261112',code_file=notebook,language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=handles,
        kernel_sources=[],competition_sources=[]),indent=2),encoding='utf-8')
    print(json.dumps(dict(notebook=str(out/notebook),code_bytes=len(payload),sha256=digest,cells=len(cells)),indent=2))
    return out


if __name__=='__main__':build()
