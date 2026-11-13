# Contrat d'export React / FastAPI

Le livrable prépare les fichiers consommables ; il ne change pas le service
historique, son seuil ni les résultats déjà publiés. `src/geofirex/serving.py`
attend un autre format de package : il ne doit pas charger ces checkpoints
compacts avec son chargeur actuel sans adaptation explicite.

`app_manifest.json` est le point d'entrée JSON. Il contient les seuils, le SHA du
verrou, les métriques communes, distributions et cas de cartes. Pour chaque cas :

```text
directory / preview                  vue côte à côte pour le suivi
directory / layers.probability       PNG RGBA, palette inferno [0,1]
directory / layers.target            PNG RGBA, vérité J+1
directory / layers.segmentation      PNG RGBA, seuil figé
directory / layers.errors            PNG RGBA, TN sombre / TP vert / FP rouge / FN bleu
directory / npz                     tableaux à résolution native
```

Les chemins sont relatifs à la racine de la nouvelle exécution. Les couches PNG
ont alpha=0 sur les pixels invalides. Dans les NPZ : `probability` float32,
`target` et `observed` binaires, `valid` booléen, `segmentation` et
`segmentation_05` booléens, `error` uint8 avec 255 pour invalide. `valid` reste
l'autorité : ne pas interpréter les zéros invalides de la cible comme du non-feu.

Pour une revue locale, servir **le seul dossier de cette exécution** avec FastAPI :

```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

EXPORT = Path("artifacts/calibration_export").resolve()
app = FastAPI()
app.mount("/calibration", StaticFiles(directory=EXPORT), name="calibration")
```

React peut lire `/calibration/app_manifest.json`, afficher la comparaison et
utiliser `directory + "/" + layers.probability` comme source d'image. Les couches
sont alignées sur une même grille ; afficher les erreurs et la vérité terrain
comme vues de validation historique. Les images ne constituent pas des tuiles
Web Mercator : le compact ne contient pas de CRS ni d'affine. Pour une projection
sur carte, récupérer ces informations depuis des métadonnées déjà certifiées et
vérifier le raccord, sans inventer de géoréférencement.

Pour appliquer un seuil à de futures inférences, le backend doit vérifier
`checkpoint_sha256[model]` et `normalization_sha256` dans le verrou, puis utiliser :

```python
threshold = np.float32(lock["thresholds"][model])
segmentation = (probability >= threshold) & valid.astype(bool)
```

L'interface doit afficher le modèle, le seuil et le groupe d'évaluation (32 ou
169). L'AP et le Brier sont associés aux probabilités : ne pas les animer ou les
recalculer depuis une segmentation lorsque l'utilisateur change un seuil pour
inspection. Un éventuel réglage interactif reste une exploration ; seul le
verrou calculé sur les 32 définit le seuil évalué sur les 169.
