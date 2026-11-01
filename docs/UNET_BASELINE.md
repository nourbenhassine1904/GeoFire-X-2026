# Première baseline U-Net 2D — protocole du suivi 3

Ce protocole poursuit le pipeline du suivi 2. Les fichiers historiques sont
conservés ; le DataLoader U-Net est distinct, car le remplissage historique par
zéro en unités physiques supprimerait l'information nécessaire à l'imputation
par la moyenne du train. T=1 uniquement pour cette première baseline 2D.

## Audit et disponibilité temporelle

La prédiction est définie **après la journée J complète**, à partir du feu détecté
à J, pour les détections à J+1. Elle suppose le masque J livré ; aucune latence
opérationnelle VIIRS n'est validée ici. Le label est une détection de feu actif,
pas une surface brûlée cumulée. `~isnan(active fire)` est la convention du projet ;
un pixel sans détection ne garantit pas l'absence physique de feu.

Sources primaires consultées le 30 octobre 2026 :

- [Zenodo v1.0 et datasheet officiel](https://zenodo.org/records/8006177).
- [Code officiel de création des bandes](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset/blob/master/DataPreparation/satellites/FirePred.py).
- [DataLoader des auteurs](https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataset.py).
- [Publication](https://openreview.net/forum?id=RgdGkPRQ03).

Le datasheet a été relu avec `pypdf` (9 pages ; copie locale dans
`artifacts/scientific_sources/`). Les fichiers bruts ont été relus avec rasterio.
L'audit reproductible est dans `reports/unet_data_audit_v1.json`.

| Bandes brutes (indices 1-based) | Décision pour le premier run | Motif |
| --- | --- | --- |
| 1–3 M11, I2, I1 | Exclues | Observations J, livraison effective non vérifiée |
| 4–5 NDVI_last, EVI2_last | Exclues | Composite multi-jours, date nominale distincte de sa disponibilité |
| 6–12 météo GRIDMET | Exclues | Agrégats J rétrospectifs ; latence de publication non vérifiée |
| 13 pente | Incluse, z-score + indicateur manquant | Relief statique |
| 14 exposition | Incluse, sinus ET cosinus + indicateur manquant | Relief statique ; encodage circulaire complet |
| 15 altitude | Incluse, z-score + indicateur manquant | Relief statique |
| 16 PDSI | Exclue | Agrégat et disponibilité à J non vérifiés |
| 17 LC_Type1 | Exclue par défaut ; one-hot optionnel | Carte annuelle rétrospective |
| 18–22 prévisions GFS | Exclues | Horizon du jour J, problème historique de direction du vent, provenance exacte de l'export non épinglée |
| 23 active fire | Incluse, masque binaire J | Cible lue exclusivement dans le fichier J+1 |

Le code de création actuellement publié sélectionne GFS à 00h J avec les
échéances F01–F24 de J : ce n'est pas directement une prévision de J+1. Il
sélectionne LC_Type1 sur l'année entière contenant J ; la publication avant J
n'est donc pas assurée. Ces constats motivent l'exclusion prudente. Ce code
actuel ne prouve pas à lui seul la version exacte ayant produit l'archive 2023.

La [note amont de février 2026](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset)
signale aussi une correction `atan(v/u)` vers `atan2(v,u)` pour la direction du
vent. L'archive locale n'est pas présumée corrigée. Ajouter un cosinus à un angle
déjà erroné ne résoudrait pas ce problème de calcul amont.

Variation empirique : 56 fichiers utilisent `total_precipitation_surface_sum`,
52 utilisent `total_precipitation_surface_last`. Le lecteur autorise uniquement
cet alias à la position 18, puis contrôle toutes les autres descriptions et leur
ordre. L'ancien lecteur ne vérifiait que le nombre de bandes d'entrée.

LC_Type1 a 14 classes observées sur l'échantillon, aucune valeur fractionnaire
et aucun pixel inconnu. L'encodeur optionnel réserve néanmoins 17 classes
(codes 1–17) et un canal inconnu pour NaN, Inf, 0, 255 ou classe fractionnaire.
Aucune normalisation ordinale, aucun arrondi implicite. L'activer exige :

```yaml
include_landcover: true
landcover_availability: retrospective_annual_assumption
```

Il s'agirait d'une expérience rétrospective avec hypothèse déclarée, pas de la
baseline conservatrice rapportée ici. Préférer à terme une carte antérieure
effectivement disponible à J, ce qui requiert une nouvelle source de données.

## Contrat, splits et prétraitement

Le contrat historique reste `X=[T,23,H,W]`, `Y=[1,H,W]`. Le nouveau chemin lit les
23 bandes mais ne transforme que les bandes retenues : 8 canaux effectifs,
`[slope_z, elevation_z, slope_missing, elevation_missing, aspect_sin, aspect_cos,
aspect_missing, fire_J]`. Avec LC optionnel : 26 canaux.

Le split historique est réutilisé en lecture seule : 3 événements train,
1 validation, 1 test. Les dates doivent être consécutives, les grilles doivent
coïncider dans chaque événement (dimensions, CRS, transform). Les doublons
événement/date, les chemins réutilisés et les assignations multiples sont rejetés.
Les chemins relatifs du manifeste se résolvent contre la racine du dépôt.

Les moyennes/écarts-types population sont calculés sur les pixels finis des
**chemins d'entrée uniques du train**, jamais sur les cibles seules ni val/test.
Les canaux constants ont un écart-type de 1 ; un canal entièrement manquant est
rejeté. Après centrage, les NaN/Inf continus sont imputés à 0 (= moyenne train),
avec indicateurs de manque séparés ; les z-scores sont bornés à ±8.
Les angles invalides ont sin=cos=0 et un indicateur manquant. Le feu suit une
convention séparée (NaN = absence de détection ; Inf = erreur).

Les statistiques pondèrent les observations et pixels, pas les événements
uniformément ; les jours d'un même événement répètent notamment le relief.

Les tuiles sont 64×64, sur une grille sans recouvrement. Au train, tirage avec
remise : quota exact de 50 % de tuiles à cible positive et 50 % de tuiles sans
feu, avec seed dépendant de l'epoch et de l'index. La présence de feu dans la
cible n'intervient que dans l'échantillonnage supervisé du train.
En validation et test, chaque pixel de chaque fenêtre est évalué une fois,
sans sélection fondée sur les cibles. Le padding est exclu de la loss et retiré
des prédictions restituées. Des raccords entre tuiles restent possibles.

## Architecture et optimisation

- Encodeurs : deux convolutions 3×3 + GroupNorm + ReLU, puis max-pooling 2×2.
- Largeurs 8, 16, 32 ; bottleneck 64.
- Décodeurs : interpolation bilinéaire, concaténation du skip de même échelle,
  deux convolutions 3×3 + GroupNorm + ReLU.
- Tête 1×1, **un logit par pixel**. Sigmoid à l'inférence pour la probabilité.
- 122 825 paramètres avec 8 canaux ; environ 0,47 Mio de poids float32.
- Largeur et profondeur configurables ; tailles impaires prises en charge.
- Loss : 0,5 BCEWithLogits + 0,5 soft Dice par tuile ; BCE seule avec
  `dice_weight: 0`. Padding masqué ; `pos_weight` configurable.
- AdamW, lr=0,001, weight_decay=0,0001, clipping de gradient à 1.
- Seed=42, opérations PyTorch déterministes, 2 threads CPU, num_workers=0.
- Budget initial : 5 epochs × 128 patches, batch=8, soit 80 mises à jour.

Le mini-overfit emploie un modèle de diagnostic séparé sur la tuile réelle de
train la plus riche en feu, 200 mises à jour Adam à lr=0,01. Réussite : loss finale
< 80 % de l'initiale et Dice à 0,5 > 0,5. Les gradients doivent être finis.
Le modèle principal est ensuite recréé avec la seed initiale.

Le checkpoint est choisi sur la **loss validation exhaustive minimale**, jamais
sur le test. Le seuil est choisi ensuite sur le Dice micro validation, dans la
grille déclarée au YAML (égalité : seuil le plus élevé). Le test est inféré
uniquement après fixation du checkpoint et du seuil.

## Métriques et limites

Dice, IoU, Precision, Recall et PR-AUC sont calculés avec les mêmes fenêtres et
pixels pour U-Net et Persistence (`fire_J`). PR-AUC désigne ici **Average Precision
non interpolée de scikit-learn**, pas une intégrale trapézoïdale. Persistence
fournit des scores binaires ; ses nombreuses égalités limitent le classement.

Rapportées : micro par pixels cumulés, macro par fenêtre, macro sur fenêtres
positives seulement, macro par événement, détails par fenêtre/événement et
nombres de métriques définies. Les fenêtres temporelles se recouvrent et ne sont
pas des répétitions indépendantes. Les longues séries pèsent davantage dans le
macro par fenêtre ; les grandes images pèsent davantage dans le micro.

Conventions explicites du nouveau rapport :

- Cible et prédiction vides : Dice=IoU=1.
- Precision sans prédiction positive : `null` ; Recall sans cible positive : `null`.
- PR-AUC sans cible positive : `null`, exclue des moyennes avec décompte déclaré.
- Le module historique `geofirex.metrics` conserve ses conventions epsilon,
  notamment precision/recall=1 dans certains cas indéfinis. Ses anciens résultats
  restent intacts ; la nouvelle comparaison recalcule les deux modèles ensemble.

37/67 cibles train, 5/23 validation et 7/13 test sont vides. Un modèle toujours
négatif peut donc afficher un Dice macro élevé. Consulter micro, PR-AUC et les
fenêtres positives conjointement. La sigmoid produit un score probabiliste qui
n'est pas une probabilité calibrée : le suréchantillonnage modifie la prévalence.

Un seul événement dans chaque split tenu à part ne permet pas de conclusion de
généralisation, d'intervalle de confiance crédible ou de comparaison OOD. La
séparation par événement élimine les jours partagés mais ne démontre pas une
indépendance géographique. Les régions et périodes sont définies
rétrospectivement à partir d'événements connus ; ce n'est pas une évaluation de
détection précoce à l'échelle de tout le territoire. Le protocole futur restera
une validation par années avec davantage d'événements.

## Commandes et artefacts

Depuis PowerShell à la racine du dépôt (utiliser le venv, pas le Python global) :

```powershell
.\.venv\Scripts\python.exe scripts/audit_unet_data.py --out reports/unet_data_audit_nouveau.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -u scripts/train_unet.py --config configs/unet_baseline.yaml --device cpu
```

Chaque entraînement crée un nouveau dossier `artifacts/unet/<UTC>_<uuid>/` et un
nouveau run dans l'expérience `geofirex-unet2d-real-v1`, distincte de Persistence.
Le mode `auto` choisit CUDA si `torch.cuda.is_available()` ; `cuda` échoue
explicitement si indisponible. Aucun téléchargement CUDA automatique.

Le run conserve configuration effective, manifeste/split copiés, SHA256 de tous
les GeoTIFF, provenance Git (commit et état non commité), snapshot des sources
avec hashes, environnement `pip freeze`, statistiques train, courbes et métriques
par epoch, diagnostic réel, meilleur checkpoint, sélection de seuil, résultats et
figures à cinq panneaux. Les fichiers du dossier sont aussi loggés dans MLflow.
Les sources non commitées sont archivées car le commit seul ne décrit pas ce run.

```powershell
# Remplacer RUN par le dossier créé, et J.tif par une observation réelle.
.\.venv\Scripts\python.exe scripts/predict_unet.py --checkpoint artifacts/unet/RUN/best.pt --input J.tif --output-dir artifacts/prediction_nouvelle
.\.venv\Scripts\python.exe scripts/evaluate_unet.py --checkpoint artifacts/unet/RUN/best.pt --split-name test --expected-results artifacts/unet/RUN/results.json --out artifacts/replay_nouveau.json
.\.venv\Scripts\python.exe scripts/verify_unet_run.py --run-dir artifacts/unet/RUN --out artifacts/verification_nouvelle.json
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

L'inférence autonome n'ouvre aucune cible : elle produit `probability.tif` et
`mask.tif` géoréférencés, avec le seuil validation du checkpoint. Le répertoire
de sortie doit être nouveau. Les runs, checkpoints et GeoTIFF restent hors Git.

Les résultats effectivement obtenus et les figures du premier entraînement
abouti sont présentés dans `reports/UNET_BASELINE_REPORT.md`.
