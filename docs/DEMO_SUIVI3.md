# Démonstration du suivi 3 — mercredi 4 novembre 2026

Depuis la racine du dépôt, avec le Python `.venv` du projet. **Ne lancer ni acquisition,
ni notebook Kaggle, ni entraînement pour cette démonstration.**

## Application et MLflow

Terminal 1 :

```powershell
.venv/Scripts/python.exe scripts/serve_demo.py --config artifacts/demo-serving/gpu-pilot-local.json --port 8000
```

Application : http://127.0.0.1:8000 ; santé : http://127.0.0.1:8000/health ;
documentation FastAPI : http://127.0.0.1:8000/docs.
Le lancement par défaut vise désormais le snapshot B1 GPU. L'ancien launch record CPU
`artifacts/demo-serving/local.json` reste conservé et sélectionnable explicitement.

Terminal 2 :

```powershell
.venv/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/suivi3-mlflow-restored/tracking.db --host 127.0.0.1 --port 5000
```

MLflow : http://127.0.0.1:5000. Expérience `geofirex-sample-t4-pilot-20261101`,
**quatre runs B0/B1/B2/B3 dans une expérience**, tous `FINISHED`.
Afficher `selected_validation_average_precision`, `selected_validation_dice`,
`selected_validation_iou`, `selected_validation_precision`, `selected_validation_recall`,
`total_optimizer_updates` et `completed_epochs`.

Les clés historiques sans préfixe `selected_` décrivent les epochs ; leur valeur « latest »
peut différer du meilleur checkpoint et les updates par epoch valent 128. La copie restaurée
ajoute des champs explicites issus des résumés certifiés. Aucun historique original n'est modifié.
Les liens d'artefacts de la copie pointent vers les fichiers locaux, plus vers `/kaggle/working`.

## Parcours de présentation (5 minutes)

1. Application : choisir `2020/fire_23654679 · val`, puis J = `2020-01-05`.
2. Cliquer **Prédire J+1**. Montrer les quatre cartes : feu observé à J, probabilité J+1,
   masque prédit et vérité terrain du `2020-01-06`. La vérité terrain est une requête séparée
   après l'inférence ; aucun pixel futur n'entre dans le modèle.
3. Montrer `B1 · pilot`, le seuil de validation, l'empreinte du checkpoint et **1 280 updates**.
   Le calcul GPU a déjà eu lieu ; la démonstration locale recharge le checkpoint sur CPU.
4. MLflow : comparer les quatre runs, montrer paramètres (`context=3`, `seed=42`, 10 epochs),
   métriques sélectionnées et `curves.png`, `example.png`, `epoch_0007.pt` du CNN.
   Persistence ne s'entraîne pas et n'a donc pas de courbe d'apprentissage.
5. Présenter les figures `reports/figures/suivi3_gpu/comparison.png` et `learning-curves.png`
   (versions SVG également fournies), puis rappeler les limites.

| Modèle | Dice | IoU | Precision | Recall | AP | Epochs / updates |
|---|---:|---:|---:|---:|---:|---|
| Persistence | **0,4814** | **0,3170** | **0,4870** | 0,4758 | 0,2319 | 0 / 0 |
| CNN B1 | 0,4800 | 0,3158 | 0,4267 | **0,5484** | **0,3535** | 10 / 1 280 |
| U-Net B2 | 0,2574 | 0,1477 | 0,2861 | 0,2339 | 0,1240 | 10 / 1 280 |
| ConvLSTM B3 | 0,3050 | 0,1800 | 0,3172 | 0,2938 | 0,1605 | 10 / 1 280 |

Dire : « B1 est le candidat de démonstration selon l'AP ; Persistence garde un Dice
légèrement supérieur. Pilote sur cinq événements, une seed, validation utilisée pour choisir
checkpoint et seuil. Aucun résultat test 2021, aucune généralisation scientifique démontrée,
probabilités non calibrées. » Détails : `reports/GPU_PILOT_2026-11-01.md`.

## État vérifié et vérification manuelle restante

- Installation Python éditable hors ligne réussie, `pip check` sans conflit ; React reconstruit
  avec les dépendances locales, sans téléchargement.
- HTTP 200 pour index React, JS construit, CSS, catalogue de cinq événements et santé B1.
  Prédiction réelle et vérité terrain vérifiées pour les cinq événements ; géométries identiques ;
  test 2021 : HTTP 403.
- MLflow : paramètres, historiques, checkpoints et figures présents pour les quatre runs ;
  interface et consultation des figures vérifiées via HTTP. Sauvegarde certifiée inchangée.
- 48 tests de contrats, splits et métriques réussis, sans lancement d'entraînement.
- **Parcours visuel validé par l'utilisateur dans Chrome** : sélection des événements et dates,
  affichage des cartes et consultation des quatre runs MLflow, paramètres, métriques et courbes.
  Le contrôle automatisé avait été bloqué par la détection d'URL ; cette validation manuelle
  est une preuve distincte, fournie par l'utilisateur lors de la finalisation GitHub.

Preuves : `reports/SUIVI3_FINAL_VERIFICATION.json`, `reports/suivi3_final_tests_20261101.xml`,
`artifacts/demo-serving/suivi3-http-final/verification.json`,
`artifacts/suivi3-mlflow-restored/restoration.json`.

## Restaurer sur un autre poste (sans refaire les calculs)

Le dépôt ne contient ni TIFF, ni checkpoints, ni environnements, ni SQLite. Copier depuis
les sauvegardes existantes l'archive certifiée et le dossier des 108 TIFF ; aucun téléchargement
ou entraînement n'est nécessaire. Archive attendue : SHA-256
`2b5f72d63df6b71f15124ce1a7e1945e428590acfc495a53251688ae34e2b0b0`.
Extraire uniquement dans un nouveau dossier vide ; conserver l'original intact.

Sur un poste déjà équipé des dépendances :

```powershell
.venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
npm --prefix frontend run build
.venv/Scripts/python.exe scripts/prepare_gpu_demo.py --snapshot artifacts/sample-launch-20261101/verified-output/snapshots/B1 --data-root data/raw/WildfireSpreadTS/real_sample
.venv/Scripts/python.exe scripts/restore_pilot_mlflow.py --source-root artifacts/sample-launch-20261101/verified-output --out artifacts/suivi3-mlflow-restored
```

Ces deux préparations refusent d'écraser une sortie existante. Sur ce PC elles sont déjà
faites : lancer directement les deux serveurs. Adapter `--snapshot` et `--data-root` si besoin.
Sur une installation neuve, installer Python 3.11, le projet avec extras `.[api,dev]` et les
dépendances frontend du lockfile avant ces étapes ; cette installation connectée n'a pas été
réalisée pendant cette mission. Les sauvegardes seules ne remplacent pas ces dépendances.
