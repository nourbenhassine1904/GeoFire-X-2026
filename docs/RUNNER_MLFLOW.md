# MLflow du runner commun

Les anciens `mlflow.db`, expériences et `mlruns` sont conservés. Par défaut,
le runner utilise une nouvelle base `artifacts/benchmark/tracking.db`, une
expérience diagnostique v2 et un artifact store sous ce même répertoire.

Chaque invocation crée un nouveau run et un dossier UTC + UUID. Une reprise
crée également un run neuf, lié par `resume.json` au checkpoint parent ; elle
n'efface jamais son parent. Train/final-test ont des tags de phase distincts.

| Artefact/tag | Contenu |
| --- | --- |
| Tags | Modèle, fold, contexte, seed, qualification, phase, protocole, identité, commit/source et dataset en train ; verrou et parent au test |
| `config.json` | Paramètres effectifs résolus, budgets, device/AMP et politique de stockage |
| `data_audit.json` / `test_audit.json` | Hashes manifeste/index/split/cibles, SHA256 des TIFF et sidecars lus, géométries, rejets temporels |
| `normalization.json`, `train_prior.json` | Chemins train uniques, moments/counts, prévalence et poids de loss effectif |
| `provenance.json` | Commit/branche/état Git, versions de toutes les distributions, Python, CPU/RAM/GPU/CUDA |
| `source/`, `source_hashes.json` | Snapshot des sources, scripts, YAML et tests, y compris non commités |
| `sampling_*.json` | Liste reproductible des indices de fenêtres et coordonnées de patches |
| `preflight.json` | Mini-overfit écarté, formes et historique de loss |
| `validation_*.json` | AP/confusions, ignitions, seuils, agrégations et timings par epoch |
| `epoch_*.pt` | Poids et, pour B1–B3, optimizer/scaler/RNG/historique/sélection |
| `summary.json` | Checkpoint sélectionné, hash, AP validation, seuil, budget et coût |
| `curves.png`, `example.png` | Courbes loss/AP et premier patch déterministe, sans sélection visuelle favorable |
| `selection_lock.json`, `test_metrics.json` | Verrou et évaluation finale, dans le run test distinct |

Les scalaires train/validation sont loggés à chaque epoch. Les artefacts sont
synchronisés après chaque checkpoint complet et à la fermeture, y compris
lors d'un échec ; un échec est marqué FAILED. Les `.bin` de tri AP et `.tmp`
incomplets restent hors upload. Les TIFF bruts ne sont jamais loggés. Le verrou
refuse de sélectionner un run MLflow qui n'est pas FINISHED.

## Stockage cloud explicite

Configurer **ensemble** `output_root`, `tracking_uri`, `artifact_root`,
`experiment` et `storage_policy`. Exemple volume persistant Runpod :

```yaml
output_root: /workspace/geofirex-runs/annual-v2
tracking_uri: sqlite:////workspace/geofirex-runs/annual-v2/tracking.db
artifact_root: file:///workspace/geofirex-runs/annual-v2/mlflow_artifacts
experiment: geofirex-benchmark-annual-v2
storage_policy: persistent
```

Vérifier que `/workspace` est réellement un volume persistant du fournisseur,
avec espace pour checkpoints **et tris AP**. Ce champ décrit le déploiement ;
le runner ne peut pas garantir la rétention réelle d'un volume cloud. Conserver
les mêmes montages pour la reprise et l'évaluation finale.

Un serveur MLflow distant peut remplacer `tracking_uri` et un artifact store
S3 compatible remplacer `artifact_root` (`s3://bucket/prefix-v2`). Installer et
configurer le SDK/credentials de ce backend dans l'environnement du serveur
ou du client selon son mode de proxy. Ne pas écrire les secrets dans le YAML.
Ce backend distant n'a pas été testé dans cette mission. Si une expérience
existe avec un autre artifact store, le runner refuse de la réutiliser : choisir
un nouveau nom, sans déplacer ni écraser l'ancienne expérience. Pour plusieurs
workers cloud, utiliser un serveur MLflow adapté plutôt que partager SQLite
entre machines.

Kaggle utilise `storage_policy: ephemeral_export`. `/kaggle/working` ne devient
durable qu'après sauvegarde des outputs de la session/version Kaggle ou copie
vers un stockage persistant. Exporter explicitement chaque run :

```bash
python scripts/verify_and_export_benchmark.py --runs /kaggle/working/geofirex-runs/<run> --export-dir /kaggle/working/exports/session-unique --report /kaggle/working/exports/verification-session-unique.json
```

L'archive contient sources, JSON, figures et checkpoints complets, sans les
tris AP régénérables, avec un SHA256 séparé. Un export existant est refusé.
Le vérificateur versionné contrôle le checkpoint, la cohérence résumé/validation,
le nombre de fenêtres audité (sans imposer 42), les sources archivées et le statut
MLflow FINISHED. Il vérifie ensuite **chaque fichier** du TAR par taille/SHA256,
sans extraction ni chargement du checkpoint. Le rapport public conserve les
basenames, hashes et métriques, sans copier les configs ou chemins absolus.

Pour revérifier des exports existants sans les réécrire :

```bash
python scripts/verify_and_export_benchmark.py --runs /kaggle/working/geofirex-runs/<run> --archive-dir /kaggle/working/exports/session-unique --report /kaggle/working/exports/reverification-unique.json
```

Plusieurs dossiers s'ajoutent après `--runs`. `--require-comparable` exige leur
même protocole/budget/données/normalisation/sources ; il ne classe aucun modèle.
Après déplacement, `--tracking-uri` désigne la base restaurée. En consultation
hors ligne sans MLflow, `--skip-mlflow` est explicite et le rapport porte
`not_checked_offline`, jamais une vérification FINISHED simulée. Les chemins de
ce script sont relatifs au répertoire courant, ou absolus selon l'argument.

Le CLI `run_benchmark.py export` reste disponible comme export simple, y compris
pour un run interrompu : le vérificateur complet exige un `summary.json` terminé.
Il ne déclenche ni entraînement, ni évaluation test, ni écriture MLflow.
Conserver aussi `tracking.db`, `mlflow_artifacts`, le verrou de sélection et le
rapport de préflight pour une restauration complète de l'interface MLflow.
Sauvegarder la base après fermeture des runs/processus. Une archive seule permet
d'inspecter/reprendre le run ; elle ne restaure pas à elle seule la base MLflow.

`mlflow_enabled: false` permet les tests locaux sans serveur ; les artefacts
et la provenance restent écrits. Ces exécutions ne sont pas présentées comme
des expériences MLflow.

Le prototype historique `artifacts/benchmark/verify_and_export.py` est conservé
localement avec les artefacts. Aucun script ne l'appelle : il ciblait les quatre
diagnostics du 30 octobre 2026 et un rapport daté. Il n'est pas nécessaire dans
un checkout GitHub/Kaggle ; sa logique réutilisable est désormais dans
`scripts/verify_and_export_benchmark.py`. Ne pas forcer l'ajout d'`artifacts/` à Git.
