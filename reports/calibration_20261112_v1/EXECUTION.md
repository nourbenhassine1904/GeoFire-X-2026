# Calibration B1/B2 — exécution Kaggle

Livrable préparé le 12 novembre 2026 pour le suivi du 18 novembre. Le notebook
embarque son code : ni clone du dépôt ni accès Internet nécessaires. Les seuils
et gains réels seront connus après son exécution avec les checkpoints Kaggle.

## 1. Ouvrir le notebook et attacher les entrées existantes

Importer `GEOFIREX_B1_B2_CALIBRATION_20261112.ipynb` dans Kaggle. Choisir GPU T4,
Internet désactivé. Ajouter **les 11 datasets déjà publiés**, sans nouvelle version :

```text
nnnnbhss/geofirex-compact-t3-20261111-p00
nnnnbhss/geofirex-compact-t3-20261111-p01
nnnnbhss/geofirex-compact-t3-20261111-p02
nnnnbhss/geofirex-compact-t3-20261111-p03
nnnnbhss/geofirex-compact-t3-20261111-p04
nnnnbhss/geofirex-compact-t3-20261111-p05
nnnnbhss/geofirex-compact-t3-20261111-p06
nnnnbhss/geofirex-compact-t3-20261111-p07
nnnnbhss/geofirex-compact-t3-20261111-p08
nnnnbhss/geofirex-compact-t3-20261111-p09
nnnnbhss/geofirex-compact-t3-20261111-p10
```

Les fichiers `.bin` contiennent les octets ZIP/NPZ d'origine. Le notebook contrôle
les 11 identités, les tailles et les SHA256, puis crée des liens vers les montages.
Il ne reconstruit pas le corpus et n'extrait pas ses archives.

Ajouter aussi, via **Add Input → Notebook Output**, les sorties **sauvegardées** des
notebooks contenant :

```text
/kaggle/working/B1-continued-20261112_135150/checkpoint_e005_u0023404_epoch_complete.pt
/kaggle/working/B2-continued-20261112_141330/checkpoint_e005_u0023404_epoch_complete.pt
```

Dans une nouvelle session, `/kaggle/working` est différent : les checkpoints seront
sous `/kaggle/input/...`. La recherche automatique examine ce nom exact et vérifie
`config.model`. Si plusieurs copies existent, renseigner les deux chemins exacts
dans `CHECKPOINTS`. Aucun réupload de dataset n'est requis. Une sortie de notebook
qui n'a jamais été conservée doit d'abord être sauvegardée dans sa session source.
Les handles de ces deux notebooks ne sont pas fournis dans la demande : le fichier
`kernel-metadata.json` contient les 11 datasets et laisse donc `kernel_sources=[]`.
Le compléter seulement avec les sources réellement identifiées si utilisation CLI.

## 2. Exécuter les cellules dans l'ordre

Paramètres préremplis : `DEVICE="cuda"`, batch 32, 4 workers, cache 64 Mio par
processus. `DEVICE="cpu"` fonctionne aussi ; diminuer les workers en cas de RAM
limitée. Seuls NumPy, PyTorch et Matplotlib sont nécessaires ; aucune installation
n'est effectuée par le notebook. Versions compatibles dans
`requirements-calibration.txt`, versions effectives enregistrées dans `protocol.json`.

La cellule de calcul exécute exactement cette commande sous forme de liste :

```python
command = [sys.executable, str(workspace / "calibrate_compact.py"),
    "--corpus", str(corpus), "--index", str(index),
    "--b1", str(CHECKPOINTS["B1"]), "--b2", str(CHECKPOINTS["B2"]),
    "--out", str(OUT), "--device", DEVICE, "--batch-size", "32",
    "--workers", "4", "--cache-mb", "64"]
subprocess.run(command, check=True, env=os.environ.copy())
```

`workspace`, `corpus`, `index`, `CHECKPOINTS` et `OUT` sont créés par les cellules
précédentes. Pour rejouer seulement la commande, conserver ces mêmes variables et
choisir un **nouveau** `OUT` ; un dossier existant n'est jamais écrasé. Le script
est aussi disponible après extraction de `calibration-code.zip` :

```bash
python calibrate_compact.py --help
```

Ordre imposé par le moteur :

1. Contrôler 250 événements train / 93 603 tuiles, 201 événements val / 78 771
   tuiles, checkpoints époque 4 / 23 404 updates, normaliseur et index identiques.
2. Lire exactement les 32 IDs sauvegardés dans les checkpoints ; contrôler leurs
   13 759 tuiles. Aucun événement n'est sélectionné selon son score.
3. Auditer les labels train sur CPU, toutes les tuiles, sans prédiction ni fit.
4. Inférer une fois chaque tuile des 32 événements, pour chacun des modèles.
   Comparer à l'historique à 0,5 ; domaine de pixels identique exigé et tolérance
   absolue 0,002 sur les métriques pour les différences de calcul entre matériels.
5. Comparer 0,10–0,95 par 0,05, puis affiner ±0,05 par 0,005 dans ces bornes.
   Maximiser Dice **micro** ; égalités : précision, puis seuil le plus élevé.
   Les comptages fins sont accumulés pendant la même inférence, sans second passage.
6. Écrire `thresholds_locked.json` et son SHA **pour les deux modèles**.
7. Inférer les 169 événements restants / 65 012 tuiles. Évaluer seulement les
   seuils 0,5 et figés ; aucune recherche de seuil sur ce groupe.
8. Exporter comparaisons, distributions, diagnostic et mosaïques réelles.

L'AP exacte utilise des tris/fusions disque avec regroupement global des scores
égaux ; elle n'est ni une AP histogramme ni une aire PR trapézoïdale. Le Brier est
la moyenne de `(p-y)**2` sur les pixels valides, accumulée en float64. Les scores
float32 du modèle ne sont ni quantifiés ni corrigés par la pondération de la loss.
Le seuil change seulement la segmentation et sa matrice de confusion.

## 3. Ressources et fin d'exécution

Prévoir environ **12–15 Go libres** pour le scratch de la plus grande phase AP,
en plus des anciens outputs présents. Les sources restent montées en lecture seule.
Le moteur vérifie l'espace avant les écritures et les fusions ; les runs AP sont
supprimés après leur calcul. Les AP par événement et globale sont calculées
séparément, ce qui ajoute du travail CPU/disque. Aucun temps total n'est annoncé
sans mesure sur cette session. L'audit train et les fusions AP n'entraînent rien.

Une validation est complète seulement si `status_complete.json` indique
`"status": "COMPLETE"`. Les résultats intermédiaires restent des résultats partiels.
Il n'y a pas de reprise automatique après interruption : choisir un nouveau `OUT`
et relancer. Ne pas mélanger des sous-dossiers de deux exécutions.

Les sorties sont dans `/kaggle/working/calibration-B1-B2-<date_heure>/`. Sauvegarder
la version du notebook avec ses outputs, puis les télécharger. Si le notebook a
été enregistré sous l'identifiant proposé dans `kernel-metadata.json` :

```powershell
kaggle kernels output nnnnbhss/geofirex-b1-b2-calibration-20261112 -p artifacts/calibration_20261112_download
```

Cette commande de téléchargement ne lance pas le notebook. Référence CLI :
[documentation officielle Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
Aucun appel Kaggle, lancement GPU, commit ou push n'a été effectué pour préparer
ce livrable.

## 4. Fichiers à lire pour le suivi

| Fichier produit | Utilisation |
|---|---|
| `protocol.json` | Checkpoints SHA, sources de validation, versions et conventions |
| `train_audit.json` | Prévalence pixel train, nouveaux feux, classes de tuiles, loss |
| `B1_historical_reproduction.json`, `B2_historical_reproduction.json` | Contrôle de la référence 0,5 |
| `thresholds_locked.json` | Deux seuils fixés sur les 32 seulement, SHA des checkpoints |
| `comparison.csv` / `.json` | B1/B2, baseline/calibré, 32/169, métriques micro et macro |
| `B*_*/per_event.csv` | TP, FP, FN, TN, Dice, IoU, précision, rappel, AP, Brier, par événement |
| `B*_*/new_fire_per_event.csv` | Même diagnostic sur les pixels sans feu observé à J |
| `B*_*/evaluation.json` | Sous-domaines nouveaux feux / feu observé, probabilités et comptages |
| `B*_*/event_*_probabilities.json` | Histogrammes conditionnels et fiabilité par événement |
| `diagnosis.json` | Contribution des nouveaux feux, FP, prévalences et expériences conditionnelles |
| `calibration_curves.png` | Courbes de recherche sur les 32 seulement |
| `*_diagnostics.png` | Histogrammes, fiabilité, distributions CDF et FP/prévalence |
| `app_manifest.json`, `B*_*/map_manifest.json` | Contrat JSON, couches PNG et NPZ pour l'application |
| `status_complete.json` | Certification de complétude de ce calcul |

Les mosaïques concernent six événements choisis par IDs régulièrement espacés
(trois dans chaque groupe), date cible médiane, toutes les tuiles de cette date.
Ces illustrations ne sont pas un échantillon représentatif et ne servent jamais
à sélectionner le seuil. Dice/IoU vide-vide=1 ; précision sans prédictions et
rappel/AP sans positifs sont `null`. Les moyennes macro excluent les valeurs
indéfinies et indiquent le nombre d'événements définis.

## 5. Reproduction locale des contrôles du livrable

Depuis la racine du dépôt, environnement existant :

```powershell
.\.venv\Scripts\python.exe scripts\prepare_calibration_notebook.py
.\.venv\Scripts\python.exe -m pytest tests\test_compact_calibration.py -q --basetemp reports\calibration_20261112_v1\synthetic-verification
```

Les tests synthétiques vérifient le calcul et l'isolation des groupes. Ils ne sont
pas des mesures des performances scientifiques des checkpoints réels.
