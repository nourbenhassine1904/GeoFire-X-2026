# GeoFire-X — lot 001 clos, publication 002–008 préparée

## Preuves du lot 001

Les trois téléchargements sont copiés sans modification dans `../kaggle_lot001_certified_20261108_v1/`. `verification.json` contient leurs tailles, SHA-256 et chemins d'origine. Le certificat final porte `LOT001_REMOTE_CPU_CERTIFIED` : 1 269 fichiers, 4 829 474 073 octets, 57 événements train, aucune anomalie. Chaque ligne de `verified.jsonl` correspond exactement au chemin, à la taille, au SHA-256 et au CRC32 du manifeste figé. Les lectures de cinq GeoTIFF à 23 bandes et du DataLoader CPU à huit canaux sont consignées. Aucun recalcul des empreintes des rasters du lot 001 n'a été exécuté.

`progress.json` conserve `RUNNING` : c'est le dernier instantané de la phase des empreintes. Ses compteurs sont cohérents avec le certificat final, qui confirme aussi les lectures rasterio et PyTorch. Le fichier est préservé tel quel. Les trois observations de contexte de décembre 2017 autorisées sont explicitement consignées dans le certificat.

## Lots restants

Tous les noms ci-dessous appartiennent à `nnnnbhss`. Les tailles sont les octets GeoTIFF exacts, avant enveloppe ZIP.

| Dataset | Rôle | Événements | GeoTIFF | Octets |
| --- | --- | ---: | ---: | ---: |
| geofirex-wsts-fold0-tv-002-v1 | train | 62 | 1 380 | 4 995 042 486 |
| geofirex-wsts-fold0-tv-003-v1 | train | 68 | 1 344 | 4 905 540 263 |
| geofirex-wsts-fold0-tv-004-v1 | train | 63 | 1 205 | 4 443 126 438 |
| geofirex-wsts-fold0-tv-005-v1 | validation | 69 | 1 305 | 4 886 889 025 |
| geofirex-wsts-fold0-tv-006-v1 | validation | 54 | 1 350 | 4 919 863 753 |
| geofirex-wsts-fold0-tv-007-v1 | validation | 61 | 1 321 | 4 833 117 096 |
| geofirex-wsts-fold0-tv-008-v1 | validation | 17 | 316 | 1 182 299 296 |

État commun : manifestes et tailles des 8 221 fichiers locaux vérifiés ; événements complets et disjoints contrôlés contre l'index initial et les tables QC existantes ; notebooks CPU privés préparés et compilés. Aucun ZIP massif reconstruit et aucun lot 002–008 publié. Aucune nouvelle lecture de pixels, préparation scientifique ou normalisation.

Les sept lots ajoutent 193 événements train et 201 validation. Avec les 57 train du lot 001 : exactement 250 train, 201 validation et 9 490 observations. Aucun événement de test ni observation de 2021 dans les sept lots.

## Budget et chaîne séquentielle

- GeoTIFF restants : **30 165 878 357 octets** (30,166 Go décimaux).
- Plafond des sept ZIP : **30 235 878 357 octets** (10 Mo de marge par ZIP). Leur taille réelle et SHA-256 seront mesurés lors du staging ; les ZIP ne sont pas prétendus déjà construits.
- Plafond proposé : **31 000 000 000 octets de corps HTTP émis**, partagé entre les sept lots et leurs contrôles CPU. Le nouveau journal `http-body.jsonl` inclut déjà les petits contrôles de préparation ; il ne sera jamais remis à zéro. L'ancien compteur du lot 001 est préservé séparément.
- Enveloppe indicative de trafic physique : **32 Go**, incluant une marge pour en-têtes/TLS et les petites preuves téléchargées. Le journal des corps HTTP ne mesure pas le trafic physique.
- Zéro retry automatique, un seul PUT par lot, aucun remplacement/versionnement de dataset. Les archives locales sont conservées ; prévoir au maximum 50,236 Go libres, réserve de 20 Go comprise.

Commande PowerShell exacte, depuis la racine du projet, **uniquement après autorisation explicite** :

```powershell
& ".\artifacts\orchestrator\kaggle-cli-env\Scripts\python.exe" ".\scripts\publish_remaining_lots.py" --plan ".\reports\kaggle_remaining_20261108_v1\plan.json" --execute --approve-body-bytes 31000000000
```

Ordre : contrôle d'automatisation CPU sans dataset → staging d'un lot avec taille/SHA/CRC → catalogue propriétaire complet/OAuth → `StartBlobUpload`/endpoint GCS validé → PUT séquentiel unique → `CreateDataset` privé → certification CPU de ce seul lot et récupération des preuves → validation locale des reçus → lot suivant. Le programme réutilise les correctifs du mécanisme du lot 001 mais en interdit l'upload dans cette phase.

À toute erreur, accès refusé, collision avec un dataset existant, incohérence de reçu ou dépassement de ressources : arrêt de la chaîne, conservation des journaux et aucune relance automatique. Un répertoire `execution/` déjà présent interdit une répétition aveugle. Le certificat de chaque lot est vérifié avant d'avancer. L'automatisation CPU n'alloue aucun GPU.

## Entraînements préparés, non lancés

Les quatre configurations et notebooks sont sous `training/`. Ils embarquent le code et les petites métadonnées ; aucun dataset de code supplémentaire n'est requis. Les notebooks sont bloqués tant que les sept certificats manquent. La publication séquentielle régénérera les quatre notebooks après validation des sept reçus, toujours sans soumettre de tâche GPU.

1. Attacher les huit datasets privés existants aux notebooks. Le montage découvre les inventaires par identité, valide fichiers/tailles et résout les chemins réels `/kaggle/input/datasets/…` sans dépendance Windows ; il réutilise les certificats et ne relance pas la certification du lot 001.
2. B0 Persistence : évaluation complète sur CPU, sans optimisation.
3. B1 CNN, B2 U-Net, B3 ConvLSTM : après autorisation séparée des entraînements, profiler chaque modèle pendant au plus 900 s, puis exécuter la première époque complète, au plus 7 200 s par modèle et sous condition que la projection du profil tienne dans ce budget. Les sorties utilisent de nouveaux répertoires.
4. Protocole repris : fold officiel 0, T=3, seed 42, huit canaux, 93 616 patches par modèle appris, batch 16, soit 5 851 mises à jour prévues. Même domaine temporel, masque de validité et traitement des features. Cette première époque est un lancement réel sur le corpus élargi ; elle ne prouve pas la convergence ni la fin du benchmark.
5. Normalisateur réutilisé, SHA-256 `aa30df4e3d6d895760f04e35d74bcbbac0d739cea045d236517401f5f9f4c7f6` ; 250 événements train, 4 948 observations de contexte utilisées lors du fit, zéro validation/test au fit. Les quatre liaisons des tables d'entrée sont vérifiées.
6. Dice, IoU, précision, rappel et Average Precision calculés par le runner existant. Sélection par AP validation, seuil par Dice validation ; aucun lancement d'évaluation test 2021 n'est généré.

Le contrôle Kaggle a confirmé que `mlflow` est absent de l'image CPU. Les nouvelles copies utilisent donc le support existant `mlflow_enabled: false`, avec métriques JSON/CSV et checkpoints conservés ; aucune configuration historique ni calcul scientifique n'a été modifié. Aucun paquet n'a été installé. Les dépendances scientifiques requises restent vérifiées avant lancement.

Huit tests ciblés de reçus, périmètre, arrêt avant rasters et URL de téléchargement sont réussis. Aucun audit scientifique complet, GPU, commit, push ou merge.

## Contrôle réel de l'automatisation CPU

Le notebook privé sans dataset `nnnnbhss/geofirex-cpu-automation-gate-20261108-v2` s'est terminé avec succès sur Kaggle. `SaveKernel`, le suivi de session et la récupération de la preuve fonctionnent désormais. La preuve de 153 octets est enregistrée dans `automation-receipt-recovered/environment.json` ; le contrôleur porte `CPU_AUTOMATION_READY`. La récupération a utilisé exactement l'URL renvoyée par le SDK sur l'hôte officiel `www.kaggleusercontent.com`, sans redirection et avec une limite de téléchargement de 2 Mo. La liste d'hôtes autorisés et ses rejets ont un test ciblé.

Les deux petits notebooks de contrôle, dont le premier a révélé `mlflow` absent, n'ont attaché aucun dataset ni transféré de raster. Tous les journaux sont conservés. Le dernier cumul du journal de préparation et récupération est de 5 440 octets de corps HTTP sur le plafond proposé de 31 Go. Le lot 001 n'a été ni reconstruit, ni réuploadé, ni recertifié.

Capacité locale au contrôle : 109 879 537 664 octets libres sur le volume de staging, pour un besoin maximal estimé de 50 235 878 357 octets en conservant tous les ZIP et la réserve de 20 Go. Le staging remesurera la place disponible avant chaque archive.
