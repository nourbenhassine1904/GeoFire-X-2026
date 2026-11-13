# GeoFire-X — point de reprise du 1er novembre 2026

Arrêt demandé par l'utilisateur pour éteindre le PC. Contrôles locaux terminés autour de 01 h 02, heure de Tunis (00 h 02 UTC). Aucun nouvel appel réseau, transfert, entraînement ou job Kaggle n'a été lancé après cette demande. Aucun commit ni push effectué. Ne pas reprendre automatiquement les transferts demain.

## Résultats confirmés du diagnostic Kaggle

Les deux acquisitions sont annulées, et ne tournent plus. Le statut officiel est `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`. Les sources distantes correspondent exactement aux SHA-256 des jobs préparés localement. Les deux jobs sont privés, version 1.

| Acquisition | Handle Kaggle | GeoTIFF finalisés présents dans les sorties | Événements complets selon les chemins officiels | Observations manquantes |
|---|---|---:|---:|---:|
| Train 2018 | `nnnnbhss/geofirex-acquire-20261031-205829-923ffb32` | 483 / 714 | 24 / 37 | 231 |
| Validation 2020 | `nnnnbhss/geofirex-acquire-20261031-205845-9e2ab980` | 460 / 546 | 24 / 30 | 86 |

Chaque sortie couvre 25 événements : 24 complets par présence des chemins et un événement partiel. Il reste donc **943 GeoTIFF finalisés, 48 événements complets par inventaire des chemins, 317 observations manquantes et deux fragments `.part`**. Ces comptes ne constituent pas une certification indépendante de l'intégrité des 943 fichiers.

L'API des sorties renvoie 928 fichiers au total pour train et 905 pour validation, outils et code inclus. `kaggle kernels files` renvoyait `[]` parce qu'il interroge les fichiers sources ; il ne faut pas l'utiliser pour conclure à l'absence de sorties. La pagination de `ListKernelSessionOutput` a permis de retrouver les données.

Les logs officiels sont `[]` et `failure_message` est vide pour les deux acquisitions. **La cause exacte de l'annulation n'est pas exposée par les réponses obtenues.** Ne pas l'attribuer sans preuve à un quota, un timeout, une erreur Zenodo ou une annulation humaine.

## Volumes et intégrité : ce qui est établi

| Mesure | Train | Validation |
|---|---:|---:|
| Volume décompressé attendu des TIFF présents, selon l'index officiel | 1 951 174 222 octets | 1 789 644 978 octets |
| Taille distante mesurée et conforme à l'index | 398 / 483 TIFF | 433 / 460 TIFF |
| Somme des tailles effectivement mesurées | 1 604 439 380 octets | 1 671 059 532 octets |
| Tailles restées inconnues à cause d'erreurs réseau | 85 TIFF | 27 TIFF |
| Fragment `.part` conservé, taille distante mesurée | 4 194 304 octets | 1 048 576 octets |
| Somme des tailles ZIP compressées des membres finalisés, selon l'index | 1 915 858 684 octets | 1 757 590 546 octets |

Le total logique attendu des 943 TIFF est **3 740 819 200 octets**. Ce total provient de l'index officiel ; la mesure distante des 112 tailles manquantes n'a pas été achevée. Les 831 tailles mesurées sont toutes conformes. Erreurs relevées : TLS (`SSLError`), connexion et délais de connexion/lecture. Aucun nouveau contrôle réseau à poursuivre ce soir.

Le point d'accès des sorties refuse `HEAD` (404). Les mesures exploitables ont utilisé des requêtes `Range: bytes=0-0` avec fermeture du flux, sans télécharger de fichier GeoTIFF sur le PC : au plus 945 octets de corps sollicités pour le relevé complet, plus la sonde préalable d'un octet. Aucun transfert massif de raster n'a été effectué pour ce diagnostic.

**Les octets Zenodo réellement transférés sont inconnus.** Les deux jobs figés n'avaient pas de journal réseau durable, et aucun rapport final `acquisition.json` ni certificat n'a été récupéré. Les sommes compressées ci-dessus sont des minima de charge utile correspondant aux membres finalisés, pas des compteurs réseau incluant index ZIP, en-têtes, fragments et retries. La limite programmée était de 3 Gio par lot, soit 6 Gio pour les deux. Le solde garanti réutilisable est donc traité comme **0 octet tant que la consommation n'est pas établie** ; ne pas réinitialiser les compteurs pour une reprise.

Les deux datasets attendus n'apparaissent pas dans l'inventaire privé authentifié :

- `nnnnbhss/gfx-wsts-train2018-20261031-205829-923ffb32`
- `nnnnbhss/gfx-wsts-val2020-20261031-205845-9e2ab980`

Aucune publication de ces deux lots n'est confirmée. Les sorties de notebook conservées ne sont pas des datasets privés certifiés. L'inventaire authentifié contient cinq datasets privés, pour 498 891 230 octets déclarés par `GetDataset` ; ce chiffre n'est pas un compteur personnalisé de quota.

## Données disponibles et certifications à conserver

- **Certifié, à ne pas recommencer :** `nnnnbhss/geofire-x-wsts-batch001/versions/1`, événement `2018/fire_21458798`, 14 GeoTIFF, 45 061 097 octets de raster. Certificat `PERSISTENCE_VERIFIED_NEW_SESSION`, session indépendante `372fa8b7497d402fa6279555564401a0`. Fichier : `artifacts/orchestrator/jobs/20261031-202857-f13d587d/retrievals/c3cea258cd874f269ff107ee0d80f702/geofirex-result/bootstrap.json`.
- **Réussi, à ne pas recommencer :** Spark, job `20261031-210700-478e4d85`, statut `SPARK_EXECUTED`, index réel de 13 607 observations et 607 événements, QC du petit export réel de 108 lignes. Les sorties Parquet et leur vérification locale sont conservées dans `artifacts/orchestrator/jobs/20261031-210700-478e4d85/retrievals/676ab33ae0da410f848de910287022cb/`.
- **Disponible mais non certifié dans ce diagnostic :** les 943 TIFF des deux acquisitions annulées. Leur contenu n'a pas été relu intégralement pour SHA-256/CRC depuis une session indépendante. Les 48 événements complets sont établis par les chemins attendus, avec les limites de mesure précisées plus haut.
- **Dataset privé existant à examiner demain :** `nnnnbhss/geofire-x-wildfirespreadts-sample`, version 1, 447 927 526 octets déclarés. Son contenu Kaggle n'a pas reçu de certificat complet ce soir. Ne pas le déclarer certifié sur la seule base de son nom ou de sa taille.
- **Échantillon local historique préservé :** 108 TIFF dans `data/raw/WildfireSpreadTS/real_sample`, cinq événements, trois train et deux validation dans le protocole local existant. Comparaison CPU historique et checkpoints conservés. La base MLflow `artifacts/demo-training/tracking.db` est présente (1 150 976 octets). Ces résultats courts ne sont pas un benchmark GPU complet.

Un contrôle CPU indépendant a été tenté avant la demande d'arrêt : `nnnnbhss/geofirex-retained-20261031-234152-9cba7fec`. Il est terminé (`COMPLETE`), mais Kaggle a refusé les deux acquisitions annulées comme `kernel_sources` (« not valid kernel sources »). Le script s'est exécuté sans leurs données et a trouvé zéro fichier valide. **Son champ technique `success: true` ne constitue aucun certificat de données.** Preuve conservée : `artifacts/orchestrator/retained-verification/20261031-234152-9cba7fec/retrieval/retained-verification/integrity.json`. Ne pas relancer ce même contrôle sans résoudre l'attachement.

## Causes du faux UNKNOWN et modifications de code

1. Le jeton OAuth avait expiré le 31 octobre à 23 h 24 UTC. Le SDK installé considère l'expiration avec un retard de 30 minutes, provoquant des refus d'accès privés pendant cet intervalle. Le jeton a été renouvelé via `KaggleCredentials.refresh_access_token()` et l'identité `nnnnbhss` vérifiée par introspection, sans afficher ni enregistrer de secret dans le projet.
2. Le parseur local ne reconnaissait pas les underscores de `CANCEL_ACKNOWLEDGED`. `src/geofirex/orchestration.py` accepte maintenant les formes avec et sans underscore. Les deux événements `REMOTE_STATUS` sont désormais enregistrés comme `CANCELACKNOWLEDGED`.
3. `tests/test_orchestration.py` : quatre cas ciblés de statut CLI ajoutés et exécutés, **4 réussis**. La précédente suite complète reste une preuve historique de **194 tests réussis**, dans `reports/orchestration_tests_final_v6.xml` ; elle n'a pas été réexécutée après ces changements. Ne pas annoncer une nouvelle suite complète de 198 tests.
4. `scripts/kaggle_account_inventory.py` : authentification OAuth explicite, renouvellement anticipé et introspection avant inventaire privé, afin qu'une réponse vide non authentifiée ne soit plus présentée comme une absence de données.
5. `scripts/kaggle_acquisition_diagnostic.py` : lecture paginée de l'API officielle, empreinte du code distant, inventaire des sorties et sondes de taille bornées. Pas de téléchargement de raster complet.
6. `scripts/kaggle_prepare_retained_verification.py` : préparation du contrôle CPU sans Internet, tailles/CRC/SHA, titre raccourci après rejet initial HTTP 400 pour titre supérieur à 50 caractères. Les versions locales suivantes refusent explicitement une entrée vide ; elles n'ont pas été resoumises.
7. `scripts/kaggle_recover_retained_outputs.py` : prototype de récupération cloud présent sur disque, **jamais exécuté**. Il copierait les TIFF depuis les sorties Kaggle et exige un fichier de spécification activé. Ce n'est pas une reprise validée respectant la contrainte de ne pas retélécharger les données valides ; ne pas l'exécuter sans réexamen du transport et des budgets. La préparation du fichier `artifacts/orchestrator/acquisition-resume-20261101.json` a été interrompue : ce fichier n'existe pas.

Les autres changements de la session précédente sont conservés : orchestration, acquisition bornée avec cache de membres ZIP et journal durable pour les futurs jobs, persistance, démo React/inférence réelle, MLflow et comparaisons CPU. Voir `docs/ORCHESTRATION_SUIVI3.md` et `reports/suivi3_orchestration_20261031T223200Z.md`. Les deux acquisitions annulées utilisaient du code figé antérieur au cache et au journal durable ; corriger le code local ne modifie pas leurs anciennes exécutions.

L'archive historique `artifacts/kaggle/geofirex-orchestrated-code-v4.2.zip` est conservée intacte. Elle précède les corrections de ce diagnostic et ne doit pas être présentée comme contenant ces nouvelles modifications. Le travail est enregistré dans l'arbre local, sur `feature/unet-baseline`, sans commit ni push.

## Processus locaux et arrêt du PC

- Le PID **43316** a été vérifié par le système : il correspondait effectivement à `scripts/kaggle_chain.py` pour les deux acquisitions, avec des cycles récents. Ce n'était pas une simple déduction à partir d'un PID sauvegardé.
- Le superviseur a été mis en pause par `artifacts/orchestrator/chain-pilot-v1/stop.request` à 23 h 35 UTC. Il a quitté sa boucle et produit `stopped-479ae155.json` à 23:35:18 UTC. Au contrôle final, le PID 43316 est absent.
- Les collecteurs du diagnostic (`--head`, arrêté après constat d'incompatibilité, puis `--sizes`, terminé avec rapport) ne sont plus actifs. Aucun CLI Kaggle, acquisition, entraînement ou serveur Python du projet ne reste actif au contrôle final.
- Les seuls Python encore présents sont ceux du serveur isort de VS Code : PID 39340 et son enfant 26188. Ils peuvent se fermer avec VS Code.
- Les PID Node 25348, 37548, 37380 et 30140 appartiennent aux runtimes/outils Codex, pas à des entraînements ou à une supervision Kaggle. Ils ne nécessitent pas de maintenir le PC allumé.
- La chaîne conserve ses anciens cycles `WAITING_FOR_ACQUISITION` et son PID historique. Ces traces ne prouvent aucune activité actuelle. Le fichier `stop.request` est volontairement conservé ; ne pas relancer aveuglément la chaîne ni retirer ce garde-fou demain.

**Le PC peut être éteint : aucun travail GeoFire-X actif identifié ne dépend encore de sa continuité.** Les deux acquisitions sont déjà annulées côté Kaggle ; le seul nouveau contrôle CPU était déjà terminé avant la demande d'arrêt. Cette conclusion porte sur les processus du projet observés, pas sur les documents éventuellement non enregistrés d'autres applications.

## Preuves locales et conservation de l'historique

Preuves du diagnostic :

- `artifacts/orchestrator/acquisition-diagnostic-20261101.json` : pagination complète des sorties, 943 chemins TIFF, fragments, statuts, empreintes des sources, logs vides.
- `artifacts/orchestrator/acquisition-diagnostic-sizes-20261101.json` : tailles mesurées et erreurs des 112 sondes non abouties.
- `artifacts/orchestrator/account-inventory-diagnostic-authenticated-20261101.json` : cinq datasets privés, compte authentifié après renouvellement. Le fichier antérieur `account-inventory-diagnostic-20261101.json` contient une réponse vide pendant la coupure d'authentification : il est préservé mais **n'est pas une preuve d'inventaire authentifié**.
- Répertoires `events/` et `retrievals/` des deux acquisitions : appels CLI, statuts corrigés et résultats `EVIDENCE_MISSING`. Aucun résultat d'acquisition complet n'a été récupéré.

Empreintes historiques revérifiées localement, identiques aux preuves précédentes :

- Certificat batch001 : `0be0864ab9cfc43bcdf2788f69e3f89505b6d80f95bf0e2d8ddbb3df38e6333a`.
- Archive code v4.2 : `28d4506a10c8545bb68362f4cc47f03053087557de20d42ce55260111cacb88c`.
- Index officiel JSON : `990fe900bd8727497d78bc51d005bc1c90dc15eed38d675e849b5776da4d2686`.
- Preuve d'adoption manuelle : `9a65bebd52e4ec646b0d60c388be66ea94806b2a0b090b88a28ba352f9fca5e9`.

Les fichiers historiques, rapports précédents, fragments, checkpoints et base MLflow restent sur disque. Aucun fichier n'a été supprimé après la demande d'arrêt. Aucune réécriture de l'historique Git, aucun commit et aucun push n'ont été effectués.

## Blocages et trois prochaines priorités pour le suivi 3

**Blocages :** annulation sans motif fourni ; journaux réseau d'origine absents ; intégrité indépendante des 943 TIFF non certifiée ; attachement des sorties annulées refusé ; 112 mesures de taille non obtenues. La récupération par téléchargement des mêmes fichiers n'est pas autorisée comme solution automatique. Les règles et quotas doivent être relus avant toute reprise ; l'ancien plafond ne constitue pas un nouveau crédit de transfert.

1. **Rendre utilisables les données déjà disponibles.** Commencer par une certification CPU en lecture seule du dataset privé existant `geofire-x-wildfirespreadts-sample`, en vérifiant le contenu réel contre l'index et les splits. En parallèle de la préparation, déterminer une méthode Kaggle supportée de conservation/réutilisation des sorties annulées sans retéléchargement des TIFF valides. Ne pas refaire session B ni Spark. Ne pas déclarer certifiés les 48 événements avant CRC/SHA et preuve de persistance indépendante.
2. **Préparer un vrai pilote GPU B1/B2/B3 limité et reproductible**, dès qu'un jeu train/validation existant est certifié : mêmes événements, contexte, patches, seed et budget d'updates, MLflow, checkpoints et validation temporelle séparée. Si le sample confirme les cinq événements locaux, annoncer explicitement trois événements train et deux validation, sans test 2021 ni prétention de benchmark complet. Proposer un plafond court avant exécution ; le scope actuel de 900 s consacré uniquement au débit avec poids jetés ne couvre pas implicitement un nouvel entraînement avec checkpoints. Avec seulement batch001 certifié, seul un contrôle d'apprentissage/débit sur un événement est actuellement défendable, sans validation indépendante.
3. **Préparer la reprise des 67 événements avec réutilisation stricte.** Les deux plans figés et leurs 943 chemins conservés sont la base : contrôler et réutiliser ces fichiers, préserver les deux `.part`, ne demander que les **231 + 86 observations manquantes** après établissement du budget restant. Utiliser le cache borné et un journal de transfert durable, vérifier les handles avant publication pour éviter les doublons, puis certifier depuis une nouvelle session. Si la consommation passée demeure inconnue, chiffrer le besoin supplémentaire et obtenir un périmètre explicite avant tout transfert. Aucun lancement automatique, entraînement long ou push GitHub.
