# GeoFire-X — reprise contrôlée du 7 novembre 2026

**État : sauvegardes récupérées et contrôlées ; finalisation suspendue par manque de RAM. Le paquet Kaggle n'est pas déclaré validé ou figé.** Aucun upload, téléchargement massif, entraînement GPU, commit, push ou merge n'a été effectué. Aucun résultat historique n'a été modifié ou supprimé. Le contrôle a comparé les 369 fichiers inventoriés au départ ; aucun fichier manquant, aucun rapport historique changé. Les modifications de code de cette reprise restent à tester.

## Récupération vérifiée

La branche active est `feature/scientific-scaleup-v1`. L'état initial comprend 7 fichiers suivis modifiés et de nombreux fichiers non suivis, sauvegardés dans `git-status-before.txt`. `tracked-before.patch` conserve le diff suivi initial ; `preservation-before.json` conserve les tailles et SHA256 des 369 fichiers déjà modifiés/non suivis inventoriés. Cette liste ne constitue pas une sauvegarde exhaustive des fichiers ignorés, archives ou checkpoints. Ceux-ci n'ont fait l'objet d'aucune mutation.

Le refus Git initial venait de la différence de propriétaire Windows. Les lectures ont utilisé exclusivement `git -c safe.directory=...`, sans modifier la configuration globale.

Les 17 fichiers correspondant au périmètre GPU décrit sont présents et ont été inspectés :

- `scripts/run_benchmark.py`, `src/geofirex/benchmark_metrics.py`, `src/geofirex/runner.py`, `src/geofirex/runner_data.py`, `src/geofirex/runner_evaluation.py` : manifeste des fenêtres, normalisateur portable, séparation train/validation et AP externe.
- `src/geofirex/gpu_operational.py`, `src/geofirex/job_control.py`, `src/geofirex/upload_budget.py` : provenance du normalisateur, lots complets, supervision et journal du budget HTTP.
- `scripts/prepare_gpu_operational.py`, `scripts/prepare_expanded_code.py`, `scripts/profile_expanded_gpu.py`, `scripts/run_expanded_benchmark.py`, `scripts/mount_expanded_corpus.py`, `scripts/stage_expanded_lot.py`, `scripts/publish_expanded_lot.py` : préparation, intégration, profilage, montage, staging et publication explicite.
- `tests/test_gpu_operational.py`, `tests/test_gpu_resources.py` : contrats de déploiement et ressources.

Cette correspondance de 17 fichiers est reconstruite à partir du contenu ; aucun manifeste historique nominatif des « 17 fichiers » n'a été fourni. Le README et `corpus.py` avaient aussi des différences plus anciennes au départ ; elles ont été préservées.

Les rapports `reports/gpu_operational_v1/` sont présents : inventaire du compte, quota GPU, intégration CPU, tests initiaux, préparations `deployment`, `deployment-v2` et `deployment-v3`. La première préparation est partielle ; v2 contient les huit lots et v3 contient en plus les quatre configurations. Les anciens dossiers restent inchangés.

Le certificat historique complet `reports/full_corpus_v1/audit-summary-1793713096154941100.json` atteste 607 événements, 13 607 TIFF et **49 294 235 723 octets** pour tout le corpus. Ce contrôle physique ancien n'a pas été rejoué. Les **34 995 352 430 octets** annoncés pour l'upload concernent seulement train/validation : 451 événements, 9 490 TIFF. Le test 2021 représente 156 événements et 4 117 TIFF supplémentaires.

D: et `D:\GeoFire-X\WildfireSpreadTS\rasters` sont accessibles. L'espace libre mesuré au départ était de 438 386 307 072 octets sur D: et d'environ 119 Go sur C:. Aucune lecture exhaustive des pixels n'a été relancée.

## Diagnostic factuel

Le journal historique `gpu_operational_v1/tests-initial.xml` contient **41 tests terminés, 40 réussites et 1 échec, 140,108 s**. L'échec est `test_checkpoint_resume_is_identical_and_test_requires_lock` : `ValueError: Source changed after training; use the saved source snapshot for final test`. Ce journal n'est donc pas la preuve d'un test encore bloqué. Il ne permet pas d'identifier le processus de la dernière exécution interrompue. Le contrôle d'intégrité du code n'a pas été affaibli : la cause exacte de ce changement doit être confirmée lors d'une exécution sans édition concurrente des sources.

Pendant cette reprise, Windows a signalé **92 à 96 % de RAM utilisée**, pour 16 837 500 928 octets physiques. La mémoire disponible a oscillé autour de 0,6 à 1,4 Go. Au dernier précontrôle de finalisation : **995 221 504 octets disponibles, 94 % utilisés**. Les plus gros processus observés appartenaient notamment à VS Code, Chrome, un programme `setup`, Dropbox et ChatGPT. Aucune application utilisateur n'a été fermée.

La commande de référence a été arrêtée par le garde-fou après 8,735 s, sous 1 GiB disponible. Aucun résultat pytest complet ni XML nouveau n'a été produit. L'arrêt initial par `taskkill /T` a renvoyé « Accès refusé » ; il ne faut pas le présenter comme un arrêt d'arbre réussi. Les PID démarrés 16112 et 24016 n'étaient plus actifs au contrôle ultérieur. Un mécanisme Windows Job Object a été ajouté pour posséder et arrêter l'arbre du lanceur `.venv\Scripts\python.exe` lors des prochaines exécutions ; ses tests sont encore en attente.

La requête ciblée du journal Système a trouvé :

- 20 événements `disk` 51 retournés, le **3 novembre à 11:46:52, heure de Tunis**, sur `\Device\Harddisk1\DR26`, pendant une opération de pagination. La requête était plafonnée à 20 événements ; ce chiffre n'est pas un inventaire exhaustif.
- Un événement Kernel-Power 41 le **4 novembre à 00:40:03**, signalant un redémarrage sans arrêt propre préalable.
- Aucun événement 2004 correspondant au fournisseur Resource-Exhaustion-Detector dans la période interrogée.

Ces événements constituent des indices réels ; ils ne prouvent ni une panne de D: ni la cause du blocage pytest. L'association `Harddisk1`/lettre de lecteur n'a pas pu être vérifiée : `Get-Partition` et CIM ont renvoyé un accès refusé. Aucun processus Java/Spark n'a été observé lors des relevés. L'absence d'un événement mémoire 2004 ne contredit pas les mesures de RAM actuelles.

Les journaux complets sont dans `windows-events-targeted.json`, les sorties des tentatives dans les sous-dossiers de supervision. `windows-events.json` contient une première recherche par identifiants, avec des événements Hyper-V/démarrage sans lien démontré ; l'analyse repose sur la recherche ciblée par fournisseur.

## Preuves scientifiques récupérées

`saved-proof-verification.json` a été produit avec succès par un audit léger indépendant, sans Torch, Arrow, lecture de pixels ou réseau :

- Les SHA256 des observations Parquet, fenêtres Parquet, split et index correspondent exactement au certificat du normalisateur et à l'intégration CPU sauvegardée.
- La reconstruction des fenêtres temporelles consécutives depuis l'index officiel retrouve **3 948 train et 3 287 validation** sur le domaine maximal de cinq jours.
- Les entrées du normalisateur sont exactement les **4 948 fichiers de contexte de 250 événements train**, sans validation ni test. Les statistiques correspondent au normalisateur source vérifié.
- Les huit identités de lots sont recalculées et correctes. Les lots couvrent exactement 9 490 fichiers et 451 événements, sans doublon de fichier ou d'événement entre lots. Les tailles et CRC déclarés correspondent à l'index ; les rôles restent séparés.
- 153 fichiers Python ont passé l'analyse syntaxique. `git diff --check` a réussi ; seuls des avertissements Git de conversion LF/CRLF ont été émis.

Cette preuve valide les liaisons entre sauvegardes et la cohérence des plans. Elle ne remplace ni une nouvelle lecture des tables par Arrow, ni une vérification SHA256 actuelle des 35 Go de TIFF. L'audit physique exhaustif n'a pas été répété.

Le certificat portable du normalisateur a pour SHA256 :

`aa30df4e3d6d895760f04e35d74bcbbac0d739cea045d236517401f5f9f4c7f6`.

L'intégration CPU conservée avait réussi en 77,030 s sur huit événements, 32 patches, huit canaux et des sorties B0/B1/B2/B3 finies. Elle indiquait zéro entrée test, zéro mise à jour optimiseur, zéro écriture brute et zéro octet réseau. Elle n'a pas été rejouée inutilement ; ses entrées ont été revérifiées par empreintes.

## Tests et opérations de cette reprise

| Opération | Résultat complet disponible | Conclusion |
|---|---|---|
| Suite historique, 3 novembre | `../gpu_operational_v1/tests-initial.xml` | 40/41 réussis ; échec d'intégrité des sources |
| Commande de référence demandée, délai 240 s | `tests_reference/output.log`, `tests_reference/supervisor.json` | Arrêt mémoire après 8,735 s ; pas de suite validée |
| Audit Arrow, délai 90 s | `metadata_audit/output.log`, `metadata_audit/supervisor.json` | Garde mémoire après 2,312 s ; une assertion incorrecte de l'auditeur sur le champ supplémentaire `targets` a aussi été trouvée et corrigée ; pas de succès revendiqué |
| Quatre nouveaux tests légers, délai prévu 45 s | `tests_lightweight/supervisor.json` | Refus avant création du processus : 911 958 016 octets disponibles, seuil 1 GiB |
| Vérification légère des preuves sauvegardées | `saved-proof-verification.json` | Réussite ; empreintes, fenêtres reconstruites, normalisateur et lots cohérents |
| Syntaxe Python et diff Git | audit léger et `git diff --check` | Réussite ; 153 fichiers Python analysés |
| Commande complète de finalisation | `finalization_attempt/status.json` | Refus avant création d'enfants ; nécessite 3 GiB disponibles |

Il n'y a eu aucune boucle automatique de relance de pytest. Les opérations légères et les vérifications de sauvegardes ont continué ; Spark, les TIFF complets et le GPU n'ont pas été relancés.

## Changements de cette reprise

Fichiers existants modifiés :

- `pyproject.toml` : extra `gpu-operational` déclarant Arrow et `psutil`, requis par le profilage. Aucune installation de dépendances effectuée.
- `scripts/prepare_expanded_code.py` : gel en lecture seule du déploiement, refus des anciennes configurations sans AP temporaire ; sorties exclusives. Le script ne réécrit plus les YAML historiques.
- `src/geofirex/job_control.py` : possession de l'arbre par Job Object Windows ou groupe POSIX et arrêt des descendants. Validation dynamique encore requise.
- `tests/test_gpu_operational.py` : fixture physique train/validation distincte, sans dossier 2021, en préservant les TIFF originaux de la fixture.
- `scripts/kaggle_account_inventory.py` : distinction entre limite documentée par dataset et quota privé personnel ; suppression de l'affirmation automatique d'un quota personnel de 200 Go.

Nouveaux fichiers de code :

- `scripts/run_bounded_local.py` : précontrôle RAM, délai, réserve disque, journaux exclusifs et arrêt de l'arbre possédé ; threads limités à deux.
- `scripts/audit_gpu_recovery.py` : audit Arrow et tailles des fichiers, sans lecture des pixels, prévu sous délai dans la finalisation.
- `scripts/audit_recovery_saved.py` : vérification légère des empreintes et plans sauvegardés ; exécutée avec succès.
- `scripts/finalize_gpu_recovery.py` : enchaînement local sans relance automatique et sans action cloud.
- `tests/test_recovery_tools.py` : tests de descendants, arrêt sur ressource, estimation ZIP/ZIP64 et gel sans modification des entrées.

Les sources corrigées ne sont pas déclarées validées fonctionnellement. Aucun ancien checkpoint, GeoTIFF, rapport, temporaire ou fichier non suivi n'a été supprimé.

## Paquet Kaggle, quotas et budget

**Aucune archive de rasters préparée dans le nouveau périmètre opérationnel ; aucun paquet code final produit.** Les huit lots sauvegardés sont des plans immuables contenant inventaires, CRC et SHA256. La finalisation créera un nouveau déploiement et un paquet uniquement après réussite des tests et de l'audit. Elle produira aussi un notebook CPU privé, accélérateur désactivé, et ses métadonnées Kaggle ; ce notebook n'a pas encore été généré.

Les chemins de rasters des quatre configurations cloud commencent par `/kaggle/input/`, sans D:. Le normalisateur portable utilise des tokens relatifs. Le split et l'index conservent les métadonnées officielles du test, mais aucun TIFF 2021 n'est demandé en mode `train_validation`. `final_test` refuse ce mode. Le test renforcé d'absence physique des TIFF 2021 reste à exécuter.

Les anciennes configurations v3 n'ont pas encore les deux options AP `evaluation_scratch_root` et `evaluation_cleanup_runs`. Elles ne doivent pas être figées telles quelles. La nouvelle finalisation configure `/kaggle/temp/geofirex-ap-v1` dans une copie distincte, après les tests.

**Quotas personnels actuels : non vérifiés.** L'environnement actuel ne contient ni `kaggle`, ni `kagglesdk`, ni `kagglehub`, ni `psutil`. Aucune valeur du 3 novembre ne doit être présentée comme actuelle, particulièrement après la date de renouvellement du 7 novembre inscrite dans l'ancien reçu.

Dernières valeurs sauvegardées : GPU autorisé 108 000 s, utilisé 9 072,533889 s, réservé 0 s, pay-to-scale désactivé ; inventaire privé de six datasets totalisant 504 962 568 octets selon l'API à cette date. Cet inventaire n'est pas un compteur de quota ni une mesure des versions historiques.

La [documentation Kaggle GPU](https://www.kaggle.com/docs/efficient-gpu-usage) indique une allocation hebdomadaire générique de 30 heures, parfois supérieure ; cela ne prouve pas le solde du compte. La [documentation notebooks](https://www.kaggle.com/docs/notebooks) indique 12 heures par session CPU/GPU. La [documentation datasets](https://www.kaggle.com/docs/datasets) indique une limite de 200 Go par dataset ; il ne faut pas la confondre avec le quota privé du compte. Les sources publiques ont été consultées le 7 novembre ; le quota authentifié devra être rafraîchi avant toute publication ou session GPU.

| Volume | Octets |
|---|---:|
| Rasters train/validation, exact selon inventaires sauvegardés | 34 995 352 430 |
| Plafond proposé, toujours non autorisé | 36 000 000 000 |
| Marge brute avant ZIP, code et HTTP | 1 004 647 570 |
| ZIP_STORED prédits d'après le format du script | 34 998 672 663 |
| Marge prédite avant code et requêtes HTTP | 1 001 327 337 |
| Borne conservatrice archives : brut + 10 Mo par lot | 35 075 352 430 |
| Marge après cette borne, avant code et HTTP | 924 647 570 |
| Plus petit lot ZIP prédit | 1 182 408 142 |

Les octets ZIP sont **prédits, non mesurés**, et le test de prédiction ajouté n'a pas encore pu tourner. Les octets HTTP exacts ne peuvent pas être connus sans archives figées et sans les requêtes réelles. Même le plus petit lot entier dépasse la marge restante : **le budget de 36 Go ne couvre pas une retransmission complète d'un lot**. Les reprises partielles consomment aussi le budget. Un même journal `BodyBudget` durable doit couvrir tous les lots et le code ; chaque corps envoyé est réservé avant émission, y compris une répétition, et aucun solde ne doit être remis à zéro. Il faut vérifier les reçus d'archives et le quota authentifié avant de demander l'autorisation d'upload.

## GPU et reprise exécutable

Les quatre configurations historiques définissent le fold 0, contexte 3, seed 42, une époque, deux threads CPU, zéro worker et un cache de lecture de 64 Mo. Les scripts prévoient : profilage 900 s maximum par modèle, 100 étapes mesurées maximum et 10 de chauffe ; proposition courante 50 étapes et 5 de chauffe. Le profilage écrit progrès, journal, reçu du superviseur et, s'il aboutit, `profile.json` ; le modèle est jetable et ne sert pas à la sélection scientifique.

Le lancement d'entraînement requiert un profil préalable, une prévision compatible avec le délai approuvé, 7 200 s maximum par modèle, 20 Go libres côté sorties et 40 Go côté scratch au démarrage ; il surveille une réserve de 5 Go et un plafond de sorties de 15 Go. Le profilage vérifie aussi RSS 8 GiB et réservation CUDA 12 GiB aux points de contrôle. Ces limites sont présentes dans les sources mais n'ont pas été validées sur GPU dans cette reprise. `psutil` manque encore localement.

Ordre prévu après autorisations : montage/vérification sur CPU ; profils B0/B1 ; examen des reçus, ressources et quota ; puis profils B2/B3 ; entraînements séparés uniquement après approbation. Aucun de ces jobs n'a été lancé.

**Action préalable nécessaire : libérer au moins 3 GiB de RAM disponible.** Aucun nouvel accord n'est nécessaire pour les tests locaux déjà autorisés. La prochaine commande utilisable est :

```powershell
.\.venv\Scripts\python.exe scripts/finalize_gpu_recovery.py --out reports/recovery_20261107_v2
```

Elle refuse de démarrer les enfants si la mémoire manque. Dans un dossier neuf, elle exécute les quatre tests légers sous 60 s, la commande pytest de référence sous 300 s puis l'audit des métadonnées/tailles sous 120 s. Elle s'arrête au premier problème. Après réussite seulement, elle copie le déploiement, corrige les chemins de la copie, crée le paquet, vérifie le ZIP et ses SHA256 et prépare le notebook CPU. **Ce chemin complet n'a pas encore été exécuté avec suffisamment de RAM.** Ne pas réutiliser un dossier de sortie existant ; choisir un suffixe neuf.

Restent soumis à accord explicite : tout upload du corpus ou de son paquet, publication distante des lots, tout profilage/entraînement GPU, téléchargement massif, commit, push et merge. La vérification authentifiée des quotas est en lecture seule ; elle reste à réaliser dans un environnement doté des SDK et d'une authentification valide.
