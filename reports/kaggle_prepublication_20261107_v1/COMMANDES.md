# Proposition d'exécution — rien n'a été transféré

Ces commandes sont préparées, pas exécutées. Chaque bloc constitue une étape distincte. Ne pas lancer tous les blocs ensemble. Autorisation de publication privée et de certification CPU nécessaire avant les blocs réseau d'écriture. Aucun GPU, entraînement, commit Git ou push Git.

Depuis `C:\Users\nourb\Projects\GeoFire-X-2026`, initialiser les variables PowerShell :

```powershell
$ml = '.\.venv\Scripts\python.exe'
$kg = '.\artifacts\orchestrator\kaggle-cli-env\Scripts\python.exe'
$cred = 'C:\Users\nourb\.kaggle\credentials.json'
$review = 'reports/kaggle_prepublication_20261107_v1'
$campaign = 'reports/kaggle_publication_20261107_v1'
$journal = "$campaign/http-body.jsonl"
```

**Arrêt initial :** confirmer dans l'interface Kaggle le stockage privé utilisé/plafond, au moins 45 Go disponibles pour cette campagne proposée et la présence de l'option GPU. Les 200 Go de la documentation ne remplacent pas cette lecture personnelle. Les wrappers refusent de démarrer sous 3 Gio disponibles, réservent 2 Gio au système, plafonnent l'arbre enfant à 1 Gio et arrêtent l'arbre qu'ils ont créé à l'échéance. Un seul bloc en cours.

**Étape 0 — publier uniquement le code**, après autorisation. Un seul PUT, dataset explicitement privé, aucun remplacement. Plafond global constant de 50 000 000 000 octets de corps HTTP ; premier point d'arrêt cumulatif à 10 000 000 octets.

```powershell
& $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/code-upload-monitor" --seconds 600 -- $kg scripts/publish_kaggle_single_attempt.py --staged reports/recovery_20261107_v3/finalized/code-package --credentials $cred --journal $journal --attempt "$campaign/code-create-attempt-01" --max-http-body-bytes 50000000000 --stop-at-cumulative-body-bytes 10000000 --license other --execute
```

La licence `other` ne concède aucune licence publique sur le code du projet. Le dataset reste privé. Si Kaggle exige une autre sélection, arrêter et revoir les métadonnées ; ne pas modifier le payload gelé.

**État distant et fichiers**, commandes CLI de lecture seule à exécuter sous votre session Windows habituelle avec l'authentification Kaggle existante :

```powershell
& $kg -m kaggle datasets status nnnnbhss/geofirex-expanded-code-recovery-v1 --format json
& $kg -m kaggle datasets files nnnnbhss/geofirex-expanded-code-recovery-v1 --format json --page-size 200
```

Attendre explicitement l'état `ready`, relever la version et vérifier la visibilité privée dans l'interface/API. Paginer si `nextPageToken` est présent. Une liste de fichiers ou un état `ready` ne certifie pas les octets.

**Certifier le code sur CPU seulement**, après autorisation de cette session CPU. Le script est sans réseau par défaut ; `--execute` soumet le notebook. Aucun GPU, Internet ou installation automatique. Le même journal compte aussi cette soumission.

```powershell
& $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/code-cpu-submit-monitor" --seconds 120 -- $kg scripts/submit_kaggle_cpu_certification.py --folder "$review/cpu-certification-v3/code" --credentials $cred --journal $journal --attempt "$campaign/code-cpu-submit-01" --max-http-body-bytes 50000000000 --stop-at-cumulative-body-bytes 10000000 --execute
& $kg -m kaggle kernels status nnnnbhss/geofirex-certify-code-20261107-v1
& $kg -m kaggle kernels output nnnnbhss/geofirex-certify-code-20261107-v1 -p "$campaign/code-cpu-receipts" --file-pattern '(^|/)runtime-certificate\.json$'
```

Ne télécharger que les petits certificats. Exiger un notebook terminé et `runtime-certificate.json` avec imports réussis, les deux Parquet lisibles, normalisateur SHA attendu. Toute conversion de Parquet, dépendance absente, manifeste modifié ou décompression inattendue impose l'arrêt avant le lot de 4,8 Go. Le certificat indique aussi si les réserves disque nécessaires à l'entraînement sont disponibles.

**Étape 1 — préparer seulement le lot 001**, après validation de l'étape 0. Cette commande future lit les seuls rasters du lot, calcule leurs empreintes en flux pendant l'archivage et conserve tout `.part` en cas d'erreur. Elle ne lance ni Spark ni audit scientifique. Racine historique relevée dans le journal v3 : `D:/GeoFire-X/WildfireSpreadTS/rasters`. Le staging exige la taille du lot + 10 Mo + 20 Go libres sur son volume ; environ 109,8 Go étaient disponibles sur C: pendant cette revue.

```powershell
& $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/lot-001-stage-monitor" --seconds 1800 -- $ml scripts/stage_expanded_lot.py --lot reports/recovery_20261107_v3/finalized/deployment/bundle/metadata/lot-001.json --raw-root D:/GeoFire-X/WildfireSpreadTS/rasters --out artifacts/kaggle_publication_20261107_v1/lot-001 --max-bytes 5000000000
```

Exiger `FINISHED`, `staging-receipt.json`, concordance du SHA256 et taille réelle du ZIP. Aucun staging de lot n'a été réalisé durant la revue.

**Publier le seul lot 001**, puis arrêter. Même plafond global, point d'arrêt cumulatif de phase porté à 6 Go ; aucune tentative automatique supplémentaire.

```powershell
& $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/lot-001-upload-monitor" --seconds 7200 -- $kg scripts/publish_kaggle_single_attempt.py --staged artifacts/kaggle_publication_20261107_v1/lot-001 --credentials $cred --journal $journal --attempt "$campaign/lot-001-create-attempt-01" --max-http-body-bytes 50000000000 --stop-at-cumulative-body-bytes 6000000000 --license CC-BY-4.0 --execute
& $kg -m kaggle datasets status nnnnbhss/geofirex-wsts-fold0-tv-001-v1 --format json
& $kg -m kaggle datasets files nnnnbhss/geofirex-wsts-fold0-tv-001-v1 --format json --page-size 200
```

Après `ready`, vérifier privé/version et paginer l'inventaire ; le lot décompressé attend 1 269 TIFF + `lot-inventory.json`. Cette liste ne remplace pas la certification CPU.

```powershell
& $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/lot-001-cpu-submit-monitor" --seconds 120 -- $kg scripts/submit_kaggle_cpu_certification.py --folder "$review/cpu-certification-v3/lot-001" --credentials $cred --journal $journal --attempt "$campaign/lot-001-cpu-submit-01" --max-http-body-bytes 50000000000 --stop-at-cumulative-body-bytes 6000000000 --execute
& $kg -m kaggle kernels status nnnnbhss/geofirex-certify-lot-001-20261107-v1
& $kg -m kaggle kernels output nnnnbhss/geofirex-certify-lot-001-20261107-v1 -p "$campaign/lot-001-cpu-receipts" --file-pattern '(^|/)(runtime-certificate|certificate)\.json$'
```

**Arrêt obligatoire avant les sept autres lots :** exiger le certificat `ONE_LOT_REMOTE_BYTES_CERTIFIED`, identité `29ccf1eaa0f4d1fb0ab4333ee307e8af881f4255d1bd5639721ccaea2c540f81`, 57 événements, 1 269 fichiers, 4 829 474 073 octets, contrôle SHA256/CRC de chaque fichier et ouverture d'une fenêtre GeoTIFF. Ce certificat n'affirme rien sur les sept autres lots.

**Lots 002 à 008, uniquement après certification du pilote et accord pour la suite.** Les arguments exacts de chaque commande sont dans [execution-plan-final.json](execution-plan-final.json), qui est un plan, jamais un ordonnanceur. Reprendre les trois commandes de lot séparément, avec les numéros/handles de `frozen-review-final.json`, les notebooks `cpu-certification-v3/lot-NNN` et un nouveau répertoire d'étape. Utiliser `--max-bytes 5100000000` pour les autres lots : le lot 002 exige plus de 5 Go avec la réserve de staging de 10 Mo. Ne pas créer de boucle automatique. Bornes cumulatives proposées : 002=11 Go, 003=16 Go, 004=21 Go, 005=26 Go, 006=31 Go, 007=36 Go, 008=37 Go. Chaque lot doit être `ready`, privé, puis certifié avant le suivant. Le plafond global du journal reste toujours 50 Go ; toute hausse d'une borne après incident demande revue des octets réservés et décision explicite.

**Après huit certificats**, le notebook `cpu-certification-v3/all` attache les neuf datasets, refait le contrôle d'intégrité distant et génère `expanded-mount-v1/mount-proof.json` et les quatre configurations relocalisées. Soumission future exacte, après autorisation de cette étape CPU :

```powershell
& $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/all-cpu-submit-monitor" --seconds 120 -- $kg scripts/submit_kaggle_cpu_certification.py --folder "$review/cpu-certification-v3/all" --credentials $cred --journal $journal --attempt "$campaign/all-cpu-submit-01" --max-http-body-bytes 50000000000 --stop-at-cumulative-body-bytes 37000000000 --execute
& $kg -m kaggle kernels status nnnnbhss/geofirex-certify-all-20261107-v1
& $kg -m kaggle kernels output nnnnbhss/geofirex-certify-all-20261107-v1 -p "$campaign/all-cpu-receipts" --file-pattern '(^|/)(runtime-certificate|mount-proof)\.json$'
```

Les scripts GPU gelés restent disponibles ensuite, mais ne sont pas exécutés par ces notebooks. Leur lancement exige une autorisation distincte, un quota rafraîchi, une calibration sur le GPU réellement disponible, 20 Go libres en sortie et 40 Go en scratch. Aucun temps d'entraînement sur les 451 événements n'est encore garanti.

**Après toute interruption :** ne pas relancer automatiquement. Conserver le journal, `.lock`, `.part` et les reçus. Vérifier d'abord si le dataset existe et son état/version, puis certifier s'il est prêt. Un dataset publié n'est pas effacé par l'arrêt du PC ; un blob partiel ne constitue pas un dataset publié. Le nouvel outil ne conserve pas de session de reprise GCS : un nouvel envoi éventuel recompte un corps entier. Une réservation compte entièrement même si la connexion coupe à mi-corps. Ne jamais remettre le journal à zéro ; un verrou orphelin ou journal tronqué impose une inspection des processus et du journal avant toute intervention.
