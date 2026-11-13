# GeoFire-X — pilote privé du seul lot 001, prêt mais non autorisé

Vos valeurs Kaggle Settings sont consignées dans `quotas-confirmes-utilisateur.json` : datasets privés 504,96 MB / 214,75 GB, modèles privés 0 B / 214,75 GB, GPU 00:00 / 30 h, TPU 00:00 / 20 h. Le stockage disponible indiqué, environ 214,25 GB, suffit. Il s'agit de votre relevé personnel, avec unités d'affichage arrondies ; aucune nouvelle requête Kaggle n'a été effectuée pendant cette préparation.

Ce pilote remplace la proposition de démarrage v1 : **un seul dataset de données privé**, sans publication préalable du dataset code. Les sept autres lots et le paquet code restent locaux. Le notebook CPU contient le manifeste canonique et le vérificateur ; il n'attache que le lot 001. Les validations cloud des Parquet et du normalisateur restent à faire lors d'une étape ultérieure ; elles ne sont pas présentées comme acquises par ce pilote.

**1. Lot et tailles mesurées**

- Destination privée : `nnnnbhss/geofirex-wsts-fold0-tv-001-v1`.
- Contenu : 57 événements train complets de 2018, 1 269 GeoTIFF, zéro raster 2021.
- Rasters : **4 829 474 073 octets**.
- ZIP_STORED créé et mesuré : **4 829 918 684 octets**, incluant `lot-inventory.json` (1 270 entrées au total).
- Fichier local : `artifacts/kaggle_publication_20261107_v1/lot-001/payload.zip`.
- SHA256 du ZIP : `464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890`.
- Identité canonique du lot : `29ccf1eaa0f4d1fb0ab4333ee307e8af881f4255d1bd5639721ccaea2c540f81`.

Le staging a vérifié en flux taille, SHA256 et CRC32 de chacun des 1 269 fichiers source pendant l'archivage, puis calculé le SHA256 du ZIP complet. Un contrôle distinct a vérifié le répertoire central ZIP, les tailles/CRC et le manifeste embarqué, sans relire les 4,8 Go. Aucun pixel raster n'a été chargé et aucun audit scientifique n'a été relancé. Reçu : `artifacts/kaggle_publication_20261107_v1/lot-001/staging-receipt.json`.

**2. Budget proposé pour le pilote uniquement**

| Poste | Octets décimaux |
|---|---:|
| Un envoi ZIP complet | 4 829 918 684 |
| Deux envois complets éventuels | 9 659 837 368 |
| Plafond persistant du pilote, corps HTTP | **11 000 000 000** |
| Marge après deux ZIP entiers | 1 340 162 632 |
| Point d'arrêt de la première tentative + certification | **6 000 000 000** |

La requête de soumission du notebook CPU a été sérialisée hors ligne : **266 649 octets** dans le contrôle SDK. Le même journal comptera l'authentification, le PUT, la création privée et la soumission CPU. Réserver 1 Mo pour les petits contrôles de lecture/certificats hors journal et prévoir **12 Go de trafic réseau indicatif**, reprises comprises. Le journal mesure des réservations de corps HTTP ; il ne mesure ni en-têtes, TLS/TCP, retransmissions réseau ni téléchargements. Cette enveloppe réseau n'est donc pas un plafond physique garanti.

**Zéro retry automatique.** L'autorisation demandée concerne une première tentative limitée à 6 Go cumulés, puis sa certification CPU. Une unique tentative complète supplémentaire est budgétée, mais reste soumise à une nouvelle autorisation après diagnostic ; elle porterait la borne cumulée à 11 Go. La commande préparée ne peut pas faire cette hausse ni lancer les autres lots. Le journal du pilote est distinct : `reports/kaggle_pilot_001_20261107_v1/http-body.jsonl`. Une éventuelle campagne ultérieure devra déduire sa consommation du budget global ; aucun compteur ne sera remis à zéro.

**3. Commande exacte de publication privée — préparée, non exécutée**

Depuis `C:\Users\nourb\Projects\GeoFire-X-2026`, après autorisation explicite seulement :

```powershell
.\reports\kaggle_prepublication_20261107_v2\PILOTE_LOT_001.ps1 -Phase Publish -Execute
```

Le script [PILOTE_LOT_001.ps1](PILOTE_LOT_001.ps1) épingle le handle, la taille et le SHA256 attendus ; il appelle le publisher avec :

```powershell
.\.venv\Scripts\python.exe scripts/run_kaggle_phase_guarded.py --out reports/kaggle_pilot_001_20261107_v1/lot-001-upload-monitor-01 --seconds 7200 -- .\artifacts\orchestrator\kaggle-cli-env\Scripts\python.exe scripts/publish_kaggle_single_attempt.py --staged artifacts/kaggle_publication_20261107_v1/lot-001 --credentials C:/Users/nourb/.kaggle/credentials.json --journal reports/kaggle_pilot_001_20261107_v1/http-body.jsonl --attempt reports/kaggle_pilot_001_20261107_v1/lot-001-create-attempt-01 --max-http-body-bytes 11000000000 --stop-at-cumulative-body-bytes 6000000000 --license CC-BY-4.0 --execute
```

Le publisher recontrôle le SHA256 entier **avant tout réseau**, vérifie le propriétaire OAuth et l'absence du dataset, envoie un seul PUT en flux, exige HTTP 200/201, puis crée avec `is_private=True`. Un dataset existant ou un état ambigu provoque l'arrêt ; aucun remplacement de version. Le ZIP gelé du code n'est pas utilisé par ce pilote et son empreinte historique a été recontrôlée inchangée.

Pour consulter la préparation sans transfert :

```powershell
.\reports\kaggle_prepublication_20261107_v2\PILOTE_LOT_001.ps1 -Phase Review
```

**4. Persistance et certification distante — uniquement après autorisation**

Après la fin du processus d'upload, effectuer une lecture avec un nouveau processus. Ces commandes futures sont elles aussi laissées inexécutées :

```powershell
$kg = '.\artifacts\orchestrator\kaggle-cli-env\Scripts\python.exe'
$campaign = 'reports/kaggle_pilot_001_20261107_v1'
& $kg -m kaggle datasets status nnnnbhss/geofirex-wsts-fold0-tv-001-v1 --format json
& $kg scripts/kaggle_prepublication_readonly.py --credentials C:/Users/nourb/.kaggle/credentials.json --out "$campaign/account-after-publication-01.json"
```

Exiger `ready`, propriétaire `nnnnbhss`, visibilité privée et relever le numéro de version dans les réponses. Ne pas considérer un simple accusé de création comme une preuve d'intégrité. Après la certification, refaire la lecture dans un autre processus avec un **nouveau** fichier `account-after-certification-02.json`, et vérifier même handle/version/visibilité. Cela observe la disponibilité au-delà du processus d'upload, sans provoquer un nouvel arrêt du PC. En cas d'interruption réelle, cette lecture doit toujours précéder une éventuelle reprise.

Soumettre ensuite, après validation de l'état distant, le notebook **CPU privé, Internet/GPU/TPU désactivés**, timeout serveur 4 200 s, contrôle des fichiers limité à 3 600 s :

```powershell
.\reports\kaggle_prepublication_20261107_v2\PILOTE_LOT_001.ps1 -Phase Certify -Execute
& $kg -m kaggle kernels status nnnnbhss/geofirex-certify-pilot-001-20261107-v2
& $kg -m kaggle kernels output nnnnbhss/geofirex-certify-pilot-001-20261107-v2 -p "$campaign/cpu-receipts" --file-pattern '(^|/)pilot-certificate\.json$'
```

Le notebook autonome est `pilot/cpu-certification/lot-001/certify.ipynb`. Il supporte le montage direct et la découverte de l'inventaire dans `/kaggle/input`, exige une copie unique du manifeste épinglé, le jeu exact de 1 269 fichiers, leurs tailles/SHA256/CRC32, puis ouvre une fenêtre GeoTIFF de 64 × 64 maximum via rasterio. Blocs de hachage 1 Mio, réserve RAM cloud 2 Gio, cache GDAL 64 Mio. Une dépendance manquante provoque l'arrêt ; aucune installation automatique.

Critère de succès : notebook terminé et `pilot-certificate.json` avec statut `ONE_LOT_REMOTE_BYTES_CERTIFIED`, identité attendue, 57 événements, 1 269 fichiers, 4 829 474 073 octets et une fenêtre GeoTIFF lue. Les 4,8 Go sont relus **sur Kaggle**, pas téléchargés sur le PC. Le SHA256 du ZIP local reste une référence : le certificat distant vérifie les fichiers extraits, pas le ZIP supprimé par décompression. Seul le petit certificat est rapatrié. Arrêt obligatoire ensuite avant tout autre lot.

**5. Mémoire locale, arrêts et reprise**

Staging réel : avant **4,39 Gio**, minimum **3,80 Gio**, après **4,23 Gio**, pic RSS de l'arbre **25 Mio**, durée **125,7 s**, sortie `FINISHED`. Dernier contrôle hors ligne : environ **4,41 Gio** disponibles après exécution. Journal : `lot-001-stage-monitor/supervisor.json` et `memory.jsonl`.

Chaque phase future mesure la RAM avant démarrage et refuse sous **3 Gio** disponibles ; elle plafonne l'arbre enfant à **1 Gio**, surveille toutes les 250 ms et arrête ses propres processus si la RAM disponible passe sous **2 Gio**, si le disque de supervision tombe sous 1 Gio ou si l'échéance est atteinte. Upload : 7 200 s maximum ; soumission CPU : 120 s maximum localement. Une seule phase en cours ; aucun processus applicatif personnel n'est fermé.

Après erreur/coupure : conserver ZIP, reçus, journal et éventuel `.lock`. Vérifier d'abord l'état distant ; si le dataset est prêt, certifier au lieu de renvoyer. Le publisher ne promet aucune reprise GCS durable. Une tentative ratée réserve conservativement tout son corps ; une nouvelle tentative recompterait le ZIP entier. Un verrou orphelin/journal tronqué impose une inspection des processus et de la comptabilité ; aucune suppression ou remise à zéro automatique. Toute reprise demande votre nouvelle autorisation.

**État final :** stockage personnel confirmé par votre relevé, ZIP local du pilote finalisé, contrôles spécifiques hors ligne réussis, script PowerShell analysé sans erreur. Zéro requête Kaggle, upload ou job pendant cette préparation ; aucune modification des 607 événements, Parquet, configurations/checkpoints ou journaux historiques. La certification distante reste **non exécutée**. Autorisation explicite attendue pour le seul premier envoi privé et son notebook CPU, sous borne de 6 Go, sans retry ni autre lot.
