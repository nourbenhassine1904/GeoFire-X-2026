# Acquisition et traitement progressifs des 607 événements

Plan du 30 octobre 2026. **Aucun téléchargement massif n'est autorisé ou lancé.**
Le budget ci-dessous constitue la proposition à approuver avant une acquisition
importante. Les volumes sont en Go décimaux (10⁹ octets), sauf mention contraire.

## État réellement vérifié et budget

Source : [Zenodo WildfireSpreadTS v1.0](https://zenodo.org/records/8006177),
archive MD5 `dc1a04e63ccc70037b277d585b8fe761`.
L'index historique ne contenait pas les tailles par fichier ni les CRC ; un
nouvel index est conservé séparément :
`data/manifests/real/corpus_zip_index_v1.json`.

Le répertoire central distant a été relu en 4 requêtes Range : **1 611 918 octets**,
sans aucun corps de GeoTIFF. Le client imposait un plafond de 8 Mio et refusait
HTTP 200. L'index recense 607 événements et 13 607 observations :

| Année | Événements | Observations |
| --- | ---: | ---: |
| 2018 | 176 | 3 773 |
| 2019 | 74 | 1 425 |
| 2020 | 201 | 4 292 |
| 2021 | 156 | 4 117 |

| Élément | Volume | Nature de la mesure |
| --- | ---: | --- |
| ZIP entier | 48,359 Go | Exact, taille officielle et index |
| Somme des charges TIFF compressées dans ZIP | 48,356 Go | Exact, répertoire central |
| GeoTIFF extraits | 49,294 Go | Exact, `ZipInfo.file_size` ; TIFF encore compressés en interne |
| Décodage 23 bandes float32 | 113,8 Go, plage extrapolée 90,8–126,9 | Estimation depuis les dimensions des 108 scènes |
| Cache numérique compact : 3×float32 + feu uint8 + LC uint8 | 17,3 Go, plage extrapolée 13,8–19,3 | Estimation, 14 octets/pixel |
| Tenseurs denses 40 canaux float32 | 197,9 Go, plage extrapolée 157,9–220,7 | Estimation ; à éviter comme stockage systématique |

Les plages ne sont **pas** des bornes garanties du corpus : les en-têtes des
13 607 TIFF n'ont pas été lus. Ni les NaN ni les pixels du corpus complet ne
sont déclarés analysés. Les 108 fichiers présents ont été vérifiés par taille
**et CRC ZIP** : 108 valides, aucun manquant/corrompu parmi les cinq événements.

Machine mesurée : Intel i7-1255U, 10 cœurs/12 processeurs logiques ;
16 837 500 928 octets de RAM ; environ **129 Go libres** sur C: (variation selon
le moment). Le venv est PyTorch CPU ; aucun coût ou gain CUDA n'est supposé.

## Scénarios de stockage et transfert

| Scénario | Besoin additionnel approximatif | Décision |
| --- | ---: | --- |
| Conserver ZIP + TIFF + cache compact + marge 20 Go | 134,97 Go | Dépasse l'espace libre actuel |
| Range par événement, TIFF + cache complet + marge 20 Go | 86,61 Go | Possible avant gros artefacts de scores, à surveiller |
| TIFF + cache dense 23 bandes + marge | Environ 183 Go | Inadapté au disque actuel |
| TIFF + cache dense 40 canaux + marge | Environ 267 Go | Inadapté |

Les 448 Mo de bruts locaux déjà présents sont inclus dans l'inventaire ; selon
le répertoire retenu et leur réutilisation vérifiée, ils peuvent éviter un
double transfert. Ne copier aucun brut sans plan de déduplication explicite.

Les scores exacts d'AP et leurs tris nécessitent aussi du scratch. L'implémentation
de référence garde ses fichiers temporaires : jusqu'à 24 octets par groupe de
score et par passe de fusion, pour chaque AP calculée. Sur la validation locale,
les quatre AP (U-Net, Persistence et domaine nouvelles ignitions) ont conservé
110,3 Mo pour 2,33 millions de pixels. Prévoir **20–60 Go de scratch** pour un
gros split, à mesurer avant exécution ; le pire cas dépend des égalités de scores
et du nombre de passes. Le scénario TIFF+cache complet+60 Go scratch+20 Go marge
atteindrait environ **146,6 Go**, donc dépasse C:.

Recommandation pour le corpus : extraction Range sans ZIP local, cache actif
borné par événements, et disque de travail externe/serveur avec **au moins
200 Go libres** avant les runs complets. C'est une marge de travail proposée,
pas un besoin mesuré pour chaque configuration. Les anciens checkpoints/runs
ne sont jamais supprimés automatiquement pour faire de la place.

Temps idéaux de transfert pour 48,356 Go : 10 Mbit/s=10,75 h ;
50 Mbit/s=2,15 h ; 100 Mbit/s=1,07 h. Ajouter les reprises, la latence des
nombreuses requêtes et la limitation éventuelle du serveur. Le débit réel n'a
pas été mesuré : aucun délai garanti n'est annoncé. Aucun tarif cloud n'est
inventé ; le budget monétaire sera heures mesurées × tarif du fournisseur choisi,
plus stockage/egress éventuels, à faire approuver séparément.

## Acquisition reprenable et contrôlée

1. Épingler version Zenodo, taille, MD5 et SHA256 du nouvel index. Garder cet
   index immuable ; une actualisation crée un nouveau fichier.
2. Planifier une liste déterministe d'événements, par année puis identifiant.
   Pour le fold 0, préparer train 2018–2019 puis validation 2020. Les données 2021
   peuvent être acquises et contrôlées sans exécuter d'évaluation de modèle.
3. Présenter la sélection, volume et destination ; obtenir l'accord explicite
   avant une session substantielle. Le CLI est en dry-run par défaut.
4. À l'exécution approuvée, plafonner les octets de **chaque session**, retries
   compris, et le nombre de nouveaux fichiers. Les en-têtes/répertoire ZIP
   consomment aussi ce budget. Pas de téléchargement implicite de l'archive.
5. Chaque fichier est écrit dans un `.part` voisin, puis validé taille+CRC et
   publié par renommage. En cas de panne, reprendre au fichier suivant ou
   recommencer le fichier incomplet ; ce n'est pas une reprise au milieu d'un TIFF.
6. Un fichier existant de bonne taille mais de mauvais CRC provoque une erreur,
   sans écrasement. Une réparation exige une décision explicite et conservation
   du fichier problématique. Les `.part` abandonnés restent visibles.
7. Comparer les observations attendues et présentes ; un événement incomplet
   ne peut pas être déclaré prêt pour le cache. Calculer ensuite SHA256 local,
   QC raster, dates, canaux, CRS, transform, NaN/Inf, classes LC et géométrie.

CRC prouve l'intégrité de l'entrée par rapport au ZIP épinglé, pas l'authenticité
cryptographique indépendante du corpus. Le MD5 global ne peut pas être recalculé
sans lire le ZIP entier. Cette limite est déclarée ; taille officielle, source
HTTPS, index hashé, CRC et SHA256 locaux constituent la traçabilité incrémentale.

```powershell
# Métadonnées seulement ; réseau plafonné à 8 Mio, sortie nouvelle.
.\.venv\Scripts\python.exe scripts/plan_corpus.py --index data/manifests/real/index_nouveau.json --refresh-remote --out reports/budget_nouveau.json
# Plan hors réseau ; aucun --execute.
.\.venv\Scripts\python.exe scripts/acquire_corpus.py --index data/manifests/real/corpus_zip_index_v1.json --destination data/raw/WildfireSpreadTS/corpus_v1 --year 2018 --max-files 100 --max-transfer-bytes 1000000000 --out reports/acquisition_plan_nouveau.json
```

Après accord sur ce budget précis seulement, ajouter `--execute` à une commande
revue. Un plafond d'un milliard d'octets ne signifie pas qu'il a été autorisé
dans la présente mission. `--verify-only` lit les fichiers locaux sans réseau.

## Format de travail choisi et alternatives

| Format | Atouts | Coûts / limites | Choix actuel |
| --- | --- | --- | --- |
| GeoTIFF brut | Source intacte, géoréférence, provenance | Décompression et lecture répétée de bandes inutilisées | Archive primaire en lecture seule |
| NumPy `.npy` memmap par événement | Déjà disponible, mmap OS, slicing temporel/spatial, fichiers simples | Pas de compression/chunks applicatifs ; cache pages OS ; pas un object store | **Pilote implémenté et validé** |
| HDF5 par événement | Slicing, compression, chunks adaptés aux patches ; pratique locale | Dépendance absente ; handles par worker ; verrou global h5py en threads | Candidat à profiler si mmap devient insuffisant |
| Zarr | Chunks et stockage objet, adapté au cloud | Dépendance absente ; nombreux objets ou configuration de sharding ; coût métadonnées | À considérer si stockage objet/distribution justifiés |
| Parquet | Colonnes, requêtes et partitions de métadonnées | Pas le format des tenseurs spatiaux denses pour PyTorch | Métadonnées/QC/export uniquement |

Ces propriétés sont documentées par [h5py datasets](https://docs.h5py.org/en/stable/high/dataset.html),
[h5py threading](https://docs.h5py.org/en/stable/threads.html),
[Zarr storage](https://zarr.readthedocs.io/en/stable/user-guide/storage/) et
[Spark Parquet](https://spark.apache.org/docs/latest/sql-data-sources-parquet.html).
Ni HDF5 ni Zarr ne sont requis simultanément. Aucun ratio de compression de ces
formats n'est supposé mesuré ici.

Le cache choisi est indépendant des folds : `numeric.npy [D,3,H,W]` float32
en unités physiques (pente, exposition, altitude, NaN conservés), `fire.npy`
uint8, `landcover.npy` uint8 avec 255=inconnu. Garder LC dans le cache ne l'autorise
pas comme entrée scientifique. L'encodage one-hot reste à la lecture, jamais
stocké systématiquement. Le cache ne contient **aucune normalisation globale**.

Pilote réel : 2018/fire_21748813, 14×338×297, **19 676 040 octets**.
La parité avec le prétraitement des bruts est testée, y compris NaN et LC.
Chaque événement est publié après création complète ; métadonnées, sources et
tableaux sont hashés. Un cache existant divergent est conservé et refusé.
Les bandes absentes du cache sont retournées comme NaN, pas comme fausses mesures.

Politique prévue : LRU de 2–4 événements/worker, handles ouverts seulement dans
le worker et fermés à l'éviction ; num_workers=0 sur ce Windows pour le premier
profil, puis 1–2 après mesure. Interdire le chargement de tous les événements
ou de toutes les fenêtres en RAM. Budgéter également les pages mmap réellement
touchées ; un mmap n'est pas une garantie de RAM nulle.

## Intégration Big Data : état et périmètre

`export_observation_metadata.py` a lu **localement** 108 rasters et créé
`data/interim/benchmark_metadata_v1/observations.parquet`, relu à l'identique.
Colonnes : événement/date/année, chemin/hash, dimensions/CRS, bbox et centre de
grille en WGS84, partition géographique CRS, pixels feu, Inf et moments des
trois bandes de relief. Ce centre n'est pas une coordonnée d'ignition.

`spark_metadata.py` prépare agrégation par année/zone et export Parquet. Il n'a
pas été exécuté : PySpark/JVM/cluster ne sont pas configurés. Un futur
`local[2]` sera un pilote local, pas un traitement distribué du corpus.

Tâches distribuables : inventaire et joins, doublons, contrôles de complétude,
distribution temporelle/géographique, statistiques de colonnes, partitions
annuelles et par zone suffisamment grosses, tables pour backend et sélection
des fichiers de train. Éviter 607 minuscules partitions par événement sans besoin.
Le CRS est un proxy géographique ; pour de vrais groupes spatiaux, dériver des
tuiles géographiques/bbox et vérifier les recouvrements.

La lecture/décodage raster actuelle reste locale. Elle ne deviendra distribuée
que si les fichiers sont accessibles aux workers et que des tâches raster
partitionnées avec quotas/mémoire sont réellement exécutées. Spark ne remplace
pas le DataLoader PyTorch ni ne distribue automatiquement la descente de gradient.

Pour des statistiques de normalisation distribuées : joindre les métadonnées
avec la liste des **chemins d'entrée train uniques du fold**, puis agréger
comptes/sommes/moments (ou Welford fusionnable). Filtrer seulement par année
n'est pas suffisant si des observations ne sont jamais des entrées valides.
Exporter statistiques versionnées + hash et tables d'observations vers le
backend futur ; aucun FastAPI/React n'est développé dans cette préparation.
