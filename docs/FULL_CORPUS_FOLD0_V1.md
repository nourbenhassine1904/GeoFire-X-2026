# Intégration du corpus complet — fold 0

Cette chaîne lit les rasters sur D: sans écriture. Elle ne télécharge rien et ne
lance aucun entraînement. Les nouveaux artefacts sont versionnés dans
`reports/full_corpus_v1/`, sans changement des manifestes du pilote ni de la démo.

**État livré après autorisation d’installation :** PySpark 4.0.2 et Py4J 0.10.9.9
sont installés. La préparation Spark complète a réussi dans
`reports/full_corpus_v1/prepared_spark_v1/` : 13 607 observations,
21 705 lignes de fenêtres, correspondance exacte avec le constructeur indépendant.
Le transfert de dépendances a consommé 434 421 883 octets HTTP, pour un plafond
cumulatif de 600 000 000. Preuve : `reports/full_corpus_v1/spark-installation.json`.
L’installateur conserve les paquets, `.part`, SHA-256 et registre des octets dans
`artifacts/local-spark-tooling-v1/`. Aucun corpus supplémentaire n’a été téléchargé.

## Audit indépendant et reprise

```powershell
Set-Location 'C:\Users\nourb\Projects\GeoFire-X-2026'
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py audit --output reports/full_corpus_v1
```

Contrôle exact des chemins attendus et de l’inventaire physique ; taille et CRC
contre l’index officiel, SHA-256 recalculé, lecture de toutes les bandes par
blocs de 256 pixels, comptages NaN/Inf/valeurs finies, CRS, dimensions, transformée,
dates, complétude et stabilité de la grille par événement. Le nom de la bande
GFS rain `last` est l’unique variante déjà admise par le contrat historique.
NaN dans active-fire signifie absence de détection, pas label invalide.

Le cache GDAL est limité à 64 Mio ; aucun cube du corpus n’est chargé. GDAL PAM
est désactivé pour éviter la création de sidecars. Chaque fichier validé est
enregistré dans un nouveau journal synchronisé. Une interruption se reprend par
la même commande ; seuls les certificats dont l’identité physique reste identique
sont réutilisés. Les traces d’erreur et queues interrompues sont conservées.
Une nouvelle version de contrat ou un autre corpus demande un autre dossier.

Les pixels 2021 sont lus uniquement pour la certification de stockage demandée.
Ils ne contribuent jamais aux fenêtres de préparation, normalisateurs ou modèles.

## Tables réellement utilisées

La préparation par défaut exige une installation Spark existante. Elle n’en
installe aucune automatiquement. Java 21 est présent sur ce PC ; pour Spark 4,
utiliser ce JDK dans le terminal de préparation :

```powershell
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py prepare --output reports/full_corpus_v1 --prepared reports/full_corpus_v1/prepared_spark_v1
```

L’exécution Spark est locale, `local[2]`, heap JVM 512 Mio, pas un cluster.
Spark SQL vérifie les clés et affectations annuelles puis construit les lags
temporels après exclusion des lignes test. Ses fenêtres sont comparées exactement
au constructeur indépendant `aligned_windows`, puis matérialisées en Parquet
avec PyArrow, pour ne pas dépendre du writer Hadoop natif sous Windows.
Le runner charge réellement ces lignes via `window_manifest` et refuse toute
différence avec le contrat temporel canonique. Les chemins restent relatifs au corpus.

Sans autorisation d’installer Spark, les mêmes contrats fonctionnels peuvent
être matérialisés par PyArrow ; le rapport marque explicitement Spark non exécuté :

```powershell
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py prepare --engine python --output reports/full_corpus_v1 --prepared reports/full_corpus_v1/prepared_v1
```

Chaque destination préparée doit être nouvelle : aucun Parquet existant n’est
remplacé. Les artefacts sont `observations.parquet`, `observations-qc.parquet`,
`events.parquet`, `windows.parquet`, un manifeste CSV et le split complet fold 0.
Spark ajoute un inventaire annuel et ses tables d’entrée. Le certificat SHA-256
d’extraction est comparé aux empreintes indépendantes, sans relire l’archive ZIP.

## DataLoaders, normalisation et preuve CPU

Après préparation, en adaptant `--prepared` au dossier réellement produit :

```powershell
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py validate --output reports/full_corpus_v1 --prepared reports/full_corpus_v1/prepared_v1
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py report --output reports/full_corpus_v1 --prepared reports/full_corpus_v1/prepared_v1
```

Pour les sorties Spark, la validation réutilise le normalisateur certifié du
premier traitement. Elle exige l’égalité des tables, du fold, de l’index et des
cibles, vérifie son SHA-256 et les identités des rasters, puis relit un échantillon
réel avec les DataLoaders et les quatre modèles CPU :

```powershell
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py validate --output reports/full_corpus_v1 --prepared reports/full_corpus_v1/prepared_spark_v1 --reuse-prepared reports/full_corpus_v1/prepared_v1
.\.venv\Scripts\python.exe scripts/integrate_full_corpus.py report --output reports/full_corpus_v1 --prepared reports/full_corpus_v1/prepared_spark_v1
```

Ces commandes de validation ont déjà produit leurs artefacts ; pour une nouvelle
exécution indépendante, choisir une nouvelle destination préparée. Les rapports
de livraison successifs utilisent un suffixe de version et ne sont pas écrasés.

La normalisation existante est ajustée sur tous les chemins contextes train
uniques du domaine T5, jamais validation/test. Les canaux disponibles restent
ceux du profil `conservative_fire_terrain` : huit canaux encodés à partir du layout
brut de 23 bandes. Les autres profils gardent leur verrou de disponibilité.

Le nombre naturel de fenêtres est rapporté pour T1/T3/T5. Les DataLoaders du
runner utilisent l’intersection des cibles, avec un domaine de validité commun
sur les cinq contextes et la cible. Le nombre de patches utilisables est exact,
y compris les bords partiels et les footprints masqués ; il est distinct des
tirages avec remise prévus par époque d’entraînement.

Le microbenchmark lit 48 patches pour chaque couple train/validation et T1/T3/T5,
sur quatre événements déterminés par leurs identifiants, avec zéro worker et
deux threads CPU. Il mesure lecture et encodage, avec un cache lecteur de 64 Mio ;
le cache système n’est pas vidé. Les quatre architectures font également un
forward CPU sans gradient. Aucune qualité prédictive n’est mesurée.

`REPORT.md`, `delivery.json`, `cpu-validation.json`, `patch-inventory.json` et
`gpu-budget.json` contiennent les résultats réels, empreintes, configurations et
commandes suivantes. Les configurations B0/B1/B2/B3 pour T1/T3/T5 ciblent un
nouveau namespace MLflow : `geofirex-full-corpus-fold0-v1`.

Les budgets GPU sont des propositions non autorisées, pas des durées mesurées.
La limite du runner est coopérative entre époques : un timeout externe dur est
nécessaire avant toute soumission avec plafond horaire. Aucun déploiement, upload,
entraînement GPU, commit ou push ne fait partie de cette chaîne.
