# Lot 001 — correction ciblée des dates

Le lot Kaggle existe selon la vérification de l'utilisateur. Son intégrité distante reste non certifiée : la précédente exécution s'est arrêtée avant le calcul des empreintes.

## Constat local

Le manifeste figé contient 1 269 GeoTIFF et 57 événements, tous affectés au split `train` dans `fold_00.csv`.

| Dossier événement | Observation | Split |
| --- | --- | --- |
| `2018/fire_21458836` | `2017-12-29.tif` | train |
| `2018/fire_21458836` | `2017-12-30.tif` | train |
| `2018/fire_21458836` | `2017-12-31.tif` | train |

Exactement 3 écarts dossier/date, dans un seul événement. Les 1 266 autres observations sont datées de 2018. Aucune observation du manifeste n'est datée de 2019, 2020 ou 2021.

La règle initiale imposant l'égalité des années était incorrecte pour une série temporelle pouvant franchir le Nouvel An. Les auteurs ajoutent quatre jours avant et après chaque événement : [article officiel, section 2.1.1](https://papers.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf). Le [DataLoader officiel](https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataset.py) regroupe les images par année de l'événement et conserve les observations quotidiennes de chaque dossier. Les trois dates sont cohérentes avec cette convention ; cette cohérence ne certifie pas les octets Kaggle.

## Correction

Le validateur vérifie indépendamment le split de l'événement et la date ISO de l'observation. Les années 2020 et 2021 restent toujours bloquantes, avec le chemin exact dans l'erreur. Aucune égalité artificielle dossier/date n'est requise entre les années train 2018 et 2019.

L'utilisateur a explicitement autorisé les trois observations de décembre 2017 le 8 novembre 2026 (« Autoriser ces trois contextes 2017 »). La liste d'exception est limitée à ces trois chemins exacts ; tout autre chemin 2017 est rejeté. Ces fichiers ne sont ni filtrés ni renommés : ils restent inclus dans les 1 269 vérifications de taille, SHA-256 et CRC32.

Les contrôles des 57 événements train, du manifeste figé, des 1 269 fichiers, des 23 bandes des lectures rasterio, de l'absence de test 2021, des fenêtres temporelles et du DataLoader CPU sont conservés. Le notebook utilise uniquement `/kaggle/input` et `/kaggle/working`, avec GPU/TPU/Internet désactivés. Chaque réexécution crée un répertoire de travail distinct, pour préserver les sorties précédentes.

## Validation ciblée

`python tests/test_lot001_date_policy.py` : 5 tests réussis. Ils couvrent les changements d'année entre 2018/2019, le rejet inconditionnel de 2020/2021, l'exception exacte et explicitement activée pour 2017, les événements hors train, les chemins invalides et les trois cas du manifeste réel.

Aucun raster ouvert localement, audit scientifique complet, nouvel upload, nouveau dataset, version de dataset, entraînement, commit ou push.

## Notebook livré

Chemin exact : `C:\Users\nourb\Projects\GeoFire-X-2026\reports\kaggle_certification_20261108_v2\cpu-certification\lot-001\certify.ipynb`.

- Taille : 186 754 octets.
- SHA-256 : `9b6d42b130fa78c45ce7b104671d456888d7dce03c184fa8833a92f689742734`.
- Dataset existant unique : `nnnnbhss/geofirex-wsts-fold0-tv-001-v1`.
- Métadonnées locales : privé, GPU/TPU/Internet désactivés.
- Vérification locale du notebook : `LOCAL_NOTEBOOK_VERIFIED_REMOTE_UNCERTIFIED`, preuve dans `kaggle_certification_20261108_v2/cpu-certification/datefix-verification.json`.
- Aucun chemin de disque Windows dans les constantes décodées du notebook, ses métadonnées embarquées ou les modules sources.

Importer ce notebook dans l'interface Kaggle avec le dataset lot 001 déjà attaché, conserver l'accélérateur `None`, puis exécuter toutes les cellules. Le suffixe local `v2` désigne uniquement la révision du notebook ; le dataset reste `v1`. La certification distante n'est acquise qu'après une exécution complète réussie et la production de `certificate.json` avec le statut `LOT001_REMOTE_CPU_CERTIFIED`.
