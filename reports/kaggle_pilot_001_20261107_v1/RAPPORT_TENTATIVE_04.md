# Lot 001 — tentative 04, 8 novembre 2026

**ZIP reçu ; certification distante bloquée par HTTP 403. Aucune relance.**

## Résultat factuel

- Destination demandée : `nnnnbhss/geofirex-wsts-fold0-tv-001-v1`.
- URL attendue, accès non confirmé : https://www.kaggle.com/datasets/nnnnbhss/geofirex-wsts-fold0-tv-001-v1
- SHA-256 du ZIP revérifié avant réseau : `464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890`.
- Un seul PUT : **4 829 918 684 octets**, réponse HTTP 200. Aucun retry.
- `CreateDataset` : HTTP 200, aucune erreur signalée au client ; requête avec `is_private=True`. Aucun appel de remplacement/versionnement.
- Contrôle authentifié indépendant après fermeture du processus de publication : `GetDataset` renvoie **HTTP 403** pour la destination demandée.
- Ce 403 indique un refus d'accès ; il ne prouve pas une absence. Cause de ce refus non établie. Création effective, visibilité distante, version, état `ready` et persistance restent **non certifiés**.
- Certification CPU : **non soumise**, car `ready` non confirmé. **0 GeoTIFF certifié à distance** ; aucune lecture PyTorch distante exécutée.

## Correction et validation locale

- `scripts/kaggle_blob_session.py` : hôte exact `www.googleapis.com`, HTTPS, port effectif 443, chemin strict `/upload/storage/v1/b/<bucket>/o`. Les règles des hôtes précédents, les interdictions de redirection, les protections contre les URL malveillantes et la confidentialité restent actives. L'URL Kaggle est utilisée sans réécriture.
- Référence officielle : [Google Cloud Storage — Request endpoints](https://docs.cloud.google.com/storage/docs/request-endpoints), [Objects: insert](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/insert).
- `tests/test_kaggle_blob_session.py` : **2 nouveaux tests ciblés réussis**, en 0,001 s, sous garde-fou 256 Mio. Acceptation de l'URL légitime inchangée ; rejets des faux domaines, chemins invalides, HTTP, mauvais port, userinfo et fragment.
- Les 33 tests précédents et les suites scientifiques n'ont pas été relancés.

## Budget et mémoire

- Compteur antérieur préservé : **3 391 octets**.
- Cumul après publication : **4 829 923 318 octets**.
- Vérification distante : **354 octets** supplémentaires de corps HTTP émis.
- **Cumul final : 4 829 923 672 / 6 000 000 000 octets** ; reste **1 170 076 328 octets**.
- Le journal comptabilise les corps des requêtes HTTP ; les réponses, en-têtes et frais TLS/TCP ne sont pas des mesures physiques de trafic dans ce compteur.
- Publication : 3 102,297 s ; RAM disponible avant **5 783 371 776 octets**, minimum **5 240 025 088 octets** (4,88 Gio), après **6 039 191 552 octets**. RSS maximal de l'arbre : **54 173 696 octets** (51,66 Mio).
- Limites conservées : 1 Gio par arbre, arrêt sous 2 Gio disponibles, délai maximal 7 200 s ; aucune alerte mémoire.

## Arrêt et suite minimale

Ne pas réenvoyer le ZIP. Le prochain contrôle utile consiste à retrouver le résultat de création dans le catalogue du propriétaire et à résoudre l'accès au slug demandé, uniquement en lecture seule. Aucun nouveau contrôle réseau n'a été lancé après le 403.

Le notebook CPU préparé a été repris localement avec une lecture de fenêtre PyTorch, `num_workers=0`, sans GPU, Internet ou entraînement (231 618 octets). Il n'a pas été soumis. Les scripts locaux `prepare-cpu-local.py`, `verify-remote-readonly.py` et `observe-upload-local.py` conservent cette préparation et l'observabilité ; ils ne constituent pas une certification réussie.

ZIP, SHA-256, données, anciens journaux et compteur préservés. Aucun autre lot, entraînement, GPU, commit, push ou merge.

Preuves : `lot-001-create-attempt-04/result.json`, `lot-001-create-attempt-04/network.jsonl`, `lot-001-upload-monitor-04/supervisor.json`, `remote-check-01/result.json`, `remote-check-01/network.jsonl`, `../kaggle_blob_validation_20261108_v1/google-json-endpoint-targeted/output.log`.
