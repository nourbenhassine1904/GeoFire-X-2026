# GeoFire-X — pilote 001 arrêté avant upload

**Résultat : arrêt propre, sans retry.** Aucun ZIP/GeoTIFF envoyé, aucune création de dataset, aucune soumission de notebook CPU ou GPU. Les sept autres lots, le code Kaggle et les résultats scientifiques historiques n'ont pas été transférés ou modifiés. Aucun commit, push ou merge.

Le pilote était limité au handle `nnnnbhss/geofirex-wsts-fold0-tv-001-v1` avec plafond dur **6 000 000 000 octets** de corps HTTP, identique dans le compteur persistant et la borne de phase. La proposition antérieure de réserve à 11 Go n'a pas été utilisée.

**ZIP vérifié avant réseau** : 4 829 918 684 octets ; SHA256 `464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890`. Il correspond au lot local de 57 événements train, 1 269 GeoTIFF, 4 829 474 073 octets bruts, aucun événement 2021. Cette vérification est réussie.

**Cause d'arrêt établie** : la requête `GetDataset`, utilisée pour vérifier l'absence du dataset avant une création sans remplacement, a reçu HTTP **403**. L'absence n'est donc pas prouvée ; le garde-fou a refusé de passer à l'upload. L'origine exacte du refus 403 reste inconnue. Cette réponse ne permet pas d'affirmer que le dataset existe ou qu'il est absent. Aucun contournement ni remplacement n'a été tenté.

| Requête unique | Réponse | Corps HTTP comptabilisé |
|---|---:|---:|
| GenerateAccessToken | 200 | 121 octets |
| IntrospectToken | 200 | 160 octets |
| GetDataset | **403** | 73 octets |
| **Total** | 3 requêtes | **354 octets** |

L'authentification existante et le propriétaire attendu ont passé leur contrôle. Le rafraîchissement OAuth est resté en mémoire ; le fichier de credentials est inchangé. Les journaux ne contiennent ni jetons ni URL signée.

**Comptabilité** : 0 octet de corps ZIP/GeoTIFF transféré ; 354 octets de corps de contrôle envoyés et réservés, soit 354 / 6 000 000 000. Solde **5 999 999 646 octets**. Le compteur ne mesure pas les en-têtes/TLS/TCP ni les réponses reçues. Aucune requête `StartBlobUpload`, aucun PUT et aucune requête `CreateDataset` n'ont été envoyés.

**État distant et CPU** : URL de destination prévue seulement : `https://www.kaggle.com/datasets/nnnnbhss/geofirex-wsts-fold0-tv-001-v1`. Aucune URL publiée ni version créée par cette tentative. Visibilité privée et persistance non certifiées ; état `ready` non obtenu. **0 / 1 269 fichiers vérifiés à distance** ; lecture de fenêtre PyTorch **non exécutée**. Aucun résultat CPU distant n'est annoncé comme réussi.

**Ressources locales** : 4 859 224 064 octets disponibles au départ (4,53 Gio), minimum 4 826 640 384 (4,50 Gio), après 4 884 045 824 (4,55 Gio). Pic RSS de l'arbre 51 605 504 octets (49,2 Mio). Durée supervisée 7,14 s, limite d'arbre 1 Gio ; aucune insuffisance mémoire ni dépassement budgétaire. L'arbre créé par cette tentative a été fermé.

**Preuves conservées** : `lot-001-upload-monitor-01/supervisor.json`, `output.log`, `memory.jsonl`, `lot-001-create-attempt-01/started.json`, `result.json`, `network.jsonl`, et `http-body.jsonl`. Le ZIP et son reçu restent à leur emplacement de staging initial. Aucun compteur réinitialisé.

**Prochaine action proposée** : diagnostiquer le refus 403 au moyen d'une consultation ciblée autorisée en lecture seule, puis revoir la preuve d'absence avant de proposer une nouvelle tentative. Aucune nouvelle requête, publication ou certification n'a été lancée après cet échec. Toute reprise nécessite une nouvelle autorisation explicite et conserve les 354 octets déjà consommés.
