# GeoFire-X — deuxième tentative, 8 novembre 2026

**Résultat : arrêt propre avant le PUT du ZIP, aucune relance.** Le précontrôle corrigé et l'authentification réussissent. Kaggle accepte `StartBlobUpload` en HTTP 200 ; le script s'arrête ensuite pendant la validation locale de cette réponse. Aucun nouveau HTTP 403 n'a été reçu.

**Contrôles avant transfert** : ADATA D: accessible, volume `ADATA HV300` prêt, racine `D:/GeoFire-X/WildfireSpreadTS/rasters` accessible, **438 386 307 072 octets libres**. La commande CIM `Get-Volume` n'était pas autorisée ; le contrôle du volume a été effectué avec `System.IO.DriveInfo`, sans modification du disque. Le ZIP existant n'a pas été reconstruit : **4 829 918 684 octets**, SHA256 recontrôlé avant tout réseau : `464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890`.

Le propriétaire OAuth est bien **nnnnbhss**. Les trois catalogues propriétaires complets et le contrôle de lecture d'un dataset privé existant réussissent : cible absente du catalogue, six datasets propriétaires recensés. Aucun écrasement autorisé. Le fichier de credentials reste inchangé.

Pour respecter votre arrêt immédiat sur tout nouveau 403, le précontrôle de publication utilise uniquement ces catalogues et le contrôle positif : il ne réinterroge pas le statut du slug non créé, qui avait retourné le 403 connu. Le garde réseau arrête désormais immédiatement sur tout HTTP 403, sans qu'un gestionnaire de sonde puisse l'absorber. Les 14 tests précédents et deux tests supplémentaires sur ces règles ont réussi, soit **16 tests ciblés**, sans suite scientifique ou test lourd.

**Réseau réellement exécuté pendant la tentative 02**

| Opération | HTTP | Corps envoyé/comptabilisé |
|---|---:|---:|
| GenerateAccessToken | 200 | 121 octets |
| IntrospectToken | 200 | 160 octets |
| ListDatasets MY | 200 | 56 octets |
| ListDatasets MY_PRIVATE | 200 | 64 octets |
| ListDatasets MY_PUBLIC | 200 | 63 octets |
| GetDataset, contrôle privé existant | 200 | 77 octets |
| StartBlobUpload | 200 | 111 octets |
| **Total tentative 02** | 7 requêtes | **652 octets** |

Aucun PUT ni appel `CreateDataset` n'a été envoyé. Une session de blob a été demandée et acceptée ; elle n'a reçu aucun octet du ZIP et ne constitue pas un dataset publié. Son URL signée et son jeton n'ont pas été journalisés ni réutilisés.

**Diagnostic** : `RuntimeError` local, étape `start_blob_upload`. Après la réponse 200, le code peut refuser un schéma/hôte de `create_url` non admis ou un jeton de blob vide. **Le journal actuel ne distingue pas ces deux causes** : il manque un code de motif local et des métadonnées non sensibles de la réponse. Il serait incorrect d'affirmer un nouveau refus Kaggle ou un hôte précis. L'exception a été capturée et le processus est terminé ; l'appel n'a pas été rejoué pour obtenir ces informations.

**Bilan factuel**

- Dataset créé par cette tentative : **non** ; aucune version obtenue.
- URL privée publiée vérifiée : **aucune**. Destination prévue seulement : `https://www.kaggle.com/datasets/nnnnbhss/geofirex-wsts-fold0-tv-001-v1`.
- Volume ZIP/GeoTIFF transféré : **0 octet**.
- Corps HTTP cumulés : **2 739 / 6 000 000 000 octets**, dont les 2 087 octets antérieurs et 652 nouveaux.
- Budget restant : **5 999 997 261 octets**. Compteur conservé, aucun reset. En-têtes/TLS/TCP et réponses reçues ne sont pas mesurés par ce compteur.
- Fichiers certifiés à distance : **0 / 1 269** ; état `ready`, visibilité privée et persistance non vérifiés faute de création.
- Certification CPU, fenêtre PyTorch, GPU et entraînement : **non exécutés**. Aucun notebook ni nouvelle publication de code.
- RAM disponible : **5,06 Gio au départ**, minimum **5,04 Gio**, environ **5,16 Gio après**. Pic RSS de l'arbre : **49,4 Mio** ; durée supervisée **7,875 s**. Aucun problème mémoire ou budget.

Les logs sont conservés dans `lot-001-create-attempt-02/` et `lot-001-upload-monitor-02/`. La tentative 01, ses journaux, le corpus historique, les Parquet, configurations et checkpoints sont préservés. Aucun transfert des sept autres lots, commit ou push.

**Prochaine action proposée, pas exécutée** : diagnostiquer spécifiquement la validation de la réponse `StartBlobUpload`, avec journalisation limitée à un code de motif, au schéma/hôte et à la présence des champs, jamais à l'URL signée ou au jeton. Si une nouvelle session de blob est nécessaire pour ce diagnostic, elle demandera une nouvelle autorisation limitée, sans PUT et sans création de dataset. Aucune troisième tentative ni publication des autres lots avant résolution et nouvel accord.
