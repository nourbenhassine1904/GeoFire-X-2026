# GeoFire-X — validation StartBlobUpload, 8 novembre 2026

**Correction locale vérifiée, sans aucune opération réseau. La cause précise de la tentative 02 reste indéterminée.** Le défaut démontré est la perte du motif de rejet dans nos journaux : seul `RuntimeError` avec étape `start_blob_upload` avait été conservé. Aucun corps de réponse, hôte, schéma ou indicateur de présence du jeton n'a été enregistré ; impossible de départager URL manquante, schéma/hôte refusé ou jeton absent à partir de ces traces. Aucun appel n'a été rejoué pour combler cette lacune.

**Contrat officiel installé, vérifié localement**

- Kaggle CLI 2.2.5 / kagglesdk 0.1.38 : réponse JSON documentée `{"createUrl": "…", "token": "…"}`.
- `FieldMetadata("createUrl", "create_url", ...)` réalise la conversion ; le SDK expose `.create_url`, `.token` et l'alias de lecture `.createUrl`.
- L'ancien accès `blob.create_url` / `blob.token` est donc conforme. Aucun défaut camelCase/snake_case n'est démontré dans cet appel. Les tests confirment la désérialisation réelle du SDK installé.
- Un JSON contenant seulement `create_url` serait ignoré par le SDK pour ce champ ; le test le reproduit. Nous ne supposons pas que le serveur a renvoyé ce format, et nous ne réparons pas silencieusement une réponse hors contrat.

**Fichiers corrigés / ajoutés**

1. `scripts/kaggle_blob_session.py` : validateur isolé et diagnostics expurgés. Réponse SDK ou dictionnaire camelCase conforme ; rejet explicite des alias contradictoires, champs absents/vides/non textuels et réponses JSON hors contrat.
2. `scripts/publish_kaggle_single_attempt.py` : projection sûre de la réponse JSON **avant** conversion SDK, puis projection de l'objet SDK ; étape dédiée `validate_blob_session`, code local précis dans le reçu d'échec. Le PUT et la création sont impossibles si la session est rejetée.
3. `tests/test_kaggle_blob_session.py` : 33 tests locaux ciblés, avec SDK réel pour la conversion et clients simulés pour le chemin de publication.
4. `reports/kaggle_prepublication_20261107_v2/PILOTE_LOT_001.ps1` : tentative future `03`, nouveaux répertoires, sans écraser les tentatives 01/02.

**Sécurité et codes d'erreur**

Aucun domaine ajouté. URL absolue **HTTPS** uniquement, hôte `storage.googleapis.com` ou sous-domaine DNS valide de cette famille ; port implicite ou **443**. Refus de l'userinfo, fragments, contrôles/espaces, antislashs, hôtes trompeurs, suffixes malveillants et hôtes non prouvés. L'URL exacte autorisée reste liée à un seul PUT ; redirects et retries restent désactivés. L'absence de preuve sur un autre hôte ne justifie pas d'élargir la liste.

Les nouveaux fichiers futurs `blob-session-wire-metadata.json` et `blob-session-sdk-metadata.json` ne contiennent que des indicateurs/types de champs, schéma, hôte DNS expurgé, port et drapeaux syntaxiques. **Aucune URL complète, chemin, query string, signature, valeur/longueur de jeton, représentation brute du SDK ou credential.** Le jeton est masqué aussi dans la représentation de l'objet de session validée.

Exemples de motifs déterministes : `BLOB_URL_MISSING`, `BLOB_URL_TYPE_INVALID`, `BLOB_URL_SCHEME_REJECTED`, `BLOB_URL_HOST_REJECTED`, `BLOB_URL_PORT_REJECTED`, `BLOB_TOKEN_MISSING`, `BLOB_TOKEN_TYPE_INVALID`, `BLOB_URL_ALIAS_CONFLICT`, `BLOB_WIRE_CASE_UNSUPPORTED`, `BLOB_WIRE_JSON_INVALID`. La réponse JSON observée est limitée à 64 Kio ; une réponse excessive reçoit `BLOB_WIRE_RESPONSE_TOO_LARGE`. Un nouvel HTTP 403 conserve son arrêt immédiat `HTTP_403_STOP`.

**Stratégie retenue**

Conserver le SDK officiel pour `StartBlobUpload` et `CreateDataset`, son contrat de désérialisation et la création privée explicite. Conserver un seul PUT en flux supervisé par notre garde réseau. Dans le CLI installé, `upload_complete()` configure `Retry(total=10)` et les chemins supérieurs permettent jusqu'à dix reprises, ainsi que des retries d'appels API ; le contexte de reprise peut persister URL signée et jeton. Remplacer directement notre publisher par `dataset_create_new()` ne respecterait donc pas les exigences sans adaptation substantielle. Son précontrôle traite aussi toute erreur HTTP de statut comme absence ; nous gardons le contrôle propriétaire plus strict.

Le correctif prouvé porte sur le diagnostic incomplet et les validations de sécurité. **Il ne démontre pas encore que la session réelle inconnue de la tentative 02 serait acceptée ni que la publication est débloquée.** Un hôte réel non autorisé restera bloqué, avec son motif enfin identifiable. Aucun domaine supplémentaire ne sera accepté automatiquement.

**Tests réellement réussis**

**33 tests**, 0,047 s pour les assertions et environ 0,56 s sous supervision dans le dernier passage. Conversion SDK, objets snake/camel, mappings, champs invalides, JSON malformé/excessif, variantes HTTPS/GCS, ports et domaines refusés, absence de secrets dans les diagnostics. Trois simulations couvrent la publication : hôte refusé et jeton absent bloquent **PUT et CreateDataset** ; session valide permet uniquement la création explicitement privée. Les envois HTTP sont interdits par le banc de tests ; aucun ZIP réel n'est lu par ces simulations. Syntaxe PowerShell vérifiée. Processus plafonné à 256 Mio, RAM disponible minimale du dernier passage environ 5,14 Gio ; aucun test lourd relancé.

**Préservation et budget**

ZIP existant : **4 829 918 684 octets**, non reconstruit et sans nouvelle lecture de son contenu. Taille et dernière modification conservées. Empreinte enregistrée inchangée : `464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890` ; dernier recalcul réussi lors de la tentative 02, prochain recalcul obligatoire avant tout futur transfert. Journaux 01/02 conservés ; budget inchangé : **2 739 / 6 000 000 000 octets**, reste **5 999 997 261 octets**. Aucune clé ni authentification manipulée pendant ce diagnostic.

**Troisième tentative proposée, non exécutée et soumise à nouvelle autorisation**

Depuis la racine du projet :

```powershell
.\reports\kaggle_prepublication_20261107_v2\PILOTE_LOT_001.ps1 -Phase Publish -PublicationAttempt 3 -Execute
```

Cible unique `nnnnbhss/geofirex-wsts-fold0-tv-001-v1`. SHA256 et taille épinglés, catalogue propriétaire exhaustif frais, refus d'un dataset existant, `CreateDataset` privé uniquement, aucun appel de version/remplacement. Répertoires de sortie exclusifs `03` et verrou du budget empêchent deux publications simultanées ; le compteur cumulé reste à 6 Go, sans reset. Un seul PUT possible et aucune relance automatique. RAM disponible initiale requise ≥3 Gio, réserve système 2 Gio, arbre enfant plafonné à 1 Gio, échéance 7 200 s. Cette commande ne lance aucun notebook ni entraînement et n'envoie aucun autre lot.

**Arrêt ici : aucun StartBlobUpload, PUT, création de dataset ou autre appel réseau pendant cette mission. Nouvelle autorisation explicite attendue avant la commande ci-dessus.**
