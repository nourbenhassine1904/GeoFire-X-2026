# GeoFire-X — diagnostic HTTP 403, 8 novembre 2026

**Résultat : précontrôle corrigé et validé en lecture seule. Aucun nouvel essai de publication.** Zéro upload ZIP/GeoTIFF, écriture de dataset, soumission de notebook ou entraînement. Aucun commit/push. Les 14 tests locaux ciblés passent ; aucune suite scientifique relancée.

**Requête identifiée et reproduite sans écriture**

```text
POST https://api.kaggle.com/v1/datasets.DatasetApiService/GetDataset
Content-Type: application/json
Authorization: Bearer <secret non journalisé>
Ressource : ownerSlug=nnnnbhss, datasetSlug=geofirex-wsts-fold0-tv-001-v1
Réponse : HTTP 403, PERMISSION_DENIED
Message serveur : Permission 'datasets.get' was denied
```

Le journal de l'échec initial contient le nom RPC et HTTP 403 ; l'endpoint exact a été reconstitué depuis le SDK installé et observé lors de cette reproduction. Il s'agit d'un POST RPC OAuth, pas d'un GET REST sans authentification. Le contrôle alternatif `GetDatasetStatus`, utilisé par le CLI, reçoit le même refus sur ce slug.

**Identité et compatibilité vérifiées** : `nnnnbhss`, jeton actif, portée retournée par introspection **`resources.admin:*`**. La lecture d'un dataset privé existant du même compte reçoit HTTP 200, version 1 ; les listes propriétaires reçoivent HTTP 200. L'authentification et l'accès privé fonctionnent donc. Le jeton d'accès enregistré étant expiré, il a été rafraîchi uniquement en mémoire ; aucune nouvelle clé durable ni modification de portées, aucun secret révélé, fichier de credentials inchangé.

CLI installé **2.2.5**, SDK **0.1.38**, correspondant aux versions publiées courantes vérifiées sur [PyPI Kaggle](https://pypi.org/project/kaggle/) et [PyPI kagglesdk](https://pypi.org/project/kagglesdk/). Aucune mise à jour/install nécessaire. Les véritables méthodes du CLI ont été exercées avec le même client authentifié : `datasets list --mine` retrouve les six datasets, `datasets status` échoue aussi en 403 sur la cible. La [documentation officielle](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/datasets.md) décrit ces commandes de liste et de statut. Aucun défaut de version ou rejet global de l'authentification n'a été démontré.

**Existence, absence, accès et incertitude**

| État | Conclusion observée / règle conservée |
|---|---|
| Dataset existant | Aucun enregistrement du slug cible trouvé. Toute présence dans une page, même partielle, ou tout statut réussi interdit la création. |
| Absence suffisamment vérifiée pour un CreateDataset neuf | Catalogues authentifiés `MY` = 6, `MY_PRIVATE` = 6, `MY_PUBLIC` = 0, chacun entièrement paginé, sans filtre ; vues cohérentes, aucun slug cible. Vue propriétaire `USER` et CLI officiel concordants, contrôle de lecture sur un dataset privé existant réussi. |
| Accès interdit | Consultation directe de la cible : `GetDataset` et `GetDatasetStatus` restent 403. Cet état est conservé explicitement dans `direct_lookup_state=ACCESS_FORBIDDEN`. |
| Inconnu | Pourquoi le serveur refuse `datasets.get` sur cette ressource, droit effectif de création et disponibilité atomique du slug côté serveur. Aucun 403 n'est interprété comme preuve d'absence. |

La liste exacte des six handles et les messages expurgés sont dans `account-probes/diagnosis.json`. L'absence retenue est limitée au catalogue complet visible du propriétaire ; elle n'est pas une garantie qu'aucun slug réservé/objet invisible n'existe côté serveur. Un conflit ou refus de création reste possible. Aucun endpoint de test non modifiant du droit de création n'est exposé par le client IAM installé ; aucune écriture n'a été tentée pour le tester.

**Cause démontrée du blocage local** : l'ancien précontrôle exigeait une réponse 404 de `GetDataset` pour continuer. Le serveur répond 403 sur la cible et le script s'arrêtait, malgré les catalogues propriétaires consultables. La cause interne exacte du 403 reste inconnue ; l'hypothèse d'une clé invalide ou d'une permission générale de lecture privée manquante n'est pas soutenue par les contrôles réussis.

**Corrections effectuées**

- `scripts/kaggle_dataset_preflight.py` : pagination complète et bornée de trois vues propriétaires, cohérence public/privé, contrôle positif de lecture et classification explicite `DATASET_EXISTS` / `ABSENT_VERIFIED` / `ACCESS_FORBIDDEN` / `UNKNOWN`. Les deux derniers états bloquent. Un 403 seul, un échec de contrôle, une vue incohérente ou une pagination incomplète ne permettent jamais de créer.
- `scripts/publish_kaggle_single_attempt.py` : utilise ce précontrôle frais avant tout blob, sauvegarde les preuves, garde l'empreinte/handle/taille épinglés, création privée uniquement, plafond persistant de 6 Go, PUT unique sans retry. Aucune méthode de version, remplacement ou mise à jour n'est autorisée. Une collision doit être refusée par `CreateDataset` ; aucun fallback vers un remplacement.
- `reports/kaggle_prepublication_20261107_v2/PILOTE_LOT_001.ps1` : nouvelle tentative explicite `02`, chemins de reçus distincts ; la tentative 01 et ses journaux ne seront pas écrasés. Aucune boucle ni relance automatique.

Le [code officiel du CLI](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py) utilise `dataset_status` pour son précontrôle, puis traite **toute** erreur HTTP comme absence. Cette règle trop large n'a pas été reprise. Seules les opérations de liste documentées et une preuve propriétaire indépendante servent ici à établir l'absence du catalogue.

**Validation réellement exécutée** : 14 tests de sécurité et pagination, contrôle syntaxique du PowerShell, puis précontrôle corrigé sur le compte réel en lecture seule. Résultat `ABSENT_VERIFIED`, `allow_create_new=true`, consultation directe toujours `ACCESS_FORBIDDEN`, `create_permission_verified=false`. Cela rend une nouvelle tentative techniquement préparée, sans garantir son succès ni la permission serveur finale. Aucun notebook CPU/PyTorch lancé pendant ce diagnostic.

**Préservation et ressources** : ZIP toujours à `artifacts/kaggle_publication_20261107_v1/lot-001/payload.zip`, **4 829 918 684 octets**, reçu SHA256 inchangé `464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890`. Aucun nouveau scan complet du ZIP de 4,83 Go. Journaux d'échec inchangés ; copie des premières lignes du budget conservée. Les diagnostics ont ajouté **1 733 octets** de corps HTTP aux **354** initiaux, soit **2 087 / 6 000 000 000**, reste **5 999 997 913 octets**. Aucun compteur remis à zéro ; l'ancien préfixe du journal est préservé.

Diagnostics séquentiels sous garde-fou : RAM disponible initiale 4,32 Gio, minimum 4,25 Gio, pic RSS de l'arbre environ 57,3 Mio. Le dernier contrôle se termine avec environ 4,91 Gio disponibles. Pas de Spark, pixels raster ou fermeture d'applications.

**Commande exacte pour une prochaine tentative, NON exécutée et nécessitant une nouvelle autorisation**

Depuis `C:\Users\nourb\Projects\GeoFire-X-2026` :

```powershell
.\reports\kaggle_prepublication_20261107_v2\PILOTE_LOT_001.ps1 -Phase Publish -PublicationAttempt 2 -Execute
```

Elle vise exclusivement `nnnnbhss/geofirex-wsts-fold0-tv-001-v1`, réutilise le ZIP existant, en recontrôle le SHA256 avant réseau, exige au moins 3 Gio disponibles et conserve le même compteur cumulé de 6 Go. Le précontrôle sera refait à l'exécution pour ne pas utiliser un état périmé. Répertoires futurs : `lot-001-upload-monitor-02` et `lot-001-create-attempt-02`, actuellement inexistants. Aucune certification CPU n'est déclenchée automatiquement par cette commande. Après tout échec, arrêt sans nouvelle tentative.

**Attente : nouvelle autorisation explicite avant d'exécuter cette commande.**
