# GeoFire-X — préparation finale après PASS Kaggle, 14 novembre 2026

**READY pour autoriser une unique expérience Focal. Exécution BLOCKED tant que l'utilisateur n'a pas explicitement autorisé ce run, son budget complet et ses destinations de sauvegarde.** `RUN_TRAINING=False`. Aucun entraînement Focal n'a eu lieu dans cette mission.

Ce bilan actualise et remplace, pour la décision courante, les anciens blocages décrits dans `focal_preparation_v1/GO_NO_GO.md` et `focal_repro_precheck_v1/technical_decision.json`. Ces documents restent des instantanés antérieurs conservés. La divergence de l'environnement CPU local ne bloque plus un lancement dans l'environnement Kaggle qualifié. Ne pas chercher à la résoudre à nouveau et ne pas remplacer l'empreinte initiale historique.

## Preuve reçue et vérifiée, sans refaire le précontrôle

Le JSON téléchargé est valide, `status=PASS`, causes vides. Son SHA-256, calculé sur ses octets locaux, est :

`d52dccc18e6caa71532c6c73de100d7294aff87d5d3a4fb785f401459a3ac3af`

Cette empreinte est désormais figée dans le reçu et le notebook. Il n'existe pas de signature distante indépendante fournie : le SHA établit l'identité du fichier reçu, pas une attestation cryptographique de Kaggle.

| Prérequis Kaggle | État attesté par le rapport |
|---|---|
| PyTorch / NumPy | `2.11.0+cu130` / `2.1.3` |
| Matériel du précontrôle | Deux Tesla T4 disponibles ; le futur run en expose une seule |
| Initialisation CPU puis CUDA | `a835383534d061787c2449bbb4a896c32c3e990d92b80fe2b3105f9c91ac6f5f` |
| Corpus | 11 identités attendues, 459 objets, 9 847 612 192 octets vérifiés intégralement |
| Ordres des quatre époques | Les quatre SHA correspondent exactement à la configuration figée |
| Opérations | 0 forward, 0 optimizer.step, 0 pixel 2021 ; aucun NPZ/raster décodé |

Le fichier et son empreinte ont été recopiés dans ce dossier. Aucun précontrôle CPU/CUDA, hash massif du corpus, téléchargement, reconstruction ou publication des datasets n'a été relancé localement.

## Notebook existant inspecté et finalisé

Le livrable reste **[le notebook existant](../focal_preparation_v1/GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb)**. Sa version avant modification est conservée ici. Le paquet scientifique embarqué et `focal_config.json` sont inchangés à l'octet près :

| Objet figé | SHA-256 |
|---|---|
| Configuration | `48a13f21fc1bf7f1d581025f585f78fc254ef9bd298fe4492c469cdd1256defc` |
| `focal-code.zip` | `8f48deba428b11be05cc20ccaaf9c4ef3dec19445a4479ccfbf9dd19355a16bd` |

Configuration vérifiée : Light U-Net B2, T=3, 8 canaux/jour, 123 977 paramètres, seed42, batch16, quatre époques sur 93 603 tuiles, 5 851 updates/époque et **23 404 updates** ; AdamW lr0,001, weight_decay0,0001, clipping1 ; Focal gamma2, poids10/1, réduction moyenne des pixels valides, Dice0. Float32, pas d'AMP ajouté, pas de changement de sampler/loader/masques, pas d'augmentation ou de recherche de paramètres.

La logistique est ajoutée dans `scripts/focal_run_logistics.py`, embarquée et hashée séparément. `CUDA_VISIBLE_DEVICES=0` est fixé avant l'import Torch ; le processus de lancement exige une unique Tesla T4 visible. Il n'utilise ni DataParallel, ni DDP, ni torchrun. Les quatre workers de chargement sont des workers CPU, pas quatre GPU.

Le lancement exige le code/config/logistique exacts, l'environnement historique, un dossier nouveau directement sous `/kaggle/working`, un contrôle d'espace et un fichier d'autorisation externe. Un verrou atomique empêche de relancer la cellule d'entraînement dans la même session. Aucun second candidat ou entraînement historique n'est prévu. Le notebook n'offre pas de reprise automatique d'entraînement.

## Évaluation et critères scientifiques conservés

Une seule évaluation finale, après l'époque4 fixe, sur les **32 événements historiques / 13 759 tuiles**. Même fonction d'évaluation, AP pooled exacte avec égalités globales, Brier, micro/macro par événement, nouveaux feux et pixels déjà actifs. Le nouveau wrapper intervient uniquement avant cet appel pour archiver le checkpoint ; il ne change pas les calculs. La baseline est lue depuis son JSON sauvegardé, jamais réentraînée ou inférée à nouveau.

Sélection32 : maximum Dice micro sur la grille figée ; égalités départagées par précision puis seuil supérieur. Un optimum en borne reste inconclusif. La comparaison de référence reste au seuil historique 0,78. Toutes les gardes préenregistrées restent nécessaires :

- gain AP pooled ≥ 0,005 ;
- variation Dice micro ≥ −0,005 ;
- gain de rappel des nouveaux pixels ≥ 0,02 ;
- FP des nouveaux pixels ≤ 1,15 × baseline ;
- variation Dice macro événement ≥ −0,01.

Le changement porte sur la famille de loss, y compris le retrait de Dice ; il n'isole pas le seul facteur de focalisation. Développement32 réutilisé : conclusion exploratoire avec une seed. Aucun nouveau seuil/critère après résultats, aucune sélection sur169 ou2021, aucune promotion Studio automatique.

## Budget complet proposé à l'autorisation

| Poste d'une session | Réserve de planification |
|---|---:|
| Démarrage, entrées existantes, identités et contrôles | 15 min |
| Entraînement unique | 45 min maximum par session avant pause existante |
| Évaluation32, sélection et AP exacte | 45 min |
| ZIP, vérifications et téléchargement | 15 min |
| **Réserve totale de session** | **120 min** |

Ces réserves ne sont pas des temps mesurés ou des bornes garanties. Le seul scénario de fit disponible reste 11,1–22,2 min, dérivé d'un U-Net apparenté mesuré à887,069s. Aucun benchmark supplémentaire n'a été lancé. Kaggle peut réserver la paire T4 même si CUDA n'en expose qu'une au code ; ne pas promettre un débit de quota équivalent à une seule carte. Pour planifier prudemment, prévoir jusqu'à deux heures de session accélérée et, si la paire est comptée carte par carte, jusqu'à quatre heures-carte. Le décompte effectif dépend du compte/session Kaggle et doit être vu dans son interface.

Si le plafond fit interrompt avant23 404 updates : statut `PAUSED_NOT_COMPLETE`, sauvegarde, aucune métrique officielle et aucun redémarrage automatique. Toute continuation ultérieure du même run demanderait une décision explicite, jamais un second candidat. Si seule l'AP échoue après entraînement complet : récupérer l'évaluation seule du checkpoint final, sans optimiseur ni nouveau fit.

**Espace : 12 GiB libres minimum vérifiés sur `/kaggle/working` et `/tmp`** (si même filesystem, il s'agit de la même réserve, pas24GiB). Proposition prudente : 6GiB AP/scratch, environ0,5GiB pour les239 checkpoints possibles si chacun tient dans2MiB, deux ZIP et marges de logs/copies/tri. La taille checkpoint de2MiB est une enveloppe de planification d'après les123 977 paramètres float32 et AdamW, pas une mesure. Les gardes AP et le calcul de place nécessaire au ZIP vérifient ensuite l'espace réel ; aucune AP approximative n'est substituée.

## Conservation préparée

Le checkpoint final et son SHA sont vérifiés avant l'AP. Une archive **TRAINING-SAFE** est alors créée et relue intégralement : checkpoints propres au run et sidecars SHA, configuration, code exact, normaliseur/index, reçu de montage, PASS, historique de loss et RNG/optimiseur/cursor du checkpoint. Les239 sauvegardes possibles sont conservées ; aucune rotation ne risque d'effacer le dernier état valide.

Après évaluation, l'archive **COMPLETE** ajoute `selection32.json`, `development32/evaluation.json` et CSV, `paired_comparison.json`, `training_complete.json`, `training_history.json`, `steps.jsonl`, `run.json` et `complete.json`. Chaque ZIP inclut un manifeste taille/SHA par fichier et un SHA externe. Une archive PARTIAL est tentée en `finally` sur erreur/pause. Le scratch AP est exclu des ZIP et n'est pas effacé par ce script.

La récupération `evaluate-only` refuse un checkpoint incomplet ou modifié et un résultat déjà complet ; elle ne construit aucun optimiseur. Si nécessaire, elle produit un nouveau dossier et une archive réunissant le checkpoint final sauvegardé et ses métriques récupérées. Ne pas appeler la cellule de fit pour réparer une erreur AP.

Après téléchargement, le script vérifie SHA du ZIP **et tous les membres**, copie dans un nouveau dossier local puis dans un nouveau dossier du disque externe et répète les vérifications. Les reçus `backup_verified.json` confirment chaque copie. Destination prévue : `C:\Users\nourb\Projects\GeoFire-X-2026\artifacts\focal-go-final-v1` puis `D:\GeoFire-X` (volume ADATA HV300 observé, environ408GiB libres au contrôle). Le lecteur peut changer : vérifier son identité avant la copie future. Aucune écriture externe n'a été faite dans cette mission.

Une panne brutale du noyau/disque ou l'absence de téléchargement ne peut être couverte par un bloc `finally`. La sauvegarde n'est considérée acquise qu'après les ZIP vérifiés, leur téléchargement et les reçus des copies locale/externe. Ne pas terminer la session avant cette étape. Un paquet local de la préparation est également conservé ; il ne contient aucun résultat Focal inventé.

## Actions exactes de l'utilisateur et ordre des travaux

1. Lire [les instructions cellule par cellule](KAGGLE_CELLULES_ET_SAUVEGARDE.md) et approuver explicitement **UN** entraînement Focal, la réserve120min et les destinations locale/externe. Aucune autorisation active n'est livrée.
2. Importer le notebook finalisé uniquement, attacher les11 datasets existants et Internet OFF. Démarrer un noyau neuf pour rendre le masque GPU effectif. Vérifier versions et espace au lancement ; le contrôle des montages de cette session ne relance pas le notebook de précontrôle.
3. Après cette autorisation seulement, créer le fichier d'autorisation correspondant au modèle désactivé, activer et exécuter une seule fois la cellule5. Puis cellule7, téléchargement et copies vérifiées.
4. Priorité scientifique suivante : [preuves de disponibilité à J et U-Net multimodal comparable](PRIORITE_MULTIMODALE.md), puis ResNet18-U-Net et U-TAE. Le T=1 et les autres ablations ne sont pas des préalables obligatoires à cette priorité.

Ni modèle principal, ni Studio V2, ni résultat/checkpoint historique, ni11 partitions compactes, ni test2021 n'ont été modifiés. Aucune opération Git n'a été exécutée ; seule la référence `.git/HEAD` a été lue pour confirmer la branche.

## Vérification finale de la préparation

**12 tests ciblés passent**, en9,53s : [JUnit final](tests-final-v2.xml). Ils vérifient le reçu PASS sans rescanner le corpus, l'identité du payload scientifique, le notebook désactivé, le refus d'autorisation absente/incomplète, l'espace, les SHA/manifeste ZIP, le rejet de corruption et les copies locales. Deux simulations avec un runner et un espace CUDA factices vérifient les chemins succès/échec AP : TRAINING-SAFE existe avant l'appel d'évaluation, PARTIAL existe après une erreur, et COMPLETE inclut sélection32 après succès. Aucun modèle, optimiseur, pixel réel ou GPU n'est exécuté par ces simulations. Aucun checkpoint historique n'est désérialisé.

Les118 fichiers de sources, frontend/Studio, configurations et références explicitement suivis sont inchangés : [preuve de préservation](preservation_after.json). Le notebook courant et ses deux payloads additionnels compilent et correspondent aux empreintes attendues. Cette vérification de logistique ne remplace pas une exécution réelle du futur entraînement ; aucune mesure de Focal ou reprise CUDA réussie n'est inventée.

Commande de validation finale exécutée :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_focal_run_logistics.py tests/test_focal_experiment.py::test_prepared_notebook_bundle_is_pinned_compilable_and_disabled -q -p no:cacheprovider --basetemp=artifacts/focal-go-final-v1/pytest-final-v2 --junitxml=reports/focal_go_final_v1/tests-final-v2.xml
```

Le sous-dossier `kit` rassemble des copies identiques du notebook finalisé, de son paquet/config figés, de la liste des11 sources et des scripts logistiques. L'archive PREPARATION et son SHA sont sauvegardés et vérifiés localement ; leur reçu est dans `artifacts/focal-go-final-v1/preparation_backup_receipt.json`. Il s'agit d'une sauvegarde de préparation, pas d'un résultat Focal. Les copies de résultats sur disque externe attendent leur création et leur téléchargement futurs.
