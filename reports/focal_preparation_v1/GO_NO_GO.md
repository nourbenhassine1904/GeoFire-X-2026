# GeoFire-X — préparation d'une expérience Focal unique

**Décision : GO pour la préparation technique CPU ; NO-GO pour une exécution GPU à ce stade.**

Branche `feature/scientific-improvements-v1`, HEAD `8c939f6`. Aucun entraînement ni job Kaggle soumis. Le notebook reste désactivé. Cette décision ne constitue aucune autorisation d'exécution.

## Prêt et conservé

Le [protocole scientifique](../../docs/SCIENTIFIC_IMPROVEMENTS_V1.md), le [contrat A→E](../../configs/scientific_improvements_v1.json), sa validation et `experiment_readiness.py` ont été lus et conservés sans modification. L'expérience A devient une préparation concrète, sans remplacer les choix déjà établis.

Les paramètres sont figés dans **une seule configuration expérimentale** : [focal_config.json](focal_config.json). Les autres JSON décrivent la provenance, les résultats historiques ou les contrôles ; ils ne proposent pas d'autres traitements.

| Paramètre | Valeur unique |
|---|---|
| Identité | `A_focal_gamma2_pos10_seed42_v1` |
| Modèle | `make_model('B2',8,3)`, même LightUNet, base8/profondeur3, 123 977 paramètres |
| Entrée | `[B,3,8,64,64]`, fusion précoce de 24 canaux ; J−2/J−1/J |
| Cible / masque | Feu J+1 ; intersection de la cible et des cinq footprints J−4…J, padding exclu |
| Train | 250 événements 2018–2019, 93 603 tuiles/époque, toutes une fois |
| Initialisation | Fraîche, seed42, **aucun checkpoint historique** ; empreinte initiale appariée obligatoire |
| Budget d'optimisation | 4 époques, batch16, dernier batch3, 5 851 updates/époque, 23 404 updates |
| Optimiseur | AdamW, lr0,001, weight_decay0,0001, clipping global1 |
| Loader | FastCompactTiles inchangé, LocalitySampler seed42/blocs256, quatre workers, generator43, prefetch2, persistent workers |
| Traitements | Pas d'augmentation, pas d'oversampling, pas d'AMP ajouté, pas de recherche de paramètres |
| Loss | Focal binaire seule sur logits, gamma2, poids positif10/négatif1, Dice0 |
| Checkpoint comparé | Époque4 fixe ; aucun choix d'époque ni validation intermédiaire |
| Développement | Les mêmes 32 événements historiques, 13 759 tuiles ; une évaluation finale du candidat |
| Environnement exigé | PyTorch `2.11.0+cu130`, NumPy `2.1.3` |
| Studio / test | Aucune promotion Studio ; 169 déjà consulté, descriptif seulement ; 2021 verrouillé |

Les quatre empreintes de l'ordre exhaustif des tuiles sont enregistrées dans la configuration. Elles ont été calculées à partir de l'index certifié avec le NumPy local ; leur concordance devra aussi être vérifiée dans l'environnement historique. L'identité de source et la seed ne suffisent pas à promettre une reproductibilité bit à bit entre versions ou GPU.

## Loss, logits, probabilités et cas limites

La [nouvelle implémentation isolée](../../src/geofirex/focal_loss.py) n'est importée ni par le serveur ni par les anciens trainers. Le U-Net retourne des **logits**. Une sigmoid est appliquée pour les métriques et l'inférence, pas avant l'entrée de la loss.

Pour p=sigmoid(z), la formule retenue est :

`FL = [10·y·(1−p)^2·softplus(−z) + (1−y)·p^2·softplus(z)]`, sommée puis divisée par le **nombre de pixels valides du batch**, sans division par la somme des poids de classe.

Les poids10/1 équivalent à alpha10/11 multiplié globalement par11. Aucun second alpha n'est appliqué. Gamma0 retrouve la BCE pondérée10. Le calcul utilise le logit signé et `sigmoid(-logit_signé)` pour éviter la soustraction de probabilités proches de1 ; aucune saturation/clamp arbitraire des probabilités n'est ajoutée.

Les pixels invalides sont sélectionnés hors calcul avant toute opération, y compris lorsque leur padding contient NaN/Inf. Une image entièrement invalide n'augmente pas le dénominateur. Un batch sans feu conserve sa loss et ses gradients sur les négatifs : il n'est pas ignoré. La loss renvoie zéro avec gradients nuls pour un batch entièrement invalide ; le futur lanceur refuse cependant un tel batch inattendu, afin de ne pas changer silencieusement le budget d'optimisation du corpus certifié. Le float16/bfloat16 est promu en float32 pour le calcul ; le run retenu reste en float32.

La baseline est `0,5 BCE(pos_weight10) + 0,5 Dice` par image, avec BCE normalisée sur les pixels valides. **La comparaison change la famille de loss, y compris le retrait de Dice. Elle ne démontre pas à elle seule l'effet causal du seul facteur de focalisation.** Ce périmètre respecte le protocole déjà préparé ; aucun contrôle gamma/alpha supplémentaire n'est lancé ou ajouté à la campagne.

## Comparaison historique prête, sans réévaluation

Les JSON du run terminé `B2_posweight_10` ont été copiés à l'identique depuis l'archive d'ablation existante : [configuration historique](baseline_config.json), [sélection historique](baseline_selection32.json), [évaluation32](baseline_evaluation32.json). Aucun `.pt` n'a été chargé. Leur provenance et leurs empreintes sont figées dans la configuration et le build.

Le [contrat de comparaison](baseline_comparison_contract.json) agrège exclusivement ces sorties sauvegardées au seuil historique0,78 :

| Métrique de développement32 | Baseline sauvegardée |
|---|---:|
| Dice micro / IoU | 0,387282 / 0,240143 |
| Précision / rappel | 0,339690 / 0,450383 |
| AP pooled / Brier | 0,267884 / 0,00278906 |
| TP / FP / FN | 36 640 / 71 223 / 44 713 |
| Ratio positifs prédits/observés | 1,325864 |
| Nouveaux pixels : précision / rappel / Dice | 0,219427 / 0,156487 / 0,182688 |
| Nouveaux pixels : TP / FP / FN | 7 850 / 27 925 / 42 314 |
| Nouveaux pixels : ratio prédits/observés | 0,713161 |
| Dice macro événement / médiane | 0,230231 / 0,207234 |

Ces valeurs concernent **32**, pas les métriques de confirmation169 présentées dans l'audit. Les deux panels ne sont pas mélangés. Aucune comparaison numérique à un protocole WSTS+ différent n'est faite.

Le futur candidat utilisera le même `evaluate`, les mêmes probabilités float32 non recalibrées, l'AP exacte avec gestion globale des égalités, le même Brier et les mêmes conventions d'événements vides. Le lanceur restreint l'évaluation aux32 et refuse toute différence de liste d'événements, nombre de tuiles, comptages de pixels valides/positifs globaux, nouveaux/anciens et par événement. L'identité des pixels repose aussi sur les archives, index, normaliseur et code de lecture certifiés : des comptages égaux, seuls, ne prouvent pas une identité de masque.

Le rapport final préparé contient Dice/IoU/précision/rappel micro, AP, Brier, nouveaux feux, pixels déjà actifs, ratios de positifs, TP/FP/FN, moyennes/médianes/quantiles et nombres définis par événement, ainsi que les différences appariées de Dice. Les comparateurs historiques ne sont jamais inférés à nouveau.

## Seuil et acceptation préenregistrés

Grille inchangée : 0,10…0,95 par0,005, puis0,96/0,97/0,98/0,99/0,995/0,999. Sélection sur le Dice micro du **développement32 uniquement** ; égalité départagée par précision puis seuil supérieur. Optimum sur une borne : conclusion inconclusive, sans élargissement opportuniste. Le modèle reste celui d'époque4.

Toutes les gardes déjà proposées doivent être satisfaites, face à la baseline32 à0,78, au seuil choisi du candidat :

- AP pooled : gain ≥0,005 ;
- Dice micro : variation ≥−0,005 ;
- rappel des nouveaux pixels : gain ≥0,02, soit deux points ;
- FP des nouveaux pixels : ≤1,15 fois la référence ;
- Dice macro événement : variation ≥−0,01.

Ce sont des critères exploratoires d'ingénierie, pas des gains attendus ni une preuve statistique avec une seule seed. Brier, précision et ratio sont rapportés sans inventer de nouvelles gardes après lecture des résultats. Une acceptation ne promeut pas le candidat dans Studio. Ni169 ni2021 ne sélectionnent la loss, le seuil ou les critères.

## Corpus et artefacts vérifiés

[Vérification des partitions](partitions_verified.json) : **11 identités distinctes, 459 objets, 9 847 612 192 octets publiés**. Les manifestes canoniques, tailles et SHA-256 ont été vérifiés sur les sources locales et les payloads de publication. Cela inclut les451 archives événementielles train/val et les métadonnées/index. Aucun NPZ ni raster réel n'a été décodé ; aucun dataset n'a été téléchargé ou republié.

Le normaliseur, l'index, la couverture250/201, les32 événements et l'ordre exhaustif du sampler passent le [précontrôle de métadonnées](metadata_preflight.json). Le [même contrôle dans le paquet embarqué](bundled_preflight.json) passe dans un processus Python isolé.

La disponibilité distante actuelle n'a pas été interrogée. Le certificat de publication/montage antérieur existe, mais ne remplace pas le contrôle au moment d'une future session : attacher les **11 sources existantes** listées dans [dataset-sources.json](dataset-sources.json). Le notebook vérifie à nouveau toutes les identités et tous les octets, puis crée des liens vers les entrées en lecture seule ; il n'extrait ou ne reconstruit aucune archive de données.

## Budget et sauvegarde

La mesure existante est **887,069 secondes sur Tesla T4**, quatre époques/23 404 updates d'un U-Net apparenté à tête supplémentaire. Le scénario hérité de l'audit applique les facteurs choisis0,75–1,5 : **11,1–22,2 minutes pour un seul nouvel entraînement**. Ce n'est ni une mesure de Focal ni une borne garantie.

Le lanceur prévoit une pause après **2 700 secondes d'entraînement/session**, avec checkpoint ; ce plafond de45minutes n'est pas une estimation. Les vérifications SHA, l'installation éventuelle dans une future mission, l'évaluation finale32/AP exacte et la sauvegarde sont **en supplément, non chronométrées pour ce candidat**. La session Kaggle peut conserver une allocation GPU pendant les fusions AP sur CPU : ne pas annoncer11–22minutes comme budget total de session. Le budget complet doit être accepté avant lancement.

Prévoir du scratch pour l'AP exacte : 46,1M pixels-jours de développement, enregistrements de tri jusqu'à24octets chacun avant regroupement, accumulateurs global/événement et fusions temporaires. Une réserve de **6GiB libres hors corpus** est une proposition prudente à vérifier, pas une mesure du pic. Les contrôles disque existants refusent une évaluation incomplète faute d'espace. La durée et le pic mémoire d'une reprise GPU ne sont pas validés ici.

Le [lanceur isolé](../../src/geofirex/focal_experiment.py) sauvegarde des fichiers nouveaux, tous les100updates et à chaque époque : modèle, AdamW, epoch/cursor/update, historique de loss, RNG Python/NumPy/Torch/CUDA, état du générateur DataLoader, règle/position du sampler, configuration et identités du code/données. Chaque checkpoint possède un sidecar taille/SHA-256. Le checkpoint final est écrit **avant** l'évaluation, pour préserver l'entraînement si l'AP échoue.

La reprise exige le hash d'un checkpoint **de cette même expérience**, les identités config/code/initialisation, un curseur de batch cohérent et un **nouveau** dossier de sortie. Aucun checkpoint historique n'initialise le candidat. L'ordre du sampler et sa projection de reprise sont testés sur CPU ; aucune reprise CUDA réelle n'est revendiquée. Les checkpoints et le modèle principal ne sont jamais écrasés.

Le notebook prépare l'export du nouveau run vers ZIP + SHA-256. Après une session future autorisée, conserver les outputs Kaggle puis copier l'archive vers la destination externe approuvée et vérifier le hash après copie. Aucune copie externe ni destination autorisée n'est créée dans cette mission. Si l'évaluation échoue après epoch4, conserver ce checkpoint et reprendre l'évaluation de ce modèle ; ne pas réentraîner une nouvelle variante.

## Blocages avant autorisation GPU

1. **Environnement non conforme ici** : PyTorch `2.15.0+cpu`, NumPy `2.4.6`, contre `2.11.0+cu130` / `2.1.3` historiquement. Aucune installation automatique ou modification de l'environnement existant.
2. **Empreinte initiale non conforme ici** : seed42 produit `69f6c443d7b12c0f45132c328dde9b82e44e42ca857475d6231a9afe31dd8fe0`, contre l'attendu historique `a835383534d061787c2449bbb4a896c32c3e990d92b80fe2b3105f9c91ac6f5f`. Voir [sonde CPU](cpu_initialization_probe.json). Les sources modèle/loss correspondent au paquet de préparation historique ([contrôle](historical_source_check.json)), ce qui ne prouve pas la cause de la divergence ni l'environnement effectivement exécuté. **Ne pas remplacer le hash attendu par le hash local.** Retrouver l'environnement et l'initialisation appariés ; sinon l'exécution reste bloquée.
3. Vérifier les quatre ordres de sampler et l'ensemble des11 montages dans la future session, avec les mêmes hashes. Internet doit rester désactivé ; aucune nouvelle partition ni téléchargement n'est requis.
4. Autoriser le budget **complet** d'une seule expérience, préciser la destination/restauration des checkpoints et vérifier le scratch. Le coût de l'évaluation est encore non mesuré.
5. Obtenir l'autorisation explicite de l'utilisateur. Le lanceur exige ensuite un fichier d'autorisation distinct, lié à l'identité d'expérience, `runs=1`, hashes config/code, plafond2 700s et destination de sauvegarde. Aucun fichier autorisant un lancement n'est fourni ici.

Ces vérifications ne demandent pas un nouveau train de qualification : les contrôles de versions, d'initialisation et d'ordre sont possibles avant tout pas d'optimisation. Aucun abaissement automatique des gardes, changement de seed, choix d'alpha/gamma alternatif ou lancement de secours n'est prévu.

## Tests et préservation

**93 tests réussis**, zéro échec, en12,16secondes : [JUnit](tests-delivery.xml). Sept avertissements `PendingDeprecationWarning` viennent d'Affine/rasterio sur les fixtures synthétiques. La suite couvre la formule et gamma0/BCE, gradients par différences finies, logits extrêmes, float32/64/16/bfloat16, masques et NaN de padding, batch sans feu, dimensions et gradients d'un petit forward du même U-Net, déterminisme local, refus du hash initial différent, sampler exhaustif/reprise, contrats figés, limites de seuil, refus d'autorisation/environnement, paquet et notebook désactivé, gardes2021 et routes scientifiques existantes.

Les backwards sont limités aux tenseurs synthétiques ; **zéro `optimizer.step`** a été exécuté. Aucun modèle réel n'a été évalué sur la validation. La première vérification avait révélé la divergence d'initialisation ; le test final vérifie explicitement que le garde la refuse, sans modifier l'empreinte historique.

**144 fichiers protégés inchangés**, y compris la livraison complète de l'audit précédent, les sources originales, le frontend, les configurations et les artefacts sélectionnés : [préservation](preservation.json). La vérification du corpus est documentée séparément par les hashes complets des partitions. Aucun fichier existant n'a été modifié pendant cette mission. La ligne de fixture déjà modifiée par la mission précédente est conservée telle quelle.

Ajouts de cette mission :

- `src/geofirex/focal_loss.py` : loss isolée ;
- `src/geofirex/focal_experiment.py` : une seule expérience, chemin par défaut métadonnées, exécution sous garde ;
- `scripts/prepare_focal_experiment.py` : préparation locale, vérification des sources existantes, empaquetage ;
- `tests/test_focal_experiment.py` : tests CPU ;
- `reports/focal_preparation_v1/` : configuration, notebook désactivé, paquet, copies JSON de référence, preuves et rapport.

Le seul ajustement de source **dans le paquet** remplace l'import des helpers `save/sha` de `fast_compact` par leurs équivalents isolés déjà utilisés dans les anciens notebooks. La logique du DataLoader et son fichier de production ne changent pas. Tous les modules embarqués et cellules de code compilent ; leurs hashes correspondent au build et le notebook contient le bon payload.

### Vérification exécutée

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_focal_experiment.py tests/test_experiment_readiness.py tests/test_science_results.py tests/test_prediction_lab.py::test_catalogue_2021_and_invalid_dates_rejected_before_pixels tests/test_prediction_lab.py::test_integrity_failure_closes_lab_instead_of_repinning tests/test_benchmark_contracts.py::test_contexts_share_exact_target_dates_and_consecutive_inputs tests/test_benchmark_contracts.py::test_alignment_handles_gaps_without_short_context_advantage tests/test_benchmark_contracts.py::test_external_ap_matches_sklearn_across_chunks_ties_and_merge_levels tests/test_unet_data.py::test_normalization_fits_only_train_inputs_not_targets_or_heldout tests/test_unet_data.py::test_nan_inf_angles_and_categorical_unknowns_have_explicit_encoding -q -p no:cacheprovider --basetemp=artifacts/focal-preparation-v1/pytest-delivery --junitxml=reports/focal_preparation_v1/tests-delivery.xml
```

Pour une nouvelle vérification, choisir de nouveaux chemins temporaires/JUnit. La préparation se reproduit vers un dossier nouveau avec `scripts/prepare_focal_experiment.py --out reports/focal_preparation_review_next` ; elle refuse tout écrasement. Le simple précontrôle local, sans GPU ni pixels, est :

```powershell
.\.venv\Scripts\python.exe -m geofirex.focal_experiment --config reports/focal_preparation_v1/focal_config.json --corpus reports/mission_20261111/compact_full --index reports/gpu_acceleration_20261111_v1/index
```

Notebook livré : [GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb](GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb). `RUN_TRAINING=False`, `AUTHORIZATION_FILE=None`, aucun output de cellule, aucune soumission. Aucun commit, push, merge, reset ou clean. Le Prediction Studio reste sur son U-Net poids10, seuil0,78.
