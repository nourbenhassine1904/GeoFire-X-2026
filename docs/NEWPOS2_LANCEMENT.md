# NEWPOS2 : un seul entraînement neuf

22 novembre 2026. Identité `B2_C8_V2_NEWPOS2_seed42`. Préparation additive sur `codex/newpos2-20261122`, sans commit ni push. Les seuils NEWPOS4 sont clos, sans nouvelle recherche. Validation exploratoire déjà observée ; aucun résultat NEWPOS2 encore mesuré. Le test 2021 reste fermé.

Le runner `src/geofirex/newpos2_experiment.py` adapte les fonctions NEWPOS4 existantes et réutilise directement le décodeur, les loaders, le sampler, le modèle, la loss générique, les métriques exactes et les sauvegardes C8. Seule différence scientifique : facteur BCE 2 des nouveaux positifs, soit poids effectifs 20/10/1, dénominateur nombre de pixels valides ; Dice inchangé. Le fichier prospectif original est conservé. La configuration exécutable séparée `configs/c8_newpos2_v1.json` garde `training_authorized=false` et `launch_enabled=false` ; seul un document d’autorisation externe peut ouvrir une action.

## Lancement court sur Kaggle

1. Importer `notebooks/GEOFIREX_C8_NEWPOS2_DISABLED.ipynb` dans un notebook privé neuf. Internet OFF. Attacher les onze datasets existants p00 à p10 ; liste exacte dans `reports/newpos2_20261122_v1/kernel_metadata_NOT_PUBLISHED.json`. Le code et tous les petits index gelés sont embarqués : aucun téléchargement, installation ou clonage nécessaire. Ne pas attacher de données 2021. Exécuter les cellules CPU avec les trois drapeaux à False. L’extraction se fait une seule fois dans un répertoire neuf ; ne pas rejouer cette cellule.
2. Après autorisation explicite de cet unique essai, sélectionner T4 et redémarrer le noyau si nécessaire avant toute importation torch. Copier les templates `authorization_gpu-preflight_DISABLED.json` et `authorization_newpos2-fit_DISABLED.json` du répertoire `ROOT/reports/newpos2_20261122_v1/` vers `/kaggle/working/newpos2_gpu_authorization.json` et `/kaggle/working/newpos2_fit_authorization.json`. Renseigner `authorized=true`, opérateur/date UTC, date du contrôle du quota, minutes restantes, heures-carte restantes/réservées, deux destinations distinctes de sauvegarde. Conserver identité, action, runs=1 et SHA du manifeste. Ces champs attestent une décision réelle, pas une autorisation à inventer.
3. Mettre uniquement `RUN_GPU_PREFLIGHT=True`, garder `RUN_FIT=False` et `RUN_RECOVERY=False`. Exiger PASS des onze montages, des ressources et de l’environnement historique : Tesla T4 visible unique, torch `2.11.0+cu130`, NumPy `2.1.3`, CUDA `13.0`, cuDNN `92700`. Aucun remplacement automatique des versions. En cas de refus : NO-GO, conserver le diagnostic.
4. Après PASS, mettre `RUN_FIT=True`, laisser `RUN_RECOVERY=False`, exécuter la cellule fit une seule fois. Quatre époques, 28 976 updates, pas de resume, arrêt précoce, choix de checkpoint ou seuil. Le verrou `B2_C8_V2_NEWPOS2_seed42_SINGLE_RUN.lock.json` est créé en mode exclusif avant le modèle ; même si le dossier de sortie est effacé, une seconde tentative dans la session est refusée. Ne jamais effacer ce verrou ni ouvrir une autre session de fit avec la même autorisation. Hors ligne, ce verrou ne peut pas coordonner des sessions Kaggle indépendantes : l’opérateur doit conserver l’autorisation consommée et les archives.
5. Télécharger immédiatement chaque ZIP et son `.sha256`, notamment TRAINING-SAFE puis COMPLETE ou PARTIAL. TRAINING-SAFE est créé et intégralement vérifié avant toute inférence. Copier sans écrasement dans deux dossiers distincts : archives locales du projet et `D:\GeoFire-X\GeoFire-X-V2`. Vérification disponible avec `scripts/run_newpos2.py backup --archive ... --sidecar ... --local-root ... --external-root ...` ; elle relit le SHA et tous les membres des deux copies. Ne pas considérer le téléchargement comme une double sauvegarde vérifiée.

Une interruption pendant le fit reste PARTIAL et n’autorise aucun second fit. Si seul le calcul d’évaluation échoue, le notebook contient une récupération depuis TRAINING-SAFE, avec autorisation `newpos2-evaluate` distincte et nouvelle sortie ; elle n’instancie aucun optimiseur. Ne pas activer simultanément fit et récupération.

## Budget vérifiable

Source : `reports/c8_propagation_20261121_v1/budget.json`, SHA conservé dans le nouveau budget. Les mesures historiques C8 sont 1 133,55 s de fit (18,89 min) et 1 213,55 s d’évaluation (20,23 min). Ce sont des références, pas une mesure NEWPOS2 ni une garantie de débit. Facteur2 fixé sans mini-entraînement de sélection.

| Poste | Réserve ou calcul |
|---|---:|
| Installation logique / précontrôles | 15 min |
| Fit neuf (limite du runner) | 60 min |
| Validation exhaustive et AP exactes (limite) | 90 min |
| Comparaison appariée | 5 min |
| Archives et transfert | 30 min |
| Marge | 25 min |
| Session réservée | **225 min = 3,75 h** |
| Quota conservateur, jusqu’à deux cartes réservées | **2 × 3,75 = 7,5 heures-carte** |

CUDA_VISIBLE_DEVICES=0 ne prouve pas la facturation d’une seule carte. Lire le quota actuel dans Kaggle avant le lancement et enregistrer la déduction réelle après la session. L’autorisation refuse un quota ou une durée insuffisants.

| Stockage / mémoire | Calcul conservateur |
|---|---:|
| Corpus monté en lecture seule, sans recopie | 459 objets ; 9 847 612 192 octets ≈ 9,17 Gio |
| Probabilités float32 + cible/validité/observé uint8 | 30 882 × 64² × 7 = 885 448 704 octets ≈ 0,825 Gio |
| Deux tris AP externes et marge de fusion | 30 882 × 64² × 24 × 4 = 12 143 296 512 octets ≈ 11,31 Gio |
| Enveloppe checkpoints avec AdamW, jusqu’à 33 écritures | 33 × 2 Mio = 66 Mio |
| Résultats + deux ZIP, avant transferts | environ 3 × (0,825 Gio + 66 Mio), hors petits manifestes |
| Minimum libre initial exigé | **18 Gio = 19 327 352 832 octets** |
| Plancher libre contrôlé pendant le run | 2 Gio |
| RAM hôte exigée | total ≥16 Gio, disponible ≥8 Gio |

Les fusions sont exclues des ZIP ; les probabilités et checkpoints y sont conservés. La taille compressée réelle et le coût des transferts seront consignés au lancement. La VRAM n’est pas mesurée dans cette mission ; le runner consigne pics alloués/réservés pendant le fit. Le coût monétaire dépend du compte et n’est pas estimé à partir d’un tarif supposé.

## Comparaison et décision préfixées

Évaluation unique sur les mêmes 80 événements et 30 882 tuiles à 0,78. Le SHA du domaine inclut chaque clé, les dimensions, les cibles, les masques valides et observed_J : une différence interdit la comparaison. La parité CPU des pixels est testée sur fixtures synthétiques ; l’exhaustivité réelle est attestée par les index et le décodeur inchangés, puis vérifiée par le SHA pendant l’évaluation. Aucune lecture exhaustive des pixels réels n’a été prétendue lors de la préparation.

Sorties : `evaluation/evaluation.json`, probabilités typées, PR exacte descriptive, `evaluation/paired_comparison/metrics.csv` (C8 officiel, NEWPOS4 historique à 0,78, NEWPOS2), différences par événement, bootstrap apparié C8 et NEWPOS4 (2 000 réplications seed20261121). Le tableau contient rappel, Dice, précision, AP nouveaux feux, FP, Dice global, AP globale, Brier, macro Dice, IoU et FPR nouveaux. L’AP désigne l’average precision exacte avec égalités de scores globales, pas une aire trapézoïdale ou approximée. Pas de bootstrap des pixels ni d’AP approchée. Le Brier décrit la calibration sans recalibrer les probabilités.

Les limites prospectives absolues se trouvent dans `acceptance.json` et `NEWPOS2_acceptance.json` est le verdict scientifique faisant foi : AP nouveaux ≥0,11512837301201887 ; AP globale ≥0,34023390370637985 ; Dice global ≥0,4339254871567759 ; rappel nouveaux ≥0,08138351147320427 ; Dice nouveaux ≥0,1106963425412671 ; FP nouveaux ≤29 134 ; macro Dice ≥0,22778327086319602 ; Brier ≤0,0031547185297208474. Domaine identique et budget autorisé restent obligatoires. `comparison.json` conserve aussi le rapport comparatif C8 historique. Même toutes les gardes passées, aucune promotion automatique ni ouverture du test 2021. Une seed ne mesure pas la variance d’initialisation.

## Reproduire les tests CPU

Créer d’abord un dossier de rapport neuf, puis lancer `.venv/Scripts/python.exe -m pytest tests/test_newpos2.py -q --basetemp=<rapport-neuf>/tmp --junitxml=<rapport-neuf>/tests.xml -o cache_dir=<rapport-neuf>/cache`. Les sauvegardes des tests portent explicitement `UNTRAINED_FIXTURE` : elles ne sont jamais des résultats d’entraînement.

Le notebook de livraison est gelé par son SHA et contient son propre payload signé par SHA. La régénération est réservée aux nouvelles révisions de préparation avant scellement ; elle ne modifie aucun fichier C8/NEWPOS4. Le dossier `draft_revisions` conserve les brouillons de cette mission.
