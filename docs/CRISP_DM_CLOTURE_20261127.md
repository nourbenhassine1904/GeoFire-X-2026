# Cloture CRISP-DM - 27 novembre 2026

Perimetre : prototype de recherche GeoFire-X, protocole V2 fige et demonstration
locale. PASS signifie que les criteres du perimetre experimental sont prouves,
pas une aptitude operationnelle. Aucun entrainement, nouvelle selection de
seuil ou nouvelle evaluation 2021 n'est necessaire ou autorise par cette cloture.

## Matrice de tracabilite

| Phase | Objectif et travail realise | Composants | Entrees et sorties | Experiences executees et resultats verifies | Preuves reproductibles | Limites, acceptation et statut |
| --- | --- | --- | --- | --- | --- | --- |
| Business Understanding | Prevoir la detection active J+1, distinguer feu global et nouveaux pixels ; fixer la cible et le compromis sensibilite/fausses alertes | `docs/PROJECT_CONTEXT.md`, `configs/protocol_v2/baseline_c8.json`, `frontend/src/Methodology.mjs` | Contrat scientifique et besoins de demonstration vers cible, metriques et interface | Les resultats C8 et variantes sont confrontes aux criteres preregistres, pas seulement au rappel | Contexte versionne, comparaison scellee `reports/jury_20261202_v1/comparison.json`, tests jury et frontend | Acceptation : objectif, exclusions et criteres explicites. Pas d'alerte, ignition ou surface brulee revendiquee. **PASS** |
| Data Understanding | Inventorier 607 evenements, 23 bandes, dates, qualite et manque ; distinguer corpus, observations, fenetres et tuiles | `reports/DATA_AUDIT.md`, `scripts/mission_spark_demo.py`, `src/geofirex/science_results.py` | Inventaires de metadonnees et qualite certifiee vers comptes annuels et tables Parquet | Spark 4.0.2 local[2] execute, doublons vides, violations annuelles 0 ; pas de nouvelle lecture des pixels 2021 | `reports/mission_20261111/spark/report.json`, ses cinq Parquet, test Big Data comparant leurs lignes au rapport | Acceptation : provenance et domaine des comptes explicites. Spark historique 250/201/156, pas le split V2. Pas de cluster. **PASS** |
| Data Preparation | Contexte causal T3, masques, split chronologique avec purge, normalisation historique train-only figee | `src/geofirex/prediction_lab.py`, `src/geofirex/protocol_v2.py`, `reports/protocol_v2_execution/split_lock.json` | Compacts normalises et routes d'index vers 318 train / 53 purges / 80 val ; test 156 reserve puis consomme une fois | Index V2 : 115 900 tuiles train / 30 882 val. Norme historique apprise seulement sur train 2018-2019 ; pas de refit sur V2 val/test | SHA index `e61448e6791dacafe95b67bf4d12b90fb6a8575b14ba9ba19ad5df5e34160655`, normaliseur `bf8b5adb6276126540786351e66076d35c2e82be09b21e14bea96a4c678b2f27`, tests causalite/masques | Acceptation : identites et absence de cible dans inference. Pas de CRS/affine exploitable dans le compact. Restauration externe des binaires requise. **PASS** |
| Modeling | Entrainer puis comparer C8, NEWPOS2, NEWPOS4 et FIRE13 avec budget et protocole explicites ; conserver les modeles historiques distincts | `src/geofirex/protocol_v2.py`, `src/geofirex/fire13_features.py`, configurations `c8_*`, adaptateur `studio_v2.py` | Checkpoints COMPLETE, configurations et normaliseurs vers predictions CPU 2020 et tableaux archives | Quatre U-Net V2 entraines, epoque 4 / 28 976 updates ; C8/NEWPOS 123 977 parametres, FIRE13 125 057. Aucun nouvel entrainement lors de cette cloture | Archives ADATA et checkpoint SHA dans `comparison.json`, inventaire de livraison, requalification CPU sur references 2020 | Acceptation : identites strictes, statut experimental non masque. FIRE13 archive : deux scenes hors tolerance 1e-6 ; pas d'assouplissement. Aucun nouveau score ConvLSTM V2. **PASS** |
| Evaluation | Evaluer les experiences sur le panel 2020 commun ; consommer une seule fois le test 2021 du C8 selectionne | `reports/jury_20261202_v1/*evaluation.json`, `src/geofirex/studio_final_results.py` | Resultats scelles 2020 et RESULTS 2021 vers metriques, provenance et exports en lecture seule | C8 2021 COMPLETE : 156 evenements / 75 540 tuiles, seuil 0,78 ; metriques ci-dessous. Variantes non promues, non testees sur 2021 | Archive RESULTS SHA officiel, membres JSON verifies, verrou durable preserve ; `/science/performance`, tests d'exports et de refus 2021 | Acceptation : identites, effectifs, COMPLETE, aucune relance ; domaines separes. Faible rappel nouveau, un seed et pas de calibration/validation operationnelle. **PASS** |
| Deployment | Construire React, servir FastAPI local, inferrer sur CPU 2020, gerer absence ADATA et exporter ; sauvegarder et surveiller | `Start-GeoFireX.ps1`, `scripts/serve_jury.py`, `requirements-studio-cpu.txt`, lock npm, guide Deployment | Sources + artefacts externes exacts + qualification vers application et fichiers exportes | Build, API et CPU, tests backend/frontend, Chrome desktop/mobile ; erreurs 403/409/503 testees ; FIRE13 desactive | Recus legers dans `reports/publication_20261127`, recettes precedentes referencees, guide de restauration/exploitation | Acceptation locale satisfaite. Installation complete sur machine vierge non essayee ; plein ecran Chrome interactif non confirme. Ne pas declarer la certification complete. **PARTIAL** |

## Evaluation officielle C8

Valeurs lues depuis les archives verifiees, arrondies ici a dix decimales :

| Indicateur | Validation 2020 | Test independant 2021 |
| --- | ---: | ---: |
| Evenements | 80 | 156 |
| Tuiles | 30 882 | 75 540 |
| AP globale | 0,3352339037 | 0,4550013955 |
| Dice global | 0,4389254872 | 0,5348317337 |
| Rappel global | 0,4358386785 | 0,5512253057 |
| AP nouveaux feux | 0,1101283730 | 0,1754763634 |
| Dice nouveaux feux | 0,1006963425 | 0,1576373262 |
| Rappel nouveaux feux | 0,0613835115 | 0,1040384448 |

Le checkpoint C8 est celui d'epoque 4, seuil 0,78 :
`c2273fea252d709f4d316eba76515d669be97eac8bfa9302ff4831da2c9aedc1`.
RESULTS officiel :
`dafc1972be8c77326d2ecde0840a3d8840468af62cc69e83f0b69fbc52742a3d`.
Le lecteur verifie l'archive entiere et ouvre seulement les membres JSON
autorises, jamais les tableaux de predictions ou des TIFF 2021.

Les populations 2020/2021 ne sont pas appariees. Une AP plus elevee en 2021
n'est pas une nouvelle amelioration du modele ; prevalences et domaines
different. Le rappel nouveaux feux de 0,1040 interdit une presentation comme
systeme fiable d'alerte. Les masques et metriques restent ceux du protocole fige.

## Axe Big Data effectivement execute

Le rapport Spark historique contient 13 607 observations de metadonnees :
5 198 train (250 evenements), 4 292 val (201), 4 117 test (156).
Les fenetres train/val T1/T3/T5 comptent chacune 3 948 / 3 287 exemples.
Le calcul qualite utilise les comptes certifies train/val, pas les pixels test.

Ingestion : `observations.parquet` et `windows.parquet` deja prepares, avec
SHA source dans le rapport. Transformations : groupements annuels, volumes de
fenetres, sommes de qualite, controles de doublons/fuite annuelle et jointure
evenement/date cible. Spark SQL a execute ces requetes en local[2] ; `collect`
puis PyArrow ont materialise `annual`, `windows`, `quality`, `annual_leakage`,
`window_join` en cinq petits Parquet. Le temps historique total est 65,110896 s,
dont 43,324014 s de demarrage JVM ; ce n'est pas un benchmark de cluster.

Les comptes de fenetres expliquent la preparation temporelle pour le Deep
Learning historique. Le compact, son index et les protocoles V2 sont une etape
distincte ; aucun calcul distribue des rasters V2 ou entrainement Spark n'est
atteste par ces fichiers. La presente mission ne relance pas Spark.

## Verification et reserves de cloture

La recette de publication reexecute build, backend, frontend/HTTP, controles
de parite 2020, liens et preservation SHA. Les preuves et leur perimetre sont
dans [le bilan de publication](PUBLICATION_20261127.md). Les recettes precedentes
restent datees ; elles ne valent pas un essai neuf sur une autre machine.

Avant fusion : reviewer la PR, verifier le plein ecran dans Chrome interactif,
restaurer et essayer une machine vierge si une reproductibilite complete est
requise, confirmer l'acces aux sauvegardes ADATA. Ne pas lever la reserve
Deployment tant que les essais manquants ne sont pas prouves.
