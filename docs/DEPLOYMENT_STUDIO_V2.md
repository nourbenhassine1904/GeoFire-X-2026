# Exploitation de Prediction Studio V2 — soutenance du 2 décembre 2026

Le lancement normal ne fait aucun entraînement, aucune évaluation de split,
aucun calcul Spark et aucune inférence 2021. FastAPI écoute seulement sur
127.0.0.1. Aucun commit, push ou merge n'est réalisé par les scripts.

## Lancement sur la machine préparée

Depuis la racine du dépôt :

```powershell
.\Start-GeoFireX.ps1
```

Ouvrir http://127.0.0.1:8026/#/overview. Le script compile le frontend puis
lance FastAPI. Si 8026 est occupé : `.\Start-GeoFireX.ps1 -Port 8028`.
Arrêter le serveur dans son terminal avec Ctrl+C. `-SkipBuild` est réservé à
un frontend déjà compilé et inchangé. Une première vérification de RESULTS.zip
lit environ 2,2 Go pour calculer son SHA-256 ; attendre le panneau de résultats.
Les chargements de modèles sont différés jusqu'à leur première utilisation.

## Installation reproductible

La machine validée utilise Python et les versions décrites dans
`reports/publication_20261127/environment.json`. Installer Python de même version
majeure/mineure et Node.js 22.12.0, puis :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-studio-cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
Push-Location frontend
npm.cmd ci
Pop-Location
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe scripts/qualify_studio_v2.py
.\Start-GeoFireX.ps1
```

Adapter `py -3.11` à la version indiquée dans `environment.json`.
`requirements-studio-cpu.txt` fige la fermeture des dépendances réellement
installées, avec l'index officiel des roues CPU PyTorch. `package-lock.json`
fige les dépendances npm. Une réinstallation complète sur machine vierge
n'est pas prétendue par les tests de cette livraison. Prévoir les roues et le
cache npm avant une démonstration hors ligne.

## Artefacts externes et ADATA

Restaurer les fichiers existants sans les réécrire ni reconstruire le corpus :

| Contenu | Chemin par défaut | Réglage facultatif |
| --- | --- | --- |
| C8 / NEWPOS2 / NEWPOS4 COMPLETE | `D:\GeoFire-X\GeoFire-X-V2` | `GEOFIREX_V2_ROOT` |
| Compact déjà normalisé | `reports/mission_20261111/compact_full` | `GEOFIREX_COMPACT_ROOT` |
| RESULTS du test terminé | `D:\GeoFire-X-Final-Evaluation-20261125-v1\results\final-2021-01-RESULTS.zip` | `GEOFIREX_RESULTS_ARCHIVE` |
| Index/normaliseur V2 | `reports/protocol_v2_20261118/index` | Identités fixes, ne pas modifier |
| Preuves comparatives locales | `reports/jury_20261202_v1` | Manifestes scellés |
| Ancien U-Net | chemins de `configs/prediction_lab_v1.json` | Contrat historique inchangé |

Les variables se définissent dans le terminal avant le lancement. Exemple :

```powershell
$env:GEOFIREX_V2_ROOT = 'E:\GeoFire-X\GeoFire-X-V2'
$env:GEOFIREX_RESULTS_ARCHIVE = Join-Path $PWD 'artifacts\final_test_v2_backups\GEOFIREX-C8-2021-20261125-01\final-2021-01-RESULTS.zip'
```

Seules des copies strictement identiques sont acceptées. Le SHA-256 officiel de
RESULTS est `dafc1972be8c77326d2ecde0840a3d8840468af62cc69e83f0b69fbc52742a3d`.
Le service vérifie l'archive entière, puis les membres JSON utilisés et les
identités du checkpoint, du protocole et du test. Il ne charge aucun tableau de
prédiction 2021 et ne possède aucune route de relance du test. Le verrou durable
`artifacts/final_test_v2_ledger` reste intact, même si ADATA est débranché.
Sans chemin explicite et si D: manque au démarrage, le panneau tente la copie C:
existante dans `artifacts/final_test_v2_backups/GEOFIREX-C8-2021-20261125-01`.

Si ADATA manque, le catalogue explique les options désactivées. Les API de calcul
renvoient 503 avec une indication de chemin/empreinte. Reconnecter le disque,
vérifier ses chemins, puis « Réessayer le catalogue ». Ne jamais remplacer une
inférence indisponible par des sorties enregistrées. Pour les comparaisons
archivées FIRE13, conserver également son archive COMPLETE d'évaluation.

## Contrat scientifique et parcours jury

- C8 V2 : référence évaluée indépendamment sur 156 événements / 75 540 tuiles
  2021. Le panneau affiche les résultats officiels en lecture seule.
- NEWPOS2 : expérimental ; NEWPOS4 : expérimental haute sensibilité. Leur
  exécution en démonstration ne constitue aucune promotion scientifique.
- U-Net historique : option distincte, ancien protocole 250 train / 201 val.
- FIRE13 : résultats de validation 2020 et sorties archivées seulement.

Studio V2 utilise 80 événements / 30 882 tuiles 2020. Le modèle est
`StackedUNet(8,3,base_channels=8,depth=3)`, 123 977 paramètres, float32 CPU,
checkpoint fixe époque 4 / 28 976 updates, seuil 0,78. Les canaux par jour sont
`slope_z, elevation_z, slope_missing, elevation_missing, aspect_sin, aspect_cos,
aspect_missing, fire_J`. Les trois jours sont J−2/J−1/J. Le compact est déjà
normalisé avec le normaliseur train-only 2018–2019 ; aucune renormalisation.
Les cinq masques passés définissent la validité de l'inférence ; le masque de
J+1 n'est lu que par la requête séparée de vérité, puis intersecté pour les erreurs.

Scène V2 proposée : `2020/fire_24332704`, observation `2020-09-09`, cible
`2020-09-10`. L'ancienne scène `2020/fire_24332880` est réservée au catalogue
historique et ne doit pas être forcée dans le panel V2.

1. Overview : illustration décorative `feu.png`, puis scores officiels 2021.
2. Studio : choisir le modèle, l'événement et J ; consulter les trois observations.
3. « Prédire J+1 » : vrai calcul CPU ; les erreurs et la vérité restent masquées.
4. Révéler J+1 ; comparer nouveaux pixels, feu global, TP/FP/FN, probabilités.
5. Agrandir, zoomer, exporter PNG et métadonnées JSON, puis JSON/CSV de l'aperçu.
6. Comparaison V2 : recalcul des trois modèles sur une même scène, révélation
   commune séparée ; plus bas, comparaisons archivées explicitement étiquetées.

Les exports raster sont des grilles de pixels sans CRS/affine, jamais des hectares.
Le `preview_stride` et le cadrage exporté décrivent exactement les pixels affichés.
Les probabilités sigmoid ne sont pas calibrées. Le faible rappel nouveaux feux
2021 (0,1040384448) doit être exposé clairement.

## Vérification et maintenance

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe scripts/qualify_studio_v2.py
.\.venv\Scripts\python.exe -m pytest tests/test_studio_v2_deployment.py tests/test_prediction_lab.py tests/test_studio_spatial.py tests/test_jury_finalization.py -q
$env:GEOFIREX_TEST_URL = 'http://127.0.0.1:8026'
Push-Location frontend
npm.cmd run test:studio
Pop-Location
```

La qualification crée un reçu horodaté puis actualise son pointeur courant,
sans toucher aux preuves scientifiques. Toute modification de l'adaptateur ou
de l'inférence historique invalide le reçu et désactive les modèles V2 jusqu'à
requalification sur 2020. Les tests React utilisent Happy DOM et de vraies API CPU ;
ils ne remplacent pas les vérifications visuelles dans un navigateur.

API utiles : `/health`, `/science/v2/deployment-health`, `/studio/models`,
`/studio/v2/C8/status`, `/studio/v2/C8/events`, `/science/final-2021`,
`/science/final-2021/export.json`, `/science/final-2021/export.csv`.
Les calculs V2 sont sérialisés : un appel concurrent reçoit 409 et peut être
réessayé. Les réponses 403 excluent les événements hors 2020 avant chargement.

La parité C8 porte sur la tuile diagnostique scellée disponible (4 096 pixels),
pas sur une nouvelle évaluation des 30 882 tuiles. L'archive C8 ne contient pas
de probabilités complètes. NEWPOS2/NEWPOS4 ajoutent trois scènes complètes
sélectionnées sans consulter leurs performances. Les reçus donnent les écarts,
la version Torch et les identités. Aucune performance agrégée n'est recalculée.

Conserver les archives C: et D:, leurs reçus, checkpoints, normaliseurs et
manifestes. Ne jamais effacer un verrou pour résoudre un problème de lancement.
Le dossier `report/`, les slides et les preuves historiques ne sont pas des
fichiers d'exploitation à régénérer.

## Restauration d'un clone et conditions d'exploitation

Le clone ne suffit pas. Restaurer depuis les sauvegardes protegees, octet pour
octet : les compacts et leurs index de tuiles `.npy`, les checkpoints historiques,
les archives COMPLETE C8/NEWPOS2/NEWPOS4 et FIRE13, les preuves de diagnostic
scellees, les metadonnees FIRE13 (geometrie, normaliseur derive et manifeste),
RESULTS et le verrou durable `artifacts/final_test_v2_ledger`. Le manifeste
`reports/jury_20261202_v1/science_manifest.json`, les configurations et le guide
de finalisation identifient les empreintes attendues. Ne jamais reconstruire
un artefact manquant par entrainement ou par un nouveau test 2021.

Les petits JSON requis sont publies avec cette release ; les index binaires,
checkpoints, compacts et RESULTS restent externes. Requalifier sur les seules
scenes 2020 autorisees apres restauration. Le recu courant et ses copies
horodatees restent locaux dans `artifacts/studio-v2-deployment` ; un clone sans
recu valide affiche des modeles desactives. FIRE13 reste archive tant que sa
parite initiale n'est pas satisfaite, meme si les classes a 0,78 sont identiques.

Surveillance : verifier `/health` pour le processus, `/studio/models` pour la
disponibilite effective des modeles et `/science/final-2021` pour les resultats.
`/health` et `deployment-health.ready` seuls ne prouvent pas que les disques et
poids sont accessibles. Conserver stdout/stderr dans un journal local ; surveiller
les 503 (artefacts), 409 (calcul occupe), latences et espace disque disponible.
Apres reconnexion, reessayer le catalogue ; redemarrer l'instance dediee si un
lecteur deja charge conserve un disque deconnecte. Ne pas effacer de verrou.

Ce service local n'a ni authentification multiutilisateur ni qualification
operationnelle. Ne pas l'exposer sur Internet ; garder le bind `127.0.0.1`.
Avant toute mise a jour : sauvegarde C:/ADATA verifiee par SHA-256, tests cibles,
qualification, build puis recette navigateur. Garder les anciens recus et
archives. Aucune modification de seuil, normaliseur ou resultat officiel n'est
une operation de maintenance autorisee. L'essai sur machine vierge et le plein
ecran natif dans Chrome interactif restent a effectuer avant une certification
de deploiement complete. La publication Git de code ne vaut pas cette certification.
