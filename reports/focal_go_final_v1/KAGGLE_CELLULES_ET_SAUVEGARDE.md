# Exécution future — une seule expérience Focal

**Préparation uniquement. L'autorisation explicite n'est pas encore donnée.** Utiliser `reports/focal_preparation_v1/GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb`, pas le notebook de précontrôle. Ne pas soumettre de job ni utiliser un Save Version qui déclencherait un second run. Ne pas utiliser Run All avant l'autorisation.

## Réglages de la session

Importer le notebook existant finalisé. Garder Internet OFF ; aucune installation pip, aucun changement Torch/NumPy. Attacher les11 sources privées déjà publiées, dont les handles exacts restent dans `../focal_preparation_v1/dataset-sources.json` : `nnnnbhss/geofirex-compact-t3-20261111-p00` jusqu'à`p10`. Aucun dataset n'est à réimporter, reconstruire ou republier. Ne pas attacher d'archive2021 ni de checkpoint historique.

Choisir l'accélérateur T4 disponible. Si Kaggle propose T4x2, le notebook expose uniquement le GPU physique0. Démarrer un noyau neuf : le masque doit être posé avant tout import Torch/CUDA. Prévoir la réserve120min, avec12GiB libres sur working/tmp et du temps de téléchargement avant fin de session.

## Cellule1 — texte du protocole

Lire le contrat : B2/T3/C8, seed42, quatre époques, batch16, AdamW, Focal2/10/1 sans Dice, 23 404 updates. Aucun changement.

## Cellule2 — bootstrap CPU et masquage GPU

Exécuter cette cellule. Elle fixe `CUDA_VISIBLE_DEVICES="0"`, vérifie le SHA du paquet scientifique inchangé, extrait uniquement du code dans un dossier temporaire neuf et vérifie les payloads logistique/PASS embarqués. Elle ne lance ni CUDA, ni forward, ni fit, ni lecture de pixels.

Ne pas recopier le code d'un autre notebook dans le dossier temporaire. Conserver la variable `workspace`.

## Cellule3 — entrées de cette session et espace

Exécuter une seule fois dans la session retenue. Le code existant vérifie les11 montages en lecture seule et crée des liens ; aucun décompactage du corpus. Il vérifie les métadonnées/couverture/ordres du sampler, sans lire de pixels, puis l'identité code/config/PASS, les versions actuelles et les12GiB disponibles sur working/tmp. Il ne refait pas les sondes d'initialisation CPU/CUDA du précontrôle réussi.

Si un contrôle échoue, arrêter avant cellule5 et conserver le message. Ne pas installer une autre version, modifier un hash, publier un dataset ou choisir une seed de secours.

## Cellule4 — autorisation externe

Pour cette mission, ne rien activer. Le fichier `authorization_TEMPLATE_DISABLED.json` fourni ici contient `training_authorized=false` et des destinations vides ; il est volontairement refusé.

**Après l'accord explicite de l'utilisateur seulement**, créer `/kaggle/working/focal_authorization.json` en recopiant ce modèle. Conserver tous les hashes fournis et renseigner uniquement :

- `training_authorized: true` ;
- `authorized_by` : auteur et référence/date de l'accord explicite ;
- `local_backup_destination` : `C:\Users\nourb\Projects\GeoFire-X-2026\artifacts\focal-go-final-v1` ;
- `checkpoint_backup_destination` : `D:\GeoFire-X` après vérification du disque ;
- conserver `runs=1`, `gpu_count=1`, `max_training_seconds=2700`, `session_reserve_minutes=120`, le nom de l'expérience et les trois SHA exacts.

Ce fichier consigne l'accord ; le modifier n'est pas une manière de contourner l'absence d'accord. Il n'est pas créé automatiquement par le notebook.

## Cellule5 — unique entraînement puis évaluation

Avant l'autorisation : garder et exécuter si souhaité la cellule telle quelle, elle affiche `DISABLED` et ne fait rien.

Après l'autorisation seulement, modifier les deux premières affectations :

```python
RUN_TRAINING=True
AUTHORIZATION_FILE=Path('/kaggle/working/focal_authorization.json')
```

Exécuter **une seule fois**. Le wrapper vérifie l'accord, les identités, le disque et l'unique T4 visible ; crée un verrou exclusif ; appelle le runner scientifique inchangé. Ne pas utiliser torchrun/DataParallel/DDP. Ne pas sélectionner le second GPU.

Déroulé automatique :

1. Initialisation fraîche appariée seed42, jamais de poids historiques ; quatre époques,23 404 updates.
2. Checkpoints tous100updates et à chaque époque, SHA/tailles ; checkpoint final écrit et vérifié.
3. `TRAINING-SAFE-....zip` et SHA, relus avant toute AP. Télécharger cette archive dès qu'elle est visible dans les outputs, si possible pendant la suite de l'évaluation.
4. Évaluation exhaustive des32 événements, probabilités float32 et AP exacte, sélection32 selon la règle figée, comparaison à la baseline JSON historique.
5. Archive `COMPLETE-....zip` + SHA en fin de run, ou `PARTIAL` en cas d'erreur/pause.

Le retour normal attendu est `status=COMPLETE`. Une acceptation scientifique `false` est un résultat valide ; elle n'autorise ni nouveau candidat ni nouveau seuil. Un statut PAUSED/PARTIAL n'est pas une expérience terminée. La limite2700s concerne le fit, pas l'ensemble AP/sauvegarde. Si l'évaluation dépasse sa réserve indicative, conserver le checkpoint et décider de la suite ; aucun second fit.

## Cellule6 — lecture des consignes de conservation

Ne pas fermer/arrêter la session. Le checkpoint final doit précéder l'évaluation. Aucune action sur les fichiers Studio ou historiques.

## Cellule7 — liens de téléchargement et contrôle

Exécuter après cellule5, y compris si celle-ci a affiché une exception et que le noyau reste actif. Elle relit les SHA des archives et affiche les liens ZIP/SHA. Si le rendu FileLink n'est pas utilisable, télécharger les mêmes fichiers via le panneau de fichiers `/kaggle/working` ; les chemins sont aussi imprimés dans les logs. Télécharger TRAINING-SAFE et COMPLETE, chacune avec son `.sha256`. Conserver également les outputs Kaggle existants sans relancer les cellules de calcul.

Contrôler `training_complete.json`: epochs4, updates23404, quatre lignes d'historique et SHA final. Contrôler `selection32.json`, `paired_comparison.json`, `development32/evaluation.json` (COMPLETE,32 événements,13 759 tuiles), `complete.json` et `artifact_manifest.json`. Seul le développement32 intervient. Ne pas ouvrir2021.

## Échec AP après entraînement complet — évaluation seule

Ne jamais réexécuter cellule5. Dans le même noyau, avec les mêmes `workspace` et `corpus`, ajouter cette cellule **uniquement si l'évaluation précédente n'a pas terminé** :

```python
TRAINING_OUT=OUT  # dossier original, contient training_complete.json et checkpoint final
RECOVERY_OUT=Path('/kaggle/working')/('geofirex-focal-eval-recovery-'+__import__('uuid').uuid4().hex)
subprocess.run([sys.executable,str(workspace/'focal_run_logistics.py'),'evaluate-only',
    '--workspace',str(workspace),'--corpus',str(corpus),
    '--training-out',str(TRAINING_OUT),'--out',str(RECOVERY_OUT)],
    check=True,env=os.environ.copy())
```

Ce chemin vérifie le SHA et l'identité du checkpoint final, recharge ce même modèle et n'instancie aucun optimiseur. Nouveau dossier, mêmes32 événements, même AP/règle/critères. Télécharger `EVALUATION-RECOVERY` et son SHA. Si le disque manque, préserver et télécharger d'abord TRAINING-SAFE ; une session avec davantage d'espace pourra réévaluer ce même checkpoint après restauration de ses propres fichiers. Le script ne supprime pas de scratch automatiquement.

Si aucune archive n'a pu être créée mais que le dossier de run est intact, appeler seulement l'export :

```python
subprocess.run([sys.executable,str(workspace/'focal_run_logistics.py'),'package',
                '--out',str(OUT)],check=True,env=os.environ.copy())
```

Un arrêt brutal du noyau peut empêcher le `finally`. Le dossier et les ZIP déjà écrits restent les seuls artefacts à récupérer ; ne pas annoncer de sauvegarde distante tant qu'ils ne sont pas téléchargés/vérifiés.

## Après téléchargement — copie locale puis disque externe

Depuis la racine locale du projet, pour **chaque** ZIP téléchargé (TRAINING-SAFE puis COMPLETE, ou récupération), adapter les deux premiers chemins au nom réellement reçu :

```powershell
$focalArchive = 'C:\Users\nourb\Downloads\NOM_EXACT_DU_ZIP.zip'
$focalSidecar = 'C:\Users\nourb\Downloads\NOM_EXACT_DU_ZIP.sha256'
.\.venv\Scripts\python.exe scripts/focal_run_logistics.py backup `
  --archive $focalArchive --sidecar $focalSidecar `
  --local-root 'C:\Users\nourb\Projects\GeoFire-X-2026\artifacts\focal-go-final-v1' `
  --external-root 'D:\GeoFire-X'
```

Vérifier d'abord que `D:\GeoFire-X` correspond toujours au disque externe attendu. Le script ne crée que des dossiers nouveaux `focal-preserved-<uuid>`, ne remplace rien, vérifie le ZIP et chaque membre avant la copie puis sur chaque destination, et écrit un reçu `backup_verified.json`. Les destinations doivent exister. Si le disque est débranché, effectuer immédiatement la copie locale en omettant `--external-root`, puis recommencer la commande avec le disque raccordé ; ne pas supprimer le téléchargement original.

La conservation est terminée seulement lorsque les ZIP+SHA sont présents et validés localement **et** sur le disque externe. Aucun commit/push/merge/reset/clean n'est nécessaire.
