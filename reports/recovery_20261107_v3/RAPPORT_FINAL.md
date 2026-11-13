# GeoFire-X — diagnostic mémoire et finalisation locale

**Préparation locale finalisée sur 32 tests indispensables isolés. Paquet code/métadonnées figé et vérifié ; aucun transfert, job GPU, entraînement scientifique, commit ou push. La cause exacte de la saturation historique reste inconnue.**

## Diagnostic établi

- La dernière commande lourde de la v2 était la commande pytest des quatre fichiers, PID 30792, arrêtée sous 1 Gio après 163,172 s.
- Les 28 tests terminés ont été reconstruits depuis les sources réellement sauvegardées par cette exécution. Le test suivant était **`tests/test_runner.py::test_resume_across_changed_readonly_mount_paths`**, ce que corrobore son dernier dossier de fixture incomplet. L'instruction exacte active n'est pas connue : aucun échantillon de pile ni compteur mémoire de processus n'avait été sauvegardé.
- Ce test a réussi seul en 44,312 s, puis en 43,453 s dans la validation finale. Ses données sont de minuscules TIFF synthétiques de 9×13 ou 11×15 pixels, pas les 607 événements de D:.
- Les DataLoaders concernés ont `workers=0`, sans mémoire épinglée sur CPU. Les lectures utilisent des fenêtres et des caches LRU bornés ; les fichiers GDAL sont ouverts dans des contextes fermés après usage. Aucune fuite ou allocation massive de rasters n'a été reproduite. Aucun correctif du modèle, des données scientifiques ou du normalisateur n'a donc été inventé.
- Les exécutables observés dans les arbres contrôlés étaient `python.exe`, `git.exe` et `cmd.exe` ; les deux Python du lanceur virtuel ne sont pas deux workers DataLoader. Aucun Spark n'a été lancé.

L'ancienne mesure de RAM système à 99 % ne suffit pas à incriminer pytest : la v2 ne mesurait ni son RSS, ni sa mémoire privée, ni les autres consommateurs. La consommation de tout Windows et celle d'un processus ne sont pas des compteurs interchangeables. La cause historique ne peut pas être rétablie rétroactivement avec ces seuls journaux.

Preuve nominative des 28 résultats et de la commande : `history-reconstruction.json`.

## Mesures et protection validées

La première mesure de cette mission indiquait **7 145 238 528 octets disponibles, environ 6,65 Gio**. Chaque nouveau test a eu un processus indépendant et un nouveau dossier.

| Contrôle pendant la validation finale | Résultat |
|---|---:|
| Tests indispensables, XML individuel sans échec ni skip | **32/32** |
| Somme des durées des processus de test | 229,735 s |
| Pic RSS cumulé de l'arbre observé | **494 940 160 octets, environ 472 Mio** |
| Pic de mémoire privée engagée échantillonnée | **565 886 976 octets, environ 540 Mio** |
| Minimum de RAM système disponible | **5 097 304 064 octets, environ 4,75 Gio** |
| RAM disponible au dernier test terminé | **5 871 255 552 octets, environ 5,47 Gio** |

Le garde-fou exige 3 Gio au départ, garde une réserve de **2 Gio** pendant l'exécution, borne chaque cas à **120 s**, limite le RSS de l'arbre à 2 Gio et configure un plafond Windows Job de 2 Gio de mémoire engagée. Il échantillonne toutes les 250 ms les PID possédés, noms, handles, RSS et mémoire privée, ainsi que la RAM et la mémoire engagée du système. Un dépassement ou une impossibilité persistante de mesurer arrête l'arbre. Aucun test n'est automatiquement relancé.

Le refus natif d'une allocation excessive, l'arrêt sur limite RSS, le refus avant démarrage et l'arrêt des descendants ont été testés sur de petites fixtures. Le compteur de pic Job peut refléter une tentative refusée : il n'est pas utilisé comme substitut du RSS ni comme preuve que cette quantité a été allouée physiquement.

## Corrections et erreurs conservées

Fichiers corrigés/créés :

- `src/geofirex/job_control.py` : plafond de mémoire engagée pour l'arbre, compteurs natifs, handles possédés et libérés ; gestion bornée des transitions de démarrage/fermeture.
- `scripts/run_memory_guarded.py` : nouveau superviseur instrumenté, réserve renforcée, sorties exclusives.
- `scripts/validate_gpu_isolated.py` : un seul cas pytest par processus, liste explicite de 32 contrats, empreintes des sources, arrêt au premier problème.
- `scripts/run_bounded_local.py` : ajout des compteurs système de mémoire engagée.
- `scripts/audit_gpu_recovery.py` : mode `--metadata-only`, sans ouverture ni `stat` des TIFF.
- `scripts/finalize_gpu_recovery.py` : suppression du relancement de la suite globale, réutilisation uniquement de reçus isolés complets avec empreintes inchangées, vérification des XML et préparation du notebook CPU.
- `tests/test_memory_guard.py` : cinq nouveaux tests du dispositif mémoire.

Deux problèmes de la nouvelle instrumentation ont été identifiés puis corrigés avant la validation finale : une assertion incorrecte sur le compteur de pic Job et une course d'accès pendant la fermeture d'un processus. Le premier essai de correction de cette course a échoué au test ciblé ; l'attente bornée de l'état du handle et les reprises limitées ont ensuite réussi, y compris sur huit démarrages/fermetures rapides. Les refus persistants restent bloquants. Un premier essai de sonde en ligne de commande avait aussi échoué sur l'échappement PowerShell ; il a été remplacé par un fichier de test. Tous ces journaux et échecs restent conservés. Ils ne sont pas présentés comme la cause de la saturation de la v2.

La validation finale est dans `essential-final/summary.json`, avec un XML et une chronologie `memory.jsonl` par cas. Les essais antérieurs interrompus ne sont pas comptés dans les 32 réussites finales.

## Tests validés et restants

Les 32 cas couvrent : cinq gardes mémoire, quatre outils de reprise, cinq contrats de déploiement/normalisateur, six contrôles AP/ressources/budget/staging, deux reprises/verrous du runner, dix contrats temporels, métriques et sorties B0/B1/B2/B3. Les seules mises à jour d'optimiseur ont concerné les petites fixtures synthétiques des tests CPU.

Une **collecte sans exécution**, également bornée, recense 57 cas dans les quatre fichiers de référence. **23 de ces cas** figurent dans les 32 tests indispensables ; les neuf autres sont les tests d'outillage de reprise/mémoire. **34 cas de référence n'ont pas été réexécutés dans cette validation finale.** Leurs noms sont conservés dans `delivery-verification.json`. Certains avaient été terminés en v2 ; cela ne vaut pas validation actuelle de toute la suite.

Ne sont pas revendiqués : suite de référence complète, audit physique du corpus, nouveau replay des anciennes statistiques ou de l'intégration CPU réelle, profilage CUDA, entraînement scientifique et validation distante Kaggle. Les ressources actuelles ont permis tous les contrôles indispensables locaux ; aucun environnement cloud payant ou gratuit n'a été lancé pour contourner une limite.

## Paquet final

Fichier : `finalized/code-package/payload.zip` — **1 930 818 octets**, 64 fichiers de code/métadonnées plus le manifeste SHA256, **zéro TIFF et zéro checkpoint**.

SHA256 :

`7f942c6cccebe17096064f3a641f1b110a5048d48a73348bcc1cf6dc0aeeb9fe`

L'audit des petites tables et des huit plans a réussi sans accès aux TIFF : 451 événements train/validation, 9 490 observations, 3 948 fenêtres train et 3 287 validation pour chaque contexte, normalisateur train-only de 250 événements/4 948 fichiers. Il réutilise les certificats physiques historiques ; il ne prétend pas avoir revérifié aujourd'hui les 35 Go de pixels.

Une vérification indépendante a contrôlé le ZIP, chaque SHA256, l'absence de données/checkpoints, les quatre YAML portables sans D:, le normalisateur lié aux tables, la syntaxe du notebook, son accélérateur et Internet désactivés. Les **78 fichiers historiques et de configuration épinglés au départ ont gardé leurs SHA256**. Les checkpoints et petits TIFF créés par les tests sont dans les nouveaux dossiers de fixtures ; les données et résultats scientifiques historiques n'ont pas été modifiés.

Le notebook CPU est `finalized/kaggle-cpu-preflight/cpu-preflight.ipynb`. Il prépare les contrôles futurs des datasets déjà attachés et affiche les commandes GPU sans les exécuter. Il n'a pas été exécuté et nécessite les datasets privés avant utilisation. Les rasters ne sont pas archivés ou publiés ; les quotas Kaggle actuels ne sont pas vérifiés, et aucun SDK n'a été installé dans cette mission.

Le volume brut planifié reste 34 995 352 430 octets ; les ZIP de rasters ne sont toujours que prédits. Le petit paquet de code ajoute maintenant exactement 1 930 818 octets au budget éventuel. Le plafond de 36 Go ne couvre toujours pas une retransmission intégrale du plus petit lot. Aucun de ces transferts n'est autorisé ou effectué ici.

Preuve indépendante : `delivery-verification.json`. État explicite : `finalized/status.json`, avec `complete_reference_suite_validated: false`.

## Prochaine action

La préparation locale est prête pour revue. Conserver les reçus et utiliser désormais **l'exécution isolée**, pas la commande globale de la v2. Pour une nouvelle validation locale après modification du code, choisir un dossier neuf :

```powershell
.\.venv\Scripts\python.exe scripts/validate_gpu_isolated.py --out reports/validation_isolee_suivante
```

Il n'est pas nécessaire de relancer ces 32 cas tant que les sources testées restent inchangées. Avant toute étape distante : vérifier en lecture seule les quotas et dépendances Kaggle, puis obtenir l'autorisation nécessaire pour les transferts et les profils GPU. Aucun nouvel entraînement n'est proposé comme prochaine commande locale.

Les distinctions RSS/mémoire privée et le plafond de mémoire Job suivent les [compteurs Windows](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex) et les [limites des Job Objects](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information). La conclusion sur la cause inconnue repose sur les journaux locaux et les nouvelles mesures, pas sur une attribution supposée.
