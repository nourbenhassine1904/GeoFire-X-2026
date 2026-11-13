# GeoFire-X — finalisation locale du 7 novembre 2026, tentative v2

**Résultat : arrêt contrôlé pour mémoire insuffisante. La récupération est préservée, mais la finalisation et le paquet Kaggle ne sont pas validés intégralement. Aucune relance automatique n'a suivi l'arrêt.**

## Autorisation et mesure préalable

La RAM a été mesurée avant toute reprise : **3 512 750 080 octets disponibles, soit environ 3,27 Gio**, pour un seuil demandé de 3 221 225 472 octets. Le dossier de sortie v2 n'existait pas.

La commande autorisée a été exécutée une fois :

```powershell
.\.venv\Scripts\python.exe scripts/finalize_gpu_recovery.py --out reports/recovery_20261107_v2
```

Les étapes ont été séquentielles, avec deux threads CPU imposés, journalisation dans un dossier neuf et supervision de l'arbre de processus. Le précontrôle demandait 3 Gio disponibles avant chaque étape ; pendant l'exécution, le garde-fou arrêtait l'arbre sous 1 Gio disponible ou 1 Gio libre sur le disque. Les délais étaient de 60 s pour les tests légers, 300 s pour la suite ciblée et 120 s pour l'audit ultérieur. Aucune étape Spark complet, audit des pixels du corpus, acquisition ou exécution GPU n'était prévue.

## Tests réellement réussis

Les **quatre tests légers ont tous réussi** en 3,206 s ; le superviseur a terminé en 3,359 s, code de sortie 0 :

1. `test_deadline_owns_venv_descendants` : le délai arrête le parent et son descendant ; contrôle Windows du descendant après arrêt.
2. `test_freezer_preserves_inputs_and_refuses_old_configs` : le gel refuse les anciennes configurations, conserve les entrées, produit un ZIP vérifiable et refuse l'écrasement d'une sortie existante.
3. `test_resource_error_stops_child` : un échec du contrôle de ressources arrête le processus créé.
4. `test_stored_zip_prediction_matches_streamed_archive` : l'estimation ZIP correspond aux archives synthétiques produites, y compris avec des seuils ZIP64 abaissés.

Preuves complètes : `tests-lightweight/output.log` et `tests-lightweight/supervisor.json`.

La suite suivante a ensuite démarré :

```text
python -m pytest tests/test_gpu_operational.py tests/test_gpu_resources.py tests/test_runner.py tests/test_benchmark_contracts.py -q
```

Le lancement utilisait aussi un nouveau répertoire temporaire, le cache pytest désactivé, un XML de résultat prévu et un diagnostic de blocage après 60 s.

**28 marqueurs pytest de réussite ont été enregistrés avant l'arrêt**. La sortie est exactement une séquence de 28 points ; aucun `F`, `E` ou traceback n'y figure. Il n'y a cependant ni résumé final pytest ni fichier `tests.xml`, puisque le processus a été interrompu. Il ne faut donc pas annoncer une suite complète réussie ni présenter ces marqueurs comme un rapport nominatif JUnit.

Les premiers tests exécutés concernent le mode sans TIFF 2021, les rejets de chemins et d'entrées test, le normalisateur portable train-only, la répartition exclusive des lots, les fenêtres de profilage, l'AP externe et les limites de ressources. La suite n'est pas allée à son terme ; aucune validation globale de tous les contrats n'est revendiquée.

Preuves : `tests-reference/output.log` et `tests-reference/supervisor.json`. Les fixtures, checkpoints synthétiques et sorties déjà produits dans `pytest-tmp/` sont conservés.

## Arrêt et diagnostic

La suite ciblée a été arrêtée après **163,172 s**, avant sa limite de 300 s, avec le statut `LOW_AVAILABLE_MEMORY` et un code de sortie 1.

| Mesure | RAM disponible | Utilisation |
|---|---:|---:|
| Avant la suite ciblée | 3 478 708 224 octets | 79 % |
| Après l'arrêt, dans le reçu du superviseur | 1 045 954 560 octets | 93 % |
| Relevé ultérieur hors suite | 86 605 824 octets, environ 83 Mio | 99 % |

La valeur maximale enregistrée par le superviseur pendant cette étape est de 93 %. La mesure de 99 % a été obtenue ensuite : elle ne permet pas d'attribuer la totalité de la pression mémoire au processus pytest.

Le finaliseur a produit `status.json` : `STOPPED_WITHOUT_RETRY`, étape `tests-reference`. L'arbre du test a été arrêté par le mécanisme Job Object Windows, lui-même vérifié par le test léger réussi. Les PID de test **2748 et 30792 n'étaient plus actifs** au contrôle après arrêt. Les processus Python encore observés occupaient environ 52 Mo et 4 Mo et n'étaient pas ces PID. Aucun processus Java n'est apparu dans le relevé effectué.

Un relevé des principaux ensembles de travail a été sauvegardé dans `process-memory-after-stop.json` : il montre notamment VS Code et plusieurs processus Chrome. Ce relevé partiel ne mesure pas toute la mémoire système et n'explique pas, à lui seul, la baisse à 83 Mio disponibles. Aucune application utilisateur n'a été fermée et aucune cause précise supplémentaire n'est affirmée.

Le disque C: conservait environ 119 Go libres à l'arrêt. L'interruption a été déclenchée par la RAM, pas par le délai ou la réserve disque.

## Corrections validées et limites

Les corrections de la reprise v1 disposent maintenant de preuves dynamiques pour :

- La possession et l'arrêt de descendants sous Windows, y compris le lanceur de l'environnement virtuel.
- L'arrêt sur erreur du contrôle de ressources.
- Le gel sans réécriture des entrées, avec refus des configurations historiques incomplètes et des sorties déjà existantes.
- Le calcul de taille ZIP sur des fixtures réelles, avec branche ZIP64.

Les tests GPU opérationnels ont commencé et leurs marqueurs de réussite sont présents dans la progression. La suite interrompue ne remplace pas une validation complète. Aucun changement de code supplémentaire n'a été effectué pendant cette tentative v2. Les corrections de la v1 n'ont pas été affaiblies pour contourner un test ou la réserve mémoire.

## État du paquet Kaggle

**Aucun nouveau paquet final n'a été produit.** Le pipeline s'est arrêté avant l'audit des métadonnées/tailles, avant la copie du déploiement, avant le gel du ZIP de code et avant la génération du notebook CPU.

Les trois préparations historiques et leurs huit plans de lots restent conservés dans `reports/gpu_operational_v1/`. Les preuves de la v1 restent disponibles : normalisateur lié aux 250 événements et 4 948 fichiers train, 3 948 fenêtres train, 3 287 fenêtres validation, huit plans couvrant 451 événements et 9 490 TIFF. Ces vérifications sauvegardées ne sont pas présentées comme un nouvel audit v2.

Les quotas Kaggle personnels n'ont pas été interrogés : **aucun appel réseau Kaggle** n'a été lancé. Ils restent non vérifiés actuellement. Aucun staging des 35 Go, upload, téléchargement, profilage GPU, entraînement GPU, commit, push ou merge n'a eu lieu.

Les sorties durables de cette tentative sont enregistrées dans `reports/recovery_20261107_v2/` : journaux, fixtures pytest, sorties de tests et rapport. Les tests légers utilisent aussi leurs propres petites fixtures dans le répertoire temporaire du système. Aucun ancien rapport, checkpoint scientifique, GeoTIFF, archive ou temporaire historique n'a été supprimé ou remplacé.

## Suite nécessaire

Les 3 Gio disponibles au départ n'ont pas empêché une nouvelle pression mémoire. **Aucun nouveau test n'est lancé dans l'état mesuré à 99 %.** La prochaine reprise doit d'abord expliquer ou stabiliser cette consommation mémoire. Il sera préférable d'isoler les groupes de tests dans des processus successifs et de journaliser les noms des cas, en conservant les mêmes délais et garde-fous. Cette adaptation n'a pas été exécutée ici.

La mission locale reste incomplète : suite ciblée à terminer, audit léger Arrow/tailles à exécuter, puis seulement gel et vérification du paquet. Il serait incorrect d'annoncer le paquet prêt pour Kaggle à ce stade.
