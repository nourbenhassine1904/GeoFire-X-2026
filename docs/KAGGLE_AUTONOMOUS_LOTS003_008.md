# Lots Kaggle 003–008 : lancement autonome

Préparé le 9 novembre 2026. Aucun transfert n'a été lancé pendant la préparation.

Fermer VS Code et Codex, ouvrir **Windows PowerShell depuis le menu Démarrer**, puis exécuter cette commande depuis n'importe quel répertoire :

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\nourb\Projects\GeoFire-X-2026\scripts\Start-KaggleRemaining.ps1" -Execute
```

Le lanceur utilise directement l'environnement Kaggle déjà installé, sans activation, installation, tâche planifiée ou dépendance à VS Code. La console PowerShell reste ouverte pendant le déploiement. La fermeture de cette console ou l'arrêt de Windows interrompt l'exécution ; aucun redémarrage automatique n'est prévu. Éviter la mise en veille pendant les transferts.

Sans `-Execute`, la même commande effectue uniquement la revue locale, sans réseau ni création de tentative. `-Status` affiche la progression et le compteur HTTP, sans lancer de travail :

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\nourb\Projects\GeoFire-X-2026\scripts\Start-KaggleRemaining.ps1" -Status
```

## Décision mémoire

| Contrôle local | Valeur | Justification |
| --- | ---: | --- |
| Démarrage autonome et de chaque phase | 3 Gio (3 221 225 472 octets) | 1 Gio au-dessus de la réserve ; consommation mesurée très inférieure |
| Réserve physique minimale | 2 Gio (2 147 483 648 octets) | Conservée ; arrêt dès qu'un échantillon descend sous ce seuil |
| Plafond natif de mémoire engagée | 512 Mio | Job Windows sur tout l'arbre de descendants, y compris les superviseurs imbriqués |
| Plafond RSS surveillé | 512 Mio | Plafond de l'arbre entier, donc également de chaque processus qu'il contient |
| Surveillance | 250 ms | Réserve physique, RSS, espace disque et délais ; fermeture du Job à la fin |
| Durée maximale du superviseur autonome | 24 heures | Phases séquentielles bornées : ZIP 3 600 s, publication 7 200 s, certification 2 400 s |

Les mesures enregistrées du lot 002 donnent environ 24,5 Mio RSS pour le ZIP, 51,6 Mio pour la publication et 51,3 Mio pour le contrôleur CPU local. La publication de 4,995 Go a été envoyée une seule fois ; son état `FAILED` provenait de l'indexation du catalogue, pas d'un dépassement mémoire. Les pics d'engagement de ces phases étaient de 13,9, 37,9 et 40,9 Mio.

Le seuil historique de 4 Gio n'est donc pas une allocation nécessaire au transfert. Les lectures de fichiers sont en flux et la taille du ZIP ne fixe pas la mémoire vive nécessaire. La certification lourde s'exécute sur Kaggle ; les chiffres du contrôleur local ne mesurent pas la mémoire du notebook distant. Ses garde-fous restent inchangés.

2,5 Gio n'est pas retenu : il ne laisserait que 512 Mio au-dessus de la réserve, soit exactement le plafond natif autorisé, sans marge pour le superviseur et les variations des autres applications. Une ancienne préparation locale de 25 Mio RSS a vu la RAM système disponible passer de 3 325 124 608 à 2 123 952 128 octets en 118 secondes. Fermer VS Code réduit la pression, mais ne garantit pas la stabilité des autres processus. La surveillance dynamique reste donc indispensable et ne garantit pas une réserve absolue entre deux échantillons.

Le profil de 3 Gio concerne uniquement cette continuation autonome. Les deux anciens modes de reprise gardent leur seuil de 4 Gio. La RAM physique réelle est lue avant chaque phase, avant la première requête réseau du processus de publication/certification et à nouveau juste avant le PUT du ZIP. Si elle est insuffisante, l'opération s'arrête ; aucun nouveau transfert n'est envoyé. Aucun ajustement automatique à 2,5 Gio et aucune boucle d'attente ou de retry.

## Séquence et conservation

Le lanceur appelle `publish_remaining_lots.py --continue-after-indexing --autonomous` : il conserve `resume-01` et vérifie le certificat privé du lot 002 sans soumettre de notebook. Le lot 001 est vérifié à partir de ses preuves locales existantes. Aucun de ces lots n'est réuploadé ou recertifié à distance.

Le journal existant doit commencer cette continuation à **4 995 735 202 octets**. Il n'est ni réinitialisé ni remplacé. Toutes les nouvelles requêtes utilisent le budget persistant existant de **31 000 000 000 octets**, sans retry HTTP. La marge restante au départ est de **26 004 264 798 octets**.

Pour chaque lot 003, 004, 005, 006, 007, 008 :

1. Créer un seul ZIP en flux, avec vérification SHA-256 et CRC32 des sources.
2. Recalculer sa signature, vérifier la RAM, publier une seule fois en privé et confirmer sa présence privée.
3. Soumettre une seule certification CPU privée et vérifier ses reçus contre le manifeste gelé.
4. Fermer les processus et handles de la phase. Supprimer uniquement le `payload.zip` de ce nouveau lot certifié, en conservant son reçu de staging, son SHA-256, le manifeste, les reçus de publication et la certification. Consigner `archive-released.json`.
5. Passer au lot suivant après le contrôle mémoire suivant.

Un seul nouveau ZIP est donc actif à la fois ; les archives historiques des lots certifiés 001/002 sont préservées. Une publication incertaine, un échec de certification ou de suppression du ZIP arrête toute la chaîne. Le ZIP du lot en échec reste disponible pour inspection. Aucun remplacement ni suppression récursive n'est effectué.

Après les six certifications, une phase locale surveillée régénère les templates GPU B1/B2/B3 avec les huit certificats. Elle ne soumet aucun notebook et ne démarre aucun entraînement. Le normaliseur gelé et le verrouillage du test 2021 restent en place. Aucun commit, push ou merge.

## Journaux et arrêt

Les nouveaux journaux sont dans :

```text
reports/kaggle_remaining_20261108_v1/resume-01/autonomous-01/
  checkpoint.json             état atomique : lot, phase, certificats, budget
  events.jsonl                historique horodaté et vidé sur disque
  monitor/memory.jsonl         télémétrie dynamique du Job autonome
  monitor/output.log          sortie persistante de l'orchestrateur
  monitor/supervisor.json      bilan final : mémoire, code de sortie, cause d'arrêt
  result.json                 résultat global
```

Les preuves par lot restent dans `resume-01/execution/lot-003` à `lot-008`, avec leurs propres journaux `stage-monitor`, `publish-monitor`, `cpu-monitor` et leurs reçus. Le journal HTTP reste `reports/kaggle_remaining_20261108_v1/http-body.jsonl`.

Le répertoire `autonomous-01` est créé de manière exclusive au premier lancement réel et sert de verrou durable de tentative. Deux lancements concurrents ne peuvent pas démarrer deux chaînes. Une tentative existante, un staging existant, un verrou du journal HTTP ou un compteur différent provoque un arrêt. Le lanceur ne supprime aucun verrou et ne reprend jamais automatiquement une tentative interrompue, même après redémarrage de Windows. Inspecter les preuves avant toute reprise explicitement autorisée ; ne pas effacer les marqueurs pour forcer une répétition.

Un checkpoint `RUNNING` sans bilan final ne prouve pas que le processus tourne encore. La commande `-Status` expose le dernier échantillon et son âge ; après fermeture brutale ou panne, l'état doit être réconcilié avant toute action.

L'analyse chiffrée, l'empreinte de conservation des preuves et le résultat des tests ciblés se trouvent dans `reports/kaggle_autonomous_20261109_v1/`.
