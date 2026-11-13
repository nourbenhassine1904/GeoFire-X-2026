# Résultat ciblé et prochaine exécution

1. **Lenteur mesurée et correction.** T4, ancien chargeur : 17,871 s d'attente
   DataLoader, 0,0077 s de copie et 0,0254 s de calcul pour 64 tuiles chronométrées.
   L'attente représente 99,8 % du temps. La préparation ouvrait aussi les en-têtes
   de 2 054 fenêtres ; elle utilise désormais les dimensions certifiées et prend
   **4,800 s**, sans audit TIFF répété. Cache normalisé borné, ordre par blocs
   spatio-temporels, index mmap de petite taille, workers persistants et pin_memory.
   Profil final complet : **72,895 s**, sans époque complète.

2. **Débit T4 mesuré**, périmètre brut strict de 119 événements :

   | Configuration B1 | Tuiles/s |
   |---|---:|
   | Ancien lecteur, workers=0 | 3,575 |
   | Cache + localité, workers=0 | 11,346 |
   | Cache + localité, workers=2, pin/persistants | 19,803 |
   | Cache + localité, workers=4, pin/persistants | **31,476** |

   Gain **×8,81** sur ce diagnostic court : même ensemble de 96 tuiles,
   32 d'échauffement et 64 chronométrées par configuration, ordre modifié par
   l'optimisation. Pas une mesure d'époque entière. Pic parent 1 967 837 184
   octets ; pic GPU alloué 71 436 800 octets. Les caches bruts et normalisés
   sont bornés à 64 Mio chacun par worker ; pas de corpus tensoriel en RAM.
   B2 : 0,063 s de calcul pour 64 tuiles, mais son débit de 971 tuiles/s utilise
   le cache déjà chaud de B1 : **ne pas l'extrapoler à une époque**.
   Le diagnostic positif T4 lit 12 tuiles avec 556 pixels positifs, sans prétendre
   fournir des métriques représentatives. Le diagnostic compact local inclut
   positifs, négatifs difficiles et négatifs : 63 tuiles, 963 pixels positifs.

3. **120e événement expliqué sans doublon.** Lots 001/002 : 57+62 événements,
   intersection vide, **48 287 tuiles**. Le sample ajoute uniquement
   `2019/fire_23300839` : **380 tuiles**, d'où exactement **48 667**.
   Ses deux autres événements train sont déjà dans les lots et étaient
   dédupliqués. Nouveau profil : train limité aux listes 001/002, sample utilisé
   seulement pour les deux événements val. Dans le corpus officiel des 250,
   `2019/fire_23300839` appartient légitimement au train. Le normaliseur commun
   train-only des 250 est conservé ; ce profil 119 est un diagnostic technique.

4. **Publication privée prête**, indépendante de l'ancien orchestrateur.
   11 partitions : métadonnées puis archives d'événements, moins de 1 Go de
   payload par partition. Originaux inchangés ; staging par hardlinks.
   **9 847 724 207 octets**, index et cartes de chemins compris ; plafond de
   corps HTTP **12 000 000 000 octets**, journal durable, retries comptés.
   Aucune publication du corpus effectuée. Les objets `.bin` conservent les
   octets des archives ; après montage, liens vers les chemins d'origine,
   sans extraction. Simulation locale : 459 objets vérifiés SHA, ancien
   CompactTiles et lecteur optimisé identiques, 93 603 train / 78 771 val.
   Le contrôle distant des SHA reste obligatoire après publication.
   La commande et les options SDK sont contrôlées contre le client officiel
   Kaggle 2.2.5 installé dans un dossier isolé ;
   [référence officielle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

   **Après accord sur ces octets uniquement**, depuis la racine du projet :

   ```powershell
   .\.venv\Scripts\python.exe scripts\compact_publication.py publish --plan reports/gpu_acceleration_20261111_v1/publication/plan.json --execute --approved-plan-sha256 87ad4e23ea8b35280b6e589e120a8e40efdbe13f5e78108bdb6d1aa204d502f8 --max-http-body-bytes 12000000000
   ```

   Reprise : même commande et même journal. Avant tout upload, une partition
   distante privée avec identité correspondante est adoptée et ignorée.
   Le SDK conserve son contexte d'upload reprenable. Identité différente,
   état distant ambigu ou plafond épuisé : arrêt, aucune création de version,
   aucune reprise automatique aveugle. 001/002 ne sont jamais des destinations.

5. **Budget GPU provisoire**, calculé avec l'attente I/O brute T4 mesurée :
   B1 **49,6 min**, B2 **50,5 min** pour les 93 603 tuiles train des 250 événements,
   hors validation/AP/checkpoints. C'est un scénario conservant cette attente
   brute, **pas une mesure du compact Kaggle**, encore non publié.
   Proposer un plafond initial de **90 min par modèle pour une époque + 32
   événements val complets**, puis le réviser à partir du profil compact.
   Le passage aux 201 val sera mesuré séparément. Aucune campagne prolongée
   ni trois époques sur la version lente n'ont été lancées.

6. **Prochaine action exacte.** Autoriser les 11 partitions sous plafond 12 Go ;
   exécuter la commande ci-dessus ; attacher les 11 handles de
   `dataset-sources-final.json` à `B1_B2_COMPACT_FINAL.ipynb`. Sa première cellule
   vérifie les SHA et restaure les chemins. Puis un test compact court, avant
   autorisation des époques :

   ```python
   import subprocess, sys
   subprocess.run([sys.executable, str(workspace/'train_compact_benchmark.py'),
       '--corpus', str(corpus), '--index', str(index), '--out', '/kaggle/working/B1-compact-short',
       '--model', 'B1', '--device', 'cuda', '--workers', '4',
       '--max-updates', '25', '--seconds', '180', '--b0', str(workspace/'b0_existing.json')], check=True)
   ```

   Après budget GPU approuvé : activer une époque B1, puis B2 dans le notebook.
   T=3, seed42, mêmes tuiles et normaliseur ; 32 événements 2020 sélectionnés
   par métadonnées, toutes leurs tuiles, sans rééquilibrage. AP exacte, Dice,
   IoU, précision, rappel, PNG/NPZ et loss. B0 est agrégé depuis ses comptes
   existants, jamais relancé ; pixels et positifs doivent coïncider exactement.
   Checkpoints toutes les 100 mises à jour, avant validation et à interruption.
   Reprise au curseur entraîné, état optimiseur/RNG/loss inclus ; vérifiée sur
   CPU, poids identiques bit à bit à deux mises à jour sans interruption.

Preuves principales : `t4-results-v3/summary.json`, `gpu-budget.json`,
`scope-difference.json`, `publication-compatibility.json`, `resume-verification.json`.
Le profil v2, sans mises à jour chronométrées après une préparation trop lente,
est conservé et exclu des débits. Aucun B0, corpus reconstruit, test 2021,
modification de 001/002, commit, push ou merge.
