# Ablation contrôlée de la pondération positive

## Question et hypothèse préenregistrées

Les probabilités des modèles actuels surestiment le feu sur les 32 événements
de réglage. La loss mélange BCE pondérée et Dice par image. Une diminution du
poids positif peut améliorer le compromis FP/FN et le Brier ; le gain n'est pas
garanti et peut coûter du rappel, notamment sur les nouveaux pixels de feu.
Changer seulement `pos_weight` permet de tester cette hypothèse.

Poids : **100, 30, 10**, exactement deux valeurs plus faibles. Le ratio train
non-feu/feu mesuré vaut 765,757 ; sa racine ≈27,67 motive 30 comme réduction
modérée, 10 comme réduction plus forte. Ces valeurs sont fixées depuis le train
et le diagnostic des 32, sans les optimiser sur les 169. Le poids 100 reste un
contrôle réentraîné, pas seulement une référence historique.

## Six runs indépendants

| Architecture inchangée | Contrôle | Variante A | Variante B |
|---|---:|---:|---:|
| B1 SpatialCNN v2, largeur 16 | 100 | 30 | 10 |
| B2 StackedUNet, base 8, profondeur 3 | 100 | 30 | 10 |

Chaque run est initialisé **de zéro**, seed 42, avec un optimiseur neuf. Il ne
charge pas les poids des checkpoints B1/B2 existants. Les initialisations sont
identiques **au sein d'une architecture**, vérifiées par `initial_state_sha256`.
Les branches sont indépendantes ; aucune variante ne repart d'une autre variante.

Paramètres identiques : 250 événements train / 93 603 tuiles ; T=3, 8 canaux
normalisés ; cible J+1 ; masque cible et cinq jours historiques ; 4 époques,
batch train 16, **23 404 updates** par run ; AdamW lr=0,001, weight_decay=0,0001,
clip gradient=1 ; Dice weight=0,5 ; même LocalitySampler seed42 et ordre par époque.
Pas d'augmentation, d'architecture ou de changement temporel ajouté. Même
environnement PyTorch/matériel à conserver pour les comparaisons contrôlées.

Total prévu : **6 runs / 140 424 updates / 24 époques**. Ce total n'est pas
autorisé ni lancé ici. Prioriser les trois runs CNN si le budget avant le
18 novembre est contraint, puis compléter le U-Net sous autorisation. Le
sélecteur final demande les six runs ; tant qu'ils manquent, ne pas présenter
une comparaison complète des deux architectures. Aucun temps GPU n'est estimé
sans débit mesuré. Un seul seed ne démontre pas la robustesse inter-seeds.

## Exécution immédiate dans Kaggle après autorisation du budget

Importer `GEOFIREX_POS_WEIGHT_INDEPENDENT_20261112.ipynb`. Attacher les **11
partitions déjà publiées** listées dans `dataset-sources.json` ; GPU T4, Internet
désactivé. Le code et les IDs des 32 événements sont embarqués. Aucun checkpoint
historique ni nouveau dataset à uploader. Dépendances : NumPy, PyTorch, Matplotlib.
Les archives du corpus sont vérifiées puis reliées, sans reconstruction.

1. Exécuter les cellules de montage et de **précontrôle**. `RUN_TRAINING=False`
   reste le défaut. Le précontrôle vérifie métadonnées et SHA sans ouvrir de pixels,
   initialiser de modèle ou créer de sortie d'entraînement.
2. Choisir un seul couple `MODEL, POS_WEIGHT` ; conserver les autres paramètres.
3. Après autorisation de la campagne, définir `RUN_TRAINING=True` et exécuter la
   cellule correspondante. Ne pas activer les six automatiquement.
4. Conserver chaque nouveau dossier de run et son `checkpoint_latest.pt`.
   Les checkpoints historiques restent inchangés.

Commande Python exacte du notebook, avec ses variables définies par les cellules :

```python
command = [sys.executable, "-m", "geofirex.loss_ablation",
    "--corpus", str(corpus), "--index", str(index),
    "--protocol", str(workspace / "protocol.json"),
    "--model", MODEL, "--pos-weight", str(POS_WEIGHT), "--out", str(OUT),
    "--device", "cuda", "--workers", "4", "--seconds", "5400"]
# Cette forme ne fait qu'un précontrôle :
subprocess.run(command, check=True, env=os.environ.copy())
# À lancer seulement après autorisation :
subprocess.run(command + ["--execute-training"], check=True, env=os.environ.copy())
```

La limite 5 400 s borne le temps d'entraînement d'une session ; elle n'est pas
une promesse de terminer quatre époques. Le script sauvegarde toutes les 100
updates et aux frontières d'époque. À la limite, il écrit `paused.json` sans
score final. Pour reprendre **la même variante indépendante**, conserver son
checkpoint, choisir un nouveau `OUT` et ajouter :

```python
command += ["--resume", "/kaggle/input/.../B1-w30-independent-.../checkpoint_latest.pt"]
```

La reprise restaure modèle, optimiseur, RNG, curseur et accumulateurs. Un
changement de modèle, poids, protocole ou configuration est refusé. Elle ne
transforme pas les runs en fine-tunings des modèles historiques. Le budget
entraînement n'inclut pas les tris AP ultérieurs ; prévoir du scratch disque
pour les 32 événements, avec contrôle d'espace par le moteur d'évaluation.

## Sélection : les 32 seulement

Le checkpoint comparé est **fixé à l'époque 4** avant l'expérience, pour donner
le même budget d'updates à tous. Aucun choix opportuniste d'époque. Chaque run
fini infère les mêmes 13 759 tuiles / 32 événements. Il n'infère **jamais** les
169 autres événements ni 2021, et ne génère pas de carte de ces groupes.

Grille de seuils commune : 0,10–0,95 par 0,005, puis 0,96, 0,97, 0,98, 0,99,
0,995, 0,999. Elle étend la borne seulement pour les **nouveaux modèles** ; la
calibration historique n'est pas relancée. Par run : Dice micro maximale,
égalité départagée par précision puis seuil supérieur. AP exacte et Brier
restent sur les probabilités, sans correction analytique ou quantification.

Sortie finale par run : `selection32.json` avec modèle, poids, seuil, métriques,
SHA initial/final, protocole et compteurs `holdout_pixels_opened=0` et
`test_2021_pixels_opened=0`. Les comptages par événement et statistiques de
probabilités restent dans `calibration32/evaluation.json`.

Après les six runs, depuis le code extrait ou le dépôt :

```powershell
.\.venv\Scripts\python.exe scripts\select_loss_ablation.py --runs artifacts\ablation\B1-w100 artifacts\ablation\B1-w30 artifacts\ablation\B1-w10 artifacts\ablation\B2-w100 artifacts\ablation\B2-w30 artifacts\ablation\B2-w10 --out artifacts\ablation\winner32.json
```

Renseigner les six vrais dossiers locaux ou leurs équivalents Kaggle. Le script
exige six runs complets et comparables, les SHA des checkpoints et les mêmes
états initiaux par architecture. Classement préenregistré : Dice micro aux seuils
sélectionnés, égalité AP plus élevée, Brier plus bas, poids plus faible, ID modèle.
Il ne lit que `selection32.json`, config et les checkpoints pour leur SHA.
Les anciennes métriques de loss pondérée ne servent pas à classer les poids.

Figer `winner32.json` avant toute confirmation éventuelle. Une seule évaluation
du gagnant sur les 169 pourra confirmer le compromis ; ni ce résultat, ni ses
cartes ne doivent déclencher de nouveaux choix de poids/seuil. Le panel est déjà
observé : il ne constitue plus un test indépendant. Le test 2021 reste verrouillé.
Signaler les pertes de rappel/nouveaux-feux et la dispersion par événement même
si la Dice globale progresse. Une amélioration réelle n'est pas présumée avant
ces nouveaux résultats.
