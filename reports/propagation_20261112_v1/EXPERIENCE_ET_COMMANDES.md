# GeoFire-X — une expérience de propagation, préparée et non lancée

Décision du 12 novembre : **U-Net avec une tête spécialisée pour les nouvelles détections à J+1**. Aucun nouvel entraînement, aucune nouvelle inférence des modèles historiques, aucun transfert et aucune publication exécutés. Le notebook est [GEOFIREX_PROPAGATION_NEW_FIRE_HEAD_V1.ipynb](GEOFIREX_PROPAGATION_NEW_FIRE_HEAD_V1.ipynb).

Le rappel de 0,146819 ne concerne pas tous les feux. Son domaine est `N = masque_valide & (feu_J == 0)` et sa cible positive est `N & (feu_Jplus1 == 1)`. Les deux sous-domaines feu_J=0 et feu_J=1 partitionnent exactement les pixels valides. Ce sont des détections actives quotidiennes, pas une surface brûlée cumulée. Le mot « propagation » désigne ici une tâche opérationnelle approchée : ces labels ne distinguent pas déplacement du front, détection isolée et réapparition.

## Pourquoi cette seule expérience

Sur les 169 événements déjà évalués, 308 168 / 562 953 feux cibles sont nouveaux (54,74 %), mais 262 923 / 300 710 faux négatifs sont dans ce sous-domaine (87,43 %). Le rappel des feux déjà observés est 85,17 %. Dix événements concentrent 46,69 % des faux négatifs nouveaux. Le rappel médian des 165 événements avec nouveaux feux vaut 8,77 % ; 19 en détectent zéro. Les quatre autres ont un rappel indéfini et ne sont pas assimilés à des succès.

Les masques et labels des **70 tuiles des trois cartes déjà enregistrées** ont été confrontés au compact : égalité exacte de la cible, du canal feu de J et de l'intersection des six masques. Les entrées sont J−2, J−1, J ; la cible est J+1. Le masque reste celui des cinq jours historiques J−4…J et de J+1. Aucun décalage n'est observé sur ces cas ; cette vérification ciblée n'est pas une nouvelle certification exhaustive du corpus.

La carte centrale `2020/fire_24332880`, cible 2020-09-04, contient 228 nouveaux feux : 27 TP, 201 FN. La probabilité médiane de ces FN est 0,434650 ; celle des feux cibles déjà observés à J est 0,913601. Deux autres dates médianes ne contiennent aucun nouveau feu, et restent présentées comme contrôles négatifs. Les probabilités conditionnelles nouveau/ancien n'ont pas été conservées sur tout le domaine : **aucune distribution conditionnelle globale ni AP des seuls nouveaux feux n'est inventée à partir de ces trois dates**.

La loss actuelle est `0,5 BCEWithLogits(pos_weight=10) + 0,5 Dice soft par image` ; le poids 100 est historique, et non celui du candidat confirmé. Elle optimise l'ensemble du feu futur sans distinguer persistance et nouveauté. Les tuiles vides sont majoritaires (85 940 négatives sur 93 603, catégories historiques), et leur Dice a peu de gradient lorsque le premier plan est absent. La pondération et le Dice rendent les sorties impropres à une interprétation automatique comme fréquences calibrées. L'AP mesure le classement ; le Brier se calcule sur les probabilités, indépendamment du seuil. Ces observations rendent la spécialisation plausible ; elles ne prouvent pas la cause du faible rappel.

L'ablation sauvegardée montre déjà la tension sur les **32** : nouveaux-feux Dice/rappel = 0,195810/0,191372 pour poids100 au seuil0,96, contre 0,182688/0,156487 pour poids10 au seuil0,78, malgré le meilleur Dice global du poids10. On conserve la décision historique et on cesse ce balayage de poids.

Le `StackedUNet` actuel empile T×C ; il n'a pas de récurrence temporelle explicite. Le ConvLSTM B3 existant encode avec stride2, huit canaux cachés, puis une tête interpolée, sans décodeur avec connexions de saut. Le tester modifierait simultanément la représentation temporelle et la restitution spatiale. La tête spécialisée conserve le U-Net éprouvé et sa résolution, tout en dirigeant directement une supervision vers les pixels manqués. La supervision auxiliaire est un principe établi, mais cette application au nouveau feu est une **hypothèse à tester**, pas un résultat publié : [U-Net, Ronneberger et al.](https://arxiv.org/abs/1505.04597), [Deeply-Supervised Nets, Lee et al.](https://proceedings.mlr.press/v38/lee15a.html).

Limite majeure : le compact huit canaux contient pente/altitude normalisées, leurs indicateurs de valeurs manquantes, sinus/cosinus d'aspect, indicateur d'aspect manquant et détection du feu. **Il ne contient ni vent, ni humidité, ni météo prévue, ni combustible.** La séquence de feu constitue sa principale information temporelle. Cette expérience ne peut pas compenser l'absence de ces forçages. Aucun corpus n'est reconstruit pour cette étape.

## Protocole figé

Un seul entraînement indépendant, identifiant `UNET_NEW_FIRE_HEAD_V1` :

- Même backbone U-Net base8/profondeur3, entrées `[B,3,8,64,64]`. Une nouvelle convolution1×1 sur les huit caractéristiques finales ajoute neuf paramètres.
- Tête principale supervisée sur tout le domaine valide, comme avant. Tête spécialisée supervisée sur `N` avec la même cible J+1 ; les anciens feux et invalides ne contribuent pas à cette deuxième loss. Tous les négatifs de N restent dans la BCE, sans filtrage par proximité ni sur-échantillonnage.
- `L = L_principale + 0,5 L_nouveau`, avec `L_nouveau = 0,5 BCE_w10(N) + 0,5 Dice_batch(N)`. Le Dice auxiliaire regroupe le batch ; sans cible nouvelle positive, sa contribution est zéro et la BCE négative reste active. Coefficient0,5 et poids10 figés, aucun balayage.
- À l'inférence, `logit = logit_nouveau` si feu_J=0, sinon `logit_principal`. La porte ne lit que J, jamais la cible. Un seul seuil global est ensuite choisi.
- 250 événements 2018–2019, 93 603 tuiles, chaque tuile une fois par époque ; LocalitySampler seed42 ; batch16 ; 4 époques, 23 404 updates. Pas d'augmentation ni changement de corpus.
- AdamW lr0,001, weight_decay0,0001, clipping1 ; float32 ; seed42 ; algorithmes déterministes, TF32 désactivé. Normaliseur train-only déjà appliqué : **ni nouvel ajustement ni seconde normalisation**.
- Environnement Kaggle conservé : Torch2.11.0, NumPy2.1.3. Le digest initial du backbone doit correspondre au digest enregistré de l'ablation `a835383534d061787c2449bbb4a896c32c3e990d92b80fe2b3105f9c91ac6f5f`. Sinon arrêt avant tout update, sans modification automatique du protocole. La machine locale Torch2.15.0+cpu donne un autre digest initial (`69f6c443…`), donc elle valide les contrats et gradients, **pas l'identité d'initialisation Kaggle**. Ce contrôle reste à passer dans l'environnement autorisé.
- Sauvegardes atomiques tous les100 updates et à chaque époque, optimiseur/RNG/curseur/temps ; reprise uniquement du nouveau run, jamais d'un checkpoint B1/B2 historique. Les checkpoints d'époque sont conservés.

## Budget et décision

**Un seul GPU T4, 7 200 secondes de processus au maximum**, pas une garantie de quatre époques. Arrêt anticipé à proximité du plafond et timeout externe de secours ; aucune reprise automatique. Une reprise explicite débite le temps du checkpoint et conserve le même plafond cumulé. Après un arrêt brutal, le journal peut contenir davantage de temps que le dernier checkpoint : utiliser le temps le plus élevé et demander un nouveau budget si nécessaire, sans remettre le compteur à zéro. Les secondes déjà allouées à une session tuée doivent être comptées intégralement.

Le fichier `gpu_budget.json` réserve le solde du budget avant les updates. Après arrêt brutal, le statut reste `RESERVED` et le runner refuse la reprise avant même d'initialiser CUDA ; le temps du dernier checkpoint ne permet pas d'effacer une allocation perdue. Seul un arrêt propre `PAUSED`, avec le bon digest de checkpoint et plus de30s restantes, permet une reprise sous le même plafond. Un budget épuisé exige une nouvelle décision explicite, pas une modification automatique.

Les journaux enregistrent temps de chargement et d'update synchronisé CUDA, temps mural, RSS processus et workers, pics mémoire GPU allouée/réservée, ainsi que des instantanés `nvidia-smi`. La capacité mémoire et le débit réels du nouveau modèle restent à mesurer lors du premier lancement autorisé. Aucun profilage GPU n'a été exécuté.

La sélection32 est séparée sur **CPU**, timeout3 600s, checkpoint fixe époque4 uniquement. Si le plafond empêche d'atteindre 23 404 updates, le résultat est incomplet et aucun score n'est comparé au budget de quatre époques. Grille prédéclarée : 0,10–0,95 par0,005, puis 0,96/0,97/0,98/0,99/0,995/0,999. Même règle historique : Dice micro **global**, égalité précision puis seuil plus élevé. Dice, IoU, AP exacte, Brier, précision, rappel, TP/FP/FN, métriques par événement, Dice/précision/rappel des nouveaux feux et cartes32 sont produits dans une seule inférence du nouveau modèle.

Acceptation sur les32 seulement : Dice global ≥0,3872822594 **et** Dice nouveaux ≥0,1826877204 **et** rappel nouveaux >0,1564867235. Sinon l'expérience est négative, sans nouvelle recherche automatique. Ce critère interdit de présenter une hausse de rappel obtenue seulement en multipliant les fausses alertes. Aucune significativité statistique n'est revendiquée avec une seed. Les32 sont un ensemble de développement réutilisé ; les169 ont déjà été consultés et ne sont plus un test aveugle. Le runner n'offre **aucune commande d'inférence169 ou2021**, et aucune promotion automatique dans l'application.

## Commandes exactes

Précontrôle local, sans pixels ni GPU et sans création du dossier de run :

```powershell
.\.venv\Scripts\python.exe -m geofirex.propagation_experiment --corpus reports/mission_20261111/compact_full --index reports/gpu_acceleration_20261111_v1/index --protocol reports/propagation_20261112_v1/protocol.json --baseline32 reports/propagation_20261112_v1/baseline32.json --out reports/propagation_20261112_v1/run-not-launched
```

Kaggle, **après autorisation GPU seulement** :

1. Ouvrir le notebook autonome livré. Attacher les onze datasets de [dataset-sources.json](dataset-sources.json), `nnnnbhss/geofirex-compact-t3-20261111-p00` à `p10`. Aucune republication ou copie des tableaux : les objets montés sont vérifiés par SHA puis liés en lecture seule. Internet OFF, un seul GPU T4.
2. Exécuter la première cellule de code : elle restaure le code et les liens, vérifie les identités et imprime le précontrôle. Les dépendances doivent déjà être disponibles ; [requirements-kaggle.txt](requirements-kaggle.txt) décrit l'environnement, sans demander une réinstallation de CUDA/PyTorch.
3. Après autorisation, mettre `RUN_TRAINING=True` dans la cellule correspondante. La cellule contrôle l'initialisation sur CPU avant de lancer la commande suivante, déjà construite avec les vrais chemins temporaires :

```python
command = common + ["--out", str(OUT), "--mode", "train", "--execute-training"]
subprocess.run(command, check=True, env=os.environ.copy(), timeout=7200)
```

4. Conserver `config.json`, `resources.jsonl`, les checkpoints et `training_complete.json`. Un checkpoint partiel ne permet pas la sélection32. Ne jamais relancer un modèle déjà terminé.
5. Désactiver l'accélérateur GPU et exécuter la sélection sur CPU, à partir du même code et des montages déjà vérifiés ; dans le notebook, définir `CHECKPOINT` puis `RUN_SELECTION32=True` :

```python
subprocess.run(common + ["--out", str(OUT)+"-selection32", "--mode", "select32",
                        "--checkpoint", str(CHECKPOINT)],
               check=True, env=os.environ.copy(), timeout=3600)
```

Les chemins `common`, `OUT`, `workspace`, `corpus` et `index` sont définis par le notebook ; aucun slug de montage n'est deviné. Si reprise nécessaire, `RESUME` pointe uniquement vers le nouveau checkpoint ; choisir un OUT inédit et ne pas dépasser le budget restant. Les résultats historiques ne sont jamais ouverts comme sortie.

Application locale :

```powershell
.\.venv\Scripts\python.exe scripts/serve_demo.py --results-only --port 8014
```

La vue « Propagation J+1 » utilise les sorties sauvegardées et les cartes dérivées, sans charger de modèle. Le lancement historique et ses contrôles de source/checkpoint restent séparés et intacts.
