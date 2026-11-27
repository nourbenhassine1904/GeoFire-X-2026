# GeoFire-X — diagnostic de reproductibilité et précontrôle Kaggle

14 novembre 2026, branche `feature/scientific-improvements-v1`.

**GO pour tester le précontrôle uniquement. NO-GO maintenu pour Focal : aucun résultat Kaggle de ce diagnostic n'est encore disponible.** Aucun entraînement, soumission, installation de packages ou changement de configuration historique n'a été effectué.

Notebook : [GEOFIREX_FOCAL_PRECHECK_ONLY.ipynb](GEOFIREX_FOCAL_PRECHECK_ONLY.ipynb).

Il s'agit d'un **compagnon de diagnostic de cinq cellules**, pas d'une nouvelle campagne. Le notebook Focal livré, son paquet, sa configuration et ses gardes restent identiques. Ses premières cellules ne mesuraient pas l'initialisation ; importer son module d'expérience complet aurait aussi embarqué un chemin d'entraînement. Le petit paquet de diagnostic ne contient que les définitions de modèle existantes, le sampler de métadonnées extrait sans modification et les sondes de ce diagnostic. Aucun optimiseur, backward ou chargement de checkpoint n'y est présent.

## 1. Ce qui est établi et ce qui ne l'est pas

| Question | Conclusion fondée sur les preuves |
|---|---|
| Le hash attendu est-il une valeur arbitraire du fichier Focal ? | Non : il est enregistré dans les configs et sélections des **trois** runs terminés poids100/30/10, puis vérifié par l'audit historique. |
| Le code de hash est-il identique ? | Oui : comparaison de l'AST de la fonction historique et de la fonction du diagnostic. |
| Le code modèle des paquets de préparation diffère-t-il ? | Non : `unet.py` et `benchmark_models.py` sont identiques entre le paquet historique conservé et le paquet Focal. Ce contrôle ne prouve pas à lui seul l'environnement exécuté sur Kaggle. |
| Une consommation Torch parasite après la seed est-elle démontrée dans le chemin initial historique ? | Non : entre la seed et la factory, le script ne contient que le réglage des threads/cuDNN. Le DataLoader et son générateur sont créés **après** le hash initial. |
| NumPy produit-il les poids de ce modèle ? | Non dans le chemin examiné : les initialiseurs utilisent Torch. Changer seulement la seed NumPy ne change pas le hash local ; son RNG n'est pas consommé pendant la création du modèle. |
| La sérialisation NumPy explique-t-elle le hash local ? | Le hash est identique avec une seconde sérialisation des octets Torch, indépendante de NumPy. Cette cause n'est pas établie localement. |
| La branche `Conv2d.reset_parameters` changée entre les sources2.11/2.15 explique-t-elle le cas présent ? | Le rejeu de la sémantique2.11 sur le backend local2.15 produit exactement le même hash local. Ce contrôle ne reproduit **pas** les anciens kernels/binaires Torch. |
| La différence PyTorch explique-t-elle définitivement l'écart ? | C'est une hypothèse compatible avec les observations, avec les différences de plateforme/build. **Elle n'est pas isolée causalement** faute de rejeu dans l'environnement historique. |
| La reproduction exacte est-elle prouvée impossible ? | Non. Elle n'est simplement pas démontrée. Le précontrôle Kaggle est l'étape nécessaire pour trancher. |

Les versions différentes ne sont pas un diagnostic causal suffisant. PyTorch ne garantit pas une reproduction identique entre versions et plateformes, même avec la même seed ([documentation officielle2.11](https://docs.pytorch.org/docs/2.11/notes/randomness.html)). NumPy exige lui aussi des conditions strictes de compatibilité des flux ; `default_rng`, utilisé par le sampler, ne doit pas être supposé invariant entre versions ([politique officielle](https://numpy.org/doc/stable/reference/random/compatibility.html)). Cela justifie les gardes sans attribuer arbitrairement l'écart à une bibliothèque.

## 2. Ordre exact de la baseline poids10

Sources : `src/geofirex/loss_ablation.py`, script/notebook historique de préparation et paquets conservés. Le script original n'a pas été exécuté pendant cette mission.

1. Imports standard, NumPy, Torch/DataLoader, factory de modèles, helpers, évaluation, loader compact et loss. Les classes de modèles sont définies, aucun modèle n'est construit par ces définitions. La grille de seuils est calculée avec des opérations NumPy déterministes.
2. Précontrôle : lecture du protocole, normaliseur/index, couverture et événements via les coordonnées des tuiles. Pas de lecture de pixels ni d'itération du DataLoader. Toute consommation éventuelle **avant** la seed serait de toute manière réinitialisée par celle-ci.
3. Contrôle CUDA et création du dossier de sortie, puis exactement :

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
network = make_model('B2', 8, 3).to('cuda')
initial_sha = model_hash(network)
```

4. La factory crée `StackedUNet(8,3)` puis `LightUNet(24,8,3)`. Les poids sont construits sur le device/dtype par défaut ; les chemins conservés ne fixent pas explicitement ces défauts. Le diagnostic refuse un défaut ambiant autre que CPU/float32 au lieu de le corriger silencieusement.
5. **15 Conv2d** consomment le RNG Torch pour leurs poids ; le biais de la convolution finale est aussi initialisé aléatoirement, puis remplacé par la constante−3. Cette dernière écriture ne rend pas son tirage initial inexistant dans le flux RNG. Les **14 GroupNorm** initialisent leurs poids à1 et biais à0 sans consommer le RNG. Le traçage local enregistre chacun de ces appels sans modifier leur résultat.
6. Le transfert CPU→CUDA précède le hash historique. Le diagnostic calcule le hash CPU puis celui après copie des paramètres vers CUDA et retour des octets sur CPU, **sans forward**. Le garde Focal actuel vérifie le hash CPU avant son transfert ; sa valeur attendue n'a pas été modifiée.
7. Le hash parcourt `sorted(network.state_dict().items())`, ajoute le nom UTF-8 puis `tensor.detach().cpu().numpy().tobytes()` au SHA-256. Il ne hache ni le fichier `.pt`, ni les états RNG, ni les versions, et n'ajoute pas de champs dtype/shape. Les préfixes de clés, dont `network.`, comptent. L'ordre de hash est lexical, distinct de l'ordre de construction des couches.
8. AdamW, la loss, le sampler et le DataLoader/générateur43 sont créés après le hash initial ; ils ne peuvent donc pas expliquer ce hash par une consommation RNG antérieure dans ce script.

La sonde locale retrouve 123 977 paramètres, 44 tenseurs de `state_dict`, tous float32, et un biais final−3. L'[inspection statique du pickle du checkpoint final](checkpoint_static_metadata.json), sans désérialisation ni chargement des poids, retrouve `FloatStorage`, `ByteStorage` et les localisations CPU/CUDA0. Ces métadonnées du checkpoint **entraîné** ne constituent pas une copie des tenseurs initiaux historiques.

## 3. Environnement et provenance du hash historique

[Provenance vérifiée](historical_provenance.json) : archive `GeoFireX_UNet_ABLATION_3_MODELS.zip`, SHA-256 `a122dfa4ac419f3b8f8d32e1ae309033ef02ccfb1891fbaed3b5ed7c6a751e07`.

Les trois configs/sélections enregistrent le même état initial :

`a835383534d061787c2449bbb4a896c32c3e990d92b80fe2b3105f9c91ac6f5f`.

La sélection poids10 référence le checkpoint final `7295eaa6…1f1b14e`. La vérification de son hash a été renouvelée en lecture seule ; l'audit existant avait déjà contrôlé l'égalité config/checkpoint et les budgets. Le hash initial est une **empreinte enregistrée avant entraînement**, pas le hash du checkpoint final ni une empreinte recalculée aujourd'hui depuis des poids initiaux archivés.

| Élément | Baseline poids10 directe | Poste local |
|---|---|---|
| PyTorch | `2.11.0+cu130` | `2.15.0+cpu` |
| NumPy | `2.1.3` | `2.4.6` |
| Device enregistré | `cuda` | `cpu` |
| Python / plateforme / build exacts | **Non enregistrés dans ces configs** | Python3.11.6, Windows/MSC64bits ; détail dans la sonde |
| CUDA | Suffixe de build `cu130` ; driver exact non enregistré | Aucun build CUDA |
| GPU exact | Non enregistré dans la config directe | Aucun GPU CUDA disponible |

Le run apparenté de propagation enregistre explicitement TeslaT4, CUDA13.0 et les mêmes versions Torch/NumPy, ainsi que le même hash du cœur initial. C'est un indice concordant, **pas une preuve du GPU/driver/Python exact du run poids10**. Les notebooks conservés ont des cellules sans outputs et une métadonnée kernelspec générique ; ils ne fournissent pas d'identifiant complet d'image exécutée.

Les sources officielles [Conv2d2.11](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/modules/conv.py) et [Conv2d2.15](https://raw.githubusercontent.com/pytorch/pytorch/v2.15.0/torch/nn/modules/conv.py) montrent un changement de gestion des poids non contigus. Notre contrôle des poids contigus ne modifie pas le hash en réappliquant la sémantique2.11 sur le backend actuel. Aucune ancienne implémentation binaire n'a été installée ou exécutée.

## 4. Résultats du diagnostic local

[Rapport complet issu du paquet isolé](local_precheck_report.json) : **BLOCKED**, résultat attendu sur ce poste.

| Contrôle CPU | Résultat |
|---|---|
| Initialisation exacte seed42 répétée | Hash local `69f6c443…dd8fe0` identique à chaque rejeu |
| Seed NumPy123, seed Torch42 inchangée | Même hash |
| Hash des octets sans sérialisation NumPy | Même hash |
| Sémantique de reset Conv2d2.11, backend local inchangé | Même hash |
| Un unique `torch.rand(1)` après seed42 et avant le modèle | Hash différent `2800d49e…07306a` ; sensibilité au décalage RNG démontrée, pas preuve d'un tel décalage historique |
| NumPy RNG pendant la factory | Inchangé |
| Hash pendant son calcul | Ne consomme pas le RNG Torch |
| Sampler : quatre époques | Les quatre empreintes concordent avec la configuration Focal |
| Boundary CUDA historique | Non testé localement : CUDA indisponible |

Le contrôle local des partitions est un aperçu de métadonnées (`all_bytes=False`), donc il reste explicitement BLOCKED pour la certification complète ; les SHA complets avaient été vérifiés dans la mission précédente. Le notebook Kaggle conserve **`VERIFY_ALL_BYTES=True`** par défaut et effectuera la vérification complète sur les montages existants, sans écrire dans les données.

Nous pouvons donc écarter les causes locales testées, mais pas distinguer à ce stade un changement de backend/RNG Torch d'une différence de plateforme/build ou d'un détail non archivé de l'exécution historique. Aucun tenseur initial complet ni état RNG d'avant modèle n'a été sauvegardé dans les artefacts disponibles ; un hash seul ne permet pas de les reconstruire. Le précontrôle commence désormais à enregistrer ces traces et les empreintes par tenseur, sans sauvegarder de nouveau checkpoint.

## 5. Instructions Kaggle, cellule par cellule

Utiliser une **copie de travail dédiée au diagnostic** du notebook Kaggle Focal existant, en conservant ses11 sources de données déjà disponibles. Importer le notebook de précontrôle ou remplacer les cellules de cette copie par ses cinq cellules de code, dans l'ordre. Ne pas exécuter simultanément les anciennes cellules de campagne. Garder le notebook original et ses fichiers locaux inchangés.

Partir d'une session fraîche pour éviter un `geofirex` déjà importé ou des défauts dtype/device modifiés. Internet désactivé ; aucune commande d'installation. Pour vérifier le transfert historique, choisir un accélérateur CUDA disponible, idéalement le même T4 lorsqu'il est proposé. Cette opération ne fera qu'interroger le GPU et copier environ0,5Mo de paramètres ; aucun calcul de modèle ni entraînement. Un contrôle sur CPU seul peut produire un diagnostic utile, mais restera BLOCKED pour la frontière CUDA.

| Cellule de code | Action exacte | Ce qu'il faut contrôler |
|---|---|---|
| **1 — chargement du diagnostic** | Vérifie le SHA du petit paquet et de la config Focal embarquée, extrait uniquement le code vers `/tmp`, crée un dossier de résultats nouveau sous `/kaggle/working/geofirex-repro-precheck-…`. | `RUN_TRAINING=False`, `AUTHORIZATION_FILE=None`, `VERIFY_ALL_BYTES=True`, `CHECK_CUDA_TRANSFER=True`. Ne changer aucun hash. Aucun paquet n'est installé. |
| **2 — environnement** | Affiche Python, plateforme, Torch, NumPy, CUDA du build, cuDNN, GPU/capacité/mémoire et driver si `nvidia-smi` est disponible. | Comparer Torch/NumPy aux valeurs2.11.0+cu130/2.1.3. Une divergence est signalée, jamais réparée automatiquement. Poursuivre le diagnostic même si les versions diffèrent pour conserver la preuve. |
| **3 — initialisation** | Rejoue la séquence de seed, trace les initialiseurs, calcule les hashes CPU et après copie CUDA, exécute les petits contrôles CPU RNG/sérialisation. | Le hash CPU **et** le hash à la frontière CUDA doivent retrouver `a8353835…c6f5f`. Les contrôles NumPy/sérialisation ne doivent pas changer les octets. Aucun forward, loss, backward ou optimiseur. |
| **4 — corpus et sampler** | Retrouve les11 `mount-map.json` attendus, vérifie les identités, chemins, tailles et SHA des459 objets **sur place**, puis lit les petits index NPY de coordonnées et vérifie les quatre ordres du sampler. | Pas de téléchargement, republication, liens ou nouvelle importation des partitions. Les ZIP sont lus comme octets compressés ; aucun membre NPZ n'est ouvert. Cette cellule peut prendre du temps pour lire environ9,85Go. Une partition manquante/dupliquée ou un hash différent bloque le résultat. |
| **5 — décision et export** | Écrit `precheck_report.json` et `precheck_report.sha256`, affiche PASS/BLOCKED avec les causes. | Sauvegarder ces deux fichiers depuis les outputs Kaggle. Le rapport contient les traces par tenseur/initialiseur. Ne pas activer Focal, même si PASS. |

Les erreurs d'initialisation ou de montages sont capturées en BLOCKED et la dernière cellule conserve leur cause. Une erreur plus tôt dans le bootstrap (paquet/config altéré) doit être conservée telle quelle ; ne pas contourner l'assertion SHA. Pour recommencer, une nouvelle exécution crée un nouveau dossier et n'écrase pas les rapports.

### Signification de PASS

PASS exige simultanément les versions Torch/NumPy figées, un GPU CUDA disponible, le hash CPU et post-transfert attendu, un rejeu local déterministe, l'accord des deux sérialisations et la certification complète corpus/sampler. Il porte **uniquement sur ces prérequis techniques**, pas sur la reproductibilité bit à bit d'un entraînement ni sur une amélioration scientifique.

Le résultat conserve toujours `training_authorized=false`, `RUN_TRAINING=false`, zéro autorisation générée et zéro pas d'optimisation. Les gardes du notebook et du lanceur Focal existants restent actives. Aucun job n'a été soumis pour vous ; le résultat réel de Kaggle est encore à obtenir.

## 6. Solutions si Kaggle diffère

1. **Versions et hashes concordent sur Kaggle** : conserver le rapport, son hash et l'environnement affiché. Les prérequis techniques contrôlés passent ; budget complet et autorisation explicite restent distincts. Aucun lancement automatique.
2. **Versions différentes, même hash initial** : preuve utile que ce modèle initialise identiquement dans ces deux environnements, mais la garde de versions reste BLOQUÉE. Le sampler et les opérations d'entraînement peuvent encore différer. Ne pas modifier les exigences pour obtenir PASS.
3. **Versions attendues, hash différent** : examiner les défauts CPU/float32, l'identité des sources, le hash avant/après transfert, les traces RNG et les tenseurs. Python/build/plateforme historiques étant incomplets, ne pas annoncer la version Torch comme cause prouvée. Ne pas chercher une seed/consommation RNG arbitraire donnant le hash attendu.
4. **Environnement historique identifié mais indisponible** : une image/installation isolée compatible peut être envisagée dans une mission ultérieure autorisée, à partir d'artefacts de provenance vérifiables. Un simple numéro de version ou une installation manuelle de roues quelconques ne garantit pas le même build. Aucune image historique Kaggle récupérable n'a été démontrée ici ; aucune commande `pip install` ou mise à jour automatique n'est livrée.

### Repli scientifique, si la reproduction exacte ne peut pas être établie

Proposer un **nouveau protocole versionné**, soumis à accord, dans un environnement commun disponible : une nouvelle baseline BCE pondérée10+Dice et un candidat Focal, tous deux depuis les mêmes tenseurs **aléatoires non entraînés** sauvegardés et hashés avant optimisation. Même corpus, exemples, ordre, seed42, quatre époques, AdamW et critères/seuils de développement32. La comparaison causale serait celle de ces deux nouveaux runs ; les résultats historiques resteraient un contexte descriptif.

Ce repli demande **un entraînement de baseline supplémentaire** par rapport à la proposition Focal unique : scénario hérité11,1–22,2min GPU en supplément, soit22,2–44,4min pour les deux entraînements, **hors deux évaluations, préparation et sauvegarde**. Ces plages ne sont pas mesurées dans le nouvel environnement et ne sont pas des bornes. Il faudrait préenregistrer le nouveau comparateur et autoriser le budget avant toute exécution.

Aucun protocole de repli exécutable, nouvelle configuration de campagne ou notebook de deux entraînements n'est créé ici. La configuration actuelle, son hash historique et ses gardes restent intégralement conservés. Aucun nouvel entraînement, réévaluation historique ou accès2021 n'est effectué.

## 7. Validation et fichiers

**55 tests CPU ciblés réussis**, zéro échec, en9,10secondes : [JUnit](tests-delivery.xml). Ils couvrent l'identité AST du hash historique, la consommation RNG des15convolutions/14GroupNorm, la sérialisation indépendante, les contrôles NumPy/Torch, les défauts ambiants, la restauration des états CPU, les décisions PASS/BLOCKED, la compilation et l'absence d'API d'optimisation/chargement de checkpoint dans toutes les cellules et le paquet, ainsi que les tests Focal précédents. Aucun entraînement n'a été exécuté pour ces vérifications.

Le petit paquet embarqué a aussi exécuté le diagnostic et le contrôle de métadonnées réelles dans un processus Python isolé. Aucune tuile scientifique n'a été décodée. L'inspection du checkpoint historique utilise uniquement les opcodes pickle en lecture statique, **sans `torch.load` ni désérialisation**.

**166 fichiers antérieurs inchangés**, vérifiés contre les empreintes des livraisons précédentes : [contrôle](preservation_check.json), [empreintes](preservation.json). Cela inclut l'intégralité de la préparation Focal, son notebook, son runner et sa config. Le modèle Studio et les données ne changent pas. Aucun commit, push, merge, reset ou clean.

Ajouts : `scripts/focal_repro_precheck.py`, `scripts/prepare_focal_repro_precheck.py`, `tests/test_focal_repro_precheck.py` et ce dossier de diagnostic uniquement. Aucun fichier préexistant n'est modifié.

Commande de tests exécutée depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_focal_repro_precheck.py tests/test_focal_experiment.py -q -p no:cacheprovider --basetemp=artifacts/focal-repro-precheck-v1/pytest-delivery --junitxml=reports/focal_repro_precheck_v1/tests-delivery.xml
```

Pour une répétition, choisir de nouveaux chemins temporaires/JUnit. Le script de préparation refuse d'écraser son dossier existant ; aucun script historique de campagne n'a été exécuté.
