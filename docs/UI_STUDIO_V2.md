# GeoFire-X Studio V2 — Flamme solaire

V2 conserve React 19.3.1, React DOM, esbuild, JavaScript `.mjs` et CSS natif. Aucune nouvelle dépendance. Branche de travail : `feature/geofire-studio-ui-v1`. Aucun commit, push, merge, reset, clean ou staging effectué.

## Changements

Le fond ivoire, les panneaux blancs et les surfaces secondaires chaudes remplacent le graphite. Les variables de `frontend/src/style.css` centralisent fonds, textes, bordures, actions, badges, hover et focus. Les boutons orange utilisent un texte brun foncé. Les contrôles disposent d'une bordure plus contrastée que les bordures décoratives. Les anciens tableaux, panneaux et dialogues partagent le thème clair.

Les 344 règles de géométrie existantes retenues dans `frontend/tests/layout-v1.json` sont vérifiées : largeurs, max-width, grilles, espacements, positions et breakpoints. Les tailles de texte des tableaux, légendes et étapes temporelles ont été relevées, sans refonte des grilles. Les nouvelles visualisations Big Data utilisent leur propre mise en page dans les panneaux existants.

Les canvas, PNG scientifiques et fonds nécessaires aux rasters restent sombres. `prediction_view.mjs`, les palettes de pixels, le seuil et les calculs ne sont pas modifiés. Les couleurs des **courbes comparatives** sont assombries pour la lecture sur fond clair ; cela ne change pas les couleurs sémantiques des cartes.

## Inventaire React Bits

Source commune : [dépôt officiel React Bits, commit figé](https://github.com/DavidHDev/react-bits/tree/4e0d091d3b1ea8b7aadef33716a8d99d164ae4ce). Versions JavaScript/CSS, licence `MIT + Commons Clause` et empreintes conservées dans `frontend/src/reactbits/`. Les adaptations locales sont explicites dans `PROVENANCE.json`.

| Composant et source locale | Importeur → page | Déclencheur réel | Dépendances déjà présentes | Preuve automatisée V2 | Vérification Chrome restante |
|---|---|---|---|---|---|
| Aurora — `frontend/src/reactbits/Aurora.jsx` et `.css` | `studio_ui.mjs` → Overview | Montage autorisé, boucle rAF WebGL2 ; suspend hors viewport/onglet masqué, détruit en quittant Overview | React ; WebGL2 natif | Appels draw, temps shader évolutif, uniform lightMode=1, contexte détruit ; fallback sans WebGL testé | Voir le mouvement orange/ambre/pêche ; juger opacité, fluidité et contraste réel |
| SplitText — `frontend/src/reactbits/SplitText.jsx` | `studio_ui.mjs` → Overview | Polices prêtes puis entrée `load`, une seule fois par montage du titre | GSAP, SplitText, ScrollTrigger, @gsap/react | Mots réellement divisés par GSAP, timeline terminée, pas de duplication des mots | Voir l'entrée du titre une fois, sans saut de mise en page |
| CountUp — `frontend/src/reactbits/CountUp.jsx` | `studio_ui.mjs` → quatre KPI Overview | Données disponibles et entrée viewport, durée 1,35 s | Motion | Quatre animations terminées et textes finaux exacts : 250 / 201 / 607 / 23 404 | Voir les nombres évoluer puis se fixer ; vérifier le déclenchement au scroll |
| SpotlightCard — `frontend/src/reactbits/SpotlightCard.jsx` et `.css` | `studio_ui.mjs` → KPI Overview | Pointeur à proximité/sur la carte ; éclairage orange, aucune boucle ambiante | React, rAF natif | Déplacement pointeur : opacité de la vraie couche lumineuse > 0,05 | Vérifier le halo perceptible et les textes lisibles au survol |
| FadeContent — `frontend/src/reactbits/FadeContent.jsx` | `studio_ui.mjs`, puis `StudioPages.mjs`, `ScienceResults.mjs`, `EventForecast.mjs` → sections Overview/Research/Big Data, sous-onglets Analytics, réponse Studio | `scroll` pour les sections, `mount` pour un changement d'onglet ou une nouvelle réponse ; 0,48 s | GSAP, ScrollTrigger | État en attente hors viewport simulé ; callback d'entrée puis vraie timeline complète ; tabs et résultat testés | Scroll lent sur les trois pages ; changement de sous-onglet ; nouvelles cartes apparaissant brièvement |
| Stepper — `frontend/src/reactbits/Stepper.jsx` et `.css` | `EventForecast.mjs` → Prediction Studio | Étapes 1 configuration / 2 requête en cours / 3 prédiction disponible / 4 vérité séparée disponible | Hugeicons ; CSS/WAAPI natifs | Étapes 2 et 4 observées, appels WAAPI ; zéro appel WAAPI dans le parcours système reduced-motion | Voir le changement d'étape selon le vrai retour API, sans progression chiffrée |
| AnimatedList — `frontend/src/reactbits/AnimatedList.jsx` et `.css` | `EventForecast.mjs` → sélecteur événements Studio | Montage seulement à l'ouverture du dialogue ; IntersectionObserver puis transitions des éléments visibles | Hugeicons ; CSS/rAF natifs | Liste absente au repos et après fermeture ; éléments `data-seen`, transitions opacity ; sélection réelle | Ouverture, scroll interne, recherche, flèches clavier/Entrée et sélection, sans latence |

Les attributs `data-animation-state` d'Aurora et SplitText reflètent le passage dans le moteur de rendu ou les callbacks réels de timeline. CountUp et Fade utilisent leurs callbacks officiels. Ce sont des diagnostics, sans animation factice. Aucun rendu de shader n'est produit par le simulateur.

La préférence `prefers-reduced-motion` prime. Le bouton de la barre supérieure permet aussi de réduire les effets et mémorise `localStorage['geofirex-reduced-motion']`. S'il vaut `true`, les effets restent volontairement statiques après rechargement. Le défaut utilisateur est animé ; une préférence système réduite ne peut pas être contournée par ce bouton. Les coches et le glissement du Stepper, ainsi que l'indicateur de liste, respectent également le mode statique.

## Intégrité scientifique

- Modèle principal inchangé : U-Net poids 10, seuil 0,78. La tête spécialisée reste une expérience comparative non retenue.
- Le rappel des nouveaux pixels, **14,68 %**, est lu depuis `confirmation.new_fire.new_fire_candidates.recall`, sans nouvelle évaluation. Le compteur expose 45 245 TP / 308 168 positifs et 262 923 FN sur le panel déjà observé.
- Les dates et événements viennent du catalogue réel. Le spinner accompagne uniquement l'attente des vraies requêtes ; aucun pourcentage ni délai inventé.
- Le conteneur des résultats est animé en opacité. Les pixels des cartes ne sont jamais interpolés. Une note signale les cartes binaires où ≤ 1 % des pixels valides de **l'aperçu** sont positifs ; ce compte d'affichage n'est ni une surface ni une métrique complète du modèle. Les cartes scientifiques sauvegardées gardent une explication de leur possible rareté.
- Les huit cartes, zoom et exports PNG/JSON/CSV restent disponibles. La vérité J+1 arrive après l'inférence par une requête séparée. Les cartes d'erreurs gardent TP vert / FP rouge / FN bleu.
- Le diagramme et les barres lisent `/science/big-data` : 2018 train 176, 2019 train 74, 2020 validation 201, 2021 test 156 (métadonnées uniquement). Les totaux par split sont dérivés des mêmes lignes. Axe commun 0–250 événements, libellés et tableau accessibles conservés.
- Spark 4.0.2 `local[2]` a été réellement exécuté auparavant ; cette interface lit ses sorties Parquet. Aucun cluster ni streaming revendiqué, aucun nouveau job lancé.
- J−4 / J−3 sont utilisés pour la validité ; J−2 / J−1 / J sont les entrées T3 ; J+1 reste une cible séparée. Aucune animation ne déroule automatiquement ces dates.
- Aucun fichier de modèle, checkpoint, prétraitement ou résultat scientifique historique n'a été modifié. Pas d'entraînement ni de lecture de pixels 2021.

## Exécuter les vérifications

Depuis la racine, démarrer le serveur local existant :

```powershell
.\.venv\Scripts\python.exe scripts/serve_demo.py --results-only --port 8014
```

Puis, dans un autre terminal :

```powershell
Push-Location frontend
npm run build
npm run test:studio
node verify-legacy-studio.mjs
Pop-Location
.\.venv\Scripts\python.exe -m pytest tests/test_prediction_lab.py tests/test_science_results.py -q -p no:cacheprovider --basetemp=artifacts/ui-studio-v2/pytest-core-rerun --junitxml=reports/ui_studio_v2/backend_targeted_tests-rerun.xml
```

Les tests `test:studio` écrivent dans `reports/ui_studio_v2` (ou le répertoire désigné par `GEOFIREX_TEST_REPORT_DIR`). Le lanceur historique conserve ses scripts d'origine et redirige leurs sorties vers V2/`artifacts/ui-studio-v2/legacy`. Il ne réutilise pas la validation manuelle V1 comme preuve pour V2. Ses sept inférences ponctuelles restent sur des événements 2020.

Résultats de cette intervention : build réussi ; 29 contrôles d'intégration frontend ; 16 tests backend ciblés ; deux vérifications frontend historiques réussies ; 344 règles CSS de géométrie conservées ; 23 couples de contraste de tokens vérifiés. Les preuves sont dans `reports/ui_studio_v2/`.

Happy DOM utilise des adaptateurs explicites pour canvas, PNG, WebGL, dialogues, IntersectionObserver, géométrie et WAAPI. Les assertions contrôlent le code effectivement monté et les déclenchements ; elles **ne prouvent pas la perception visuelle**, le rendu GPU, la console Chrome, la géométrie responsive réelle ou le contraste de tous les pixels composites. L'audit axe est structurel sur les cinq pages, sans certification globale WCAG.

## Checklist Chrome à réaliser

L'outil navigateur a répondu « The browser access host is unavailable ». Aucune animation V2 n'a été vue par l'agent. Contrôles manuels à effectuer à **100 %**, après build et démarrage local :

1. Ouvrir `http://127.0.0.1:8014/#/overview`. Vérifier le fond ivoire, les panneaux blancs et l'absence de larges surfaces sombres hors rasters. Le titre entre une fois, l'Aurora chaude se déplace, les quatre KPI atteignent exactement leurs valeurs, le halo suit le pointeur. Si statique, consulter le bouton de mouvement et la préférence système ; ne pas désactiver une préférence d'accessibilité sans intention.
2. Faire défiler Overview, Big Data et Research : apparitions courtes à l'entrée du viewport, sans contenu durablement caché. Les barres Big Data croissent une fois ; ouvrir la répartition par split et comparer aux tableaux. Les étapes temporelles ne défilent pas seules.
3. Prediction Studio : choisir `2020/fire_24332880`, date J `2020-09-03`, cible `2020-09-04`. Tester dates précédente/suivante et recherche d'événement. Prédire : attente sans pourcentage, puis huit cartes, vérité séparée, légendes cohérentes. Aucun fond Aurora actif sur cette page.
4. Agrandir une carte, zoomer/défiler, fermer par bouton et Échap. Exporter un PNG de chaque type de couche, le JSON de provenance et le CSV ; vérifier dimensions et valeurs avec les mêmes données. Aucun lissage des pixels au zoom.
5. Analytics : ouvrir les quatre sous-onglets, vérifier leur entrée, les courbes/tableaux lisibles, le rappel 14,68 %, les notes sur les cartes rares et la tête comparative non retenue. Changer les événements/cartes sauvegardées. Aucun chiffre ne doit se transformer en promesse de supériorité.
6. Contrôler les modales navigation rapide, mobile, événements et cartes ; elles doivent être claires. Tester Ctrl/Cmd+K, recherche/Entrée, Tab/Shift+Tab, flèches des sous-onglets, retour de focus et Échap. Inspecter la console et l'onglet Réseau pour erreurs inattendues.
7. Refaire la navigation et les exports sur desktop, tablette ~768 px et mobile ~390 px. Les largeurs V1 doivent rester stables ; tableaux défilables, aucun bouton inaccessible, aucun débordement global. Tester l'accès au clavier et les focus visibles.
8. Activer le mode réduit via l'interface, puis via le système/émulation DevTools : titre et compteurs exacts immédiatement, pas d'Aurora animée, pas de reveal masquant le contenu, ni coche/glissement animé. L'inférence, les dialogues et exports restent utilisables. Recharger et vérifier la persistance du choix utilisateur.
9. Vérifier le contraste réel sur le hero animé, au survol des KPI, dans les tableaux et les dialogues. Les ratios calculés portent sur les tokens, pas sur une capture composite du navigateur.

La copie locale des sources V1 avant intervention est dans `artifacts/ui-studio-v2/before/`, ignorée par Git. Les rapports V1 et preuves scientifiques historiques sont conservés. Il appartient à l'utilisateur de réaliser ses commits et pushes.
