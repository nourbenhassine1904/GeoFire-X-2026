# GeoFire-X Studio UI V1

Branche de travail : `feature/geofire-studio-ui-v1`, base locale `39ed50c`.
Cette phase modifie la présentation et les contrôles du frontend. Elle ne modifie
aucune route FastAPI, architecture de modèle, configuration scientifique,
checkpoint, normaliseur, sortie enregistrée ou formule d'export.

## Parcours

| Espace | Adresse locale | Fonction |
| --- | --- | --- |
| Overview | `/#/overview` | Carte de segmentation enregistrée réelle, inventaires et métriques sourcés depuis les API |
| Prediction Studio | `/#/studio` | Catalogue 2020, recherche, navigation par date, inférence réelle puis vérité séparée, cartes et exports |
| Model Analytics | `/#/analytics` | U-Net poids 10, ablations, résultats historiques CNN/U-Net, propagation et schémas fonctionnels |
| Big Data Observatory | `/#/data` | Inventaire, fenêtres, qualité, partitions, volumes, requêtes SQL et agrégats Parquet déjà calculés |
| Research | `/#/research` | Contrats temporels, protocole, limites, liens vers les preuves et export de provenance |

Les URL utilisent un fragment, pour permettre le rechargement et les liens
directs avec le serveur statique FastAPI existant, sans nouvelle règle serveur.
La navigation rapide fonctionne avec Ctrl/Cmd+K. Les onglets d'analyse acceptent
les flèches, Home et End. La navigation mobile utilise un dialogue natif.
Le Predictor TIFF historique reste accessible depuis Studio lorsque `/health`
indique qu'un snapshot compatible est disponible.

## Sources scientifiques

| Affichage | Source existante |
| --- | --- |
| Inventaire officiel, observations, SQL, Parquet | `/science/big-data` |
| 250 train, 201 validation, 11 partitions, volume compact | `/lab/lineage` |
| Historique CNN/U-Net, époques et mises à jour | `/science/summary` |
| Poids 10, seuil 0,78, confirmation, cartes sauvegardées | `/science/unet-ablation`, `/science/unet-assets/...` |
| Expérience spécialisée non retenue | `/science/propagation-comparison` |
| Diagnostic de nouveaux pixels du modèle principal | `/science/propagation` |
| Prévision à la demande et vérité lue ensuite | `/lab/predict`, puis `/lab/ground-truth` |

Les cartes d'accueil sont explicitement des prédictions **enregistrées**.
L'ouverture des pages de résultats ne lance pas de calcul d'inférence agrégé.
La grille compacte n'a pas de CRS disponible : aucun fond géographique,
MapLibre, hectare, météo, pas horaire ou intervalle d'incertitude n'a été ajouté.
Les palettes, masques, tableaux, conventions TP/FP/FN, CSV et JSON existants
restent inchangés. Les résultats du panel 169 sont déjà observés et ne sont pas
présentés comme un nouveau test indépendant.

## Architecture et effets

React et React DOM restent en **19.3.1**, avec esbuild **0.28.2**, `.mjs` et CSS
classique. Les sources React Bits `.jsx` sont compilées directement par esbuild.
Elles possèdent une liaison React explicite compatible avec sa transformation
JSX classique. Aucun Vite, TypeScript, Tailwind ou shadcn/ui n'est introduit.

- `StudioApp.mjs` : coque, fragments d'URL, navigation, dialogues, préférences.
- `StudioPages.mjs` : accueil, analyse, recherche et observatoire.
- `studio_ui.mjs` : composants partagés, lecture des sources, effets adaptés.
- `EventForecast.mjs` : interface enrichie autour du même `requestForecast` et des
  mêmes `ForecastView`, `imagePixels`, `comparisonLayers` et `forecastCSV`.
- `style.css` : surfaces graphite, bordures fines, accent violet, typographie,
  tables défilantes, grilles et points de rupture 540/800/1000/1250 px.

Les variantes JavaScript/CSS de [React Bits](https://reactbits.dev/) sont
épinglées au commit `4e0d091d3b1ea8b7aadef33716a8d99d164ae4ce` : Aurora,
SplitText, SpotlightCard, CountUp, FadeContent, Stepper et AnimatedList.
Les dépendances associées sont verrouillées dans `package-lock.json` : GSAP,
`@gsap/react`, Motion et Hugeicons. Les tests DOM utilisent Happy DOM et axe-core
en dépendances de développement. La [référence Linear](https://getdesign.md/linear.app/design-md)
inspire la hiérarchie et les surfaces ; aucune marque ou page Linear n'est copiée.

La licence amont, son copyright et la provenance des fichiers sont conservés
dans `frontend/src/reactbits/`. Le build inclut aussi
`dist/react-bits-LICENSE.txt`. Les empreintes de source locale dans la provenance
sont calculées après normalisation des fins de ligne LF.

Les effets décoratifs sont limités à l'accueil. Aurora est plafonné à DPR 1,
sans suivi de souris ; il se démonte en quittant l'accueil. La préférence système
`prefers-reduced-motion` est prioritaire et réagit aux changements. Une commande
supplémentaire permet à l'utilisateur de réduire les animations, avec préférence
locale persistante. Dans ce mode, les titres, compteurs et sections restent
immédiatement lisibles, sans instancier leurs animations. Les valeurs accessibles
des statistiques sont fixes ; l'animation du compteur est décorative.

## Relancer et vérifier

Les poids et données restent locaux, aux chemins prévus dans le contrat du Lab.
Consulter la [procédure de restauration](GITHUB_PRESERVATION_20261113.md) sur une
autre machine. Depuis la racine :

```powershell
Push-Location frontend
npm ci
npm run build
Pop-Location
.\.venv\Scripts\python.exe scripts/serve_demo.py --results-only --port 8014
```

Puis, dans un autre terminal, sans entraînement ni Spark :

```powershell
Push-Location frontend
npm run test:studio
Pop-Location
.\.venv\Scripts\python.exe -m pytest tests/test_prediction_lab.py tests/test_science_results.py tests/test_unet_confirmation.py -q
```

`test:studio` monte réellement les composants React dans Happy DOM et appelle
l'API locale réelle. Il contrôle la navigation, les sous-onglets, le catalogue,
les dates, une prédiction, l'ordre prédiction/vérité, les huit cartes, les données
PNG/JSON/CSV, l'agrandissement, le zoom, la navigation mobile, le clavier,
les erreurs/réessais et les préférences d'animation. Les API natives de dialogue
et de canvas sont adaptées pour le simulateur ; les RGBA réellement fournis par
le composant sont encodés en PNG pour vérifier dimensions et signature.
Les rapports sont écrits dans `reports/ui_studio_v1/` ; les anciens rapports de
vérification ne sont pas écrasés.

Les contrôles structurels axe-core sont séparés d'une validation visuelle :
**un simulateur DOM ne vérifie ni la géométrie responsive ni les contrastes
réellement rendus**. Le connecteur navigateur était indisponible lors de cette
mission et l'accès natif à Chrome a expiré. Aucun screenshot ou contrôle visuel
de navigateur n'est donc présenté comme réalisé.

À vérifier dans un navigateur réel avant validation finale :

1. Overview à 1440 px, puis 390 et 768 px : lisibilité, grille et absence de
   débordement ; menu mobile et palette Ctrl/Cmd+K.
2. Studio : `2020/fire_24332880`, date J `2020-09-03` ; huit cartes après calcul,
   agrandir/zoom/Échap et exports PNG, JSON et CSV.
3. Les quatre onglets Analytics, les requêtes et tables de l'observatoire,
   l'export de provenance Research.
4. Clavier seul, focus, réduction des animations système et commande locale.

Aucun commit, push, merge, reset destructif ou nettoyage Git n'est exécuté dans
cette phase. Les commits et publications restent à la charge de l'utilisateur.
