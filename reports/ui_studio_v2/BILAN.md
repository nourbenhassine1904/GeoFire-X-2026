# Studio V2 — bilan de livraison

Implémentation locale sur `feature/geofire-studio-ui-v1`. Thème clair « Flamme solaire », composants React Bits officiels animés selon les préférences, cinq pages et fonctions scientifiques conservées. Les dimensions V1 validées sont préservées. Aucun commit, push, merge, reset, clean ou staging.

## Validation

| Contrôle | Résultat |
|---|---|
| `npm run build` | Réussi ; cinq fichiers servis avec HTTP 200 et octets identiques au build |
| `npm run test:studio` | 29 contrôles d'intégration réussis ; navigation, clavier, erreurs/reprise, inférence réelle, huit cartes, dates, zoom, exports, animations et reduced-motion |
| Géométrie CSS | 344 règles V1 conservées, y compris breakpoints ; comparaison de déclarations, sans mesure de navigateur |
| Contraste | 23 couples de tokens conformes aux seuils testés (4,5:1 texte, 3:1 contrôles/focus) ; contraste composite du navigateur à vérifier |
| Accessibilité axe | Aucune violation parmi les règles structurelles testées sur les cinq pages ; pas de certification WCAG globale |
| Tests Python ciblés | 16 réussis : `tests/test_prediction_lab.py`, `tests/test_science_results.py` |
| Vérificateurs frontend historiques | Deux réussis via `node verify-legacy-studio.mjs`, sorties isolées ; sept inférences ponctuelles 2020, pas d'évaluation complète |
| Préservation | 63 fichiers scientifiques identiques par SHA-256 ; quatre contrats API identiques à V1 ; moteur `prediction_view.mjs` inchangé |
| Nouveaux entraînements / Spark / pixels test 2021 | Aucun |
| Navigateur réel | Indisponible : « The browser access host is unavailable ». Aucun aperçu V2 vu par l'agent |

Les tests d'animation exécutent les moteurs GSAP/Motion et contrôlent les couches Spotlight, les appels WebGL et WAAPI, avec adaptateurs de simulateur. Le nombre d'appels de dessin ne mesure pas les performances d'un navigateur. La fluidité, le rendu GPU, les contrastes composites et le responsive réel restent à valider dans Chrome.

Les premiers essais ont servi à compléter les adaptateurs Happy DOM (constructeur EventTarget de base pour Motion, géométrie hors viewport pour ScrollTrigger). Le dernier état des preuves ci-dessous est PASS ; le diagnostic intermédiaire est conservé dans `artifacts/ui-studio-v2/`, sans modifier les rapports V1.

## Fichiers de source et documentation

**17 fichiers modifiés par rapport à l'état local V1 :**

- `README.md`
- `frontend/index.html`
- `frontend/verify-studio.mjs`
- `frontend/src/EventForecast.mjs`
- `frontend/src/PropagationComparison.mjs`
- `frontend/src/ScienceResults.mjs`
- `frontend/src/StudioApp.mjs`
- `frontend/src/StudioPages.mjs`
- `frontend/src/UnetAblation.mjs`
- `frontend/src/studio_ui.mjs`
- `frontend/src/style.css`
- `frontend/src/reactbits/AnimatedList.jsx`
- `frontend/src/reactbits/Aurora.jsx`
- `frontend/src/reactbits/SplitText.jsx`
- `frontend/src/reactbits/Stepper.jsx`
- `frontend/src/reactbits/PROVENANCE.json`
- `frontend/tests/studio-dom.mjs`

**Quatre nouveaux fichiers de source/documentation :**

- `frontend/verify-legacy-studio.mjs` — relance isolée des tests existants sans écraser leurs preuves historiques.
- `frontend/tests/layout-v1.json` — référence de géométrie CSS V1.
- `frontend/tests/theme-contract.mjs` — contrastes des tokens et conservation de géométrie.
- `docs/UI_STUDIO_V2.md` — inventaire de chaque composant, déclencheur, source, dépendances, preuves et checklist Chrome précise.

Aucune dépendance ajoutée en V2. Les autres changements locaux V1 restent présents. Les sources avant intervention sont sauvegardées dans `artifacts/ui-studio-v2/before/`, ignoré par Git.

## Preuves légères nouvelles

- `AUDIT_AVANT.md` — audit des imports, montages et réglages V1, avant modification.
- `frontend_before.json` — empreintes des sources locales avant V2.
- `backend_targeted_tests.xml` — 16 tests Python.
- `frontend_dom_tests.json` — 29 contrôles et instrumentation d'animation.
- `accessibility_dom.json` — résultats axe par page.
- `theme_contract.json` — 344 règles de géométrie et ratios de contraste.
- `prediction_lab_smoke.json` — vérification du modèle réel, exports et six dates supplémentaires 2020.
- `propagation_component_check.json` — composant historique rendu avec cinq couches réelles.
- `verification_summary.json` — SHA-256, contrats API, build servi et inventaire détaillé des fichiers.

Le modèle reste U-Net poids 10, seuil 0,78. Le faible rappel des nouveaux pixels (14,68 %) est explicitement présenté comme une limite. Les cartes rares restent de vraies cartes ; aucune image artificielle ni géoréférence n'est ajoutée. La vérité terrain est demandée après la prédiction, par une requête distincte.

La vérification manuelle restante est décrite étape par étape dans [UI_STUDIO_V2.md](../../docs/UI_STUDIO_V2.md#checklist-chrome-à-réaliser) : ouvrir les cinq pages à 100 %, observer chaque animation, tester huit cartes/exports, tablette/mobile, clavier, console et mode réduit.
