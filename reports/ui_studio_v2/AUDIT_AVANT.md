# Audit avant Studio V2

Branche contrôlée : `feature/geofire-studio-ui-v1`, HEAD `39ed50c`. Changements V1 conservés ; copie des sources avant intervention dans `artifacts/ui-studio-v2/before/` (ignorée par Git), empreintes dans `frontend_before.json`.

Les sept composants sont du code React Bits officiel, commit `4e0d091d3b1ea8b7aadef33716a8d99d164ae4ce`, provenance et licence présentes. React 19, esbuild et CSS natif sont compatibles ; aucune migration nécessaire.

| Composant | Montage V1 constaté dans le code | Limite identifiée |
|---|---|---|
| Aurora | `studio_ui.mjs` → Overview, seulement si animations autorisées | Couleurs froides, luminosité 0,6 et deux opacités cumulées ; absence de mode clair. Aucun rendu sans WebGL2. |
| SplitText | `studio_ui.mjs` → titre Overview | Vrai GSAP/SplitText, déclencheur `load` valide, attend les polices. Désactivé en reduced-motion. |
| CountUp | `studio_ui.mjs` → quatre KPI lorsque les données arrivent | Vrai Motion, entrée viewport ; nombres accessibles exacts. Désactivé en reduced-motion. |
| SpotlightCard | `studio_ui.mjs` → quatre KPI | Vrai suivi pointeur/focus ; intensité 0,1 et palette violette peu perceptibles. |
| FadeContent | `studio_ui.mjs` → sections | `trigger='load'` n'est pas une valeur reconnue par cette version : retombe dans ScrollTrigger. Employer explicitement `scroll` ou `mount`. Pas de transition des sous-onglets. |
| Stepper | `EventForecast.mjs` → étapes réelles | Vrai composant CSS/WAAPI ; thème sombre par défaut. Pas de pourcentage inventé. |
| AnimatedList | `EventForecast.mjs` → dialogue événements | Vrai composant WAAPI/IntersectionObserver ; monté même dialogue fermé, thème sombre par défaut. |

La préférence système et `localStorage['geofirex-reduced-motion'] === 'true'` neutralisent les effets. Le défaut utilisateur est animé, sans outrepasser le système. Il faut vérifier cette préférence dans Chrome avant de conclure à un défaut de montage.

Les imports CSS sont bien regroupés dans `dist/app.css`, chargé avant `style.css`. Les dépendances GSAP, Motion, Hugeicons sont déjà installées. Aucun composant vide ou substitut factice constaté. Les cartes sont de vrais canvas et PNG scientifiques ; leur rendu et leurs palettes doivent rester identiques.

Limite de l'audit : nouvel appel à l'outil navigateur le 13 novembre 2026, réponse « The browser access host is unavailable ». Aucune animation vue, aucune console Chrome inspectée. Les assertions DOM et les tests d'activation sont distincts de la validation visuelle manuelle.
