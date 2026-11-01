# Publication GitHub du suivi 3 — staging autorisé, commit en attente

Branche locale : `feature/unet-baseline`, commit de départ `e8ff4b0`.
`main` local (`620f8a1`) est ancêtre de cette branche ; quatre commits existants la séparent
de `main`. Les modifications non commitées couvrent aussi les sessions précédentes.
**Staging autorisé uniquement pour la liste validée ; aucun commit, merge ou push autorisé.**
Les références distantes n'ont pas
été actualisées : recontrôler l'état distant après accord, avant intégration.

## Périmètre concret

La liste exacte à examiner et publier est `reports/suivi3_publish_paths.txt` :

- Frontend React source + package/lockfile, serveur et inférence FastAPI.
- B1 CNN versionné, variantes déterministes B2/B3, runner GPU, correction des chemins
  de stockage et dépendances corpus/persistance déjà modifiées.
- Scripts de lancement, restauration locale MLflow, figures et préparation du launch record.
- Configurations/notebooks reproductibles, tests concernés, documents et résultats sélectionnés.
- Les fichiers déjà modifiés de la couche cloud sont inclus explicitement : ce travail antérieur
  n'est pas présenté comme une nouvelle acquisition ou une nouvelle autorisation de calcul.

`reports/suivi3_deferred_paths.txt` liste les changements locaux **hors de cette publication** :
diagnostics temporaires, plans/logs anciens, superviseurs et récupération prototype des sorties
annulées. Ils restent sur disque. Leur publication éventuelle fera l'objet d'un examen séparé.

Ne jamais utiliser `git add .` pour ce lot. Les candidats ont été contrôlés pour signatures
de secrets et volumes : aucune signature évidente et aucun fichier >10 Mio. Ce contrôle
n'est pas une garantie exhaustive ; relire le diff avant commit.

Exclus du dépôt : `.venv/`, `node_modules/`, `frontend/dist/`, TIFF, checkpoints, archives,
SQLite/MLflow, `.aws/`, `.kaggle/`, identifiants Kaggle et `.env`. Les résultats numériques,
figures légères et manifestes scientifiques sont versionnables. Les artefacts ignorés restent
indispensables à la démonstration : GitHub seul ne les sauvegarde pas.

## Staging validé et publication soumise à accord

Depuis la racine, après examen du périmètre :

```powershell
git switch feature/unet-baseline
git add --pathspec-from-file=reports/suivi3_publish_paths.txt
git diff --cached --check
git diff --cached --stat
git diff --cached
# Seulement après revue et autorisation explicite :
git commit -m "feat: finalize certified GPU pilot demo and restored MLflow"
git push origin feature/unet-baseline
```

Créer une PR vers `main`, titre **Finalize verified GPU pilot demo and restored MLflow**.
Description proposée :

> Rend la démonstration J+1 utilisable avec le checkpoint B1 GPU existant et des TIFF réels,
> avec vérité terrain séparée et test 2021 verrouillé. Restaure les quatre runs MLflow en local,
> leurs artefacts et des métriques sélectionnées explicites, sans modifier les sauvegardes.
> Inclut tableaux, figures et guide du suivi 3. Validation : 48 tests sans entraînement,
> installation hors ligne, rebuild React et HTTP réel de l'application/MLflow.
> Parcours React/FastAPI et MLflow validé visuellement dans Chrome par l'utilisateur.

Après push autorisé, comparer :
https://github.com/nourbenhassine1904/GeoFire-X-2026/compare/main...feature/unet-baseline

Après revue et autorisation de merge, depuis un arbre propre (les fichiers différés restent
à préserver ; ne pas les effacer pour rendre l'arbre propre), actualiser les références,
recontrôler la divergence et faire merger la PR par GitHub. Préférer la PR au merge local
tant que le travail différé reste dans cet arbre. Ne pas forcer de push, rebase ou reset.
L'absence de conflit n'est établie que pour les références locales actuelles.
