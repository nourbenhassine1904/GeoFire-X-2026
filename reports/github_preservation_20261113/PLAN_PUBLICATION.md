# Plan de publication soumis à accord — 13 novembre 2026

Branche unique : `feature/scientific-scaleup-v1`. Aucun commit ni push effectué.
Remote : `https://github.com/nourbenhassine1904/GeoFire-X-2026.git`.
Branche distante absente au contrôle ; fetch effectué, HEAD/origin/main identiques
à `7428561f607914eb2f7cd286ee50435d142f8e09` (0 devant / 0 derrière).

La liste exacte est `publication_files.txt`, les tailles et SHA-256 sont dans
`publication_manifest.csv`. Les exclusions récursives et leurs raisons sont dans
`excluded_files.csv`. `off_git_manifest.csv` conserve les empreintes des résultats
et données scientifiques à préserver hors Git. `checkpoint_members.json`
identifie les poids contenus dans les archives critiques.

Vérification : 29 tests ciblés réussis, compilation frontend réussie et analyse
syntaxique de 258 fichiers Python réussie. Les avertissements Rasterio sont des
dépréciations existantes. Pas d'entraînement, transfert Kaggle ou refonte.

Risques : les poids et données hors Git restent locaux tant qu'une copie externe
vérifiée n'est pas réalisée. Un clone exige leur restauration (voir
`docs/GITHUB_PRESERVATION_20261113.md`). Les références distantes doivent être
revérifiées avant publication ; aucun push forcé ni écriture sur main.
Les chemins absolus historiques et le compte Kaggle dans les preuves décrivent
la provenance, sans identifiant secret. Le contrôle de motifs ne couvre pas
l'interprétation de toutes les charges binaires ni l'historique Git antérieur.

Message proposé :
`feat: preserve scientific scale-up, verified results and Prediction Lab`

Commandes exactes (avec gardes et sélection explicite) :
`COMMANDES_APRES_ACCORD.ps1`. Ce fichier est fourni pour revue, pas exécuté.
L'autorisation explicite du commit et du push est encore requise.
