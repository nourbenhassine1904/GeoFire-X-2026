# Publication controlee du 27 novembre 2026

Branche autorisee : `release/geofirex-jury-20261202`. PR #5 vers `main` :
<https://github.com/nourbenhassine1904/GeoFire-X-2026/pull/5>.
Publication de code et preuves legeres seulement ; aucune fusion vers main.
Le SHA publie est fourni dans le bilan de livraison et l'historique Git, pas
insere de maniere autoreferentielle dans son propre commit.

## Cloture et preuves

La [matrice CRISP-DM](CRISP_DM_CLOTURE_20261127.md) donne les objectifs,
entrees/sorties, composants, experiences, resultats, criteres et limites des
six phases. Les cinq phases scientifiques sont PASS dans le protocole fige.
Deployment reste PARTIAL : pas d'installation vierge certifiee et plein ecran
natif Chrome interactif encore a verifier. La publication n'efface pas ces reserves.

Recette de publication, executee sur la machine preparee :

- [Backend](../reports/publication_20261127/backend.xml) : 47 tests cibles API,
  CPU, causalite, parite, refus 2021, ADATA absent et exports.
- [Garde-fous 2021](../reports/publication_20261127/guards.xml) : 38 tests sur
  fixtures synthetiques/temporaires seulement, aucun pixel reel 2021.
- [Frontend](../reports/publication_20261127/frontend/propagation-dom.json) :
  44 controles React/HTTP avec calculs CPU reels, navigation et exports.
- [Acceptation](../reports/publication_20261127/acceptance.json) : build,
  reponses API, dates des preuves et limites.
- [Qualification](../reports/publication_20261127/qualification.json) :
  C8/NEWPOS2/NEWPOS4 PASS ; FIRE13 archive, tolerance initiale inchangee.
- [Evaluation en lecture seule](../reports/publication_20261127/science-readonly.json) :
  COMPLETE, identites et metriques exactes 2020/2021.
- [Big Data](../reports/publication_20261127/big-data.json) : lignes et SHA des
  cinq Parquet compares au rapport execute local[2], sans nouveau job Spark.
- [Environnement](../reports/publication_20261127/environment.json) et
  [guide Deployment](DEPLOYMENT_STUDIO_V2.md) : dependances, installation,
  artefacts externes, erreurs, sauvegarde et surveillance.
- [Lanceur](../reports/publication_20261127/launcher.json) : demarrage dedie
  sur 8031, catalogue reel et API, sans entrainement.
- [Preservation](../reports/publication_20261127/preservation.json) : comparaison
  SHA des fichiers proteges avant/apres, sans modification scientifique.
- [Selection explicite](../reports/publication_20261127/selection.json) et
  [audit Git](../reports/publication_20261127/git-content-audit.json) : volume,
  references et scan de secrets des fichiers candidats et deja suivis.

La [recette Chrome precedente](../reports/finalisation_20261127/browser/browser-audit.json)
reste datee et identifiee comme telle : cinq pages sur desktop/mobile, mouvement
normal/reduit, absence de debordement de page, flamme, cartes, zoom et vrais
telechargements. Les PNG complets restent dans la livraison locale ; ce commit
publie le recu leger, pas les profils Chrome ni toutes les captures. Elle ne
prouve pas le plein ecran natif, refuse en headless, ni une installation vierge.

Les [bilans precedents](FINALISATION_20261202.txt) sont des instantanes historiques
anterieurs a l'autorisation de push ; leurs mentions « publication non executee »
se rapportent a ces missions et non a la presente synchronisation.

## Controle du contenu

`scripts/audit_release_publication.py inventory` recense tous les fichiers
modifies/non suivis et le volume suivi dans un inventaire local non publie.
La selection est explicite : sources, tests, lanceur, dependances, documentation,
diagnostic de reference scelle et petites metadonnees requises, puis recus.
Les runners 2021 sont conserves comme sources d'audit verrouillees ; ils ne sont
ni importes par le service Studio ni executes par cette recette.

Les TIFF, checkpoints, archives ZIP, compacts, index binaires, environnements,
profils, telechargements et fichiers confidentiels restent hors de ce commit.
Tous les autres fichiers non suivis sont conserves localement, pas supprimes.
Le rapport LaTeX, les slides, `feu.png`, les normaliseurs et les archives scellees
ne sont pas reecrits. Les membres 2021 consultes sont uniquement JSON ; aucun
pixel 2021 n'est decode et aucune seconde evaluation n'est lancee.

Les sources scientifiques sont aussi figees en octets par leurs manifestes.
Certains blobs Git historiques avaient ete normalises LF alors que les
empreintes attendues de ces fichiers sont celles des originaux CRLF. Des regles
`-text` nominatives et une remise en Git des octets originaux corrigent ce point
de portabilite. Les fichiers locaux ne sont pas modifies ; aucun changement de
logique scientifique n'est inclus. L'audit compare les blobs prepares aux SHA
attendus et prouve que ces corrections ne changent que les fins de ligne.
Les attributs reconnaissent CR comme fin de ligne (`cr-at-eol`) tout en gardant
les controles `blank-at-eol`, `blank-at-eof` et `space-before-tab` : les octets
scelles CRLF ne doivent pas etre effaces pour satisfaire un lint prevu pour LF.

Le scan recherche des motifs de tokens, cles privees et URL avec identifiants ;
il ne garantit pas mathematiquement l'absence de toute information sensible.
Les chemins techniques locaux dans les configurations scientifiques font partie
de leur provenance figee ; ne pas les modifier pour cosmetique. L'inventaire
exhaustif local et les profils navigateur ne sont pas publies.

Avant commit : branche verifiee, fetch, absence de divergence, revue des diffs,
selection nominative sans `git add -A`, controle de l'index et de ses octets,
`git diff --cached --check`, puis commit descriptif et push normal. Aucun
force push, reset, rebase destructif, suppression ou merge n'est autorise.

## Actions avant fusion

1. Faire relire les changements et preuves de la PR #5 par un autre membre.
2. Essayer le plein ecran natif dans Chrome interactif.
3. Tester une installation restauree sur machine vierge avant toute affirmation
   de reproductibilite complete ; garder Deployment PARTIAL jusque-la.
4. Verifier l'acces aux copies ADATA/C: et a leurs empreintes, et le verrou durable.
5. Conserver FIRE13 archive tant que le critere original de parite echoue.
6. Decider separement la fusion dans main ; elle n'est pas realisee ici.
