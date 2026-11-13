# WildfireSpreadTS complet sur ADATA D: — exécution autonome

État du 2 novembre 2026 : acquisition C: arrêtée, aucun redémarrage automatique.
Le processus est terminé et son verrou libéré. Un marqueur `STOP` protège l'ancienne
session. Les GeoTIFF et `.part` sur C: restent en place.

## Progrès conservés sur C:

- 206 nouveaux TIFF, 208 déjà présents, soit **414 TIFF valides au CRC ZIP**.
- SHA-256 calculés dans l'inventaire ; 18 événements complets ; 4 sous-lots certifiés.
- Un `.part` de zéro octet conservé ; 1 360 observations manquent encore au palier 80.
- Corps HTTP confirmés : **839 458 790 octets**. Réservations cumulatives, retries
  compris : **839 496 472 octets** sur 6 611 516 695.
- Solde de l'ancien plafond : 5 772 020 223 octets, **non transféré au nouveau projet**.
- Sauvegarde des journaux/certificats et inventaire :
  `reports/acquisition_stopped_20261102_v2/`.

L'interruption console a terminé le processus ; le rapport de fin automatique
n'a pas été émis. L'inventaire d'arrêt indépendant capture le dernier checkpoint
durable, vérifie les fichiers et sauvegarde les journaux sans les réécrire.

## Choix recommandé

D: est l'**ADATA HV300**, NTFS, capacité 1 000 187 359 232 octets ;
espace libre mesuré : **536 110 944 256 octets**.

| Solution | Charge utile nouvelle | Disque D: | Reprise | Limite |
|---|---:|---:|---|---|
| ZIP officiel intégral, recommandé | **48 359 369 821 octets** | ZIP + TIFF + réserve = **117 653 605 544 octets** | À l'octet du `.zip.part`, même journal et plafond | Les fichiers déjà sur C: sont aussi présents dans le ZIP ; C: reste intact |
| HTTP Range, corpus entièrement sur D: | Somme des charges TIFF compressées de l'index, proche de 48,36 Go | TIFF ~49,29 Go + réserve ; enveloppe du script ~118,59 Go avec staging | Par fichier ; lots par événement ; `.part` conservés | Beaucoup de petites requêtes, lenteur observée pendant la session C: |
| Range avec réutilisation des trois racines C: | **46 736 191 861 octets** au constat d'arrêt | Seulement les nouveaux TIFF sur D: | Même mécanisme, CRC des sources C: avant réutilisation | Corpus complet réparti sur C:+D:, pas entièrement présent physiquement sur D: |

Le ZIP est recommandé ici parce que D: offre largement la place pour conserver
archive et rasters et qu'un flux continu évite les requêtes par entrée. Son débit
réel n'a pas été mesuré dans cette intervention. Les deux commandes sont
indépendantes de Codex et affichent leur progression dans votre terminal.

**Budget proposé : 55 000 000 000 octets supplémentaires**, commun aux reprises
d'une même méthode/session. Ce n'est pas une autorisation consommée dans Codex.
Le programme bloque au plafond des lectures de corps HTTP ; il réserve avant
chaque lecture, conserve les réservations des erreurs et refuse une session dont
le budget a changé. TCP/TLS et en-têtes réseau sont hors de cette mesure.
Les anciennes consommations restent comptabilisées séparément : plafond combiné
ancien réservé + nouveau proposé = 55 839 496 472 octets.

Le script natif `curl.exe -C -` sait reprendre, mais `--max-filesize` limite la
taille du fichier, pas un cumul durable de retransmissions. Les commandes ci-dessous
utilisent donc un compteur persistant. Voir la [documentation curl](https://curl.se/docs/manpage.html).

## Commandes PowerShell — ZIP recommandé

Ouvrir un terminal PowerShell normal, depuis votre session Windows. Le terminal
reste autonome ; Codex peut être fermé. Conserver les mêmes chemins à la reprise.
Les dossiers D: sont créés par le programme lors de votre lancement, pas par Codex.

```powershell
Set-Location -LiteralPath 'C:\Users\nourb\Projects\GeoFire-X-2026'

# Télécharger / reprendre le ZIP officiel, plafond cumulatif nouveau de 55 Go.
.\.venv\Scripts\python.exe scripts/full_corpus_external.py zip --archive 'D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip' --session-dir 'D:/GeoFire-X/WildfireSpreadTS/logs/zip-v1' --budget-bytes 55000000000 --max-mbps 40 --execute

# Vérification locale : taille, MD5 officiel, SHA-256, index et CRC de tout le ZIP.
# Aucun réseau ; peut prendre plusieurs minutes sur disque externe.
.\.venv\Scripts\python.exe scripts/full_corpus_external.py verify --archive 'D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip'

# Extraction locale reprenable, CRC/SHA par TIFF, sans écrasement des fichiers valides.
# Aucun réseau ; .part conservés si interruption.
.\.venv\Scripts\python.exe scripts/full_corpus_external.py extract --archive 'D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip' --destination 'D:/GeoFire-X/WildfireSpreadTS/rasters' --execute
```

Après chaque commande, vérifier `$LASTEXITCODE` (0 = réussite) avant de passer à
la suivante. Le téléchargement vérifie déjà le MD5 avant de renommer le `.part`
en `.zip`. La commande `verify` ajoute le contrôle complet des CRC internes ;
`extract` recontrôle l'archive puis les fichiers extraits, et garde un certificat
`rasters/.extraction-session/complete-*.json`.

MD5 officiel attendu : `dc1a04e63ccc70037b277d585b8fe761`.
Source : [Zenodo WildfireSpreadTS v1.0](https://zenodo.org/records/8006177).
Les 607 événements / 13 607 TIFF sont contrôlés contre l'index épinglé du dépôt.

Pour interrompre le téléchargement : **Ctrl+C une fois**, puis attendre le
message de fin. Le programme finit le bloc courant et garde le préfixe `.part`.
Relancer **exactement la même commande ZIP** pour reprendre ; ne pas créer une
autre session afin de remettre le compteur à zéro. Les erreurs critiques,
un MD5 divergent, un serveur ignorant la reprise, un disque trop plein ou un
plafond épuisé bloquent le processus et préservent les fichiers.

Une alternative d'arrêt depuis un autre terminal est de créer le marqueur :

```powershell
New-Item -ItemType File -Path 'D:/GeoFire-X/WildfireSpreadTS/logs/zip-v1/STOP'
```

Le programme s'arrête au prochain bloc sûr. Le marqueur doit être retiré
explicitement par l'opérateur pour reprendre. Ne retirer aucun journal ou `.part`.
Éviter de débrancher D: avant la fin effective du processus et l'éjection Windows.

## Alternative HTTP Range — choisir à la place du ZIP

Ne pas lancer les deux méthodes en même temps. Cette session a elle aussi un
budget nouveau de 55 Go, pas un crédit supplémentaire automatique.

```powershell
Set-Location -LiteralPath 'C:\Users\nourb\Projects\GeoFire-X-2026'

# Tout le corpus physiquement sur D:, sans toucher C:.
.\.venv\Scripts\python.exe scripts/full_corpus_external.py range --destination 'D:/GeoFire-X/WildfireSpreadTS/rasters' --session-dir 'D:/GeoFire-X/WildfireSpreadTS/logs/range-v1' --budget-bytes 55000000000 --max-mbps 40 --execute
```

Pour réutiliser C: sans recopier ses fichiers, ajouter **dès le premier lancement** :

```powershell
--reuse-root 'C:/Users/nourb/Projects/GeoFire-X-2026/data/raw/WildfireSpreadTS/real_sample' --reuse-root 'C:/Users/nourb/Projects/GeoFire-X-2026/data/raw/WildfireSpreadTS/corpus_v1' --reuse-root 'C:/Users/nourb/Projects/GeoFire-X-2026/data/raw/WildfireSpreadTS/scientific_corpus_v1'
```

Dans ce cas, le manifeste futur doit référencer C:+D:. Aucune copie/migration de
C: vers D: n'est exécutée ou proposée implicitement. Reprendre avec les mêmes
options conserve le même compteur et ignore les TIFF valides. Ctrl+C demande
un arrêt après le fichier courant ; `logs/range-v1/STOP` bloque aussi le départ
du fichier suivant. Un certificat complet est produit à la fin et les certificats
par sous-lot restent disponibles en cas d'arrêt intermédiaire.

## Séparation scientifique

Le mode corpus complet autorise uniquement l'acquisition et le contrôle des octets
2021. Il ne produit aucune métrique et n'entraîne aucun modèle. Le fold 0 reste
train2018–2019 / validation2020 / test2021 verrouillé pour entraînement, tuning
et sélection. Préparer ensuite un manifeste géospatial certifié pour le DataLoader.

Validation de cette préparation : 14 tests de transport simulé et d'extraction de
fixtures réussis (`reports/external_acquisition_tests_delivery.xml`). Aucun ZIP
complet réel, téléchargement sur D: ou copie des TIFF n'a été lancé par Codex.
