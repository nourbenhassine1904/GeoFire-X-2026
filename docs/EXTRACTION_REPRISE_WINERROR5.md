# Reprise de l’extraction après WinError 5 — 3 novembre 2026

Le fichier `2018/fire_21890115/2018-07-09.tif.459a753a50264d4f9c9f6704007da800.part`
est complet : **4 328 884 octets**, CRC attendu et vérifié **6eb853c4**,
SHA-256 `fc84d1fe85e379bd6506f81bc03dde82b1f85688caa18925641ffa016db9c196`.
Le fichier final n’existe pas. Le chemin résolu reste dans la destination attendue.
Attribut du fichier : Archive (32), sans attribut lecture seule.

## Diagnostic et limites de la conclusion

Le code ferme le flux ZIP, le fichier de sortie et le lecteur de validation avant
`temporary.rename(output)` : aucun handle Python laissé ouvert n’a été identifié.
Il effectuait en revanche un seul renommage et ignorait les `.part` complets à la reprise.

Dans le contexte utilisateur habituel, Restart Manager ne signale aucun processus
utilisant ce `.part`. Une ouverture native demandant l’accès DELETE, immédiatement
refermée sans disposition de suppression, réussit. Un renommage Windows requiert cet
accès et peut être empêché par le mode de partage d’un autre handle, même avec des ACL
correctes : [CreateFileW, FILE_SHARE_DELETE](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).
Le renommage réel de ce fichier n’a pas été tenté pendant le diagnostic.

Un premier diagnostic dans le compte sandbox indiquait accès refusé sur D: ; il ne
représente pas les droits du terminal utilisateur. Le rapport à retenir est
`reports/extraction_publication_diagnostic-73892474cf3244db979135d52bb939bb.json`.
Aucun processus Python d’extraction n’était actif lors de la vérification.

Windows déclare Defender, McAfee et 360 Total Security. Defender indique « Not running » ;
aucun événement Defender 1123/1124 concernant GeoFire/python sur les deux derniers jours,
ni événement Application pertinent des fournisseurs concernés dans la fenêtre de
30 minutes autour du `.part`, n’a été trouvé. Cela ne permet pas d’exclure un filtre
antivirus tiers. Aucune protection ou exclusion n’a été modifiée.
Snapshot : `reports/extraction_security_snapshot-7293083013734323a867ad29c3c1ce69.json`.

**L’auteur précis du refus initial reste indéterminé.** Un verrou ou un refus transitoire
est compatible avec les observations, mais n’est pas établi. Aucune corruption n’a été
observée sur le `.part`. Un test synthétique reproduit un refus de renommage avec un
vrai handle Windows sans partage DELETE, puis valide la reprise après libération.

## Correction

- Recherche des seuls `.part` correspondant exactement au nom attendu et à un UUID
  de 32 caractères hexadécimaux ; validation taille/CRC avant toute réutilisation.
- Aucun contenu du ZIP n’est relu pour réextraire un `.part` valide.
- L’extraction complète utilise maintenant une création atomique de **lien physique**
  du `.part` validé vers le `.tif`. D: est NTFS, sain, avec 479 826 927 616 octets libres
  au diagnostic. Le `.part` reste présent ; les deux noms désignent les mêmes données,
  sans deuxième copie. Ce ne sont pas des sauvegardes indépendantes : ne modifier aucun
  des deux noms. [Fonction CreateHardLinkW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createhardlinkw).
- Aucun remplacement, suppression, changement de droits ou repli par copie.
  Si les liens physiques ne sont pas disponibles, l’opération échoue proprement.
- Cinq tentatives au maximum pour les erreurs Windows 5/32/33 ; attentes de
  0,25 / 0,5 / 1 / 2 secondes. Les autres erreurs sont immédiatement propagées.
- Une destination apparue entre-temps est acceptée uniquement après CRC et SHA-256
  conformes au temporaire. Toute divergence est fatale ; les deux fichiers restent.
- Contrôles des chemins résolus, des liens/reparse points de fichiers, de l’identité
  du temporaire avant publication et du contenu final après publication.
- Les nouveaux temporaires sont vidés et synchronisés sur disque avant publication.
- Chaque tentative utilise un nouveau journal append-only synchronisé dans
  `rasters/.extraction-session/publication-<UUID>.jsonl`, incluant le membre courant,
  les erreurs, les fichiers validés avec SHA-256, l’arrêt ou la fin. Aucun ancien
  journal n’est écrasé. Une reprise vérifie également les `.tif` déjà publiés.
- Le certificat MD5/CRC complet reste obligatoire avant extraction ; réserve de
  20 milliards d’octets libres contrôlée également avant chaque fichier.

Fichiers de code : `src/geofirex/corpus.py`, `src/geofirex/archive_transfer.py`.
Diagnostic reproductible : `scripts/diagnose_extraction_publication.py` (lecture seule).
Tests : `tests/test_extraction_publication.py` ; régressions corpus, archive et session.

Bilan final : **53 tests réussis, 1 ignoré** (création d’un lien symbolique interdite
au compte de test Windows). Le test avec un vrai handle Windows bloquant le
renommage a réussi, ainsi que la reprise après libération, les collisions de
destination, les CRC incorrects, les erreurs persistantes et la réutilisation sans
lecture des données du ZIP. Rapport :
`reports/extraction_publication_tests_final_20261103.xml`.

## Commande PowerShell de reprise

À exécuter dans votre terminal habituel. Cette commande vérifie d’abord le certificat
MD5/CRC lié à l’identité actuelle de l’archive dans le répertoire de preuves existant,
puis reprend les fichiers existants et les `.part` valides. Le certificat actuel
indique VERIFIED, 14 218 CRC vérifiés, zéro non vérifié ; les erreurs historiques
restent consignées. Aucun téléchargement n’est effectué.

```powershell
Set-Location 'C:\Users\nourb\Projects\GeoFire-X-2026'
.\.venv\Scripts\python.exe scripts/full_corpus_external.py extract --archive 'D:\GeoFire-X\WildfireSpreadTS\archive\WildfireSpreadTS.zip' --destination 'D:\GeoFire-X\WildfireSpreadTS\rasters' --verification-dir 'reports/full_zip_verify_v2' --execute
```

Pour refaire uniquement le diagnostic du fichier concerné, sans publier de fichier :

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_extraction_publication.py
```

L’extraction réelle n’a pas été relancée pendant cette correction. Aucun fichier du
corpus sur C: ou D: n’a été déplacé, supprimé ou modifié, aucun téléchargement,
entraînement, commit ou push n’a été lancé.
