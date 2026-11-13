# Vérification ZIP v2 — correction du 3 novembre 2026

Archive en lecture seule : `D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip`.
Pas de téléchargement, pas de réparation du ZIP, pas d'extraction automatique.

## Diagnostic et degré de certitude

Le reçu de téléchargement et le MD5 recalculé indépendamment par l'utilisateur
concordent : 48 359 369 821 octets,
`dc1a04e63ccc70037b277d585b8fe761`. Le reçu conserve aussi le SHA-256
`25bbbf1d71e316610c552ef611a155b69869606ec41e5b3fbec093ceba95538c`.
Ces empreintes sont des preuves antérieures ; le diagnostic par membre ne
prétend pas avoir recalculé une nouvelle fois ce MD5 global.

Python installé : CPython3.11.6, 64 bits. Le répertoire central est lisible :
14 218 membres, 13 607 TIFF, 611 répertoires. Aucun offset négatif ou hors archive.
Les membres proches de 2, 4, 8, 16, 32 et 48 Go ont été lus intégralement et passent
leur CRC, avec et sans tampon Python. Une limite générale ZIP64/32 bits n'est
donc pas reproduite.

Windows associe D: au disque1, ADATA HV300, busUSB. Le journal système révèle
1 149 événements disqueID51 sur Harddisk1 pendant les dernières24 heures,
dont une série le 3 novembre à08:41:38 UTC (09:41:38 à Tunis).
Le disque est actuellement Online/Healthy ; ce statut n'efface pas les erreurs
d'E/S précédentes. La documentation Microsoft explique que ces événements
peuvent concerner des lectures de fichiers en E/S tamponnées, pas uniquement un
fichier d'échange :
[Event ID51](https://learn.microsoft.com/en-us/troubleshoot/windows-server/backup-and-storage/event-id-51-information).

**Conclusion étayée : incident d'E/S intermittent sur le chemin disque/USB
probable ; corruption du corpus non démontrée.** Le composant physique précis
(câble, port, boîtier/contrôleur, alimentation, disque ou pilote) n'est pas
identifié par ces seules preuves. Le rapport diagnostique complet indique si
l'exception initiale a été reproduite ; ne pas confondre hypothèse et certitude.

Le `testzip()` de Python3.11.6 ne capture que `BadZipFile` ; une `OSError` remonte
sans journal du membre courant. La [documentation Python](https://docs.python.org/3.11/library/zipfile.html#zipfile.ZipFile.testzip)
décrit le contrôle CRC, mais un `Errno22` isolé n'identifie pas sa cause physique.
Le défaut applicatif corrigé est l'absence de journal, de reprise et de diagnostic
par membre, pas une suppression du contrôle d'intégrité.

## Comportement corrigé

- `archive_verification.py` : MD5/SHA-256 globaux, structure/index/offsets,
  ouverture par `ZipInfo`, lecture par blocs de1 Mio, CRC32 explicite et taille
  décompressée comparée pour **tous** les membres, répertoires compris.
- Lecture sans tampon Python par défaut ; `--io-mode buffered` disponible pour
  le diagnostic. `buffering=0` ne désactive ni le cache Windows ni la pile USB et
  ne constitue pas une réparation du matériel.
- Avant chaque membre : journal durable de son nom, ordinal et offset. Après EOF
  et comparaison : résultat CRC durable. Une erreur conserve phase, type,
  errno/winerror et traceback. Le membre interrompu n'est jamais validé.
- Reprise à partir du premier membre non certifié, en relançant la même commande.
  Aucun état de décompression partiel n'est présenté comme un CRC complet.
- MD5 complet effectué une fois par journal, puis réutilisé seulement si chemin,
  taille, inode/périphérique, mtime/ctime et index restent identiques. Un changement
  refuse la réutilisation des anciennes preuves. Une interruption pendant le MD5
  nécessite de refaire ce premier passage ; aucun hash incomplet n'est certifié.
- Journaux et résumés nouveaux par tentative ; pas d'écrasement des anciens.
  Une dernière ligne tronquée/non terminée n'est pas acceptée comme preuve.
- Résultats séparés : `md5_status`, `structure_status`,
  `crc_verified_members`, `crc_unverified_members`, `crc_verified_tiffs`,
  `crc_unverified_tiffs`, `errors_encountered`, `unresolved_member_errors`,
  `last_member_started`, `last_member_verified`, `all_crc_checked` et `status`.
- Une erreur, une interruption ou un passage borné partiel renvoie un code
  non nul. Une erreur d'E/S passée reste visible même si une reprise vérifie
  ensuite réellement le membre. `--verify-limit N` est un diagnostic partiel,
  jamais une réussite du corpus.
- L'extraction exige maintenant la preuve CRC complète **avant de créer son
  dossier de rasters**. Elle réutilise le même journal et lit aussi le ZIP sans
  tampon Python. Les règles existantes CRC/SHA, `.part`, reprise et refus
  d'écrasement des rasters restent appliquées.

## Commandes exactes, depuis PowerShell

Les journaux sont placés sur C: pour rester accessibles même si D: rencontre
un problème USB. La première vérification relit localement48,36 Go pour les
empreintes, puis les membres pour leurs CRC ; aucune donnée réseau.

```powershell
Set-Location -LiteralPath 'C:\Users\nourb\Projects\GeoFire-X-2026'

.\.venv\Scripts\python.exe scripts/full_corpus_external.py verify --archive 'D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip' --verification-dir 'reports/full_zip_verify_v2'

if ($LASTEXITCODE -ne 0) { throw 'Vérification incomplète ou en échec : extraction bloquée. Consulter le résumé JSON.' }

# Uniquement après status=VERIFIED et all_crc_checked=true :
.\.venv\Scripts\python.exe scripts/full_corpus_external.py extract --archive 'D:/GeoFire-X/WildfireSpreadTS/archive/WildfireSpreadTS.zip' --destination 'D:/GeoFire-X/WildfireSpreadTS/rasters' --verification-dir 'reports/full_zip_verify_v2' --execute
```

Pour reprendre après Ctrl+C, relancer la commande `verify` inchangée. Codes :
0 = vérification complète ; 2 = erreur ou vérification partielle ; 130 = arrêt
demandé pendant la vérification. Le JSON fournit le chemin du résumé et le
dernier membre commencé/vérifié. Ne supprimer aucun journal pour contourner
un résultat négatif. Si le disque s'interrompt encore, conserver le journal,
stabiliser la connexion USB, puis reprendre ; aucun retéléchargement automatique.

## Preuves et limites

- `reports/zip_io_reproduction_20261103.jsonl` : reproduction réelle, chaque
  membre contrôlé et éventuelle erreur avec traceback.
- `reports/zip_offset_probes_20261103.json` : sondes64 bits réelles.
- `reports/zip_windows_storage_events_20261103.json` : événementsWindows ciblés.
- `tests/test_archive_verification.py` : erreurs22 à l'ouverture/lecture,
  CRC divergent, reprise sans relecture des membres certifiés, interruption,
  cacheMD5, journal tronqué, archive/index changé, mémoire bornée, code de sortie
  partiel et extraction bloquée sans preuve.
- `reports/archive_verification_tests_delivery_20261103.xml` : tests ciblés.

Les CRC des données2021 sont des contrôles d'acquisition, sans métrique modèle,
entraînement, tuning ou sélection. Aucun GPU, commit ou push n'est lancé.
