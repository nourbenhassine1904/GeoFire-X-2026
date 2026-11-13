# GeoFire-X — résultat de publication des lots 002–008

Arrêt obligatoire avant tout upload de rasters : `LOW_AVAILABLE_MEMORY` pendant le staging du lot 002. RAM disponible minimale : **2 123 952 128 octets**, sous la réserve de **2 147 483 648 octets**. RSS maximal de l'arbre de préparation : **25 665 536 octets** ; il s'agit d'un manque de réserve système, pas d'une limite RSS atteinte par le processus. Arrêt après 117,813 secondes, sans relance automatique.

Le fichier `artifacts/kaggle_remaining_20261108_v1/lot-002/payload.zip` est conservé, taille observée **4 995 525 361 octets**. Le reçu de staging et le SHA-256 final de l'archive n'ont pas été produits : ne pas traiter cette archive comme prête à publier. Une reprise devra d'abord la vérifier et la réutiliser si elle est conforme, sans reconstruction aveugle.

Aucun appel de publication du lot 002 n'a été lancé ; les lots 003–008 n'ont pas démarré. Les sept datasets n'ont donc pas été créés par cette exécution. Zéro octet de raster transféré. Le seul job Kaggle exécuté est le petit contrôle CPU privé sans dataset, terminé et vérifié (`CPU_AUTOMATION_READY`).

Corps HTTP supplémentaires de cette exécution autorisée : **1 905 octets**. Le cumul conservateur du journal inclut aussi **5 440 octets** de préparation antérieure, soit **7 345 octets** au total. Les profils GPU restent verrouillés en attente des sept certifications ; aucun profil ni entraînement n'est lancé.

État : **FAILED_STOP_NO_RETRY**.

| Lot | Rôle | Privé confirmé | Événements certifiés / prévus | GeoTIFF certifiés / prévus | SHA-256 / CRC32 | Corps HTTP réservés | État |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| 002 | train | non confirmé | 0 / 62 | 0 / 1380 | NOT_REMOTE_CERTIFIED | 0 | STARTED_STOPPED |
| 003 | train | non confirmé | 0 / 68 | 0 / 1344 | NOT_REMOTE_CERTIFIED | 0 | NOT_STARTED |
| 004 | train | non confirmé | 0 / 63 | 0 / 1205 | NOT_REMOTE_CERTIFIED | 0 | NOT_STARTED |
| 005 | val | non confirmé | 0 / 69 | 0 / 1305 | NOT_REMOTE_CERTIFIED | 0 | NOT_STARTED |
| 006 | val | non confirmé | 0 / 54 | 0 / 1350 | NOT_REMOTE_CERTIFIED | 0 | NOT_STARTED |
| 007 | val | non confirmé | 0 / 61 | 0 / 1321 | NOT_REMOTE_CERTIFIED | 0 | NOT_STARTED |
| 008 | val | non confirmé | 0 / 17 | 0 / 316 | NOT_REMOTE_CERTIFIED | 0 | NOT_STARTED |

Cumul du journal : **7,345 / 31 000 000 000 octets**. Les réservations ne sont pas une mesure physique du trafic.

Cause d'arrêt, le cas échéant : Stopped at lot-002/stage-monitor; inspect logs, no automatic retry

Le lot 001 n’a été ni reconstruit, ni réuploadé, ni recertifié. Aucun GPU, entraînement, commit, push ou merge.

Les identités des manifestes, SHA-256 des archives, tailles et preuves détaillées figurent dans deployment-report.json et les sous-répertoires de chaque lot.
