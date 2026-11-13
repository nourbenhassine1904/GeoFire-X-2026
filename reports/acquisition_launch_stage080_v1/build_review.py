"""Render this launch review from the existing plan and measured preflight."""
import json
from pathlib import Path
import shutil
from geofirex.corpus import json_new,sha256_file

base=Path(__file__).resolve().parent
pre=json.loads((base/'preflight.json').read_text())
storage=json.loads((base/'storage_and_reuse.json').read_text())
execution=json.loads(Path('reports/scientific_scaleup_v4/execution.json').read_text())
sources=['scripts/acquire_scientific_stage.py','src/geofirex/acquisition_session.py',
         'configs/acquisition_stage080_v1.json','tests/test_acquisition_session.py']
pins={f'acquisition_{n}.json':sha256_file(f'reports/scientific_scaleup_v4/acquisition_{n}.json')
      ==execution['hashes'][f'acquisition_{n}.json'] for n in (80,200,451)}
assert all(pins.values())
disk=shutil.disk_usage('.')._asdict()
command=r'.\.venv\Scripts\python.exe scripts/acquire_scientific_stage.py --launch-config configs/acquisition_stage080_v1.json --execute'
summary=dict(status='READY_AWAITING_EXPLICIT_USER_APPROVAL',destination=pre['destination'],
             disk_now=disk,project_bytes_measured=storage['project_bytes_measured_before_preflight'],
             raw_directory_bytes=storage['raw_directory_bytes'],local_files_crc_sha_verified=208,
             reused_in_stage80=len(pre['reused']),new_files=pre['missing_files'],
             expected_observations=1768,expected_events=80,sub_batches=len(pre['batches']),
             max_batch_payload_bytes=max(b['compressed_payload_bytes'] for b in pre['batches']),
             payload_bytes=pre['compressed_payload_bytes'],new_geotiff_bytes=pre['additional_extracted_bytes'],
             write_allocation_bytes=12488063369,free_reserve_bytes=20000000000,
             maximum_additional_disk_with_reserve_bytes=32488063369,
             stage451_conservative_with_reserve_bytes=89184988138,
             stage451_headroom_bytes=disk['free']-89184988138,
             network_cap_bytes=6611516695,network_rate_target_mbps=20,concurrency=1,
             network_cap_scope='cumulative HTTP response bodies reserved including retries/ZIP metadata; TCP/TLS overhead excluded',
             network_started=False,exact_command=command,validated_v4_plans_unchanged=pins,
             source_sha256={p:sha256_file(p) for p in sources})
json_new(base/'launch_ready.json',summary)
text=f'''# Palier 80 : lancement prêt, autorisation explicite en attente

Destination persistante : `{pre['destination']}`. Même racine aux paliers200/451, sans copie ou redémarrage. Leurs sessions/budgets exigeront un nouvel accord.

Disque libre : **{disk['free']} octets** lors de cette mesure. Projet : 7 707 106 812 octets lors de l'inventaire initial. Dossier brut : 805 735 845 octets (real_sample : 447 927 526 ; corpus_v1 : 357 789 196). Voir storage_and_reuse.json pour les volumes et 208 vérifications CRC/SHA actuelles, concordantes avec les empreintes précédentes.

Stocker les451 événements est réaliste : 34 189 635 708 octets de TIFF supplémentaires ; enveloppe conservatrice de89 184 988 138 octets incluant une copie de staging et20 Go de réserve. Marge après cette enveloppe : {summary['stage451_headroom_bytes']} octets. Cache et scratch AP des futurs entraînements à mesurer séparément. Les607 événements ne sont pas couverts par cet accord ; aucun fichier2021 sélectionné.

Palier80 : **80 événements /1 768 observations**, 202 TIFF réutilisés, **1 566 nouveaux fichiers**. {summary['sub_batches']} sous-lots par événements, cible256 Mio compressés ; plus grand lot : {summary['max_batch_payload_bytes']} octets. Pas de ZIP entier local, pas de copie des fichiers valides des autres racines.

Charge compressée supplémentaire exacte :5 741 850 510 octets ; TIFF supplémentaires :5 853 750 530 octets. Allocation maximale de destination :12 488 063 369 octets, anciens.part compris. Réserve de20 000 000 000 octets libres contrôlée avant chaque fichier. Besoin conservateur maximal :**32 488 063 369 octets**, réserve comprise.

Plafond cumulatif :**6 611 516 695 octets** de corps HTTP réservés, retries et répertoire ZIP compris. Journal fsync avant chaque tentative, compteur conservé à la reprise, journal tronqué/divergent bloquant. TCP/TLS et en-têtes réseau ne sont pas mesurés par urllib. Une connexion active ; débit cible20 Mbit/s avec rafales de lecture bornées à4 Mio. Environ38 minutes de corps au débit cible, hors latence/erreurs/contrôles : pas de durée garantie.

Verrou OS contre lancements simultanés. TIFF validés ignorés au transfert, fichiers.part conservés. Un fichier interrompu reprend depuis son début dans un nouveau.part, avec les octets précédents toujours débités du plafond. Certificats CRC/SHA par lot, relecture SHA à la reprise et comparaison finale. Si disque ou budget deviennent insuffisants : arrêt ; jamais de nouvelle allocation automatique.

Tests exécutés :20 tests initiaux puis20 tests finaux ciblés. Fixtures réseau simulées uniquement ; aucun transfert réel. Plansv4 hashés et inchangés ; corpus.py historique inchangé. Le correctif concerne la persistance du budget entre exécutions, les lots, le verrou, la limite de débit et la garde disque permanente.

Commande exacte depuis la racine, **après accord seulement** :

```powershell
{command}
```

Reprise : exactement la même commande/configuration. Session :artifacts/scientific-acquisition/stage080-v1 ; ledgertransfer.jsonl, rapportsattempt-*.json distincts, certificatsbatch-*.json etcomplete-*.json. Aucun entraînement GPU.

Après acquisition : bilan réel fichiers/événements/CRC/SHA/octets confirmés et réservés, puis QC raster et manifeste multi-racines avant DataLoader. Pour200 puis451 : réutiliser les plans existants, dédupliquer tous les TIFF acquis valides et présenter uniquement un nouveau budget résiduel à approuver.
'''
with (base/'README.md').open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps({k:summary[k] for k in ('disk_now','sub_batches','max_batch_payload_bytes','stage451_headroom_bytes')},indent=2))
