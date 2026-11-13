# Palier 80 : lancement prêt, autorisation explicite en attente

Destination persistante : `C:\Users\nourb\Projects\GeoFire-X-2026\data\raw\WildfireSpreadTS\scientific_corpus_v1`. Même racine aux paliers200/451, sans copie ou redémarrage. Leurs sessions/budgets exigeront un nouvel accord.

Disque libre : **123333505024 octets** lors de cette mesure. Projet : 7 707 106 812 octets lors de l'inventaire initial. Dossier brut : 805 735 845 octets (real_sample : 447 927 526 ; corpus_v1 : 357 789 196). Voir storage_and_reuse.json pour les volumes et 208 vérifications CRC/SHA actuelles, concordantes avec les empreintes précédentes.

Stocker les451 événements est réaliste : 34 189 635 708 octets de TIFF supplémentaires ; enveloppe conservatrice de89 184 988 138 octets incluant une copie de staging et20 Go de réserve. Marge après cette enveloppe : 34148516886 octets. Cache et scratch AP des futurs entraînements à mesurer séparément. Les607 événements ne sont pas couverts par cet accord ; aucun fichier2021 sélectionné.

Palier80 : **80 événements /1 768 observations**, 202 TIFF réutilisés, **1 566 nouveaux fichiers**. 26 sous-lots par événements, cible256 Mio compressés ; plus grand lot : 266052121 octets. Pas de ZIP entier local, pas de copie des fichiers valides des autres racines.

Charge compressée supplémentaire exacte :5 741 850 510 octets ; TIFF supplémentaires :5 853 750 530 octets. Allocation maximale de destination :12 488 063 369 octets, anciens.part compris. Réserve de20 000 000 000 octets libres contrôlée avant chaque fichier. Besoin conservateur maximal :**32 488 063 369 octets**, réserve comprise.

Plafond cumulatif :**6 611 516 695 octets** de corps HTTP réservés, retries et répertoire ZIP compris. Journal fsync avant chaque tentative, compteur conservé à la reprise, journal tronqué/divergent bloquant. TCP/TLS et en-têtes réseau ne sont pas mesurés par urllib. Une connexion active ; débit cible20 Mbit/s avec rafales de lecture bornées à4 Mio. Environ38 minutes de corps au débit cible, hors latence/erreurs/contrôles : pas de durée garantie.

Verrou OS contre lancements simultanés. TIFF validés ignorés au transfert, fichiers.part conservés. Un fichier interrompu reprend depuis son début dans un nouveau.part, avec les octets précédents toujours débités du plafond. Certificats CRC/SHA par lot, relecture SHA à la reprise et comparaison finale. Si disque ou budget deviennent insuffisants : arrêt ; jamais de nouvelle allocation automatique.

Tests exécutés :20 tests initiaux puis20 tests finaux ciblés. Fixtures réseau simulées uniquement ; aucun transfert réel. Plansv4 hashés et inchangés ; corpus.py historique inchangé. Le correctif concerne la persistance du budget entre exécutions, les lots, le verrou, la limite de débit et la garde disque permanente.

Commande exacte depuis la racine, **après accord seulement** :

```powershell
.\.venv\Scripts\python.exe scripts/acquire_scientific_stage.py --launch-config configs/acquisition_stage080_v1.json --execute
```

Reprise : exactement la même commande/configuration. Session :artifacts/scientific-acquisition/stage080-v1 ; ledgertransfer.jsonl, rapportsattempt-*.json distincts, certificatsbatch-*.json etcomplete-*.json. Aucun entraînement GPU.

Après acquisition : bilan réel fichiers/événements/CRC/SHA/octets confirmés et réservés, puis QC raster et manifeste multi-racines avant DataLoader. Pour200 puis451 : réutiliser les plans existants, dédupliquer tous les TIFF acquis valides et présenter uniquement un nouveau budget résiduel à approuver.
