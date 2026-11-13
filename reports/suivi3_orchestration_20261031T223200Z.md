# GeoFire-X — état réel du suivi 3

Instantané : 2026-10-31T22:32:00.046585+00:00

- Connexion OAuth officielle effectuée ; identifiants hors du dépôt.
- Session B terminée : 14 GeoTIFF, 45 061 097 octets, version 1 privée, nouvelle session indépendante.
- Sonde privée : création observée, puis récupération en lecture seule vérifiée dans une nouvelle session.
- PySpark 4.0.4 exécuté : 13 607 entrées, 607 événements, zéro doublon ; Parquet récupéré et comparé à l’index.
- FastAPI/React compilé et vérifié par HTTP avec un checkpoint réellement entraîné ; vérité J+1 séparée.
- Test 2021 verrouillé. Vérification visuelle Chrome bloquée par le contrôle Computer Use.
- 194 tests réussis. Aucun commit ou push.

## Diagnostic local mesuré

3 événements train, 2 validation, 108 observations ; seed 42, deux epochs de 32 patches pour B1/B2/B3.

| Modèle | Updates | AP validation |
|---|---:|---:|
| B0 | 0 | 0.231948 |
| B1 | 8 | 0.004120 |
| B2 | 8 | 0.001941 |
| B3 | 8 | 0.000537 |

Persistence a l’AP la plus élevée dans ce diagnostic court. Aucune sélection scientifique n’est annoncée.

## Jobs cloud à cet instant

| Job | Type | Dernier état distant lu | Preuve vérifiée |
|---|---|---|---|
| 20261031-202857-f13d587d | session-b | COMPLETE | oui |
| 20261031-203241-aae74499 | publication-probe | ERROR | non |
| 20261031-204157-6dad4c9c | publication-probe | COMPLETE | oui |
| 20261031-205829-923ffb32 | acquire | UNKNOWN | non |
| 20261031-205845-9e2ab980 | acquire | UNKNOWN | non |
| 20261031-205945-97cf9126 | bigdata | ERROR | non |
| 20261031-210700-478e4d85 | bigdata | COMPLETE | oui |
| 20261031-221724-b1312212 | publication-probe | ERROR | non |

Les 37 événements train / 714 observations et 30 événements validation / 546 observations restent des objectifs de lots tant que leurs preuves ne sont pas récupérées.
Les budgets de 6 Gio Zenodo et 6 Gio de payloads d’upload ont été autorisés explicitement. La chaîne poursuit certifications et sonde T4, puis s’arrête avant une campagne longue.

## Contrôle

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_orchestrate.py status
.\.venv\Scripts\python.exe scripts\serve_demo.py
```

Supervision locale : `artifacts/orchestrator/chain-pilot-v1`. Son `stop.request` arrête sa boucle ; les jobs cloud gardent leurs contrôles propres.
La supervision nécessite un PC actif et possède une échéance enregistrée. Ce rapport est un instantané ; la commande status lit les derniers cycles.

## Preuves et livrables

- [Documentation](../docs/ORCHESTRATION_SUIVI3.md)
- [Comparaison et courbes](../artifacts/demo-comparison/real-cpu-v1/comparison.json)
- [Contrôle HTTP sur validation](../artifacts/demo-serving/http-verification-v2-validation/verification.json)
- [Figure d’inférence réelle](../artifacts/demo-serving/http-verification-v2-validation/real-inference.png)
- [Archive de code v4.2](../artifacts/kaggle/geofirex-orchestrated-code-v4.2.zip)
- Le JSON homonyme contient tous les chemins, hashes, preuves et états détaillés.

Prochaine autorisation pour poursuivre au-delà de la sonde : un budget GPU long, chiffré à partir des mesures réelles.
