"""Seal final jury evidence and local application changes; never train/publish."""
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from geofirex import v2_execution as old
from geofirex import auxnew_experiment as auxiliary
from geofirex.protocol_v2 import sha256,save
from geofirex.jury_app import evidence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/jury_20261202_v1'


def main():
    if (OUT/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed jury delivery')
    old.identity(ROOT);auxiliary.identity(ROOT)
    before=old.read(OUT/'preservation_before.json')['files']
    # Includes original FIRE13 SAFE/PARTIAL archives, sidecars and frozen inputs.
    before.update(old.read(ROOT/'reports/fire13_recovery_20261123_v1/preservation_before.json')['files'])
    for name,digest in before.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha256(path)!=digest:raise ValueError('Protected source changed: '+name)
    tree=ET.parse(OUT/'tests_CPU_API.xml').getroot();cases=tree.findall('.//testcase')
    if len(cases)!=28 or any(tree.findall('.//'+k) for k in ['failure','error','skipped']):
        raise ValueError('28 actually executed CPU/API tests must pass')
    dom=old.read(OUT/'jury-dom.json')
    if dom['status']!='PASS' or len(dom['checks'])!=13 or any(v['violations'] for v in dom['a11y'].values()):
        raise ValueError('Complete React DOM/export/inference receipt required')
    for name in old.read(OUT/'science_manifest.json')['files']:evidence(name)
    http=[]
    for path in ['/health','/lab/status','/science/v2/comparison','/science/v2/diagnostic-tile',
                 '/science/v2/comparable-fp','/science/v2/provenance','/science/v2/export.csv','/']:
        with urllib.request.urlopen('http://127.0.0.1:8026'+path,timeout=30) as response:
            content=response.read()
            http.append(dict(path=path,status=response.status,content_type=response.headers.get('Content-Type'),bytes=len(content)))
            if response.status!=200:raise ValueError('HTTP failure')
            if path=='/lab/status':
                status=json.loads(content)
                if not status['ready'] or status['threshold']!=.78 or not status['checkpoint_sha256'].startswith('7295eaa6f957'):
                    raise ValueError('Serving model changed')
            if path=='/science/v2/export.csv' and '95053' not in content.decode():raise ValueError('CSV differs')
    save(OUT/'http_restart_verified.json',dict(status='PASS',server_restarted=True,checks=http,
        inference_contract='React DOM receipt + 28 CPU/API tests, including real historical checkpoint and saved-probability equality',
        browser_visual_check=False))
    build=old.read(ROOT/auxiliary.REPORT/'notebook_build.json')
    notebook=ROOT/'notebooks/GEOFIREX_AUXNEW_V2_DISABLED.ipynb'
    if sha256(notebook)!=build['notebook_sha256']:raise ValueError('Candidate notebook changed after CPU qualification')
    subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'diff','--check'],cwd=ROOT,check=True,capture_output=True,text=True)
    comparison=old.read(OUT/'comparison.json');profile=old.read(ROOT/auxiliary.REPORT/'cpu_feasibility.json')
    verification=dict(status='CPU_API_DOM_PASS_VISUAL_REVIEW_PENDING',date='2026-11-23',jury_date='2026-12-02',
        tests=dict(CPU_API_passed=len(cases),failed=0,skipped=0,React_DOM_checks=13,
            CPU_API_receipt_sha256=sha256(OUT/'tests_CPU_API.xml'),DOM_receipt_sha256=sha256(OUT/'jury-dom.json')),
        original_scientific_identities_verified=77,protected_files_verified=len(before),protected_changed=[],
        archive_verification='All members SHA-256 verified for the four complete model archives',
        FIRE13_closed=True,FIRE13_new_fit=False,FIRE13_new_recovery=False,FIRE13_new_inference=False,
        new_aggregate_evaluation=False,training_executed=False,optimizer_created=False,test_2021_pixels_opened=0,
        scientific_results_scope='already observed validation2020 V2 80 events; legacy201 kept separate',
        served_checkpoint='7295eaa6f957e59a31b7155403881a9b3bf19933a35b6872768896bda1f1b14e',
        served_threshold=.78,automatic_promotion=False,Big_Data_or_Spark_development=False,
        auxiliary_candidate=dict(status='GO_CPU_PREPARATION_ONLY',GPU_launch='NO_GO_PENDING_NEW_AUTHORIZATION_QUOTA_AND_PREFLIGHT',
            additional_parameters=9,CPU_derivative_time_ratio=profile['CPU_time_ratio'],GPU_measured=False,
            initialization_guard='Exact archived initial C8 digest before optimizer; local CPU initialization differs across torch versions',
            notebook_sha256=build['notebook_sha256'],performance_obtained=False),
        blockers=['Real browser layout/native dialogs/responsive visual review pending: browser host unavailable and native Chrome approval timed out',
                  'Candidate current Kaggle quota and exact GPU runtime/memory not measured; launch not authorized'],
        commit=False,push=False,publication=False,git_diff_check='PASS')
    save(OUT/'verification_finale.json',verification)
    lines=['GeoFire-X — finalisation scientifique et fonctionnelle, 23 novembre 2026',
        'Jury : 2 décembre 2026. État : CPU/API/React validés ; contrôle visuel manuel restant.',
        '', 'COMPARAISON OFFICIELLE : VALIDATION 2020, PROTOCOLE V2',
        'Même domaine SHA-256, 80 événements, 30882 tuiles, seuil0,78 ; validation déjà observée.',
        'Modèle | Rappel nouveau % | Dice nouveau % | Précision nouveau % | AP nouveau % | FP nouveau | Dice global % | AP globale % | Brier']
    for r in comparison['models']:
        lines.append(f"{r['model']} | {100*r['recall_new']:.4f} | {100*r['dice_new']:.4f} | {100*r['precision_new']:.4f} | {100*r['AP_new']:.4f} | {r['FP_new']} | {100*r['dice_global']:.4f} | {100*r['AP_global']:.4f} | {r['Brier']:.6f}")
    lines += ['', 'Aucune variante ne satisfait les huit critères figés. C8 est la référence V2 ;',
        'NEWPOS2, NEWPOS4 et FIRE13 restent expérimentaux et non promus. Un rappel accru',
        'ne suffit pas : les gains s’accompagnent de FP et/ou de dégradation AP/Brier.',
        'FIRE13 : quatre critères sur huit passent ; absence de supériorité démontrée',
        'dans l’analyse sauvegardée à budget FP comparable. Aucune relance FIRE13.',
        'AP exacte par égalités globales de scores, pas aire trapézoïdale. Aucun score2021.',
        'comparison.json/CSV : résultats et sources. Les quatre *_evaluation.json sont',
        'les octets extraits des archives vérifiées. paired_results.json contient cinq',
        'comparaisons appariées sauvegardées, avec SHA des membres ; aucun nouveau bootstrap.',
        'Les intervalles portent sur les métriques de comptage/Brier, pas sur AP.',
        '', 'APPLICATION EXISTANTE FINALISÉE EN LOCAL',
        'Entrée additive scripts/serve_jury.py et geofirex/jury_app.py : les modules',
        'scientifiques historiques gelés ne sont pas modifiés. React garde Prediction',
        'Studio, les huit couches, le zoom et les exports. Nouvelle comparaison V2',
        'traçable, critères consultables, statut expérimental explicite et même tuile',
        'pour les quatre modèles. Cette tuile est sélectionnée sans regarder les scores',
        'et n’est pas représentative. Les scores NEWPOS2/4/FIRE13 proviennent des sorties',
        'sauvegardées ; seul C8 a une nouvelle inférence CPU ponctuelle depuis ses poids',
        'originaux. Aucune évaluation agrégée supplémentaire des 80 événements.',
        'Le modèle interactif reste le U-Net poids10 historique, checkpoint7295eaa6…,',
        '23404 mises à jour, protocole201 : 32 réglage +169 confirmation déjà observée.',
        'Il est explicitement distinct du C8 V2 c2273fea…, 28976 mises à jour.',
        'Grilles de pixels sans CRS/affine, aucune surface en hectares inventée.',
        'CSV des tuiles : correction des offsets ligne/colonne, avec test non nul.',
        'Big Data/Spark retiré de la navigation jury ; modules historiques inchangés.',
        '', 'DERNIER CANDIDAT — GO PRÉPARATION, PAS DE GO LANCEMENT',
        'AUXNEW V2 : 123986 paramètres au lieu de123977, soit neuf paramètres ajoutés.',
        'Backbone C8, huit canaux, index, normaliseur, sampler et loss principale',
        'inchangés. Ajout d’une BCE auxiliaire pondérée10, coefficient0,25, sur',
        'valid & feu_J==0. Seule la tête principale sert à l’inférence ; aucun routage',
        'ou mélange de scores. Ce choix diffère de l’ancienne tête spécialisée',
        'gâtée par le masque observé et non retenue sur le protocole32.',
        f"Profil CPU synthétique, dérivées sans optimiseur : ratio médian {profile['CPU_time_ratio']:.3f}x.",
        'Ce ratio n’est pas une mesure de débit GPU. Référence C8 archivée : fit1133,55s,',
        'pic RSS4,81GB et pic VRAM70,9MB ; aucun de ces chiffres ne mesure AUXNEW.',
        'Réserves proposées : session300min /10h-carte ; fit5400s, évaluation8100s,',
        'RAM16GiB totale /8GiB disponible, disque20GiB, VRAM disponible2GiB minimum.',
        'Un seul fit neuf seed42, quatre époques,28976 mises à jour, checkpoint final',
        'uniquement, seuil0,78, sans sweep, sans reprise automatique, critères inchangés.',
        'Risques : validation déjà consultée, une seed, météo/combustible absents ;',
        'la supervision auxiliaire n’ajoute aucune information physique et ne garantit',
        'pas de gain. La précédente tête spécialisée n’établit pas son succès sur V2.',
        'Le digest initial C8 a été retrouvé dans le checkpoint archivé à zéro mise',
        'à jour. Le CPU local initialise différemment avec sa version PyTorch ;',
        'le runner exige le runtime GPU historique et le digest exact avant tout',
        'optimiseur. Si quota/runtime/mémoire indisponibles, rester NO-GO sans autre modèle.',
        'Notebook autonome désactivé, protocole préenregistré et grants DISABLED livrés.',
        'Aucun entraînement ni métrique AUXNEW obtenu. Nouvelle autorisation requise.',
        '', 'VÉRIFICATION EFFECTIVEMENT EXÉCUTÉE',
        '28 tests CPU/API réussis, zéro échec/skip. Sources et archives protégées',
        'vérifiées, refus2021 avant lecture, inférence du checkpoint historique réel,',
        'égalité avec les probabilités sauvegardées à1e-6, plusieurs événements/dates,',
        'contrôles de masques, corruption refusée, architecture auxiliaire et gradients',
        'synthétiques sans optimiseur, autonomie du payload et imports transitifs.',
        '13 contrôles React/HTTP réussis : quatre modèles, pixels canvas, exports',
        'JSON/CSV/PNG, zoom/fermeture en simulateur, navigation, inférence réelle puis',
        'vérité séparée, aucun appel Spark ; règles axe structurelles sans violation.',
        'Le serveur a été arrêté puis redémarré et huit routes HTTP ont répondu200.',
        'La fin du test libère les timers GSAP et sort proprement.',
        '', 'LIMITES ET BLOCAGES CONCRETS',
        'Le navigateur intégré est indisponible. L’autorisation du contrôle Chrome',
        'natif a expiré. Aucun rendu visuel, screenshot, contraste final ou géométrie',
        'responsive n’est prétendu validé ; revue manuelle avant le jury nécessaire.',
        'Les dialogues, canvas et animations du test React utilisent des adaptateurs',
        'DOM explicites. Ce test ne constitue pas une validation dans Chrome.',
        'Quota Kaggle actuel et précontrôle GPU AUXNEW non effectués ; pas de lancement.',
        '', 'PRÉSERVATION',
        f"77 identités scientifiques et {len(before)} fichiers protégés vérifiés inchangés.",
        'Archives, checkpoints, critères et autorisations consommées conservés.',
        'Aucun fit, entraînement, récupération FIRE13, nouvel accès2021, commit, push',
        'ou publication. Les seuls calculs de dérivées sont des fixtures CPU non entraînées.',
        '', 'DÉMONSTRATION',
        'Suivre docs/DEMONSTRATION_JURY_20261202.txt. Lancer serve_jury.py --port8026,',
        'ouvrir Overview, inférer2020/fire_24332880 au2020-09-03, examiner les huit',
        'couches et exports, puis Comparaison V2, critères, provenance et limites.',
        'Le serveur de vérification est arrêté après préparation ; le relancer pour',
        'la démonstration. Aucun notebook d’entraînement n’est nécessaire au jury.']
    (OUT/'RAPPORT_FINALISATION.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    names=['README.md','docs/DEMONSTRATION_JURY_20261202.txt',
        'scripts/serve_jury.py','scripts/prepare_jury_science.py','scripts/prepare_auxnew.py','scripts/finalize_jury.py',
        'src/geofirex/jury_app.py','src/geofirex/auxnew_model.py','src/geofirex/auxnew_experiment.py',
        'tests/test_jury_finalization.py','tests/test_auxnew.py','notebooks/GEOFIREX_AUXNEW_V2_DISABLED.ipynb',
        'frontend/build.mjs','frontend/verify-jury.mjs','frontend/tests/jury-dom.mjs',
        'frontend/src/StudioApp.mjs','frontend/src/StudioPages.mjs','frontend/src/ScientificComparison.mjs',
        'frontend/src/prediction_view.mjs','frontend/src/style.css',
        'frontend/dist/app.js','frontend/dist/index.html','frontend/dist/style.css']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/auxiliary.REPORT).glob('*') if p.is_file()]
    for name in names:
        dest=OUT/'kit'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as target,(ROOT/name).open('rb') as source:shutil.copyfileobj(source,target)
    files={p.relative_to(OUT).as_posix():sha256(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'delivery_manifest.json',dict(schema='geofirex-jury-delivery-1',files=files,
        scientific_results='verified saved validation2020',status=verification['status'],commit=False,push=False))
    (OUT/'delivery_manifest.sha256').write_text(sha256(OUT/'delivery_manifest.json')+'  delivery_manifest.json\n',encoding='ascii')
    archive=old.package(OUT,'JURY-CPU-API-VERIFIED')
    save(ROOT/'reports/jury_20261202_v1_delivery.json',dict(archive=str(archive),sha256=sha256(archive),
        bytes=archive.stat().st_size,status=verification['status'],CPU_API_passed=28,React_DOM_checks=13,
        candidate='GO_PREPARATION_ONLY_NO_GPU_AUTHORIZATION',visual_review='MANUAL_PENDING'))
    print(json.dumps(old.read(ROOT/'reports/jury_20261202_v1_delivery.json'),indent=2))


if __name__=='__main__':main()
