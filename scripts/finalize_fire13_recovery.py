"""Seal CPU-verified recovery delivery, without publishing or GPU execution."""
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

from geofirex import fire13_experiment as c
from geofirex import fire13_recovery as r

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/r.REPORT
    if (out/'delivery_manifest.json').exists():
        raise FileExistsError('Preserve sealed recovery delivery')
    identity=r.identity(ROOT)
    before=c.read(out/'preservation_before.json')['files']
    for name,digest in before.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if c.sha256(path)!=digest:
            raise ValueError('Protected input changed: '+name)
    receipts=[]; unique=set()
    for name in ['tests_CPU.xml','tests_historical_CPU.xml']:
        tree=ET.parse(out/name).getroot()
        cases=tree.findall('.//testcase')
        if not cases or any(tree.findall('.//'+kind) for kind in ['failure','error','skipped']):
            raise ValueError('All selected CPU tests must actually pass: '+name)
        unique.update((row.attrib['classname'],row.attrib['name']) for row in cases)
        receipts.append(dict(path=(r.REPORT/name).as_posix(),passed=len(cases),sha256=c.sha256(out/name)))
    required=['test_restore_real_training_safe_checkpoint_and_consumed_grant',
        'test_isolated_notebook_payload_imports_every_module',
        'test_candidate_evaluation_cpu_chain_writes_every_output',
        'test_historical_results_counts_and_fixed_acceptance_compatible',
        'test_comparable_fp_all_thresholds_locked_before_assessment']
    if not all(any(test_name==wanted for _,test_name in unique) for wanted in required):
        raise ValueError('Required executed CPU coverage missing')
    maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    c.save(out/'cpu_preflight_final.json',c.preflight(ROOT,maps))
    total,available=c.old.host_memory()
    verification=c.read(out/'archive_verification.json')
    training=Path(verification['archives'][0]['path'])
    upload=out/'kit/kaggle-upload';upload.mkdir(parents=True,exist_ok=False)
    for src,name in [(training,training.with_suffix('.bin').name),
                     (training.with_suffix('.sha256'),training.with_suffix('.sha256').name)]:
        with (upload/name).open('xb') as target,src.open('rb') as stream:
            shutil.copyfileobj(stream,target)
    if c.sha256(upload/training.with_suffix('.bin').name)!=identity['training_archive_sha256']:
        raise ValueError('Kaggle upload copy changed')
    delivery_files=['notebooks/GEOFIREX_FIRE13_EVALUATION_RECOVERY_ONLY.ipynb',
        'src/geofirex/fire13_recovery.py','src/geofirex/c8_thresholds.py',
        'scripts/prepare_fire13_recovery.py','scripts/finalize_fire13_recovery.py',
        'tests/test_fire13_recovery.py','docs/FIRE13_EVALUATION_RECOVERY.txt']
    for name in delivery_files:
        dst=out/'kit'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as target,(ROOT/name).open('rb') as source:
            shutil.copyfileobj(source,target)
    # Include the unchanged tests underlying every selected final receipt.
    for name in ['tests/test_fire13.py','tests/test_fire13_saves.py',
                 'tests/test_c8_propagation.py','tests/test_v2_evaluation.py']:
        dst=out/'kit'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as target,(ROOT/name).open('rb') as source:
            shutil.copyfileobj(source,target)
    git=['git','-c','safe.directory='+ROOT.as_posix()]
    subprocess.run(git+['diff','--check'],cwd=ROOT,check=True,capture_output=True,text=True)
    c.save(out/'verification_finale.json',dict(status='CPU_PASS_GPU_NOT_LAUNCHED',
        tests=dict(unique_passed=len(unique),failed=0,skipped=0,receipts=receipts),
        protected_files_verified=len(before),protected_changed=[],original_scientific_identities=77,
        original_manifest_sha256=identity['original_manifest_sha256'],
        original_runtime_single_run_lock='Not present in supplied ZIPs; never deleted, replaced or fabricated',
        checkpoint_sha256=identity['checkpoint_sha256'],new_optimizer_created=False,
        recovery_path_calls_fit=False,training_executed=False,test_pixels_opened=0,
        GPU_executed=False,published=False,commit=False,push=False,
        performance='FIRE13_NOT_YET_MEASURED_REQUIRES_MANUAL_KAGGLE_EVALUATION',
        quota='Current Kaggle receipt required at manual launch',
        host_local=dict(ram_total_bytes=total,ram_available_bytes=available,
                        disk_free_bytes=shutil.disk_usage(ROOT).free),
        resource_requirements=dict(session_minutes=300,card_hours_reserved=10,
            evaluation_deadline_seconds=8100,host_ram_total_bytes=16*c.old.GIB,
            host_ram_available_bytes=8*c.old.GIB,
            disk_free_bytes=18*c.old.GIB+4*verification['archives'][0]['expanded_bytes']),
        duplicate_evaluation_guard='Exclusive checkpoint-bound lock, fixed fresh output, no automatic retry; local to Kaggle working filesystem',
        git_diff_check='PASS'))
    text='''FIRE13 — rapport de correction et de récupération, 23 novembre 2026

État : PRÉPARATION CPU VALIDÉE. Aucun lancement GPU, entraînement, commit ou push.
Les performances FIRE13 restent à mesurer dans le notebook Kaggle livré.

Cause confirmée : le payload FIRE13 importait c8_thresholds sans l'embarquer.
Correction additive : module historique exact et pilote fire13_recovery séparé.
Le manifeste FIRE13 original et les 77 identités scientifiques restent inchangés.
La provenance identifie l'archive C8 Thresholds A et le membre exact, avec SHA-256.

MODULE c8_thresholds.py :
c4c34c3aba4d59b4e49bfb1e27129802040fee75316e106b35608368bec13520
ARCHIVE TRAINING-SAFE ADATA :
b76dd0696342267a5dafa0a6cde780a5c2d3aae1b7dc38a28983d9d88c8e3a35
ARCHIVE PARTIAL ADATA :
3743f2b256eedcafeee9f2db68d65fb352bbca9faab82f17e2bd5e82e6e2baa7
CHECKPOINT FINAL COMMUN (4 époques, 28976 mises à jour, curseur 0) :
5afca67a87d88c25fad2eadb2e12c8aef57b5d294a4af19b56622af51f10909c
Les deux archives ont 176 membres vérifiés, sans résultats d'évaluation.
Leurs copies Téléchargements sont identiques à ADATA. Les quatre originaux
de chaque emplacement sont préservés. La restauration CPU du checkpoint réel
et le chargement strict des poids ont réussi, avec une entrée synthétique.
Ce contrôle de forme ne constitue aucune mesure de performance scientifique.

Audit : 35 modules Python transitifs, imports internes aux fonctions compris.
Seul module historique absent : c8_thresholds. Ajout technique : fire13_recovery.
Les 35 modules ont été importés dans un processus isolé -I depuis le payload
extrait, sans accès au code du dépôt. Le notebook refait ce contrôle avant GPU.

Tests finaux : UNIQUE_COUNT tests CPU distincts réussis, zéro échec ou skip.
Reçus JUnit : tests_CPU.xml et tests_historical_CPU.xml ; détails et SHA dans
verification_finale.json. Couverture : restauration réelle, SHA invalide refusé,
modèle chargé strictement, autonomie du payload, absence d'appels d'apprentissage
dans la récupération, autorisations invalides refusées avant CUDA, verrou exclusif,
comptages et égalités de scores, masques et refus 2021, folds 64/16, verrouillage
des cinq seuils avant assessment, AP exacte comparée à sklearn, bootstrap apparié,
comparaison avec les résultats historiques et chaîne complète sur données synthétiques.
Le bootstrap du test synthétique de chaîne est remplacé par un témoin de routage ;
le vrai bootstrap de 2000 réplications est testé séparément et avec les historiques.
Les tests de gradients/entraînement et fixtures construisant un optimiseur ont
été exclus. Un premier passage, conservé dans tests_CPU_initial.xml, comporte
10 tests historiques appelant uniquement le garde d'entrée fit avec des grants
invalides : tous ont refusé avant modèle/CUDA/optimiseur, sans apprentissage.
Le passage final exclut également ces appels. Le notebook ne contient aucun fit.

Protection : les archives d'origine ne sont jamais réécrites ; restauration neuve,
copie du checkpoint avec contrôle SHA avant/après, autorisation consommée conservée,
nouveaux grants fire13-evaluate et gpu-preflight liés aux identités corrigées,
sortie neuve et verrou exclusif par SHA du checkpoint avant inférence.
Le verrou SINGLE_RUN historique du fit n'était pas dans les ZIP : il n'est ni
supprimé ni reconstruit. Le nouveau verrou ne protège que le filesystem Kaggle
de cette session : télécharger et conserver le reçu, ne pas dupliquer la session.

Budget inchangé : 300 minutes de session, réserve 10 heures-carte ; délai de
8100 secondes pour inférence/AP/comparaisons. Minimum 16 GiB RAM totale et
8 GiB disponibles ; environ 18,28 GiB disque libre, dont 18 GiB de réserve
historique plus quatre copies de la taille décompressée TRAINING-SAFE.
Le quota, la RAM, le disque et le runtime GPU Kaggle sont vérifiés à l'exécution.
Aucun quota actuel ni temps FIRE13 d'évaluation n'est prétendu mesuré ici.

Science inchangée : 30882 tuiles, 80 événements de validation 2020, seuil 0,78,
AP exacte par groupes d'égalités de scores, critères originaux et trois références
C8/NEWPOS4/NEWPOS2. comparable_fp reste exploratoire et ne remplace aucun seuil.
Aucun pixel de test 2021, aucune promotion ni intégration Studio automatique.

Livraison : notebook autonome, code et manifeste additionnels, provenance,
tests et reçus, procédure docs/FIRE13_EVALUATION_RECOVERY.txt, et copie .bin
exacte TRAINING-SAFE avec son sidecar pour upload privé Kaggle. La livraison
est archivée avec un manifeste de contenu et un SHA-256. Elle est distincte
des archives de résultats que produira l'évaluation manuelle.
'''.replace('UNIQUE_COUNT',str(len(unique)))
    (out/'RAPPORT_CORRECTION.txt').write_text(text,encoding='utf8')
    members={p.relative_to(out).as_posix():c.sha256(p) for p in out.rglob('*') if p.is_file()}
    c.save(out/'delivery_manifest.json',dict(schema='fire13-recovery-delivery-1',files=members,
        status='CPU_PASS_GPU_NOT_LAUNCHED',notebook_sha256=c.sha256(ROOT/delivery_files[0])))
    (out/'delivery_manifest.sha256').write_text(c.sha256(out/'delivery_manifest.json')+'  delivery_manifest.json\n',encoding='ascii')
    archive=c.old.package(out,'EVALUATION-RECOVERY-PREPARATION-CPU-PASS')
    print(json.dumps(dict(status='CPU_PASS_GPU_NOT_LAUNCHED',unique_tests=len(unique),
        archive=str(archive),sha256=c.sha256(archive)),indent=2))


if __name__=='__main__':
    main()
