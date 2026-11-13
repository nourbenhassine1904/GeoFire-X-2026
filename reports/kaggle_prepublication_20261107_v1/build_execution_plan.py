"""Write exact argument arrays for each proposed single-lot phase; never execute."""
import json
from pathlib import Path

root=Path(__file__).parent
review=json.loads((root/'frozen-review-final.json').read_text())
ml='.venv/Scripts/python.exe'; kg='artifacts/orchestrator/kaggle-cli-env/Scripts/python.exe'
campaign='reports/kaggle_publication_20261107_v1'; frozen='reports/recovery_20261107_v3/finalized'
common=['--credentials','C:/Users/nourb/.kaggle/credentials.json','--journal',campaign+'/http-body.jsonl',
        '--max-http-body-bytes','50000000000']
phases=[]
for row,stop in zip(review['lots'],[6,11,16,21,26,31,36,37]):
    tag=f"lot-{row['number']:03d}"; staged='artifacts/kaggle_publication_20261107_v1/'+tag
    def guard(label,seconds,command):
        return [ml,'scripts/run_kaggle_phase_guarded.py','--out',f'{campaign}/{tag}-{label}-monitor',
                '--seconds',str(seconds),'--']+command
    phases.append(dict(lot=row['number'],handle=row['handle'],private=True,identity=row['identity'],
        prerequisite=('Code runtime certificate' if row['number']==1 else f"Lot {row['number']-1:03d} certificate AND permission to continue"),
        cumulative_body_stop_at=stop*1_000_000_000,
        stage=guard('stage',1800,[ml,'scripts/stage_expanded_lot.py','--lot',f'{frozen}/deployment/bundle/metadata/{tag}.json',
            '--raw-root','D:/GeoFire-X/WildfireSpreadTS/rasters','--out',staged,'--max-bytes','5100000000']),
        publish=guard('upload',7200,[kg,'scripts/publish_kaggle_single_attempt.py','--staged',staged]+common+[
            '--attempt',f'{campaign}/{tag}-create-attempt-01','--stop-at-cumulative-body-bytes',str(stop*1_000_000_000),
            '--license','CC-BY-4.0','--execute']),
        status=[kg,'-m','kaggle','datasets','status',row['handle'],'--format','json'],
        files_first_page=[kg,'-m','kaggle','datasets','files',row['handle'],'--format','json','--page-size','200'],
        certify_cpu=guard('cpu-submit',120,[kg,'scripts/submit_kaggle_cpu_certification.py','--folder',
            f'{root.as_posix()}/cpu-certification-v3/{tag}']+common+['--attempt',f'{campaign}/{tag}-cpu-submit-01',
            '--stop-at-cumulative-body-bytes',str(stop*1_000_000_000),'--execute']),
        kernel_status=[kg,'-m','kaggle','kernels','status',f'nnnnbhss/geofirex-certify-{tag}-20261107-v1'],
        download_small_receipts=[kg,'-m','kaggle','kernels','output',f'nnnnbhss/geofirex-certify-{tag}-20261107-v1',
            '-p',f'{campaign}/{tag}-cpu-receipts','--file-pattern',r'(^|/)(runtime-certificate|certificate)\.json$']))
plan=dict(status='PROPOSED_NOT_EXECUTED_NOT_AUTHORIZED',cwd=str(Path.cwd()),global_upload_body_ceiling=50_000_000_000,
          stop_after_first_lot=True,automatic_retries=0,phases=phases)
with (root/'execution-plan-final.json').open('x',encoding='utf-8') as f: json.dump(plan,f,indent=2)
print('Prepared exact argument arrays for eight single-lot phases; nothing executed.')
