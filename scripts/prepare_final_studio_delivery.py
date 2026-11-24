"""Inventory only: explicit Git publication plan, no staging, commit or upload."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/final_studio_20261124'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(name):
 p=ROOT/name
 return dict(path=name,bytes=p.stat().st_size,sha256=digest(p))
def git(*args):return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),*args],cwd=ROOT).decode('utf8')

# The final UI is an overlay on the user's already modified scientific app.
final=['frontend/index.html',*[f'frontend/src/{n}' for n in ('EventForecast.mjs','PropagationView.mjs','SceneTimeline.mjs','ScientificComparison.mjs','StudioPages.mjs','V2Views.mjs','studio_ui.mjs','style.css')],
 'frontend/tests/propagation-dom.mjs','scripts/verify_final_studio.py','scripts/prepare_final_studio_delivery.py','docs/SOUTENANCE_FINALE_20261202.txt','README.md']
dirty={line[3:] for line in git('status','--porcelain','--untracked-files=all').splitlines() if line[:2] in (' M','??','A ','AM','MM')}
modules=set();pending=['jury_app','studio_spatial','prediction_lab','v2_execution']
while pending:
 name=pending.pop()
 if name in modules:continue
 p=ROOT/'src/geofirex'/f'{name}.py'
 if not p.is_file():continue
 modules.add(name)
 for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
  if isinstance(node,ast.ImportFrom) and node.module=='geofirex':pending.extend(a.name for a in node.names)
  if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:pending.append(node.module.split('.')[0])
  names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
  for imported in names:
   if imported and imported.startswith('geofirex.'):pending.append(imported.split('.')[1])
prerequisites={f'src/geofirex/{m}.py' for m in modules}
prerequisites.update(['Start-GeoFireX.ps1','scripts/serve_jury.py','tests/test_jury_finalization.py','tests/test_studio_spatial.py','frontend/build.mjs','frontend/package.json','frontend/package-lock.json','frontend/verify-studio-propagation.mjs'])
for folder in ['frontend/src','frontend/tests']:
 prerequisites.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file())
prerequisites.update(['reports/studio_propagation_20261123_v1/visualization_sources.json','reports/studio_propagation_20261123_v1/visualization_manifest.json','reports/protocol_v2_20261118/split_summary.json'])
seal=json.loads((ROOT/'reports/jury_20261202_v1/science_manifest.json').read_text())
prerequisites.add('reports/jury_20261202_v1/science_manifest.json')
prerequisites.update('reports/jury_20261202_v1/'+n for n in seal['files'] if n!='diagnostic_tile.json')
frozen=json.loads((ROOT/'reports/protocol_v2_execution/execution_manifest.json').read_text())
prerequisites.update(['reports/protocol_v2_execution/execution_manifest.json','reports/jury_20261202_v1/preservation_before.json','reports/studio_propagation_20261123_v1/preservation_before.json'])
binary={'.pt','.pth','.npy','.npz','.zip','.bin','.tif','.tiff'}
prerequisites.update(n for n in frozen['files'] if Path(n).suffix.lower() not in binary)
prerequisites=sorted((prerequisites & dirty)-set(final))
receipts=['reports/final_studio_20261124/'+n for n in ['RAPPORT_FINAL.txt','api-cpu.xml','preservation.json','browser-audit.json','browser-exports.json','frontend-release/propagation-dom.json','frontend-release/theme_contract.json'] if (OUT/n).is_file()]
captures=[p.relative_to(ROOT).as_posix() for p in (OUT/'captures').glob('*-native.png')]
sources=json.loads((ROOT/'reports/studio_propagation_20261123_v1/visualization_sources.json').read_text())
adata=[dict(path=r['archive'],sha256=r['archive_sha256'],role=r['model']+' COMPLETE, never Git') for r in sources['archives']]
adata.append(dict(path=sources['hero']['path'],sha256=sources['hero']['sha256'],role='historical raster; restore original path from local/ADATA backup'))
adata.append(dict(**entry('reports/jury_20261202_v1/diagnostic_tile.json'),role='saved pixel arrays; restore locally, not in this commit'))
lock=json.loads((ROOT/'reports/jury_20261202_v1/preservation_before.json').read_text())
for name,sha in lock['files'].items():
 if Path(name).suffix.lower() in ('.pt','.pth','.npy','.npz','.zip','.bin','.tif','.tiff'):
  adata.append(dict(path=name,sha256=sha,role='protected historical artifact; keep existing bytes on local/ADATA storage'))
selected=set(final+prerequisites+receipts+captures)
selected.update({'reports/final_studio_20261124/git-manifest.json','reports/final_studio_20261124/git-files-to-publish.txt'})
live=json.loads((ROOT/'configs/prediction_lab_v1.json').read_text())
manifest=dict(schema='geofirex-final-publication-plan-v1',date='2026-11-24',soutenance='2026-12-02',
 policy='Read-only inventory; no Git mutations. UI overlay requires original local/ADATA data. Existing tracked binaries are preserved, not deleted.',
 commit_ui=[entry(n) for n in final],commit_prerequisites=[entry(n) for n in prerequisites],commit_receipts=[entry(n) for n in receipts],commit_clean_captures=[entry(n) for n in captures],
 ADATA_only=adata,ADATA_only_collections=[dict(path=live[k],role=k+' runtime data; retain existing bytes and local mount mapping; do not stage new binary data') for k in ('corpus','index')],local_only=['.venv/','frontend/node_modules/','frontend/dist/','reports/final_studio_20261124/prior-work-local-only.txt','reports/final_studio_20261124/captures/* except explicitly listed native captures','reports/final_studio_20261124/frontend*/ intermediate DOM HTML and raster exports'],
 prior_work_not_selected_count=len(dirty-selected),prior_work_not_selected_inventory='prior-work-local-only.txt',git_commit_push_merge=False)
(OUT/'prior-work-local-only.txt').write_text('\n'.join(sorted(dirty-selected))+'\n',encoding='utf8')
(OUT/'git-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
# One exact path per line; this is a review list, never executed as git add.
(OUT/'git-files-to-publish.txt').write_text('\n'.join(sorted(selected|{'reports/final_studio_20261124/git-manifest.json','reports/final_studio_20261124/git-files-to-publish.txt'}))+'\n',encoding='utf8')
(OUT/'git-status.txt').write_text(git('status','--short'),encoding='utf8')
print(json.dumps({k:len(manifest[k]) for k in ['commit_ui','commit_prerequisites','commit_receipts','commit_clean_captures','ADATA_only']}))
