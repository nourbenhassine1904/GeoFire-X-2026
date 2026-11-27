// Actual HTTP and React/canvas contract checks. No claim of browser layout QA.
import {build} from 'esbuild';
import {mkdir,access} from 'node:fs/promises';
import {randomUUID} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {verifyTheme} from './tests/theme-contract.mjs';
import {verifyProfessionalCSS} from './tests/professional-contract.mjs';
const root=dirname(fileURLToPath(import.meta.url));process.chdir(root);
process.env.GEOFIREX_TEST_REPORT_DIR ||= resolve(root,'../reports/studio_checks/'+new Date().toISOString().replace(/[:.]/g,'-')+'-'+randomUUID().slice(0,8));
const reportDirectory=resolve(root,process.env.GEOFIREX_TEST_REPORT_DIR);
const testURL=process.env.GEOFIREX_TEST_URL||'http://127.0.0.1:8026';
const readinessStarted=Date.now();
for(;;){
 try{const response=await fetch(testURL+'/health',{signal:AbortSignal.timeout(2000)});if(response.ok)break;}catch{}
 if(Date.now()-readinessStarted>60000)throw Error('FastAPI indisponible sur '+testURL+' ; lancer Start-GeoFireX.ps1 avant les tests.');
 await new Promise(resolve=>setTimeout(resolve,500));
}
let sealed=false;
for(const marker of ['delivery_manifest.json','science_manifest.json','artifact_manifest.json']){
 try{await access(resolve(reportDirectory,marker));sealed=true;}catch(e){if(e.code!=='ENOENT')throw e;}
}
if(sealed)throw Error('Refusing to overwrite a sealed delivery; choose a new GEOFIREX_TEST_REPORT_DIR');
process.env.GEOFIREX_TEST_REPORT_DIR=reportDirectory;
await verifyTheme();
await verifyProfessionalCSS(reportDirectory);
const directory=resolve(root,'node_modules/.geofirex-propagation-tests');
await mkdir(directory,{recursive:true});
await build({absWorkingDir:root,tsconfigRaw:{},entryPoints:[resolve(root,'tests/propagation-dom.mjs')],bundle:true,
 platform:'node',format:'esm',outfile:directory+'/propagation-dom.mjs',external:['happy-dom','axe-core'],
 define:{'process.env.NODE_ENV':'"development"'},logLevel:'warning'});
execFileSync(process.execPath,[directory+'/propagation-dom.mjs'],{stdio:'inherit',env:{...process.env,
 GEOFIREX_TEST_URL:process.env.GEOFIREX_TEST_URL||'http://127.0.0.1:8026',
 GEOFIREX_TEST_REPORT_DIR:process.env.GEOFIREX_TEST_REPORT_DIR||'../reports/studio_propagation_20261123_v1'}});
