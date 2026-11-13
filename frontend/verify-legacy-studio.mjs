// Run the two existing smoke verifiers with isolated V2 outputs. Preserve all
// historical reports and their original verification scripts byte-for-byte.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
const directory='node_modules/.geofirex-studio-tests';
await mkdir(directory,{recursive:true});
await mkdir('../artifacts/ui-studio-v2/legacy',{recursive:true});
await mkdir('../reports/ui_studio_v2',{recursive:true});
for(const [source,name] of [['verify-prediction-lab.mjs','lab'],['verify-propagation.mjs','propagation']]){
 let script=await readFile(source,'utf8');
 script=script.replaceAll('../artifacts/prediction-lab-','../artifacts/ui-studio-v2/legacy/prediction-lab-');
 script=script.replaceAll('../reports/propagation_20261112_v1/react-ssr-check.cjs','../artifacts/ui-studio-v2/legacy/propagation-ssr.cjs');
 script=script.replaceAll('../reports/propagation_20261112_v1/react-component-check.json','../reports/ui_studio_v2/propagation_component_check.json');
 // V1's earlier manual review is not evidence of a browser review of V2.
 script=script.replace("browser_interactions_verified:'manual user confirmation 2026-11-13: eight maps, enlargement/close, date controls and PNG/JSON/CSV exports'","browser_interactions_verified:false,browser_review:'V2 manual Chrome review pending; browser host unavailable'");
 const isolated=directory+'/legacy-'+name+'.mjs';await writeFile(isolated,script);
 execFileSync(process.execPath,[isolated],{stdio:'inherit'});
}
