// Actual HTTP and React/canvas contract checks. No claim of browser layout QA.
import {build} from 'esbuild';
import {mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=dirname(fileURLToPath(import.meta.url));process.chdir(root);
const directory=resolve(root,'node_modules/.geofirex-jury-tests');
await mkdir(directory,{recursive:true});
await build({absWorkingDir:root,tsconfigRaw:{},entryPoints:[resolve(root,'tests/jury-dom.mjs')],bundle:true,
 platform:'node',format:'esm',outfile:directory+'/jury-dom.mjs',external:['happy-dom','axe-core'],
 define:{'process.env.NODE_ENV':'"development"'},logLevel:'warning'});
execFileSync(process.execPath,[directory+'/jury-dom.mjs'],{stdio:'inherit',env:{...process.env,
 GEOFIREX_TEST_URL:process.env.GEOFIREX_TEST_URL||'http://127.0.0.1:8026',
 GEOFIREX_TEST_REPORT_DIR:process.env.GEOFIREX_TEST_REPORT_DIR||'../reports/jury_20261202_v1'}});
