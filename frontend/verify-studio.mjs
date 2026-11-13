// Component integration in a DOM simulator. This is not a browser layout test.
import {build} from 'esbuild';
import {mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {verifyTheme} from './tests/theme-contract.mjs';
await verifyTheme();
const directory='node_modules/.geofirex-studio-tests';
await mkdir(directory,{recursive:true});
await build({entryPoints:['tests/studio-dom.mjs'],bundle:true,platform:'node',format:'esm',outfile:directory+'/studio-dom.mjs',external:['happy-dom','axe-core'],define:{'process.env.NODE_ENV':'"development"'},logLevel:'warning'});
execFileSync(process.execPath,[directory+'/studio-dom.mjs'],{stdio:'inherit',env:process.env});
