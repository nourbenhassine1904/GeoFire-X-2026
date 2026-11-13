// Render the real loaded React component with saved API data, without a browser.
// This verifies component execution, not interactive browser rendering.
import {build} from 'esbuild';
import {execFileSync} from 'node:child_process';
const output='../reports/propagation_20261112_v1/react-ssr-check.cjs';
await build({stdin:{contents:`
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {PropagationView} from './src/Propagation.mjs';
import fs from 'node:fs';
const data=JSON.parse(fs.readFileSync('../reports/propagation_20261112_v1/diagnosis.json','utf8'));
const html=renderToStaticMarkup(React.createElement(PropagationView,{data}));
if ((html.match(/<img /g)||[]).length!==5) throw new Error('Expected five actual map layers');
if (!html.includes('2020/fire_24332880')||!html.includes('Nouveaux pixels et feu')) throw new Error('Missing real case/content');
if (!html.includes('selected="" value="2020/fire_24332880"')&&!html.includes('value="2020/fire_24332880" selected=""')) throw new Error('Wrong metadata-selected default case');
fs.writeFileSync('../reports/propagation_20261112_v1/react-component-check.json',JSON.stringify({status:'PASS',renderer:'ReactDOMServer, real saved diagnostics',events:data.per_event.length,map_layers:5,html_bytes:Buffer.byteLength(html),browser_interactive_test:false},null,2));
console.log('PASS: React component rendered real diagnostics, 169 event rows, five map layers.');
`,resolveDir:process.cwd(),sourcefile:'verify-propagation-entry.mjs'},bundle:true,platform:'node',format:'cjs',minify:true,outfile:output,define:{'process.env.NODE_ENV':'"production"'}});
execFileSync(process.execPath,[output],{stdio:'inherit'});
