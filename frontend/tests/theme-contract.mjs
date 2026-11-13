import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';

const geometryProperties=/^(--sidebar|(?:min-|max-)?(?:width|height)|grid-template-(?:columns|rows)|grid-(?:column|row)|gap|row-gap|column-gap|padding(?:-.+)?|margin(?:-.+)?|inset|top|bottom|left|right|display|position|flex(?:-.+)?)$/;
export function geometryOf(css){
 // Native stylesheet uses no brace-containing strings/URLs. Walk its blocks,
 // including media queries; keep declaration order and duplicate selectors.
 const result={};css=css.replace(/\/\*[\s\S]*?\*\//g,'');
 function visit(source,context=''){
  const occurrences={};let cursor=0;
  while(cursor<source.length){const open=source.indexOf('{',cursor);if(open<0)break;const header=source.slice(cursor,open).trim();let end=open+1,depth=1;while(depth&&end<source.length){if(source[end]==='{')depth++;if(source[end]==='}')depth--;end++;}assert.equal(depth,0,'Unbalanced CSS');const body=source.slice(open+1,end-1);cursor=end;
   if(header.startsWith('@media')||header.startsWith('@supports')){visit(body,context+' / '+header);continue;}if(header.startsWith('@'))continue;
   const selector=context+' / '+header,key=selector+' #'+(occurrences[selector]=(occurrences[selector]||0)+1),properties={};
   for(const declaration of body.split(';')){const colon=declaration.indexOf(':');if(colon<0)continue;const name=declaration.slice(0,colon).trim();if(geometryProperties.test(name))properties[name]=declaration.slice(colon+1).trim();}
   if(Object.keys(properties).length)result[key]=properties;
  }
 }
 visit(css);return result;
}
function luminance(hex){const c=hex.slice(1).match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return .2126*c[0]+.7152*c[1]+.0722*c[2];}
const contrast=(a,b)=>(Math.max(luminance(a),luminance(b))+.05)/(Math.min(luminance(a),luminance(b))+.05);
export async function verifyTheme(){
 const css=await readFile('src/style.css','utf8'),baseline=JSON.parse(await readFile('tests/layout-v1.json','utf8')),current=geometryOf(css);
 assert(Object.keys(baseline).length>100,'Complete V1 geometry baseline required');
 for(const [rule,properties]of Object.entries(baseline))assert.deepEqual(current[rule],properties,'V1 geometry changed: '+rule);
 const root=css.match(/:root\{([^}]+)\}/)[1],tokens=Object.fromEntries([...root.matchAll(/(--[\w-]+):([^;}]+)/g)].map(m=>[m[1],m[2]]));
 const resolve=name=>tokens[name].startsWith('var(')?resolve(tokens[name].slice(4,-1)):tokens[name];
 const tests=[];
 for(const fg of ['--text','--text-secondary','--accent'])for(const bg of ['--bg','--surface','--surface-secondary','--accent-soft','--hover'])tests.push([fg,bg,4.5]);
 tests.push(['--button-text','--button-bg',4.5],['--button-text','--button-hover',4.5],['--danger-ink','--danger-bg',4.5],['--warning','#FFF5D6',4.5],['--success','--success-bg',4.5],['--focus','--surface',3],['--border-control','--surface',3],['--border-control','--bg',3]);
 const pairs=tests.map(([fg,bg,minimum])=>({foreground:fg,background:bg,ratio:contrast(fg.startsWith('--')?resolve(fg):fg,bg.startsWith('--')?resolve(bg):bg),minimum}));
 for(const pair of pairs)assert(pair.ratio>=pair.minimum,JSON.stringify(pair));
 assert.equal(tokens['--bg'],'#FFF8F0');assert(css.includes('color-scheme:light'));
 assert(css.includes('[data-motion=reduced]'));assert(css.includes('@media(prefers-reduced-motion:reduce)'));
 const report={status:'PASS',preserved_geometry_rules:Object.keys(baseline).length,contrast_pairs:pairs,limits:'Token contrast calculation and CSS declaration comparison only; no browser compositing, visual motion or responsive layout measurement.'};
 const output=process.env.GEOFIREX_TEST_REPORT_DIR||'../reports/ui_studio_v2';await mkdir(output,{recursive:true});await writeFile(output+'/theme_contract.json',JSON.stringify(report,null,2));return report;
}
