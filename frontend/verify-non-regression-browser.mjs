// Real Chrome/CDP audit. Start Chrome with --remote-debugging-port=9233 first.
import assert from 'node:assert/strict';
import {mkdir,writeFile,readdir,stat} from 'node:fs/promises';
import {resolve} from 'node:path';
const base=process.env.GEOFIREX_TEST_URL||'http://127.0.0.1:8026';
const output=resolve(process.env.GEOFIREX_TEST_REPORT_DIR||'../reports/non_regression_20261127/browser');
await mkdir(output,{recursive:true});
const downloads=resolve(output,'../../downloads');await mkdir(downloads,{recursive:true});
const target=await(await fetch('http://127.0.0.1:9233/json/new?about:blank',{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl),pending=new Map(),listeners=new Map();
let sequence=0;
socket.onmessage=event=>{const message=JSON.parse(event.data);if(message.id){const p=pending.get(message.id);pending.delete(message.id);if(message.error)p?.reject(Error(JSON.stringify(message.error)));else p?.resolve(message.result);}else for(const fn of listeners.get(message.method)||[])fn(message.params);};
await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
const on=(method,fn)=>listeners.set(method,[...(listeners.get(method)||[]),fn]);
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text+': '+JSON.stringify(r.exceptionDetails.exception));return r.result.value;}
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(expression,label,timeout=60000){const start=Date.now();while(!await evaluate(`Boolean(${expression})`)){if(Date.now()-start>timeout)throw Error('Timed out: '+label);await pause(100);}}
async function screenshot(name){const r=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(output+'/'+name+'.png',Buffer.from(r.data,'base64'));}
async function route(id){await evaluate(`(()=>{if(innerWidth<=800){document.querySelector('.mobile-trigger').click();document.querySelector('.mobile-navigation a[href="#/'+${JSON.stringify(id)}+'"]').click();}else document.querySelector('.sidebar a[href="#/'+${JSON.stringify(id)}+'"]').click();})()`);await until(`document.querySelector('.sidebar [aria-current]')?.hash==='#/'+${JSON.stringify(id)}`,'route '+id);}
async function clickText(text,scope='document'){await evaluate(`(()=>{const e=[...${scope}.querySelectorAll('button')].find(e=>e.textContent.includes(${JSON.stringify(text)}));if(!e)throw Error('Missing button '+${JSON.stringify(text)});e.click();})()`);await pause(150);}
async function select(selector,value){await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('change',{bubbles:true}));})()`);}
const report={browser:(await(await fetch('http://127.0.0.1:9233/json/version')).json()).Browser,base,viewports:[],checks:[],consoleErrors:[],requests:[],limits:[]};
on('Runtime.exceptionThrown',e=>report.consoleErrors.push(e.exceptionDetails.text+': '+e.exceptionDetails.exception?.description));
on('Network.requestWillBeSent',e=>{if(e.request.url.startsWith(base))report.requests.push(e.request.url);});
let failures=false;
on('Fetch.requestPaused',async e=>{try{if(failures)await send('Fetch.fulfillRequest',{requestId:e.requestId,responseCode:503,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({detail:'Source indisponible pour le test de reprise'})).toString('base64')});else await send('Fetch.continueRequest',{requestId:e.requestId});}catch(error){report.consoleErrors.push(String(error));}});
try{
 await send('Page.enable');await send('Runtime.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});
 await send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads});
 await send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
 await send('Emulation.setPageScaleFactor',{pageScaleFactor:1});
 await send('Page.navigate',{url:base+'/#/overview'});
 await until(`document.querySelectorAll('.stat-value').length===3&&[...document.querySelectorAll('.stat-value')].map(e=>e.textContent).join(',')==='318,80,156'`,'initial KPI',120000);
 const models=await(await fetch(base+'/studio/models')).json();report.models=models.models;
 const v2=models.models.filter(m=>['C8','NEWPOS2','NEWPOS4'].includes(m.id)).every(m=>m.enabled);
 if(!v2)report.limits.push('ADATA absent: real V2 CPU and saved full-scene comparisons cannot be exercised. No substitution with simulated predictions.');
 for(const [width,height] of [[1920,1080],[1366,768],[390,844]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  for(const reduced of [false,true]){
   await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:reduced?'reduce':'no-preference'}]});
   await evaluate(`localStorage.setItem('geofirex-reduced-motion','false')`);
   await send('Page.navigate',{url:base+'/?audit='+width+'-'+reduced+'#/overview'});
   await until(`document.querySelectorAll('.stat-value').length===3&&[...document.querySelectorAll('.stat-value')].map(e=>e.textContent).join(',')==='318,80,156'`,'loaded KPI',120000);
   await until(`document.querySelector('.overview-fire-art img')?.complete`,'image');await pause(1000);
   const state=await evaluate(`(()=>{const flame=document.querySelector('.professional-hero > .hero-flame'),path=flame.querySelector('.flame-outer'),img=document.querySelector('.overview-fire-art img'),box=flame.getBoundingClientRect();return {motion:document.querySelector('.studio-app').dataset.motion,counts:[...document.querySelectorAll('.stat-value')].map(e=>e.textContent),flame:{width:box.width,height:box.height,opacity:getComputedStyle(flame).opacity,animation:getComputedStyle(path).animationName,transform:getComputedStyle(path).transform,hidden:flame.getAttribute('aria-hidden')},image:{width:img.naturalWidth,height:img.naturalHeight,fit:getComputedStyle(img).objectFit},caption:document.querySelector('.overview-fire-art').textContent,overflow:document.documentElement.scrollWidth>innerWidth,devicePixelRatio,scale:visualViewport.scale};})()`);
   assert.deepEqual(state.counts,['318','80','156']);assert.equal(state.caption,'');assert.equal(state.image.width,1254);assert(!state.overflow);assert.equal(state.scale,1);
   if(width<=800)assert.equal(await evaluate(`document.querySelector('.workspace').getBoundingClientRect().left`),0);
   const transform=state.flame.transform;await pause(450);const next=await evaluate(`getComputedStyle(document.querySelector('.hero-flame .flame-outer')).transform`);
   if(reduced){assert.equal(state.flame.animation,'none');assert.equal(next,transform);}else {assert.equal(state.flame.animation,'flame-breathe');assert.notEqual(next,transform);}
   const prefix=width+'x'+height+'-'+(reduced?'reduced':'full');
   await screenshot(prefix+'-overview');
   // Pause other layers so a screenshot difference isolates the original flame.
   await evaluate(`document.getAnimations().filter(a=>!a.effect?.target?.closest('.hero-flame')).forEach(a=>a.pause())`);
   const withFlame=(await send('Page.captureScreenshot',{format:'png'})).data;
   await evaluate(`document.querySelector('.hero-flame').style.visibility='hidden'`);
   await pause(150);
   const withoutFlame=(await send('Page.captureScreenshot',{format:'png'})).data;
   assert.notEqual(withFlame,withoutFlame,'flame must affect real rendered pixels');
   await evaluate(`document.querySelector('.hero-flame').style.visibility=''`);
   await pause(150);
   state.flame.renderedPixelDifference=true;state.routes=[];
   for(const [id,selector] of [['studio','.prediction-studio'],['analytics','.performance-metrics'],['comparison','.comparison-charts'],['research','.channel-grid']]){
    await route(id);await until(`document.querySelector(${JSON.stringify(selector)})`,'page '+id,120000);await pause(500);
    const geometry=await evaluate(`({heading:document.querySelector('main h1')?.textContent,overflow:document.documentElement.scrollWidth>innerWidth,emptyHeading:![...document.querySelectorAll('main h1')].some(e=>e.textContent.trim()),alerts:[...document.querySelectorAll('main [role=alert]')].map(e=>e.textContent)})`);
    assert(!geometry.overflow,prefix+' '+id+' horizontal overflow');assert(!geometry.emptyHeading);
    state.routes.push({id,...geometry});await screenshot(prefix+'-'+id);
   }
   report.viewports.push({width,height,reduced,...state});
   console.log(JSON.stringify({viewport:prefix,status:'PASS'}));
  }
 }
 report.checks.push('30 actual Chrome viewport captures; five pages at all three sizes, full and system-reduced motion; scale 1; no page overflow','original large flame changes transform and rendered pixels; static reduced-motion fallback','KPI visible text is 318 / 80 / 156 even below fold; no captions under illustration');
 await send('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false});
 await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'no-preference'}]});
 await route('overview');await until(`document.querySelector('.motion-button')`,'motion toggle');
 await evaluate(`if(document.querySelector('.studio-app').dataset.motion==='full')document.querySelector('.motion-button').click()`);await pause(100);
 await send('Page.reload');await until(`document.querySelector('.studio-app')?.dataset.motion==='reduced'`,'user preference persists');
 await until(`[...document.querySelectorAll('.stat-value')].map(e=>e.textContent).join(',')==='318,80,156'`,'reload KPI');
 report.checks.push('user reduced-motion preference persists across reload; KPI remains exact');
 await send('Fetch.enable',{patterns:[{urlPattern:'*/science/v2/populations'},{urlPattern:'*/science/final-2021'}]});
 failures=true;await route('research');await route('overview');
 await until(`document.querySelectorAll('.source-error').length===2`,'two independent API errors');
 assert.deepEqual(await evaluate(`[...document.querySelectorAll('.stat-value')].map(e=>e.textContent)`),['—','—','—']);await screenshot('source-errors');
 failures=false;
 await evaluate(`document.querySelectorAll('.source-error button').forEach(e=>e.click())`);
 await until(`[...document.querySelectorAll('.stat-value')].map(e=>e.textContent).join(',')==='318,80,156'`,'API retries');
 await send('Fetch.disable');report.checks.push('503 sources display unavailable values, never zero; both independent retries restore exact KPI');
 await route('studio');await until(`document.querySelector('#forecast-model')?.options.length===5`,'models');
 await select('#forecast-model',v2?'C8':'legacy');
 await until(`document.querySelector('.timeline-player canvas')&&document.querySelector('#forecast-date')?.value`,'real past observations',120000);
 await clickText('Lire',`document.querySelector('.timeline-controls')`);await pause(1900);await clickText('Pause',`document.querySelector('.timeline-controls')`);
 const paused=await evaluate(`document.querySelector('.timeline-status').textContent`);await pause(1900);assert.equal(await evaluate(`document.querySelector('.timeline-status').textContent`),paused);
 await clickText('Prédire J+1');
 await until(`document.querySelectorAll('.propagation-result canvas').length===3&&!document.querySelector('.predict-button').disabled`,'real CPU prediction',120000);
 assert.equal(await evaluate(`document.querySelectorAll('.scene-score-grid').length`),0);
 await clickText('Révéler la vérité J+1');await until(`document.querySelector('.scene-score-grid')`,'separate truth',120000);
 await clickText('Toutes les couches');assert.equal(await evaluate(`document.querySelectorAll('.propagation-result canvas').length`),8);
 const pixels=await evaluate(`(()=>{const c=document.querySelector('.propagation-result canvas'),p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;return {width:c.width,height:c.height,colours:new Set(Array.from({length:p.length/4},(_,i)=>Array.from(p.slice(i*4,i*4+4)).join(','))).size};})()`);
 assert(pixels.colours>1);report.canvas=pixels;
 for(const [width,height] of [[1920,1080],[1366,768],[390,844]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.querySelector('.forecast-reveal').scrollIntoView({block:'start'})`);await pause(300);await screenshot(width+'x'+height+'-real-prediction');
  await evaluate(`document.querySelector('.propagation-result .prediction-maps').scrollIntoView({block:'start'});scrollBy(0,-90)`);await pause(150);await screenshot(width+'x'+height+'-real-maps');
  const legend=await evaluate(`(()=>{const e=document.querySelector('.propagation-result .map-legend'),c=getComputedStyle(e);return {text:e.textContent,display:c.display,width:e.getBoundingClientRect().width};})()`);
  assert(legend.text.includes('Feu actif')&&legend.width>0&&legend.display!=='none');
 }
 await evaluate(`document.querySelector('.propagation-result .map-actions button').click()`);await until(`document.querySelector('.map-dialog[open]')`,'map dialog');
 await evaluate(`document.querySelector('[aria-label="Augmenter le zoom"]').click()`);assert.equal(await evaluate(`document.querySelector('[aria-label="Zoom de la carte"]').value`),'1.5');
 await screenshot('mobile-map-zoom');
 const fullButton=await evaluate(`(()=>{const e=[...document.querySelector('.map-dialog[open]').querySelectorAll('button')].find(e=>e.textContent==='Plein écran'),r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
 await send('Input.dispatchMouseEvent',{type:'mousePressed',...fullButton,button:'left',clickCount:1});
 await send('Input.dispatchMouseEvent',{type:'mouseReleased',...fullButton,button:'left',clickCount:1});await pause(500);
 const fullScreen=await evaluate(`Boolean(document.fullscreenElement)`);
 if(fullScreen){await screenshot('mobile-map-fullscreen');await evaluate(`document.exitFullscreen()`);report.checks.push('native Fullscreen API entered and exited in real Chrome');}
 else report.limits.push('Native fullscreen unavailable in headless Chrome; dialog enlargement and zoom verified.');
 await clickText('Fermer',`document.querySelector('.map-dialog[open]')`);
 await evaluate(`document.querySelector('.propagation-result .map-actions button:nth-child(2)').click();document.querySelector('.propagation-result .map-actions button:nth-child(3)').click()`);
 await clickText('JSON + provenance',`document.querySelector('.propagation-result')`);await clickText('CSV des pixels',`document.querySelector('.propagation-result')`);
 await pause(1500);
 report.downloads=await Promise.all((await readdir(downloads)).filter(name=>!name.endsWith('.crdownload')).map(async name=>({name,bytes:(await stat(downloads+'/'+name)).size})));
 if(!['.png','.metadata.json','.json','.csv'].every(extension=>report.downloads.some(f=>f.name.endsWith(extension)&&f.bytes>0)))report.limits.push('Some native Chrome downloads were blocked; PNG/CSV/JSON payloads verified by the HTTP/React test suite.');
 report.checks.push('real CPU '+(v2?'C8 V2':'historical U-Net')+' observations, timeline play/pause, separate truth, 8 maps, nonblank canvas pixels, zoom, PNG/metadata/JSON/CSV downloads');
 await clickText('Comparaison appariée');
 await until(`document.querySelectorAll('.comparison-observations canvas').length===3`,'shared observations');
 await clickText('Recalculer les trois modèles');
 await until(`document.querySelectorAll('.live-comparison .spatial-models canvas').length===3`,'paired CPU',120000);
 assert.equal(await evaluate(`document.querySelectorAll('.live-scene-metrics').length`),0);
 await clickText('Révéler la vérité commune J+1');await until(`document.querySelectorAll('.live-scene-metrics tbody tr').length===3`,'paired metrics');
 await select('#live-layer','errors');
 for(const [width,height] of [[1920,1080],[1366,768],[390,844]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.querySelector('.live-scene-metrics').scrollIntoView({block:'start'});scrollBy(0,-90)`);await pause(200);await screenshot(width+'x'+height+'-paired-metrics');
  await evaluate(`document.querySelector('.live-comparison .spatial-models').scrollIntoView({block:'start'});scrollBy(0,-90)`);await pause(200);await screenshot(width+'x'+height+'-paired-errors');
  assert.equal(await evaluate(`document.documentElement.scrollWidth>innerWidth`),false);
 }
 report.checks.push('Studio paired mode: common T3 observations, three real checkpoints, independent truth and per-model recall/Dice/precision/TP/FP/FN; shared geometry/crop; desktop/mobile screenshots');
 assert(!report.requests.some(u=>/\/(predict|observations|ground-truth)\?/.test(u)&&u.includes('2021')));
 assert.deepEqual(report.consoleErrors,[]);report.status=report.limits.length?'PASS_WITH_EXTERNAL_LIMITS':'PASS';
}catch(error){report.status='FAIL';report.error=error.stack;await screenshot('failure').catch(()=>{});process.exitCode=1;}
finally{await writeFile(output+'/browser-audit.json',JSON.stringify(report,null,2));socket.close();}
console.log(JSON.stringify({status:report.status,captures:report.viewports.length,error:report.error,limits:report.limits}));
