import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {deflateSync,inflateSync} from 'node:zlib';
import {Window,EventTarget as DOMEventTarget} from 'happy-dom';
const base=process.env.GEOFIREX_TEST_URL||'http://127.0.0.1:8014';
const output=process.env.GEOFIREX_TEST_REPORT_DIR||'../reports/ui_studio_v2';await mkdir(output,{recursive:true});
const requests=[],downloads=[],errors=[],checks=[];
const nodeFetch=globalThis.fetch;
let failCatalog=false;
const window=new Window({url:base+'/#/overview'}),document=window.document;
document.documentElement.lang='fr';document.body.innerHTML='<div id="root"></div>';
for(const key of ['window','document','navigator','CSS','Document','DocumentFragment','ShadowRoot','HTMLElement','HTMLInputElement','HTMLSelectElement','HTMLCanvasElement','SVGElement','Element','Node','EventTarget','Event','MouseEvent','KeyboardEvent','CustomEvent','MutationObserver','getComputedStyle','requestAnimationFrame','cancelAnimationFrame','localStorage']){
 const value=window[key];Object.defineProperty(globalThis,key,{value:typeof value==='function'&&['getComputedStyle','requestAnimationFrame','cancelAnimationFrame'].includes(key)?value.bind(window):value,configurable:true,writable:true});
}
globalThis.IS_REACT_ACT_ENVIRONMENT=true;
// Happy DOM's per-window EventTarget constructor differs from the base inherited
// by its nodes. Motion resolves elements with instanceof EventTarget.
globalThis.EventTarget=DOMEventTarget;
// Happy DOM's rAF is unthrottled. Use a 16 ms test clock, not a claimed browser FPS.
globalThis.requestAnimationFrame=window.requestAnimationFrame=callback=>setTimeout(()=>callback(performance.now()),16);
globalThis.cancelAnimationFrame=window.cancelAnimationFrame=timer=>clearTimeout(timer);
let systemPrefersReduced=true;
const mediaListeners=new Set();
window.matchMedia=query=>({get matches(){return query.includes('prefers-reduced-motion')&&systemPrefersReduced;},media:query,addEventListener(event,fn){if(query.includes('prefers-reduced-motion'))mediaListeners.add(fn);},removeEventListener(event,fn){mediaListeners.delete(fn);},addListener(){},removeListener(){}});
window.localStorage.setItem('geofirex-reduced-motion','true');
globalThis.matchMedia=window.matchMedia;
class ResizeObserver{observe(){}unobserve(){}disconnect(){}}
class IntersectionObserver{constructor(callback){this.callback=callback;}observe(target){queueMicrotask(()=>this.callback([{target,isIntersecting:true,intersectionRatio:1}],this));}unobserve(){}disconnect(){}}
globalThis.ResizeObserver=window.ResizeObserver=ResizeObserver;
globalThis.IntersectionObserver=window.IntersectionObserver=IntersectionObserver;
window.scrollTo=()=>{};
window.HTMLElement.prototype.scrollTo=()=>{};
const waapiCalls=[];
window.Element.prototype.getAnimations=()=>[];
window.Element.prototype.animate=function(frames,options){waapiCalls.push({className:this.getAttribute('class'),frames,options});const animation={cancel(){},finish(){},finished:Promise.resolve()};queueMicrotask(()=>animation.onfinish?.());return animation;};
const dialogProto=window.HTMLDialogElement.prototype;
dialogProto.showModal=function(){this.setAttribute('open','');this.querySelector('[autofocus],button,input')?.focus();};
dialogProto.close=function(){this.removeAttribute('open');this.dispatchEvent(new window.Event('close'));};
const crc32=data=>{let crc=0xffffffff;for(const byte of data){crc^=byte;for(let b=0;b<8;b++)crc=crc>>>1^((crc&1)?0xedb88320:0);}return (crc^0xffffffff)>>>0;};
function chunk(type,body){const name=Buffer.from(type),size=Buffer.alloc(4),crc=Buffer.alloc(4);size.writeUInt32BE(body.length);crc.writeUInt32BE(crc32(Buffer.concat([name,body])));return Buffer.concat([size,name,body,crc]);}
function png(canvas){const width=canvas.width,height=canvas.height,header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=8;header[9]=6;const pixels=Buffer.from(canvas.__pixels||new Uint8Array(width*height*4)),lines=[];for(let y=0;y<height;y++)lines.push(Buffer.from([0]),pixels.subarray(y*width*4,(y+1)*width*4));return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(Buffer.concat(lines))),chunk('IEND',Buffer.alloc(0))]);}
// Instrument WebGL calls without pretending to render a GPU image.
let webglAvailable=false,glDraws=0,glDisposed=0;const glValues={};
const uniforms=['uResolution','uTime','uStops','uStopCount','uAmplitude','uBlend','uHeight','uScale','uGlow','uBrightness','uLayers','uFlip','uPointer','uReveal','uOpacity','uLightMode'];
const gl=new Proxy({},{get(target,key){if(key==='getProgramParameter')return (p,k)=>k==='ACTIVE_UNIFORMS'?uniforms.length:true;if(key==='getShaderParameter')return()=>true;if(key==='getActiveUniform')return(p,i)=>({name:uniforms[i]});if(key==='getUniformLocation')return(p,name)=>name;if(key==='getExtension')return()=>({loseContext(){glDisposed++;}});if(key==='drawArrays')return()=>glDraws++;if(key==='uniform1f'||key==='uniform1i')return(name,value)=>{(glValues[name]||=[]).push(value);};if(String(key).startsWith('create'))return()=>({});if(String(key).toUpperCase()===key)return key;return()=>{};}});
window.HTMLCanvasElement.prototype.getContext=function(kind){if(kind==='webgl2')return webglAvailable?gl:null;if(kind!=='2d')return null;const canvas=this;return {createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData:image=>{canvas.__pixels=image.data;},clearRect(){},fillRect(){},getImageData(){const hex=this.fillStyle.replace('#','');return {data:canvas.__pixels||new Uint8ClampedArray([parseInt(hex.slice(0,2),16),parseInt(hex.slice(2,4),16),parseInt(hex.slice(4,6),16),255])};},fillStyle:'#000000'};};
window.HTMLCanvasElement.prototype.toDataURL=function(){return 'data:image/png;base64,'+png(this).toString('base64');};
let nextBlob=0;const blobs=new Map();
globalThis.Blob=window.Blob;
globalThis.URL.createObjectURL=blob=>{const url='blob:studio-test-'+nextBlob++;blobs.set(url,blob);return url;};
globalThis.URL.revokeObjectURL=()=>{};
window.HTMLAnchorElement.prototype.click=function(){downloads.push({name:this.download,href:this.href,blob:blobs.get(this.href)});};
globalThis.fetch=async(input,options)=>{
 const url=new URL(String(input),base).href;requests.push(url);
 if(failCatalog&&url.endsWith('/lab/events'))return new Response(JSON.stringify({detail:'Catalogue temporairement indisponible'}),{status:503,headers:{'Content-Type':'application/json'}});
 return nodeFetch(url,options);
};
const originalError=console.error;console.error=(...args)=>{const text=args.map(String).join(' ');if(!text.includes('not wrapped in act'))errors.push(text);};
const React=await import('react'),{act}=React,{createRoot}=await import('react-dom/client'),{default:StudioApp,pageFromHash,pages}=await import('../src/StudioApp.mjs');
const {imagePixels,forecastCSV}=await import('../src/prediction_view.mjs');
const h=React.createElement,root=createRoot(document.getElementById('root'));
const {default:axe}=await import('axe-core');const a11yByPage={};
async function accessibilityAudit(page){const data=await axe.run(document,{runOnly:{type:'rule',values:['aria-valid-attr','aria-valid-attr-value','aria-required-attr','aria-required-children','aria-required-parent','aria-dialog-name','button-name','link-name','image-alt','label','document-title','html-has-lang','landmark-one-main','heading-order']}});a11yByPage[page]={violations:data.violations,incomplete:data.incomplete.map(v=>v.id),passed_rules:data.passes.map(v=>v.id)};assert.deepEqual(data.violations.map(v=>v.id),[],page+' accessibility');}
async function tick(){await act(async()=>{await new Promise(resolve=>setTimeout(resolve,35));});}
async function until(test,label,timeout=25000){const start=Date.now();while(!test()){if(Date.now()-start>timeout)throw Error('Timed out: '+label);await tick();}}
async function click(el){assert(el,'Missing clickable element');await act(async()=>el.dispatchEvent(new window.MouseEvent('click',{bubbles:true})));await tick();}
const button=text=>[...document.querySelectorAll('button')].find(e=>e.textContent.includes(text));
const ariaButton=label=>document.querySelector(`button[aria-label="${label}"]`);
async function route(id){await click(document.querySelector(`.sidebar a[href="#/${id}"]`));await until(()=>document.querySelector('.sidebar [aria-current]')?.getAttribute('href')==='#/'+id,'route '+id);assert.equal(document.querySelectorAll('main').length,1);assert.equal(document.querySelectorAll('h1').length,1);checks.push('navigation:'+id);}
async function change(select,value){await act(async()=>{const proto=select instanceof window.HTMLInputElement?window.HTMLInputElement.prototype:window.HTMLSelectElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(select,value);select.dispatchEvent(new window.Event('input',{bubbles:true}));select.dispatchEvent(new window.Event('change',{bubbles:true}));});await tick();}
const {default:ScientificComparison,tileCounts}=await import('../src/ScientificComparison.mjs');
try{
 await act(async()=>root.render(h(ScientificComparison)));
 await until(()=>document.querySelectorAll('tbody tr').length===4&&document.querySelectorAll('canvas').length===8,'four real V2 results and eight maps');
 assert(document.body.textContent.includes('95')&&document.body.textContent.includes('FIRE13'));
 const data=await (await nodeFetch(base+'/science/v2/diagnostic-tile')).json();
 const offsetResult={...data.cases[0].result,tile_origin:[64,128]};
 const offsetRow=forecastCSV(offsetResult,data.cases[0].truth).split('\n')[1].split(',');
 assert.equal(offsetRow[6],'64');assert.equal(offsetRow[7],'128');
 assert.deepEqual(tileCounts({truth:[[1,0,1,0]],result:{observed_fire:[[0,0,0,0]],predicted_fire:[[1,1,0,0]]}}),{tp:1,fp:1,fn:1,tn:1,recall:.5,precision:.5,dice:.5});
 for(const model of ['C8','NEWPOS2','NEWPOS4','FIRE13']){
  await change(document.getElementById('v2-model'),model);
  const item=data.cases.find(c=>c.model===model),counts=tileCounts(item);
  assert(document.body.textContent.includes(`TP ${counts.tp}`));
  assert(document.body.textContent.includes(`FP ${counts.fp}`));
  assert(document.body.textContent.includes(`FN ${counts.fn}`));
  assert.equal(document.querySelectorAll('canvas').length,8);
  assert.deepEqual(Buffer.from(document.querySelector('canvas').__pixels),Buffer.from(imagePixels(item.result.observed_fire).pixels));
  checks.push('saved_layers:'+model);
 }
 await click(button('Exporter JSON + provenance'));
 const jsonExport=downloads.find(d=>d.name==='geofirex-validation2020-V2-80.json');
 assert(jsonExport?.blob);const exported=JSON.parse(await jsonExport.blob.text());
 assert.equal(exported.models.length,4);assert.equal(exported.models[3].FP_new,95053);
 const csv=await (await nodeFetch(base+'/science/v2/export.csv')).text();
 assert.equal(csv.trim().split('\n').length,5);assert(csv.includes('95053'));
 await click(button('Agrandir'));assert(document.querySelector('dialog[open]'));
 await click(button('Fermer'));assert(!document.querySelector('dialog[open]'));
 await click(button('Exporter PNG'));
 const pngDownload=downloads.find(d=>d.href.startsWith('data:image/png'));
 assert(pngDownload);const pngBytes=Buffer.from(pngDownload.href.split(',')[1],'base64');
 assert.equal(pngBytes.subarray(1,4).toString(),'PNG');
 checks.push('exports:JSON/CSV/PNG','map_enlargement_and_close:DOM');
 await act(async()=>root.render(h(StudioApp)));
 await until(()=>document.querySelector('.stat-value')?.getAttribute('aria-label')==='250','jury overview');
 assert(pages.some(p=>p.id==='comparison'));assert(!pages.some(p=>p.id==='data'));
 await route('comparison');await until(()=>document.querySelectorAll('tbody tr').length===4,'V2 page');
 await accessibilityAudit('comparison');
 await route('studio');await until(()=>document.getElementById('forecast-event')?.options.length===202,'real historical catalogue');
 await change(document.getElementById('forecast-event'),'2020/fire_24332880');
 await change(document.getElementById('forecast-date'),'2020-09-03');
 await click(document.querySelector('.predict-button'));
 await until(()=>document.querySelectorAll('canvas').length===8&&!document.body.textContent.includes('Inf\u00e9rence en cours'),'actual PyTorch inference and separate truth',60000);
 const predictIndex=requests.findIndex(u=>u.includes('/lab/predict?'));
 const truthIndex=requests.findIndex(u=>u.includes('/lab/ground-truth?'));
 assert(predictIndex>=0&&truthIndex>predictIndex);
 assert(document.body.textContent.includes('7295eaa6f957'));
 assert(document.body.textContent.includes('sans g\u00e9or\u00e9f\u00e9rencement'));
 assert(!requests.some(u=>u.includes('/science/big-data')));
 await accessibilityAudit('studio');
 checks.push('live_CPU_inference_legacy201','truth_requested_after_prediction','no_Spark_route');
 await route('analytics');await until(()=>document.body.textContent.includes('Mod\u00e8le principal'),'analytics');
 await route('research');await until(()=>document.body.textContent.includes('Lire les r\u00e9sultats'),'research');
 await accessibilityAudit('research');
 await writeFile(output+'/jury-dom.html',document.documentElement.outerHTML);
 assert.deepEqual(errors,[]);
 const receipt={status:'PASS',kind:'React DOM simulator; not a browser layout check',checks,a11y:a11yByPage,downloads:downloads.map(d=>d.name),requests,visual_browser:'NOT_VERIFIED: browser host unavailable, native app approval timed out',training_executed:false};
 await writeFile(output+'/jury-dom.json',JSON.stringify(receipt,null,2));
 console.log(JSON.stringify({status:'PASS',checks:checks.length,exports:downloads.length,actual_inference:true,browser_layout_verified:false}));
}finally{
 console.error=originalError;
 await act(async()=>root.unmount());
 await window.happyDOM.abort();
 const {ScrollTrigger}=await import('gsap/ScrollTrigger');
 const {gsap}=await import('gsap');ScrollTrigger.killAll();ScrollTrigger.disable();gsap.ticker.sleep();window.close();
}
// All assertions, receipt writes and teardown must complete before exit.
process.exit(0);
