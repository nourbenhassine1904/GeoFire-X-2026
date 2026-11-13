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
try{
 assert.equal(pageFromHash('#/studio'),'studio');assert.equal(pageFromHash('#/unknown'),'overview');assert.equal(pages.length,5);
 await act(async()=>root.render(h(StudioApp)));
 await until(()=>document.querySelector('.stat-value')?.getAttribute('aria-label')==='250'&&document.querySelector('.raster-preview img'),'real overview');
 assert(document.body.textContent.includes('23 404')||document.body.textContent.includes('23 404'));
 assert.equal(document.querySelector('.studio-app').dataset.motion,'reduced');
 const overviewImage=document.querySelector('.raster-preview img');assert.equal((await nodeFetch(new URL(overviewImage.getAttribute('src'),base))).status,200);
 checks.push('overview:real-counts-saved-raster-reduced-motion');
 await click(ariaButton('Navigation rapide'));assert(document.querySelector('.command-dialog').open);
 await click([...document.querySelectorAll('.command-results button')].find(e=>e.textContent.includes('Research')));
 await until(()=>button('Exporter la provenance')&&!button('Exporter la provenance').disabled,'research metadata');
 await click(button('Exporter la provenance'));const provenance=JSON.parse(await downloads.at(-1).blob.text());assert.equal(provenance.compact.roles.train.events,250);assert.equal(provenance.selection.threshold,.78);checks.push('research:provenance-export');await accessibilityAudit('research');
 await route('analytics');await until(()=>document.body.textContent.includes('169 événements · seuils fixés'),'unet evidence');
 const tabs=[...document.querySelectorAll('[role=tab]')];
 await click(tabs[1]);await until(()=>document.body.textContent.includes('Chaque événement compte'),'historical metrics');
 await click(button('32 événements'));await until(()=>document.body.textContent.includes('32 événements complets.'),'32-event scope');
 await click(tabs[2]);await until(()=>document.body.textContent.includes('Critères d’acceptation : FALSE'),'comparative head');assert(document.body.textContent.includes('Le modèle principal reste U-Net poids10'));
 await click(tabs[3]);assert(document.body.textContent.includes('Connexions de saut'));checks.push('analytics:all-tabs-frozen-model-metrics');await accessibilityAudit('analytics');
 await route('data');await until(()=>document.body.textContent.includes('Inventaire annuel'),'big data');assert(document.body.textContent.includes('13 607')||document.body.textContent.includes('13 607'));assert(document.body.textContent.includes('local[2]'));checks.push('data:spark-parquet-existing-results');await accessibilityAudit('data');
 await route('studio');await until(()=>document.querySelector('#forecast-event')?.options.length===202,'201-event catalogue');
 await click(ariaButton('Rechercher un événement'));assert(document.querySelector('.event-picker').open);
 const list=document.querySelector('.event-picker [role=listbox]');assert.equal(list.getAttribute('aria-label'),'Événements historiques disponibles');assert.equal(document.querySelectorAll('.event-picker [role=option]').length,201);
 await click([...document.querySelectorAll('.event-picker [role=option]')].find(e=>e.textContent.includes('2020/fire_24332880')));assert(!document.querySelector('.event-picker').open);
 await change(document.querySelector('#forecast-date'),'2020-09-03');assert.equal(document.querySelector('.target-date').textContent,'2020-09-04');
 await click(button('Date suivante'));assert.equal(document.querySelector('#forecast-date').value,'2020-09-04');await click(button('Date précédente'));
 await click(button('Prédire J+1'));await until(()=>document.querySelectorAll('canvas[role=img]').length===8&&!button('Prédire J+1')?.disabled,'real inference and separate truth',45000);
 assert(document.body.textContent.includes('Aucun pixel de J+1 lu'));assert(document.body.textContent.includes('0.78'));
 const forecastRequests=requests.filter(u=>u.includes('/lab/predict?')||u.includes('/lab/ground-truth?'));assert.equal(forecastRequests.length,2);assert(forecastRequests[0].includes('/predict?'));assert(forecastRequests[1].includes('/ground-truth?'));
 checks.push('studio:event-list-date-navigation-real-inference-target-separated');
 await click(button('JSON + provenance'));const exported=JSON.parse(await downloads.at(-1).blob.text());assert.equal(exported.prediction.threshold,.78);assert.equal(exported.prediction.target_pixels_read,false);assert.equal(exported.prediction.optimizer_updates,23404);assert.equal(exported.prediction.crs,null);
 await click(button('CSV des pixels'));const csv=await downloads.at(-1).blob.text();assert.equal(csv,forecastCSV(exported.prediction,exported.ground_truth));
 await click(button('Exporter PNG'));const pngData=Buffer.from(downloads.at(-1).href.split(',')[1],'base64'),pixels=imagePixels(exported.prediction.observed_fire);assert.equal(pngData.readUInt32BE(16),pixels.width);assert.equal(pngData.readUInt32BE(20),pixels.height);assert.equal(pngData.subarray(1,4).toString(),'PNG');assert.equal(document.querySelector('canvas').__pixels.length,pixels.pixels.length);
 await click(button('Agrandir'));assert(document.querySelector('.map-dialog').open);const zoom=document.querySelector('.map-dialog input');await change(zoom,'2');assert(document.querySelector('.map-dialog img').style.width==='200%');await click(button('Fermer'));assert(!document.querySelector('.map-dialog').open);
 checks.push('exports:json-csv-canvas-rgba-png-dialog-zoom');await accessibilityAudit('studio');
 await route('overview');await until(()=>document.querySelector('.raster-preview img'),'overview reload');
 failCatalog=true;await route('studio');await until(()=>document.querySelector('[role=alert]')?.textContent.includes('temporairement'),'catalogue error');assert(!document.querySelector('canvas'));failCatalog=false;await click(button('Réessayer le catalogue'));await until(()=>document.querySelector('#forecast-event')?.options.length===202&&!document.querySelector('[role=alert]'),'catalogue recovery');checks.push('errors:catalogue-unavailable-retry-no-fabricated-result');
 await route('overview');await until(()=>document.querySelector('.raster-preview img'),'accessible overview');
 await click(ariaButton('Ouvrir la navigation'));assert(document.querySelector('.mobile-navigation').open);await click(document.querySelector('.mobile-navigation a[href="#/research"]'));await until(()=>document.querySelector('h1')?.textContent==='Research','mobile navigation');assert(!document.querySelector('.mobile-navigation').open);checks.push('mobile:navigation-dialog-open-select-close');
 await act(async()=>window.dispatchEvent(new window.KeyboardEvent('keydown',{key:'k',ctrlKey:true,bubbles:true})));assert(document.querySelector('.command-dialog').open);await change(document.querySelector('#command-input'),'Model');assert.equal(document.querySelectorAll('.command-results button').length,1);await act(async()=>document.querySelector('#command-input').dispatchEvent(new window.KeyboardEvent('keydown',{key:'Enter',bubbles:true})));await until(()=>document.querySelector('h1')?.textContent==='Model Analytics','keyboard command navigation');
 const firstTab=document.querySelector('[role=tab]');await act(async()=>firstTab.dispatchEvent(new window.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true})));assert.equal(document.querySelector('[aria-selected=true]').textContent,'CNN / U-Net');checks.push('keyboard:command-search-enter-and-analytics-arrow-keys');
 await route('overview');await until(()=>document.querySelector('.raster-preview img'),'motion overview');
 assert.equal(waapiCalls.length,0,'No native step animations in system reduced-motion');
 assert(ariaButton('Animations réduites par le système').disabled);await act(async()=>{systemPrefersReduced=false;for(const fn of mediaListeners)fn();});await click(ariaButton('Activer les animations'));await until(()=>document.querySelector('.studio-app').dataset.motion==='full','animations enabled');
 assert.equal(document.querySelector('[data-react-bits=Aurora]').dataset.animationState,'webgl-unavailable');
 checks.push('motion:aurora-no-webgl-static-warm-backdrop');
 await click(ariaButton('Réduire les animations'));webglAvailable=true;await click(ariaButton('Activer les animations'));
 await until(()=>glDraws>2&&document.querySelector('.hero-title .split-word'),'Aurora draw and actual GSAP word split');
 assert.equal(document.querySelector('[data-react-bits=Aurora]').dataset.animationState,'running');assert(glValues.uLightMode.every(v=>v===1));assert(new Set(glValues.uTime).size>1);
 assert.equal(document.querySelectorAll('[data-react-bits=SpotlightCard]').length,4);
 assert.equal(document.querySelectorAll('[data-react-bits=CountUp]').length,4);
 await until(()=>[...document.querySelectorAll('[data-react-bits=CountUp]')].every(e=>e.dataset.animationState==='complete'),'four real counters finish',10000);
 assert.deepEqual([...document.querySelectorAll('.stat-value')].map(e=>e.getAttribute('aria-label')),['250','201','607','23 404']);
 assert.deepEqual([...document.querySelectorAll('.stat-value>span:first-child')].map(e=>e.textContent),['250','201','607','23 404']);
 await until(()=>document.querySelector('.hero-title').dataset.animationState==='complete','GSAP title completes');
 const words=document.querySelectorAll('.hero-title .split-word').length;assert(words>=5);await tick();assert.equal(document.querySelectorAll('.hero-title .split-word').length,words);
 checks.push('motion:aurora-time-light-mode-split-text-and-four-exact-countups');
 const spotlight=document.querySelector('.stat-card');spotlight.getBoundingClientRect=()=>({left:0,top:0,right:250,bottom:200,width:250,height:200,x:0,y:0});
 await act(async()=>window.dispatchEvent(new window.MouseEvent('pointermove',{clientX:100,clientY:80,bubbles:true})));
 await until(()=>Number(spotlight.querySelector('.spotlight-card__light').style.opacity)>.05,'real spotlight pointer light');
 checks.push('motion:spotlight-pointer-activates-real-light-layer');
 const {ScrollTrigger}=await import('gsap/ScrollTrigger');
 assert.equal(document.querySelector('[data-react-bits=FadeContent]').dataset.animationState,'complete');
 // Give an isolated below-viewport Fade realistic geometry: Happy DOM has no layout.
 const {Fade,MotionContext}=await import('../src/studio_ui.mjs');const probeHost=document.createElement('div');document.body.appendChild(probeHost);const probeRoot=createRoot(probeHost);
 const originalBox=window.Element.prototype.getBoundingClientRect;
 window.Element.prototype.getBoundingClientRect=function(){return this.classList?.contains('viewport-probe')?{top:1800,bottom:2100,left:0,right:600,width:600,height:300,x:0,y:1800}:originalBox.call(this);};
 Object.defineProperty(document.documentElement,'scrollHeight',{value:5000,configurable:true});
 await act(async()=>probeRoot.render(h(MotionContext.Provider,{value:{enabled:true}},h(Fade,{className:'viewport-probe'},h('p',null,'Viewport probe')))));
 const fadeTrigger=ScrollTrigger.getAll().find(t=>t.trigger?.classList.contains('viewport-probe'));assert(fadeTrigger,'Below-viewport ScrollTrigger created');assert.equal(fadeTrigger.trigger.dataset.animationState,'waiting');
 await act(async()=>fadeTrigger.vars.onEnter());await until(()=>document.querySelector('.viewport-probe').dataset.animationState==='complete','viewport callback runs actual fade timeline');
 await act(async()=>probeRoot.unmount());probeHost.remove();window.Element.prototype.getBoundingClientRect=originalBox;delete document.documentElement.scrollHeight;
 checks.push('motion:viewport-trigger-plays-fadecontent-timeline');
 await route('analytics');await until(()=>document.querySelector('.recall-limit strong'),'new fire limit');assert(document.querySelector('.recall-limit strong').textContent.includes('14,68'));assert(document.querySelector('.sparse-map-note'));
 await click(document.querySelectorAll('[role=tab]')[3]);await until(()=>document.querySelector('#analytics-panel [data-react-bits=FadeContent]').dataset.animationState==='complete','tab actual fade');checks.push('motion:analytics-tab-fade-and-actual-low-recall');
 await route('data');await until(()=>document.querySelectorAll('.annual-bar').length===7,'annual and split chart');assert.equal(document.querySelectorAll('.pipeline-node').length,4);assert(document.querySelector('.annual-chart').getAttribute('aria-label').includes('2021 test 156'));assert.equal(document.querySelector('.annual-bar[data-split=val]').style.width,'80.4%');checks.push('data:annual-split-bars-actual-metadata-and-pipeline');
 await route('studio');await until(()=>document.querySelector('#forecast-event')?.options.length===202,'animated studio');assert(!document.querySelector('.animated-list'));assert(!document.querySelector('.hero-aurora'));
 const stoppedDraws=glDraws;await tick();assert.equal(glDraws,stoppedDraws);assert(glDisposed>0);
 await click(ariaButton('Rechercher un événement'));await until(()=>document.querySelector('.animated-list__item[data-seen=true]'),'animated list entrance');assert(document.querySelector('.animated-list[data-animated]'));assert(document.querySelector('.animated-list__item').style.transition.includes('opacity'));
 await click(ariaButton('Fermer les événements'));assert(!document.querySelector('.animated-list'));
 const waapiBefore=waapiCalls.length;await click(button('Prédire J+1'));assert(document.querySelector('.inference-spinner'));assert.equal(document.querySelector('.prediction-stepper').dataset.currentStep,'2');
 await until(()=>document.querySelectorAll('canvas[role=img]').length===8&&!button('Prédire J+1')?.disabled,'normal motion inference',45000);
 assert.equal(document.querySelector('.prediction-stepper').dataset.currentStep,'4');assert(waapiCalls.length>waapiBefore);await until(()=>document.querySelector('.forecast-reveal').dataset.animationState==='complete','result fade');
 const rasterCopy=[...document.querySelectorAll('canvas[role=img]')].map(c=>Buffer.from(c.__pixels).toString('base64'));await tick();assert.deepEqual([...document.querySelectorAll('canvas[role=img]')].map(c=>Buffer.from(c.__pixels).toString('base64')),rasterCopy);
 checks.push('motion:list-dialog-entrance-stepper-real-busy-result-fade-raster-stable-no-aurora');
 await route('overview');await until(()=>document.querySelector('[data-react-bits=SpotlightCard]'),'motion return');
 await click(ariaButton('Réduire les animations'));assert.equal(document.querySelector('.studio-app').dataset.motion,'reduced');assert.equal(localStorage.getItem('geofirex-reduced-motion'),'true');assert(!document.querySelector('[data-react-bits=Aurora]'));assert(!document.querySelector('[data-react-bits=CountUp]'));assert(!document.querySelector('[data-react-bits=FadeContent]'));assert(!document.querySelector('.split-word'));assert.equal(document.querySelectorAll('[data-motion-fallback=SpotlightCard]').length,4);
 checks.push('motion:system-preference-live-change-user-toggle-static-fallback-and-persistence');
 await accessibilityAudit('overview');await writeFile(output+'/accessibility_dom.json',JSON.stringify({environment:'DOM simulator; structural checks only, not browser contrast or geometry',pages:a11yByPage},null,2));checks.push('a11y:five-pages-structural-axe-rules-and-single-main-heading');
 assert(!requests.some(u=>u.includes('/predict?')&&u.includes('2021')));
 assert.deepEqual(errors,[]);
 const report={status:'PASS',checks,animation_instrumentation:{webgl_draw_calls:glDraws,webgl_contexts_disposed:glDisposed,webgl_light_mode:glValues.uLightMode.every(v=>v===1),webgl_time_changes:new Set(glValues.uTime).size,waapi_calls:waapiCalls.length,observed_countups_final:[250,201,607,23404]},requests:requests.map(u=>u.replace(base,'')),exports:downloads.map(d=>({name:d.name,kind:d.blob?'Blob':'canvas PNG data URL'})),rendering:'ReactDOM in Happy DOM with real FastAPI endpoints; canvas RGBA captured and encoded as PNG in test adapter',limits:['No browser layout, console or screenshot validation: browser host unavailable.','Final composited colour contrast, perceptible motion and responsive geometry require real-browser review.','Native dialogs, canvas, WebGL calls, IntersectionObserver and WAAPI use explicit simulator adapters. Shader pixels are not rendered; viewport callback is exercised with synthetic geometry.'],backend_pixels_2021_requested:false,new_training:false,spark_jobs_started:false};
 await writeFile(output+'/frontend_dom_tests.json',JSON.stringify(report,null,2));console.log(JSON.stringify({status:'PASS',checks:checks.length,exports:downloads.length}));
}catch(error){await writeFile(output+'/frontend_dom_failure.json',JSON.stringify({message:error.message,stack:error.stack,checks,errors,animations:[...document.querySelectorAll('[data-animation-state]')].map(e=>({component:e.dataset.reactBits,state:e.dataset.animationState,text:e.textContent.slice(0,80)})),text:document.body.textContent.slice(-18000)},null,2));throw error;}
finally{console.error=originalError;act(()=>root.unmount());const {ScrollTrigger}=await import('gsap/ScrollTrigger');const {gsap}=await import('gsap');ScrollTrigger.killAll();ScrollTrigger.disable();gsap.ticker.sleep();window.close();}
// Browser libraries retain global timers in Node. Exit only after every assertion,
// report write and synchronous component teardown has completed successfully.
process.exit(0);
