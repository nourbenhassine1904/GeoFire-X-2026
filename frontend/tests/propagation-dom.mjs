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
const {activeBounds,cropGrid,previewMetrics}=await import('../src/prediction_view.mjs');
try{
 assert.deepEqual(tileCounts({truth:[[1,0,1,0]],result:{observed_fire:[[0,0,0,0]],predicted_fire:[[1,1,0,0]]}}),{tp:1,fp:1,fn:1,tn:1,recall:.5,precision:.5,dice:.5});
 const synthetic={probability:[[.9,.9,.1,null]],predicted_fire:[[1,1,0,null]],observed_fire:[[0,0,0,null]]};
 assert.equal(previewMetrics(synthetic,[[1,0,1,1]],true).AP,(.5+2/3)/2);
 const tile=await (await nodeFetch(base+'/science/v2/diagnostic-tile')).json();
 const offsetRow=forecastCSV({...tile.cases[0].result,tile_origin:[64,128]},tile.cases[0].truth).split('\n')[1].split(',');
 assert.equal(offsetRow[6],'64');assert.equal(offsetRow[7],'128');
 checks.push('counts_and_tied_AP','CSV_absolute_pixel_offsets');
 await act(async()=>root.render(h(StudioApp)));
 await until(()=>document.querySelector('.hero-evidence canvas')&&document.querySelector('.stat-value')?.getAttribute('aria-label')==='250','real saved overview');
 assert(document.body.textContent.includes('318 train')&&document.body.textContent.includes('156 test'));
 await accessibilityAudit('overview');checks.push('overview_real_cropped_raster_and_split_labels');
 await click(button('Explorer la propagation'));
 await until(()=>document.getElementById('forecast-event')?.options.length===202,'historical catalogue');
 await change(document.getElementById('forecast-event'),'2020/fire_24332880');
 await change(document.getElementById('forecast-date'),'2020-09-03');
 await until(()=>document.querySelectorAll('.history-maps canvas').length===3,'three past observations');
 assert.equal(document.getElementById('forecast-date').value,'2020-09-03');
 const timeline=()=>document.querySelector('.timeline-player');
 assert.equal(timeline().querySelectorAll('.timeline-steps button').length,3);
 await click(timeline().querySelector('[aria-label="Image suivante"]'));
 assert(timeline().querySelector('.timeline-status').textContent.includes('2020-09-02'));
 await change(timeline().querySelector('select'),'2');
 await click([...timeline().querySelectorAll('button')].find(b=>b.textContent==='Lire'));
 await click([...timeline().querySelectorAll('button')].find(b=>b.textContent==='Pause'));
 const paused=timeline().querySelector('.timeline-status').textContent;
 await act(async()=>{await new Promise(resolve=>setTimeout(resolve,1000));});
 assert.equal(timeline().querySelector('.timeline-status').textContent,paused,'pause freezes frame');
 await click([...timeline().querySelectorAll('button')].find(b=>b.textContent==='Lire'));
 await until(()=>timeline().querySelector('.timeline-status').textContent.includes('2020-09-03'),'timeline timer');
 await until(()=>[...timeline().querySelectorAll('button')].some(b=>b.textContent==='Lire'),'timeline stop');
 checks.push('timeline_real_dates_manual_steps_speed_play_stop');
 assert.equal(requests.filter(u=>u.includes('/ground-truth?')).length,0);
 await click(document.querySelector('.predict-button'));
 await until(()=>document.querySelectorAll('.propagation-result canvas').length===3&&!document.querySelector('.predict-button').disabled,'real CPU prediction',60000);
 assert.equal(requests.filter(u=>u.includes('/ground-truth?')).length,0,'no automatic truth read');
 assert.equal(timeline().querySelectorAll('.timeline-steps button').length,4);
 await click(timeline().querySelector('.timeline-steps button:last-child'));
 assert(timeline().querySelector('.timeline-status').textContent.includes('2020-09-04'));
 assert(timeline().querySelector('.timeline-status').textContent.includes('Prévision du modèle'));
 checks.push('timeline_prediction_label_separate_truth');
 const beforeCanvas=document.querySelector('.propagation-result canvas');const beforeShape=[beforeCanvas.width,beforeCanvas.height];
 assert(document.body.textContent.includes('7295eaa6f957'));
 await writeFile(output+'/prediction-before-truth.html',document.documentElement.outerHTML);
 await click(button('R\u00e9v\u00e9ler la v\u00e9rit\u00e9 J+1'));
 await until(()=>document.querySelectorAll('.propagation-result canvas').length===4&&document.querySelector('.scene-score-grid'),'explicit truth reveal');
 assert.deepEqual([beforeCanvas.width,beforeCanvas.height],beforeShape,'truth cannot change viewport');
 const predictIndex=requests.findIndex(u=>u.includes('/lab/predict?')),truthIndex=requests.findIndex(u=>u.includes('/lab/ground-truth?'));
 assert(predictIndex>=0&&truthIndex>predictIndex);
 const q='?event_id=2020%2Ffire_24332880&as_of=2020-09-03&max_side=512';
 const r=await(await nodeFetch(base+'/lab/predict'+q)).json(),t=await(await nodeFetch(base+'/lab/ground-truth'+q)).json();
 const expected=previewMetrics(r,t.ground_truth,true);
 for(const [key,value] of [['TP',expected.tp],['FP',expected.fp],['FN',expected.fn]])assert(document.querySelector('.score-'+key).textContent.includes(String(value)));
 await writeFile(output+'/historical-scene-metrics.json',JSON.stringify({event_id:r.event_id,as_of:r.as_of,target_date:r.target_date,new_fire:expected,global:previewMetrics(r,t.ground_truth),checkpoint:r.checkpoint_sha256,scope:'full scene; stride1; joint mask'},null,2));
 checks.push('past_observations_2020_only','live_CPU_inference','truth_requires_user_action','crop_unchanged_after_truth','real_scene_counts');
 await click(button('Prévision / vérité côte à côte'));
 assert.equal(document.querySelectorAll('.comparison-pair canvas').length,2);
 await click(button('Vue des couches'));
 checks.push('prediction_truth_side_by_side');
 await click(ariaButton('Image précédente'));
 assert(timeline().querySelector('.timeline-status').textContent.includes('2020-09-03'));
 checks.push('timeline_pause_and_previous');
 await click(button('Feu global'));assert(document.querySelector('.propagation-result').textContent.includes('Pr\u00e9diction globale J+1'));
 await click(button('Nouveaux feux'));await click(button('Toutes les couches'));assert.equal(document.querySelectorAll('.propagation-result canvas').length,8);
 await click(button('Vue essentielle'));await click(button('Afficher toute la grille'));assert.equal(document.querySelector('.propagation-result canvas').width,256);
 await click(button('Zoom zone active'));const bounds=activeBounds([r.observed_fire,r.predicted_fire]);
 assert.deepEqual(Buffer.from(document.querySelector('.propagation-result canvas').__pixels),Buffer.from(imagePixels(cropGrid(r.observed_fire,bounds)).pixels));
 checks.push('domain_and_layers_controls','full_grid_and_active_crop_pixel_exact');
 await click(document.querySelector('.propagation-result .map-actions button'));
 assert(document.querySelector('dialog[open]'));await click(button('Fermer'));assert(!document.querySelector('dialog[open]'));
 await click(document.querySelector('.propagation-result .map-actions button:nth-child(2)'));
 await click(document.querySelector('.propagation-result .map-actions button:nth-child(3)'));
 const pngDownload=downloads.find(d=>d.href.startsWith('data:image/png'));
 assert(pngDownload);const pngBytes=Buffer.from(pngDownload.href.split(',')[1],'base64');
 assert.equal(pngBytes.subarray(1,4).toString(),'PNG');
 const metaDownload=downloads.find(d=>d.name.endsWith('.metadata.json'));const meta=JSON.parse(await metaDownload.blob.text());
 assert.equal(meta.checkpoint_sha256,r.checkpoint_sha256);assert.deepEqual(meta.viewport,bounds);
 await writeFile(output+'/rendered-observation-crop.png',pngBytes);
 await click(button('JSON + provenance'));await click(button('CSV des pixels'));
 assert(downloads.some(d=>d.name.endsWith('.csv')));
 checks.push('enlargement_open_close','PNG_and_metadata_export','scene_JSON_CSV_export');
 await accessibilityAudit('studio');await writeFile(output+'/studio-after-truth.html',document.documentElement.outerHTML);
 await route('comparison');await until(()=>document.querySelectorAll('tbody tr').length===4&&document.getElementById('saved-event')?.options.length===80,'V2 aggregates and saved catalogue',60000);
 await change(document.getElementById('chart-metric'),'FP_new');assert(document.querySelector('.metric-chart').textContent.includes('138'));
 await change(document.getElementById('chart-metric'),'AP_new');checks.push('metric_chart_selection_and_FP_scatter');
 await click(button('Afficher les sorties sauvegard\u00e9es'));
 await until(()=>document.querySelectorAll('.saved-spatial .spatial-models canvas').length===3,'three saved V2 full scenes',60000);
 const savedEvent=document.getElementById('saved-event').value,savedDate=document.getElementById('saved-date').value;
 const scene=await(await nodeFetch(base+'/science/v2/scene?'+new URLSearchParams({event_id:savedEvent,target_date:savedDate}))).json();
 assert.equal(scene.cases.length,3);assert.equal(scene.new_inference,false);
 await change(document.getElementById('saved-layer'),'probability');await change(document.getElementById('saved-domain'),'global');
 await change(document.getElementById('saved-layer'),'errors');await change(document.getElementById('saved-domain'),'new');
 await click(button('Exporter les trois mod\u00e8les JSON'));await click(button('Exporter JSON + provenance'));
 const official=downloads.find(d=>d.name==='geofirex-validation2020-V2-80.json');assert.equal(JSON.parse(await official.blob.text()).models[3].FP_new,95053);
 const csv=await(await nodeFetch(base+'/science/v2/export.csv')).text();assert.equal(csv.trim().split('\n').length,5);
 checks.push('saved_80_event_catalogue','three_models_same_scene_saved_only','spatial_layer_and_domain_controls','official_metrics_exports_unchanged');
 await accessibilityAudit('comparison');await writeFile(output+'/comparison-dom.html',document.documentElement.outerHTML);
 await route('analytics');await until(()=>document.querySelector('.comparison-charts')&&document.body.textContent.includes('Mod\u00e8le principal'),'analytics');
 await accessibilityAudit('analytics');
 await route('research');await until(()=>document.body.textContent.includes('Lire les r\u00e9sultats'),'research');await accessibilityAudit('research');
 assert(!requests.some(u=>u.includes('/science/big-data')));checks.push('no_BigData_route');
 await click(ariaButton('Ouvrir la navigation'));assert(document.querySelector('.mobile-navigation[open]'));await click(ariaButton('Fermer la navigation'));checks.push('mobile_navigation_dialog');
 assert.deepEqual(errors,[]);
 await writeFile(output+'/propagation-dom.json',JSON.stringify({status:'PASS',kind:'React DOM simulator; actual HTTP/CPU; not browser layout',checks,a11y:a11yByPage,requests,downloads:downloads.map(d=>d.name),actual_CPU_inference:true,truth_explicit_user_action:true,training_executed:false},null,2));
 console.log(JSON.stringify({status:'PASS',checks:checks.length,actual_CPU_inference:true,truth_explicit:true}));
}finally{
 console.error=originalError;await act(async()=>root.unmount());await window.happyDOM.abort();
 const {ScrollTrigger}=await import('gsap/ScrollTrigger');const {gsap}=await import('gsap');ScrollTrigger.killAll();ScrollTrigger.disable();gsap.ticker.sleep();window.close();
}
process.exit(0);
