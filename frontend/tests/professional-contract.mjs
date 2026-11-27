// CSS cascade checks in a DOM simulator; explicitly not browser layout/screenshots.
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {Window} from 'happy-dom';

export async function verifyProfessionalCSS(output){
 const base=await readFile('src/style.css','utf8'),professional=await readFile('src/professional.css','utf8');
 const checks=[];
 for(const [width,height] of [[1920,1080],[1366,768],[390,844]]){
  const window=new Window({width,height}),doc=window.document;
  doc.head.innerHTML='<style>'+base+'</style><style>'+professional+'</style>';
  doc.body.innerHTML='<section class="overview-hero professional-hero"><span class="fire-emblem hero-flame"></span><figure class="overview-fire-art"><img width="1254" height="1254"></figure></section>';
  const figure=window.getComputedStyle(doc.querySelector('figure')),img=window.getComputedStyle(doc.querySelector('img')),flame=window.getComputedStyle(doc.querySelector('.hero-flame'));
  assert.equal(img.maxHeight,'none');assert.equal(img.objectFit,'contain');assert.equal(img.height,'auto');
  assert.equal(img.aspectRatio.replaceAll(' ',''),'1/1');assert.equal(flame.position,'absolute');
  assert.equal(flame.pointerEvents,'none');assert(Number(flame.opacity)>=.2);
  assert(['transparent','rgba(0, 0, 0, 0)'].includes(figure.backgroundColor));
  checks.push({width,height,image_max_height:img.maxHeight,image_fit:img.objectFit,image_ratio:img.aspectRatio,
    figure_background:figure.backgroundColor,flame_opacity:flame.opacity});window.close();
 }
 assert(professional.includes('[data-motion=reduced]'));
 assert(professional.includes('@media(prefers-reduced-motion:reduce)'));
 const luminance=hex=>{const c=hex.slice(1).match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return .2126*c[0]+.7152*c[1]+.0722*c[2];};
 const pairs=[['#352c27','#fff8f0'],['#66594f','#fff8f0'],['#ffe0ad','#3b2c22'],['#85420d','#fff0d6'],['#fff1dd','#30251e'],['#e1c3a3','#21150d'],['#766454','#fffaf3']].map(([foreground,background])=>{const a=luminance(foreground),b=luminance(background);const ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);assert(ratio>=4.5);return {foreground,background,ratio};});
 await mkdir(output,{recursive:true});await writeFile(output+'/professional-css.json',JSON.stringify({status:'PASS',checks,contrast_pairs:pairs,
  scope:'CSS cascade at simulated viewport sizes; no pixel layout, overflow measurement or Chrome screenshot claim'},null,2));
}
