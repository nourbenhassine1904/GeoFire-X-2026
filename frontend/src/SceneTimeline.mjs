import React,{useContext,useEffect,useMemo,useState} from 'react';
import {FireMap} from './PropagationView.mjs';
import {activeBounds} from './prediction_view.mjs';
import {MotionContext} from './studio_ui.mjs';
const h=React.createElement;

export function timelineFrames(observations,result){
 return [...(observations?.observations||[]).map((o,i)=>({date:o.date,label:['J−2','J−1','J'][i],state:'Observation réelle',grid:o.fire})),...(result?[{date:result.target_date,label:'J+1',state:'Prévision du modèle',grid:result.predicted_fire}]:[])];
}
export default function SceneTimeline({observations,result,truth,onReveal,revealing}){
 const {enabled}=useContext(MotionContext),[index,setIndex]=useState(0),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[focus,setFocus]=useState(true);
 const frames=useMemo(()=>timelineFrames(observations,result),[observations,result]);
 const bounds=useMemo(()=>activeBounds(frames.map(f=>f.grid)),[frames]);
 useEffect(()=>{setIndex(0);setPlaying(false);},[observations]);
 useEffect(()=>{setIndex(i=>Math.min(i,Math.max(0,frames.length-1)));setPlaying(false);},[frames.length]);
 useEffect(()=>{if(!enabled)setPlaying(false);},[enabled]);
 useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setIndex(i=>{if(i>=frames.length-1){setPlaying(false);return i;}return i+1;}),1800/speed);return()=>clearInterval(timer);},[playing,speed,frames.length]);
 const frame=frames[Math.min(index,frames.length-1)];if(!frame)return null;
 const meta={event_id:observations.event_id,as_of:observations.as_of,full_shape:observations.full_shape,preview_stride:observations.preview_stride,crs:observations.crs,transform:observations.transform,source_archive_sha256:observations.source_archive_sha256,checkpoint_sha256:result?.checkpoint_sha256};
 return h('section',{className:'timeline-player','aria-label':'Lecteur chronologique scientifique'},
  h('div',{className:'section-heading'},h('h3',null,'Le feu, jour après jour'),h('span',{className:'badge'},'GRILLES RÉELLES · SANS INTERPOLATION')),
  h('div',{className:'timeline-steps',role:'group','aria-label':'Dates du lecteur'},...frames.map((f,i)=>h('button',{key:f.date+f.label,'aria-pressed':index===i,onClick:()=>{setPlaying(false);setIndex(i);}},h('strong',null,f.label),h('span',null,f.date),h('small',null,f.state)))),
  h('div',{className:'timeline-controls'},h('button',{onClick:()=>{setPlaying(false);setIndex(i=>Math.max(0,i-1));},disabled:index===0,'aria-label':'Image précédente'},'Précédent'),
   h('button',{className:'button-primary','aria-pressed':playing,onClick:()=>{if(index===frames.length-1)setIndex(0);setPlaying(v=>!v);}},playing?'Pause':'Lire'),
   h('button',{onClick:()=>{setPlaying(false);setIndex(i=>Math.min(frames.length-1,i+1));},disabled:index===frames.length-1,'aria-label':'Image suivante'},'Suivant'),
   h('label',null,'Vitesse ',h('select',{'aria-label':'Vitesse de lecture',value:speed,onChange:e=>setSpeed(Number(e.target.value))},...[.5,1,2].map(v=>h('option',{value:v,key:v},'×'+v)))),
   h('button',{'aria-pressed':focus,onClick:()=>setFocus(v=>!v)},focus?'Grille complète':'Cadrer le feu')),
  h('p',{className:'timeline-status',role:'status'},frame.label+' · '+frame.date+' · '+frame.state+(playing?' · lecture en cours':'')),
  h('div',{className:'timeline-stage'},h(FireMap,{title:frame.state+' · '+frame.date,grid:frame.grid,bounds:focus?bounds:null,metadata:{...meta,layer_date:frame.date,state:frame.state},exportPrefix:observations.event_id.replace('/','__')}),
   h('aside',{className:'timeline-insight'},h('span',{className:'eyebrow'},'LIRE LA PROPAGATION'),h('h3',null,index<3?'Observer avant d’anticiper':'Une prévision à confronter au réel'),h('p',null,index<3?'Chaque image est une observation quotidienne disponible. Le contexte J−2, J−1, J alimente la prévision du lendemain.':'Le lecteur montre la segmentation prédite au seuil figé de 0,78. La vérité est consultée séparément après l’inférence.'),
    !result&&h('p',{className:'reveal-note'},'Lancez « Prédire J+1 » pour ajouter la quatrième étape.'),
    result&&!truth&&h('button',{className:'button-secondary',disabled:revealing,onClick:onReveal},revealing?'Lecture de la vérité…':'Révéler J+1 pour comparer'),
    truth&&h('p',{className:'reveal-note'},'Vérité du '+result.target_date+' révélée. Les cartes comparatives et les métriques sont disponibles ci-dessous.'),
    h('p',{className:'caption'},'Grille de pixels non géoréférencée. Le cadre commun utilise seulement l’historique et la prévision. La détection active ne mesure pas une surface brûlée.'))),
  !enabled&&h('p',{className:'caption'},'Mouvement réduit : démarrage manuel et images statiques. Les commandes restent disponibles.'));
}
