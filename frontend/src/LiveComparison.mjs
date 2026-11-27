import React,{useEffect,useMemo,useRef,useState} from 'react';
import {h,Badge,SourceState,useResource,downloadJSON,number} from './studio_ui.mjs';
import {FireMap,ForecastView} from './PropagationView.mjs';
import {activeBounds,comparisonLayers,requestForecast,previewMetrics} from './prediction_view.mjs';
import {readJSON} from './api.mjs';

export function assertCommonScene(rows){
 const first=rows[0];if(rows.length!==3)throw Error('Trois modèles requis');
 const keys=['event_id','as_of','target_date','input_dates','full_shape','preview_stride','crs','transform','observed_fire'];
 if(rows.some(r=>keys.some(k=>JSON.stringify(r[k])!==JSON.stringify(first[k]))||JSON.stringify(r.probability.map(row=>row.map(v=>v===null)))!==JSON.stringify(first.probability.map(row=>row.map(v=>v===null)))))throw Error('Domaines de comparaison incompatibles');
}
function CommonObservations({event,day,bounds,focus}){
 const state=useResource('/studio/v2/C8/observations?'+new URLSearchParams({event_id:event,as_of:day}));
 const frames=state.data?.observations||[],crop=focus?(bounds||activeBounds(frames.map(o=>o.fire))):null;
 return h('section',{className:'comparison-observations'},h('h3',null,'Observations communes · J−2, J−1, J'),h(SourceState,{state}),
  h('div',{className:'common-observation-maps'},...frames.map((o,i)=>h(FireMap,{key:o.date,title:['J−2','J−1','J'][i]+' · '+o.date,grid:o.fire,bounds:crop,metadata:{...state.data,observations:undefined,layer_date:o.date},exportPrefix:'observations_communes'}))));
}

export default function LiveComparison(){
 const catalog=useResource('/studio/v2/C8/events'),[event,setEvent]=useState(''),[day,setDay]=useState(''),[cases,setCases]=useState([]),[truth,setTruth]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState(''),[layer,setLayer]=useState('new'),[detail,setDetail]=useState('');
 const events=catalog.data||[],selected=events.find(e=>e.event_id===event);
 const [focus,setFocus]=useState(true);
 const request=useRef(null);
 useEffect(()=>()=>request.current?.abort(),[]);
 useEffect(()=>{if(events.length&&!event){const e=events[0];setEvent(e.event_id);setDay(e.forecast_dates[Math.floor(e.forecast_dates.length/2)]);}},[catalog.data]);
 const bounds=useMemo(()=>cases.length?activeBounds([cases[0].observed_fire,...cases.map(r=>r.predicted_fire)]):null,[cases]);
 const metrics=useMemo(()=>truth?cases.map(r=>({model:r.model_id,...previewMetrics(r,truth,true)})):[],[cases,truth]);
 function reset(){request.current?.abort();setCases([]);setTruth(null);setError('');setDetail('');setLayer('new');setBusy(false);}
 async function compute(){reset();const c=new AbortController();request.current=c;setBusy(true);
  try{const rows=[];for(const id of ['C8','NEWPOS2','NEWPOS4']){const response=await requestForecast('/studio/v2/'+id,event,day,{signal:c.signal,revealTruth:false});rows.push(response.result);}assertCommonScene(rows);if(!c.signal.aborted)setCases(rows);}
  catch(e){if(e.name!=='AbortError')setError(e.message);}finally{if(request.current===c)setBusy(false);}}
 async function reveal(){if(truth||busy||!cases.length)return;const c=new AbortController();request.current=c;setBusy(true);setError('');
  try{const b=await readJSON('/studio/v2/C8/ground-truth?'+new URLSearchParams({event_id:event,as_of:day,max_side:512}),{signal:c.signal});if(['full_shape','preview_stride','crs','transform','target_date','event_id'].some(k=>JSON.stringify(b[k])!==JSON.stringify(cases[0][k])))throw Error('Géométrie ou cible incompatible');if(!c.signal.aborted)setTruth(b.ground_truth);}
  catch(e){if(e.name!=='AbortError')setError(e.message);}finally{if(request.current===c)setBusy(false);}}
 return h('section',{className:'panel live-comparison','aria-label':'Comparaison des inférences CPU V2'},h('h2',null,'Comparer trois inférences réelles sur une scène 2020'),h(Badge,null,'CPU · C8 / NEWPOS2 / NEWPOS4'),
  h('p',null,'Recalcule chaque checkpoint gelé sur les mêmes observations J−2, J−1, J. Les variantes restent expérimentales. La vérité J+1 se révèle ensuite, séparément.'),h(SourceState,{state:catalog}),
  events.length>0&&h('div',{className:'spatial-form'},h('label',null,'Événement ',h('select',{id:'live-event',value:event,disabled:busy,onChange:e=>{reset();setEvent(e.target.value);setDay(events.find(r=>r.event_id===e.target.value).forecast_dates[0]);}},...events.map(e=>h('option',{key:e.event_id,value:e.event_id},e.event_id)))),
   h('label',null,'Observation J ',h('select',{id:'live-date',value:day,disabled:busy,onChange:e=>{reset();setDay(e.target.value);}},...(selected?.forecast_dates||[]).map(d=>h('option',{key:d,value:d},d)))),h('button',{className:'button-primary',disabled:busy||!day,onClick:compute},busy?'Calcul CPU en cours…':'Recalculer les trois modèles')),
  event&&day&&h(CommonObservations,{key:event+day,event,day,bounds,focus}),
  error&&h('p',{role:'alert',className:'lab-error'},error),cases.length>0&&h(React.Fragment,null,
   h('div',{className:'view-controls'},h('label',null,'Couche ',h('select',{id:'live-layer',value:layer,onChange:e=>setLayer(e.target.value)},h('option',{value:'new'},'Nouveaux pixels prédits'),h('option',{value:'global'},'Feu global prédit'),h('option',{value:'probability'},'Probabilités'),truth&&h('option',{value:'errors'},'Erreurs TP / FP / FN'))),h('button',{'aria-pressed':focus,onClick:()=>setFocus(v=>!v)},focus?'Afficher toute la grille':'Zoom zone active'),!truth&&h('button',{onClick:reveal,disabled:busy},'Révéler la vérité commune J+1'),truth&&h('span',{className:'truth-revealed',role:'status'},'Vérité commune révélée'),h('button',{onClick:()=>downloadJSON({cases,ground_truth:truth,new_fire_metrics:truth?metrics:null,new_inference:true,scope:'validation2020_V2_80',viewport:focus?bounds:null},'geofirex-V2-real-CPU-comparison.json')},'Exporter les trois inférences JSON')),
   h('p',{className:'caption'},'Cadrage commun fondé sur J et les trois prédictions. Aperçu au pas '+cases[0].preview_stride+' ; grilles de pixels sans CRS. Les résultats agrégés officiels restent inchangés.'),
   truth&&h('div',{className:'table-scroll live-scene-metrics'},h('table',{className:'science-table'},h('caption',null,'Nouveaux pixels de feu · '+(cases[0].preview_stride===1?'scène complète':'aperçu échantillonné')+' · masque valide commun avec J+1 · seuil 0,78'),
    h('thead',null,h('tr',null,...['Modèle','Rappel','Dice','Précision','TP','FP','FN'].map(label=>h('th',{key:label,scope:'col'},label)))),
    h('tbody',null,...metrics.map(m=>h('tr',{key:m.model},h('th',{scope:'row'},m.model),...['recall','dice','precision'].map(k=>h('td',{key:k},Number.isFinite(m[k])?number(m[k]*100,2)+' %':'—')),...['tp','fp','fn'].map(k=>h('td',{key:k},number(m[k])))))))),
   truth&&h('p',{className:'scientific-note'},'Un rappel supérieur peut s’accompagner de plus de fausses alertes. Ces comptes décrivent uniquement cette scène ; ils ne modifient ni les scores officiels ni le choix de C8. Aucun résultat indépendant 2021 des variantes n’existe.'),
   h('div',{className:'spatial-models'},...cases.map(r=>h('article',{key:r.model_id},h('h3',null,r.model_id),h(Badge,null,r.model_id==='C8'?'RÉFÉRENCE ÉVALUÉE 2021':'EXPÉRIMENTAL'),h(FireMap,{title:r.model_id+' · '+({new:'Nouveaux pixels prédits',global:'Feu global',probability:'Probabilités',errors:'Erreurs nouveaux feux'})[layer],grid:layer==='probability'?r.probability:layer==='global'?r.predicted_fire:layer==='errors'?comparisonLayers(r,truth).newErrors:r.new_predicted_fire,kind:layer==='probability'?'probability':layer==='errors'?'errors':'binary',bounds:focus?bounds:null,exportPrefix:r.model_id+'_'+day,metadata:{...r,probability:undefined,predicted_fire:undefined,observed_fire:undefined,new_predicted_fire:undefined}}),h('button',{onClick:()=>setDetail(detail===r.model_id?'':r.model_id)},'Détail et exports '+r.model_id)))),
   detail&&h(ForecastView,{key:detail,result:cases.find(r=>r.model_id===detail),truth})));
}
