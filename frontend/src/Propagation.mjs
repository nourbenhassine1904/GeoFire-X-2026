import React,{useEffect,useState} from 'react';
import PropagationComparison from './PropagationComparison.mjs';
const h=React.createElement;
const number=(v,d=4)=>v===null||v===undefined?'—':Number(v).toLocaleString('fr-FR',{maximumFractionDigits:d});
const percent=v=>number(100*v,2)+' %';
export default function Propagation(){
  const [data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{const controller=new AbortController();fetch('/science/propagation',{signal:controller.signal})
    .then(async r=>{if(!r.ok)throw new Error(`Diagnostic indisponible (${r.status})`);setData(await r.json());})
    .catch(e=>{if(e.name!=='AbortError')setError(e.message);});return()=>controller.abort();},[]);
  if(error)return h('main',{className:'science'},h('p',{role:'alert'},error));
  if(!data)return h('main',{className:'science'},h('p',{role:'status'},'Lecture des diagnostics enregistrés…'));
  return h(PropagationView,{data});
}
export function PropagationView({data}){
  const [choice,setChoice]=useState('');
  const c=data.cases.find(c=>c.event_id===choice)||data.cases[Math.floor(data.cases.length/2)],s=data.summary;
  const rows=[['Tous les feux J+1',data.full],['Nouveaux : feu J = 0',data.new_fire],['Déjà détectés : feu J = 1',data.observed_fire]];
  const columns=[['positive_pixels','Feux cibles'],['tp','TP'],['fp','FP'],['fn','FN'],['dice','Dice'],['precision','Précision'],['recall','Rappel']];
  const table=(rows,cols)=>h('div',{className:'table-scroll'},h('table',null,h('thead',null,h('tr',null,...cols.map(([key,label])=>h('th',{key,scope:'col'},label)))),
    h('tbody',null,...rows.map((r,i)=>h('tr',{key:i},...cols.map(([key])=>h('td',{key},r[key]===null||r[key]===undefined?'—':typeof r[key]==='number'?number(r[key],['n','positive_pixels','tp','fp','fn'].includes(key)?0:6):r[key])))))));
  return h('main',{className:'science'},h('header',{className:'science-header'},h('span',{className:'eyebrow'},'Prévoir demain · diagnostic scientifique'),
    h('h1',null,'Propagation J+1 : résultats et compromis'),h('p',null,'Expérience comparative sur les32 ; diagnostic du modèle principal sur les169 présentés séparément. Sorties historiques conservées.')),
    h(PropagationComparison),
    h('section',{className:'panel'},h('h2',null,'Nouveaux pixels et feu déjà observé · 169 événements'),
      h('p',null,'Nouveau feu = cible J+1 positive, absence de détection à J et masque valide commun. Contexte : J−2, J−1, J ; cible : J+1. Le domaine valide inclut aussi J−4 et J−3 pour conserver la comparaison historique.'),
      table(rows.map(([domain,r])=>({domain,...r})),[['domain','Domaine'],...columns]),
      h('p',null,`${percent(s.new_positive_share)} des feux cibles sont nouveaux ; ils concentrent ${percent(s.new_false_negative_share)} des faux négatifs. ${percent(s.observed_true_positive_share)} des vrais positifs concernent des pixels déjà détectés à J.`),
      h('p',null,`${s.events_zero_new_true_positives} / ${s.events_with_new_fire} événements avec nouveaux feux n’en détectent aucun. Rappel médian : ${percent(s.new_recall_event.median)} ; moyenne par événement : ${percent(s.new_recall_event.mean)}. ${s.events_without_new_fire} événements sans nouveau feu : rappel indéfini.`),
      h('p',{className:'legend'},'Détection active nouvelle ≠ surface brûlée cumulée. Une réapparition, une détection isolée ou une variation d’observation peut contribuer ; ces labels seuls ne prouvent pas une propagation physique.')),
    h('aside',{className:'boundary-note'},h('strong',null,'Limite des informations disponibles'),
      h('p',null,'Le corpus compact utilise pente, altitude, aspect, indicateurs de valeurs manquantes et détections du feu. Il ne contient ni vent, ni humidité, ni météo prévue. Une nouvelle architecture ne peut pas recréer ces variables absentes.')),
    h('section',{className:'panel'},h('h2',null,'Cartes réelles de nouveaux feux · sans nouvelle inférence'),
      h('label',null,'Événement ',h('select',{value:c.event_id,onChange:e=>setChoice(e.target.value)},...data.cases.map(c=>h('option',{key:c.event_id,value:c.event_id},`${c.event_id} · ${c.target_date}`)))),
      h('p',null,'Même sélection de trois événements et dates médianes que la confirmation. Grille de pixels, pas de géoréférencement inventé. Transparent : invalide ; pour les probabilités et erreurs, les pixels déjà détectés à J sont également exclus.'),
      h('div',{className:'map-grid'},...[
        ['observed','Détection observée J'],['new_target','Vérité : nouveaux pixels J+1'],['new_prediction','Prédiction nouveaux pixels · 0,78'],
        ['new_probability','Probabilité sur absence à J'],['new_errors','TP vert · FP rouge · FN bleu']
      ].map(([key,label])=>h('figure',{key},h('figcaption',null,label),h('img',{src:'/science/propagation-assets/'+c.layers[key],alt:`${label} · ${c.event_id}`,loading:'lazy'})))),
      h('p',null,`Cette date seulement : ${number(c.new_positive_pixels,0)} nouveaux pixels cibles, ${number(c.new_tp,0)} TP, ${number(c.new_fn,0)} FN. Ce cas n’est pas une estimation du corpus.`),
      h('details',null,h('summary',null,'Probabilités enregistrées · cette date seulement'),table(Object.entries(c.probability_groups).map(([group,r])=>({group,...r})),[['group','Sous-ensemble'],['n','Pixels'],['mean','Moyenne'],['q10','q10'],['median','Médiane'],['q90','q90']]))),
    h('details',{className:'panel'},h('summary',null,'Métriques de nouveaux feux par événement'),table(data.per_event,[['event_id','Événement'],...columns])),
    h('details',{className:'panel'},h('summary',null,'Preuves et vérifications temporelles'),h('pre',null,JSON.stringify({label_checks:data.label_checks,sources:data.sources},null,2))));
}
