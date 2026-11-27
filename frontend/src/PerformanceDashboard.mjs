import React,{useState} from 'react';
import {h,PageHeader,Badge,SourceState,useResource,number,Fade} from './studio_ui.mjs';
import {MetricGuide} from './ScientificUI.mjs';
import {Results} from './ScienceResults.mjs';
import UnetAblation from './UnetAblation.mjs';
import Propagation from './Propagation.mjs';

const metrics=[['average_precision','AP exacte',3],['dice','Dice',3],['iou','IoU',3],['precision','Précision',3],['recall','Rappel',3],['brier','Brier',6],['fp','Faux positifs',0]];

export function eventHistogram(events,domain,metric){
 const values=events.map(e=>e[domain]?.[metric]).filter(Number.isFinite),bins=Array(10).fill(0);
 for(const v of values)bins[Math.min(9,Math.max(0,Math.floor(v*10)))]++;
 return {bins,defined:values.length,excluded:events.length-values.length};
}
function Histogram({events,domain,metric,title}){
 const {bins,defined,excluded}=eventHistogram(events,domain,metric),max=Math.max(1,...bins);
 return h('figure',{className:'distribution-chart'},h('figcaption',null,title),
  h('svg',{viewBox:'0 0 520 220',role:'img','aria-label':title+' : effectifs par intervalle de 0,1'},
   h('line',{x1:32,y1:184,x2:500,y2:184,stroke:'currentColor'}),
   ...bins.map((n,i)=>h('g',{key:'bin-'+i},h('rect',{x:38+i*46,y:180-n/max*140,width:34,height:n/max*140,rx:3},h('title',null,`${i/10}–${(i+1)/10} : ${n} événements`)),h('text',{x:55+i*46,y:174-n/max*140,textAnchor:'middle'},n))),
   ...[0,.2,.4,.6,.8,1].map(v=>h('text',{key:'tick-'+v,x:32+v*460,y:208,textAnchor:'middle'},number(v,1)))),
  h('p',{className:'caption'},defined+' événements avec valeur définie'+(excluded?' · '+excluded+' exclus (valeur absente)':'')+'. Distribution descriptive ; chaque événement compte une fois.'));
}

export default function PerformanceDashboard(){
 const state=useResource('/science/performance'),data=state.data;
 const [domain,setDomain]=useState('pooled'),[metric,setMetric]=useState('dice'),[population,setPopulation]=useState('independent_test'),[page,setPage]=useState(0),[historical,setHistorical]=useState('unet');
 const selected=data?.[population],rows=selected?.per_event||[],pageSize=20;
 const [decimals,setDecimals]=useState(6);
 return h('div',{className:'page-width performance-page'},h(PageHeader,{kicker:'MESURER · INTERPRÉTER · TRACER',title:'Performances',description:'C8 V2 sur la validation 2020 et le test indépendant 2021. Des populations distinctes, des résultats officiels et leurs limites.',action:h(Badge,null,'C8 V2 · RÉSULTATS ARCHIVÉS')}),
  h(SourceState,{state}),data&&h(React.Fragment,null,
   h('div',{className:'population-strip'},...[[data.validation,'Validation 2020'],[data.independent_test,'Test indépendant 2021']].map(([r,title])=>h('div',{key:title},h('span',{className:'eyebrow'},title),h('strong',null,r.status==='COMPLETE'?'Terminé · lecture seule':'Panel de développement déjà observé'),h('span',null,number(r.events)+' événements · '+number(r.tiles)+' tuiles')))),
   h(Fade,{trigger:'mount'},h('section',{className:'panel performance-metrics'},h('div',{className:'section-heading'},h('h2',null,'Les résultats, domaine par domaine'),h(Badge,null,'SEUIL FIGÉ · 0,78'),h('label',null,'Décimales ',h('select',{id:'performance-decimals',value:decimals,onChange:e=>setDecimals(Number(e.target.value))},...[3,6,10].map(n=>h('option',{key:n,value:n},n))))),
    h('div',{className:'segmented-control',role:'group','aria-label':'Domaine des performances'},...[['pooled','Feu global'],['new_fire','Nouveaux pixels de feu']].map(([key,label])=>h('button',{key,'aria-pressed':domain===key,onClick:()=>{setDomain(key);setPage(0);}},label))),
    h('p',{className:'caption'},domain==='new_fire'?'Pixels valides absents du feu observé à J. Une nouvelle détection n’établit pas une ignition ni une surface brûlée.':'Tous les pixels du domaine valide commun. AP et Brier utilisent les scores originaux ; les métriques de segmentation utilisent le seuil 0,78.'),
    h('div',{className:'table-scroll'},h('table',{className:'science-table'},h('caption',null,'Valeurs arrondies à l’écran. Précision originale et empreintes conservées dans les exports.'),h('thead',null,h('tr',null,...['Métrique','Validation 2020','Test indépendant 2021','Écart descriptif'].map(t=>h('th',{key:t,scope:'col'},t)))),
     h('tbody',null,...metrics.map(([key,label,digits])=>{const a=data.validation[domain][key],b=data.independent_test[domain][key],delta=Number.isFinite(a)&&Number.isFinite(b)?b-a:null,places=digits===0?0:decimals;return h('tr',{key},h('th',{scope:'row'},label),h('td',null,number(a,places)),h('td',null,number(b,places)),h('td',{className:'delta-value'},delta===null?'—':(delta>0?'+':'')+number(delta,places)));})))),
    h('p',{className:'scientific-note'},'Les événements, lieux et années diffèrent : ces écarts ne mesurent pas un gain apparié ni une amélioration du modèle. « — » signifie non disponible dans l’archive ; aucune valeur n’est reconstituée.'),
    h(MetricGuide),h('div',{className:'lab-exports'},h('a',{className:'button-secondary',href:'/science/performance/export.json',download:true},'Exporter JSON + provenance'),h('a',{className:'button-secondary',href:'/science/performance/events.csv',download:true},'Exporter CSV par événement')))),
   h('section',{className:'panel distribution-panel'},h('div',{className:'section-heading'},h('h2',null,'Variabilité entre événements'),h('label',null,'Indicateur ',h('select',{id:'distribution-metric',value:metric,onChange:e=>setMetric(e.target.value)},...metrics.filter(r=>['dice','recall','precision'].includes(r[0])).map(([key,label])=>h('option',{key,value:key},label))))),
    h('div',{className:'distribution-grid'},h(Histogram,{events:data.validation.per_event,domain,metric,title:'Validation 2020 · '+metric}),h(Histogram,{events:data.independent_test.per_event,domain,metric,title:'Test indépendant 2021 · '+metric}))),
   h('section',{className:'panel event-results'},h('div',{className:'section-heading'},h('h2',null,'Résultats archivés par événement'),h('label',null,'Population ',h('select',{id:'performance-population',value:population,onChange:e=>{setPopulation(e.target.value);setPage(0);}},h('option',{value:'validation'},'Validation 2020'),h('option',{value:'independent_test'},'Test indépendant 2021')))),
    h('div',{className:'table-scroll'},h('table',{className:'science-table'},h('caption',null,'Ordre archivistique des événements · '+(domain==='pooled'?'feu global':'nouveaux pixels de feu')+' · aucune sélection de scène par performance.'),h('thead',null,h('tr',null,...['Événement','Dice','Précision','Rappel','FP'].map(t=>h('th',{key:t,scope:'col'},t)))),h('tbody',null,...rows.slice(page*pageSize,(page+1)*pageSize).map(e=>h('tr',{key:e.event_id},h('th',{scope:'row',className:'mono'},e.event_id),...['dice','precision','recall','fp'].map(k=>h('td',{key:k},number(e[domain]?.[k],k==='fp'?0:3)))))))),
    h('div',{className:'pagination'},h('button',{disabled:page===0,onClick:()=>setPage(v=>v-1)},'Précédent'),h('span',{role:'status'},`Page ${page+1} / ${Math.max(1,Math.ceil(rows.length/pageSize))}`),h('button',{disabled:(page+1)*pageSize>=rows.length,onClick:()=>setPage(v=>v+1)},'Suivant'))),
   h('details',{className:'panel performance-provenance'},h('summary',null,'Provenance et lecture du test indépendant'),h('p',null,'Le SHA-256 officiel de RESULTS.zip et les membres JSON sont vérifiés avant lecture. Seul C8 a été évalué indépendamment sur 2021. Le verrou durable est conservé ; aucune nouvelle lecture des rasters ou inférence 2021.'),h('pre',null,JSON.stringify({validation:data.validation.provenance,test:data.independent_test.provenance},null,2)))),
  h('details',{className:'panel historical-analytics'},h('summary',null,'Résultats historiques CNN / U-Net · protocole distinct'),h('p',null,'250 train et 201 validation 2020 : 32 événements de réglage et 169 de confirmation déjà observés. Ces résultats ne sont pas le test indépendant 2021.'),
   h('div',{className:'segmented-control',role:'group','aria-label':'Analyses historiques'},...[['unet','U-Net et ablations'],['cnn','CNN / U-Net'],['propagation','Tête de propagation']].map(([key,label])=>h('button',{key,'aria-pressed':historical===key,onClick:()=>setHistorical(key)},label))),
   historical==='unet'?h(UnetAblation):historical==='cnn'?h(Results):h(Propagation)));
}
