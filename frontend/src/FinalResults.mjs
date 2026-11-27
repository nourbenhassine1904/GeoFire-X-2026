import React from 'react';
import {h,Badge,SourceState,useResource,number} from './studio_ui.mjs';

export default function FinalResults(){
 const state=useResource('/science/final-2021'),r=state.data;
 return h('section',{className:'panel final-test-results','aria-label':'Test indépendant 2021 terminé'},
  h('div',{className:'section-heading'},h('h2',null,'Test indépendant 2021 · C8 V2'),h(Badge,null,'COMPLETE · LECTURE SEULE')),
  h('p',null,'156 événements · 75 540 tuiles · checkpoint époque 4 · seuil 0,78. Résultats officiels chargés depuis RESULTS.zip après vérification de son SHA-256. Aucune nouvelle inférence 2021.'),
  h(SourceState,{state}),r&&h(React.Fragment,null,
   h('div',{className:'table-scroll'},h('table',null,h('caption',null,'Test 2021 : C8 seulement. Les variantes figurent dans la validation 2020 distincte.'),
    h('thead',null,h('tr',null,...['Domaine','AP exacte','Dice','Rappel'].map(s=>h('th',{key:s,scope:'col'},s)))),
    h('tbody',null,...[['pooled','Feu global'],['new_fire','Nouveaux pixels de feu']].map(([key,label])=>h('tr',{key},h('th',{scope:'row'},label),...['average_precision','dice','recall'].map(m=>h('td',{key:m,className:'mono'},number(r[key][m],10)))))))),
   h('div',{className:'lab-exports'},h('a',{href:'/science/final-2021/export.json',download:true,className:'button-secondary'},'Exporter JSON officiel'),h('a',{href:'/science/final-2021/export.csv',download:true,className:'button-secondary'},'Exporter CSV officiel')),
   h('details',null,h('summary',null,'Provenance vérifiée du test terminé'),h('pre',null,JSON.stringify(r.provenance,null,2))),
   h('p',{className:'caption'},'Le faible rappel des nouveaux feux reste une limite majeure. Les sorties sigmoid ne sont pas des probabilités calibrées. Le verrou durable du test est conservé.')));
}
