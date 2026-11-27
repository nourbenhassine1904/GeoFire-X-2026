import React from 'react';
import {h,Badge,SourceState,useResource,StatCard,Icon,WelcomeTitle,FireEmblem,Fade} from './studio_ui.mjs';
import {MetricTiles,ProcessFlow} from './ScientificUI.mjs';

export default function Overview({navigate}){
 const populations=useResource('/science/v2/populations'),test=useResource('/science/final-2021');
 return h('div',{className:'overview page-width professional-overview'},
  h('section',{className:'overview-hero professional-hero'},h(FireEmblem,{hero:true}),h('div',{className:'hero-copy'},
   h(Badge,null,'GEOFIRE-X · PRÉVISION SCIENTIFIQUE'),h(WelcomeTitle),
   h('p',{className:'hero-description'},'Transformer trois observations quotidiennes en une prévision de détection active du feu. Explorer les cartes, confronter les modèles et suivre chaque résultat jusqu’à sa source.'),
   h('div',{className:'hero-actions'},h('button',{className:'button-primary',onClick:()=>navigate('studio')},h(Icon,{name:'studio'}),'Explorer une prévision'),h('button',{className:'button-secondary',onClick:()=>navigate('analytics')},'Voir les performances')),
   h('div',{className:'hero-proof'},h('span',null,'C8 V2'),h('span',null,'CPU · J+1'),h('span',null,'Test indépendant 2021 terminé'))),
   h('figure',{className:'overview-fire-art'},h('img',{src:'/feu.png',alt:'Illustration conceptuelle, sans sortie du modèle : relief forestier et propagation du feu',width:1254,height:1254,fetchPriority:'high'}))),
  h('section',{className:'overview-stats','aria-label':'Populations du protocole V2'},
   h(StatCard,{icon:'layers',label:'Événements train',value:populations.data?.roles?.train?.events,detail:'Protocole chronologique V2'}),
   h(StatCard,{icon:'studio',label:'Événements de validation',value:populations.data?.roles?.val?.events,detail:'2020 · démonstrations autorisées'}),
   h(StatCard,{icon:'shield',label:'Événements du test indépendant',value:test.data?.events,detail:'2021 · C8 V2 seulement'})),
  h(SourceState,{state:populations}),
  h(Fade,null,h('section',{className:'panel overview-workflow'},h('div',{className:'section-heading'},h('h2',null,'De l’observation à une prévision vérifiable'),h(Badge,null,'CONTEXTE T3')),h(ProcessFlow,{compact:true}))),
  h(Fade,null,h('section',{className:'panel overview-result-summary','aria-label':'Résumé du test indépendant C8'},
   h('div',null,h('span',{className:'eyebrow'},'RÉSULTATS INDÉPENDANTS · 2021'),h('h2',null,'Une référence mesurée, avec ses limites'),h('p',null,'C8 V2 est évalué sur 156 événements indépendants. Les variantes expérimentales restent comparées sur la validation 2020.'),h(SourceState,{state:test}),
    test.data&&h('p',{className:'caption'},'Les scores des nouveaux pixels de feu sont détaillés dans Performances. Leur faible rappel constitue une limite majeure.')),
   h('div',null,h(MetricTiles,{metrics:test.data?.pooled,compact:true}),h('button',{className:'text-link',onClick:()=>navigate('analytics')},'Consulter les résultats et leur provenance')))),
  h('div',{className:'overview-next'},h('div',null,h(Icon,{name:'shield'}),h('p',null,'Validation 2020 pour les interactions. Résultats 2021 archivés en lecture seule.')),h('button',{className:'button-secondary',onClick:()=>navigate('research')},'Comprendre les données et la méthode')));
}
