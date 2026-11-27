import React,{useState} from 'react';
import {h,PageHeader} from './studio_ui.mjs';
import EventForecast from './EventForecast.mjs';
import LiveComparison from './LiveComparison.mjs';

export default function PredictionStudio({apiBase=''}){
 const [compare,setCompare]=useState(false);
 if(apiBase!=='/lab')return h(EventForecast,{apiBase});
 const modes=h('div',{className:'segmented-control',role:'group','aria-label':'Mode du studio'},
  h('button',{'aria-pressed':!compare,onClick:()=>setCompare(false)},'Prévision individuelle'),
  h('button',{'aria-pressed':compare,onClick:()=>setCompare(true)},'Comparaison appariée'));
 return compare?h('section',{className:'science prediction-studio'},h(PageHeader,{kicker:'COMPARER · RÉVÉLER · INTERPRÉTER',title:'Studio de prédiction',description:'C8, NEWPOS2 et NEWPOS4 sur les mêmes observations de validation 2020.'}),modes,h(LiveComparison)):
  h(EventForecast,{apiBase,modeSwitch:modes});
}
