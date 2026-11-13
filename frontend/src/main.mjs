import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import EventForecast from './EventForecast.mjs';
import {Results, BigData} from './ScienceResults.mjs';
import UnetAblation from './UnetAblation.mjs';
import Propagation from './Propagation.mjs';
const h=React.createElement;
function App() {
  const [view,setView]=useState('lab');
  const [live,setLive]=useState(true);
  useEffect(()=>{
    const controller=new AbortController();
    fetch('/health',{signal:controller.signal}).then(r=>r.json()).then(data=>setLive(data.live_inference!==false)).catch(()=>{});
    return ()=>controller.abort();
  },[]);
  return h(React.Fragment,null,
    h('nav',{className:'app-nav','aria-label':'Navigation principale'},h('a',{href:'#',onClick:e=>{e.preventDefault();setView('lab');},className:'brand'},'GEOFIRE-X'),
      ...[['lab','Prediction Lab'],['propagation','Propagation J+1'],['unet','Ablation U-Net'],['results','CNN / U-Net historiques'],['data','Big Data'],['forecast','Prévision historique']].map(([key,label])=>h('button',{key,'aria-current':view===key?'page':undefined,disabled:key==='forecast'&&!live,title:key==='forecast'&&!live?'Mode résultats : un snapshot historique compatible est nécessaire pour l’inférence.':undefined,onClick:()=>setView(key)},label)),
      h('span',{className:'locked'},'2021 · verrouillé')),
    view==='lab'?h(EventForecast,{apiBase:'/lab'}):view==='propagation'?h(Propagation):view==='unet'?h(UnetAblation):view==='results'?h(Results):view==='data'?h(BigData):h(EventForecast));
}
createRoot(document.getElementById('root')).render(h(App));
