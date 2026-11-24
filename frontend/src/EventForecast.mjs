import React,{useContext,useEffect,useMemo,useRef,useState} from 'react';
import SceneTimeline from './SceneTimeline.mjs';
import {activeBounds,requestForecast} from './prediction_view.mjs';
import {Icon,Badge,MotionContext,Fade} from './studio_ui.mjs';
import Stepper,{Step} from './reactbits/Stepper.jsx';
import AnimatedList from './reactbits/AnimatedList.jsx';
import {FireMap,ForecastView} from './PropagationView.mjs';
export {FireMap,ForecastView} from './PropagationView.mjs';
const h=React.createElement;
const n=(v,d=2)=>Number(v).toLocaleString('fr-FR',{maximumFractionDigits:d});

export default function EventForecast({apiBase=''}){
  const [events,setEvents]=useState([]),[eventId,setEventId]=useState(''),[day,setDay]=useState('');
  const [result,setResult]=useState(null),[truth,setTruth]=useState(null),[busy,setBusy]=useState(false),[loading,setLoading]=useState(true);
  const [error,setError]=useState(''),[truthError,setTruthError]=useState('');
  const [catalogueRevision,setCatalogueRevision]=useState(0);
  const [observations,setObservations]=useState(null),[observationError,setObservationError]=useState(''),[revealing,setRevealing]=useState(false);
  const request=useRef(null);
  const picker=useRef(null),browseButton=useRef(null),[query,setQuery]=useState(''),[pickerOpen,setPickerOpen]=useState(false);
  const {enabled:motion}=useContext(MotionContext);
  useEffect(()=>{const c=new AbortController();setLoading(true);setError('');
    fetch(`${apiBase}/events`,{signal:c.signal}).then(async r=>{const body=await r.json();if(!r.ok)throw new Error(body.detail||'Catalogue indisponible');setEvents(body);
      const initial=body.find(e=>e.event_id==='2020/fire_24332880')||body[0];
      if(initial){setEventId(initial.event_id);setDay(initial.event_id==='2020/fire_24332880'&&initial.forecast_dates.includes('2020-09-03')?'2020-09-03':initial.forecast_dates[Math.floor(initial.forecast_dates.length/2)]||'');}
    }).catch(e=>{if(e.name!=='AbortError')setError(e.message);}).finally(()=>{if(!c.signal.aborted)setLoading(false);});
    return()=>{c.abort();request.current?.abort();};
  },[apiBase,catalogueRevision]);
  const selected=events.find(e=>e.event_id===eventId);
  useEffect(()=>{setObservations(null);setObservationError('');if(apiBase!=='/lab'||!eventId||!day)return;const c=new AbortController();
    fetch(`${apiBase}/observations?${new URLSearchParams({event_id:eventId,as_of:day})}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok)throw Error(b.detail||'Observations indisponibles');if(!c.signal.aborted)setObservations(b);}).catch(e=>{if(e.name!=='AbortError')setObservationError(e.message);});return()=>c.abort();
  },[apiBase,eventId,day]);
  const dateIndex=selected?.forecast_dates.indexOf(day)??-1;
  const filtered=events.filter(e=>e.event_id.toLowerCase().includes(query.toLowerCase()));
  const target=day?new Date(new Date(day+'T00:00:00Z').getTime()+86400000).toISOString().slice(0,10):'—';
  const stage=truth?4:result?3:busy?2:1;
  function chooseEvent(id){reset();setEventId(id);const chosen=events.find(e=>e.event_id===id);setDay(chosen?.forecast_dates[Math.floor(chosen.forecast_dates.length/2)]||'');picker.current?.close();}
  function reset(){request.current?.abort();setBusy(false);setRevealing(false);setResult(null);setTruth(null);setError('');setTruthError('');}
  async function reveal(){
    if(!result||busy||revealing)return;const c=new AbortController();request.current=c;setRevealing(true);setTruthError('');
    try{const q=new URLSearchParams({event_id:result.event_id,as_of:result.as_of,max_side:512});const r=await fetch(`${apiBase}/ground-truth?${q}`,{signal:c.signal}),b=await r.json();if(!r.ok)throw Error(b.detail||'Vérité indisponible');
      if(JSON.stringify(b.full_shape)!==JSON.stringify(result.full_shape)||b.preview_stride!==result.preview_stride||b.crs!==result.crs||JSON.stringify(b.transform)!==JSON.stringify(result.transform))throw Error('Géométrie de vérité incompatible');
      if(!c.signal.aborted&&request.current===c)setTruth(b.ground_truth);
    }catch(e){if(e.name!=='AbortError')setTruthError(e.message);}finally{if(request.current===c)setRevealing(false);}
  }
  async function predict(){
    reset();const c=new AbortController();request.current=c;setBusy(true);
    try{
      const response=await requestForecast(apiBase,eventId,day,{signal:c.signal,revealTruth:false,onPrediction:body=>{
        if(!c.signal.aborted&&request.current===c)setResult(body);
      }});
      if(c.signal.aborted||request.current!==c)return;
      setTruth(response.truth);setTruthError(response.truthError);
    }catch(e){if(e.name!=='AbortError')setError(e.message);}
    finally{if(request.current===c)setBusy(false);}
  }
  return h('section',{className:'science prediction-studio'},h('header',{className:'science-header studio-header'},h('div',null,h('span',{className:'eyebrow'},'EXPLORER · PRÉDIRE · COMPARER'),
    h('h1',null,apiBase==='/lab'?'Prediction Studio':'Predictor TIFF historique'),h('p',null,apiBase==='/lab'?'Une inférence réelle sur un événement historique. Des cartes lisibles, une provenance vérifiable.':'Snapshot historique vérifié · modèle et seuil propres au snapshot.')),
    h(Badge,{tone:'violet'},h(Icon,{name:'studio',size:14}),apiBase==='/lab'?'U-NET POIDS 10 · 0,78':'SNAPSHOT HISTORIQUE')),
    h('div',{className:'demo-brief'},h('span',{className:'eyebrow'},'PARCOURS SOUTENANCE · 6 MINUTES'),h('h2',null,'De trois observations à une prévision vérifiable'),h('p',null,'1 min : historique et lecteur · 1 min : inférence CPU · 1 min : vérité et erreurs · 1 min : propagation et métriques · 2 min : comparaison V2 et méthode.'),h('button',{className:'button-secondary',disabled:busy||loading,onClick:()=>{reset();setEventId('2020/fire_24332880');setDay('2020-09-03');}},'Charger la scène de soutenance'),h('p',{className:'caption'},'2020/fire_24332880 · observation 2020-09-03 · cible 2020-09-04 · U-Net historique interactif')),
    h('div',{className:'studio-contract-strip'},h('span',null,h('span',{className:'status-dot green'}),'PyTorch CPU'),h('span',null,apiBase==='/lab'?'Contexte T3 → J+1':'Contexte du snapshot → J+1'),h('span',null,apiBase==='/lab'?'Validation historique 2020':'Corpus du snapshot historique'),h('span',null,h(Icon,{name:'lock',size:13}),'Test 2021 fermé')),
    apiBase==='/lab'&&h('nav',{className:'scene-jump-nav','aria-label':'Parcours de la scène'},...[
      ['Observations','.observation-history',false],['Prévision','.forecast-reveal',!result],['Révélation J+1','.reveal-control',!result],['Propagation','.timeline-player',!observations],['Erreurs','.forecast-reveal .scene-metrics',!truth]
    ].map(([label,selector,disabled])=>h('button',{key:label,disabled,onClick:()=>document.querySelector(selector)?.scrollIntoView({behavior:motion?'smooth':'instant',block:'start'})},label))),
    h('section',{className:'panel prediction-controls','aria-label':'Configuration de la prédiction'},
      h('div',{className:'section-heading'},h('h2',null,'Votre espace de prédiction'),h('span',{className:'caption'},'01 / Configuration')),
      loading&&h('p',{role:'status',className:'loading-note'},'Vérification du checkpoint, des sources et du catalogue…'),
      h('div',{className:'prediction-form'},
        h('div',{className:'form-field event-field'},h('label',{htmlFor:'forecast-event'},'Événement historique'),h('div',{className:'event-select-row'},h('select',{id:'forecast-event',value:eventId,disabled:loading||busy,onChange:e=>chooseEvent(e.target.value)},
          h('option',{value:''},'Sélectionner un événement'),...events.map(e=>h('option',{key:e.event_id,value:e.event_id},`${e.event_id} · ${e.scope==='calibration32'?'réglage (32)':e.scope==='observed_confirmation169'?'confirmation observée (169)':e.split}`))),
          h('button',{ref:browseButton,className:'icon-button browse-events',disabled:loading||busy,'aria-label':'Rechercher un événement',onClick:()=>{setQuery('');setPickerOpen(true);picker.current?.showModal();}},h(Icon,{name:'search'})))),
        h('div',{className:'form-field'},h('label',{htmlFor:'forecast-date'},'Observation · date J'),h('select',{id:'forecast-date',value:day,disabled:loading||busy,onChange:e=>{reset();setDay(e.target.value);}},
          h('option',{value:''},'Choisir une date'),...(selected?.forecast_dates||[]).map(d=>h('option',{key:d,value:d},d)))),
        h('div',{className:'form-field target-field'},h('span',{className:'field-label'},'Cible · J+1'),h('div',{key:eventId+day,className:'target-date mono selection-feedback'},h(Icon,{name:'arrow',size:15}),target)),
        h('button',{className:'button-primary predict-button',onClick:predict,disabled:loading||busy||!day||!eventId},h(Icon,{name:busy?'clock':'studio',size:17}),busy?'Inférence en cours…':'Prédire J+1')),
      h('div',{className:'temporal-nav'},h('div',null,h('button',{className:'button-secondary',disabled:busy||dateIndex<=0,onClick:()=>{reset();setDay(selected.forecast_dates[dateIndex-1]);}},'← Date précédente'),
        h('span',{className:'mono'},dateIndex>=0?`${dateIndex+1} / ${selected.forecast_dates.length}`:'—'),
        h('button',{className:'button-secondary',disabled:busy||dateIndex<0||dateIndex>=((selected?.forecast_dates.length||0)-1),onClick:()=>{reset();setDay(selected.forecast_dates[dateIndex+1]);}},'Date suivante →')),
        h('span',{className:'caption'},'Les dates changent le contexte. Relancez « Prédire J+1 » pour obtenir les cartes.')),
      h('div',{className:'prediction-stepper','data-react-bits':'Stepper','data-current-step':stage},h(Stepper,{key:motion?'animated':'static',step:stage,theme:'light',style:{'--stepper-ink':'var(--text)','--stepper-muted':'var(--text-secondary)'},showFooter:false,disableStepIndicators:true,frame:false,width:'100%',radius:8,accentColor:'#9A3412',transition:motion?'fade':'none','aria-label':'Étapes réelles de la prédiction'},
        ...[['Configurer','Choisissez un événement de validation et une date disponible.'],['Inférer J+1','Lecture des tuiles vérifiées puis calcul du modèle. Aucun pixel cible dans l’inférence.'],['Comparer','Prédiction calculée. La vérité J+1 est demandée séparément, ensuite.'],['Exporter','Cartes et comparaison disponibles. Exports PNG, JSON avec provenance et CSV des pixels.']].map(([title,text])=>h(Step,{key:title,title},h('p',null,text))))),
      busy&&h('p',{role:'status',className:'inference-status'},h('span',{className:'inference-spinner','aria-hidden':true}),'Calcul CPU en cours. La vérité J+1 restera masquée.'),
      error&&h('p',{role:'alert',className:'lab-error'},error),error&&!events.length&&h('button',{className:'button-secondary',onClick:()=>setCatalogueRevision(v=>v+1)},'Réessayer le catalogue'),truthError&&h('p',{role:'alert',className:'lab-error'},truthError)),
    apiBase==='/lab'&&h('section',{className:'panel observation-history'},h('h2',null,`Observer le passé · ${eventId}`),h('p',null,'Les observations J−2, J−1 et J alimentent le modèle historique avec huit canaux C8 par jour. La vérité J+1 se consulte séparément après l’inférence.'),observationError?h('p',{role:'alert'},observationError):!observations?h('p',{role:'status'},'Lecture des trois observations passées…'):h(React.Fragment,null,h(SceneTimeline,{observations,result,truth,onReveal:reveal,revealing}),h('details',null,h('summary',null,'Voir les trois observations côte à côte'),h('div',{className:'prediction-maps history-maps'},...observations.observations.map(o=>h(FireMap,{key:o.date,title:'Observation · '+o.date,grid:o.fire,bounds:activeBounds(observations.observations.map(v=>v.fire)),metadata:{...observations,observations:undefined,layer_date:o.date},exportPrefix:eventId.replace('/','__')+'_'+o.date})))))),
    result&&h('section',{className:'panel reveal-control'},h('h2',null,truth?'Vérité J+1 révélée':'Confronter la prévision à J+1'),h('p',null,truth?'Les erreurs sont calculées après intersection des masques valides.':'La prédiction est affichée ci-dessous. Révélez ensuite la cible dans une requête séparée.'),!truth&&h('button',{className:'button-primary',onClick:reveal,disabled:busy||revealing},revealing?'Lecture de la vérité…':'Révéler la vérité J+1')),
    result?h(Fade,{key:result.request_id||eventId+day,trigger:'mount',className:'forecast-reveal'},h(ForecastView,{result,truth})):h('section',{className:'studio-empty','aria-label':'Cartes en attente'},h('h2',null,'Observer J. Prédire J+1. Examiner les erreurs.'),h('p',null,'Lancez une prédiction, puis révélez la vérité pour mesurer ses limites.')),
    h('dialog',{ref:picker,className:'event-picker','aria-labelledby':'event-picker-title',onClose:()=>{setPickerOpen(false);browseButton.current?.focus();}},
      h('div',{className:'dialog-heading'},h('h2',{id:'event-picker-title'},'Explorer les événements'),h('button',{className:'icon-button','aria-label':'Fermer les événements',onClick:()=>picker.current?.close()},h(Icon,{name:'close'}))),
      h('label',{className:'sr-only',htmlFor:'event-query'},'Rechercher par identifiant'),h('input',{id:'event-query',type:'search',autoFocus:true,placeholder:'Rechercher par identifiant…',value:query,onChange:e=>setQuery(e.target.value)}),
      h('p',{className:'caption',role:'status'},`${filtered.length} événements · ${apiBase==='/lab'?'validation historique 2020 uniquement':'catalogue du snapshot historique'}`),
      pickerOpen&&(filtered.length?h(AnimatedList,{key:query,theme:'light',style:{'--al-ink':'var(--text)','--al-muted':'var(--text-secondary)','--al-tile':'var(--surface)','--al-tile-edge':'var(--line)'},items:filtered.map(e=>({id:e.event_id,title:e.event_id,description:e.scope==='calibration32'?'Réglage · 32 événements':e.scope==='observed_confirmation169'?'Confirmation déjà observée · 169 événements':`Snapshot historique · ${e.split||'périmètre du snapshot'}`,meta:e.forecast_dates.length+' dates',icon:h(Icon,{name:'flame'})})),initialSelectedIndex:filtered.findIndex(e=>e.event_id===eventId),selectOnHover:false,animation:motion?'fade':'none',stagger:.035,frame:false,width:'100%',maxHeight:380,showGradients:false,accentColor:'#9A3412',onItemSelect:item=>chooseEvent(item.id)}):h('p',null,'Aucun événement ne correspond à cette recherche.'))));
}
