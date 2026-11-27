import React,{useContext,useEffect,useMemo,useRef,useState} from 'react';
import SceneTimeline from './SceneTimeline.mjs';
import {activeBounds,requestForecast} from './prediction_view.mjs';
import {Icon,Badge,MotionContext,Fade} from './studio_ui.mjs';
import Stepper,{Step} from './reactbits/Stepper.jsx';
import AnimatedList from './reactbits/AnimatedList.jsx';
import {FireMap,ForecastView} from './PropagationView.mjs';
import {ModelStatus} from './ScientificUI.mjs';
import {readJSON} from './api.mjs';
export {FireMap,ForecastView} from './PropagationView.mjs';
const h=React.createElement;
const n=(v,d=2)=>Number(v).toLocaleString('fr-FR',{maximumFractionDigits:d});

export default function EventForecast({apiBase='',modeSwitch}){
  const [modelId,setModelId]=useState('C8'),[models,setModels]=useState([]),[modelError,setModelError]=useState('');
  const service=apiBase==='/lab'&&modelId!=='legacy'?`/studio/v2/${modelId}`:apiBase;
  const modelLabel=models.find(m=>m.id===modelId)?.label||modelId;
  const demoEvent=modelId==='legacy'?'2020/fire_24332880':'2020/fire_24332704';
  const demoDay=modelId==='legacy'?'2020-09-03':'2020-09-09';
  const [events,setEvents]=useState([]),[eventId,setEventId]=useState(''),[day,setDay]=useState('');
  const [result,setResult]=useState(null),[truth,setTruth]=useState(null),[busy,setBusy]=useState(false),[loading,setLoading]=useState(true);
  const [error,setError]=useState(''),[truthError,setTruthError]=useState('');
  const [catalogueRevision,setCatalogueRevision]=useState(0);
  const [loadedService,setLoadedService]=useState('');
  const [observations,setObservations]=useState(null),[observationError,setObservationError]=useState(''),[revealing,setRevealing]=useState(false);
  const request=useRef(null);
  const picker=useRef(null),browseButton=useRef(null),[query,setQuery]=useState(''),[pickerOpen,setPickerOpen]=useState(false);
  const {enabled:motion}=useContext(MotionContext);
  useEffect(()=>{if(apiBase!=='/lab')return;const c=new AbortController();
    fetch('/studio/models',{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok)throw Error(b.detail||'Modèles indisponibles');if(!c.signal.aborted)setModels(b.models);}).catch(e=>{if(e.name!=='AbortError')setModelError(e.message);});return()=>c.abort();
  },[apiBase,catalogueRevision]);
  useEffect(()=>{const c=new AbortController();setLoading(true);setError('');
    setEvents([]);setEventId('');setDay('');setResult(null);setTruth(null);setLoadedService('');
    readJSON(`${service}/events`,{signal:c.signal}).then(body=>{if(c.signal.aborted)return;setEvents(body);setLoadedService(service);
      const initial=body.find(e=>e.event_id===demoEvent)||body[0];
      if(initial){setEventId(initial.event_id);setDay(initial.forecast_dates.includes(demoDay)?demoDay:initial.forecast_dates[Math.floor(initial.forecast_dates.length/2)]||'');}
    }).catch(e=>{if(e.name!=='AbortError')setError(e.message);}).finally(()=>{if(!c.signal.aborted)setLoading(false);});
    return()=>{c.abort();request.current?.abort();};
  },[service,catalogueRevision]);
  const selected=events.find(e=>e.event_id===eventId);
  useEffect(()=>{setObservations(null);setObservationError('');if(apiBase!=='/lab'||loading||loadedService!==service||!eventId||!day)return;const c=new AbortController();
    readJSON(`${service}/observations?${new URLSearchParams({event_id:eventId,as_of:day})}`,{signal:c.signal}).then(b=>{if(!c.signal.aborted)setObservations(b);}).catch(e=>{if(e.name!=='AbortError')setObservationError(e.message);});return()=>c.abort();
  },[service,eventId,day,loadedService,loading]);
  const dateIndex=selected?.forecast_dates.indexOf(day)??-1;
  const filtered=events.filter(e=>e.event_id.toLowerCase().includes(query.toLowerCase()));
  const target=day?new Date(new Date(day+'T00:00:00Z').getTime()+86400000).toISOString().slice(0,10):'—';
  const stage=truth?4:result?3:busy?2:1;
  function chooseEvent(id){reset();setEventId(id);const chosen=events.find(e=>e.event_id===id);setDay(chosen?.forecast_dates[Math.floor(chosen.forecast_dates.length/2)]||'');picker.current?.close();}
  function reset(){request.current?.abort();setBusy(false);setRevealing(false);setResult(null);setTruth(null);setError('');setTruthError('');}
  async function reveal(){
    if(!result||truth||busy||revealing)return;const c=new AbortController();request.current=c;setRevealing(true);setTruthError('');
    try{const q=new URLSearchParams({event_id:result.event_id,as_of:result.as_of,max_side:512});const r=await fetch(`${service}/ground-truth?${q}`,{signal:c.signal}),b=await r.json();if(!r.ok)throw Error(b.detail||'Vérité indisponible');
      if(JSON.stringify(b.full_shape)!==JSON.stringify(result.full_shape)||b.preview_stride!==result.preview_stride||b.crs!==result.crs||JSON.stringify(b.transform)!==JSON.stringify(result.transform))throw Error('Géométrie de vérité incompatible');
      if(!c.signal.aborted&&request.current===c)setTruth(b.ground_truth);
    }catch(e){if(e.name!=='AbortError')setTruthError(e.message);}finally{if(request.current===c)setRevealing(false);}
  }
  async function predict(){
    reset();const c=new AbortController();request.current=c;setBusy(true);
    try{
      const response=await requestForecast(service,eventId,day,{signal:c.signal,revealTruth:false,onPrediction:body=>{
        if(!c.signal.aborted&&request.current===c)setResult(body);
      }});
      if(c.signal.aborted||request.current!==c)return;
      setTruth(response.truth);setTruthError(response.truthError);
    }catch(e){if(e.name!=='AbortError')setError(e.message);}
    finally{if(request.current===c)setBusy(false);}
  }
  return h('section',{className:'science prediction-studio'},h('header',{className:'science-header studio-header'},h('div',null,h('span',{className:'eyebrow'},'EXPLORER · PRÉDIRE · COMPARER'),
    h('h1',null,apiBase==='/lab'?'Studio de prédiction':'Predictor TIFF historique'),h('p',null,apiBase==='/lab'?'Choisir une scène 2020, calculer J+1 sur CPU, puis confronter la prévision à une vérité révélée séparément.':'Snapshot historique vérifié · modèle et seuil propres au snapshot.')),
    h(Badge,{tone:'violet'},h(Icon,{name:'studio',size:14}),apiBase==='/lab'?(modelId==='legacy'?'U-NET HISTORIQUE':modelId+' V2')+' · CPU · 0,78':'SNAPSHOT HISTORIQUE')),modeSwitch,
    h('div',{className:'demo-brief'},h('div',null,h('strong',null,'Scène de référence'),h('p',{className:'caption'},`${demoEvent} · observation ${demoDay} · contexte prédéfini pour le protocole sélectionné`)),h('button',{className:'button-secondary',disabled:busy||loading,onClick:()=>{reset();setEventId(demoEvent);setDay(demoDay);}},'Charger la scène de référence')),
    h('div',{className:'studio-contract-strip'},h('span',null,h('span',{className:'status-dot green'}),'PyTorch CPU'),h('span',null,apiBase==='/lab'?'Contexte T3 → J+1':'Contexte du snapshot → J+1'),h('span',null,apiBase==='/lab'?'Validation 2020 uniquement':'Corpus du snapshot historique'),h('span',null,h(Icon,{name:'lock',size:13}),'Test 2021 terminé · lecture seule')),
    apiBase==='/lab'&&h('nav',{className:'scene-jump-nav','aria-label':'Parcours de la scène'},...[
      ['Observations','.observation-history',false],['Prévision','.forecast-reveal',!result],[truth?'Vérité J+1':'Révélation J+1','.reveal-control',!result],['Chronologie','.timeline-player',!observations],['Erreurs','.forecast-reveal .scene-metrics',!truth]
    ].map(([label,selector,disabled])=>h('button',{key:label,disabled,onClick:()=>document.querySelector(selector)?.scrollIntoView({behavior:motion?'smooth':'instant',block:'start'})},label))),
    h('section',{className:'panel prediction-controls','aria-label':'Configuration de la prédiction'},
      h('div',{className:'section-heading'},h('h2',null,'Votre espace de prédiction'),h('span',{className:'caption'},'01 / Configuration')),
      apiBase==='/lab'&&h('div',{className:'model-selector'},h('label',{htmlFor:'forecast-model'},'Modèle à exécuter sur CPU'),h('select',{id:'forecast-model',value:modelId,disabled:busy||revealing,onChange:e=>{reset();setModelId(e.target.value);}},...models.map(m=>h('option',{key:m.id,value:m.id,disabled:!m.enabled},m.label+(!m.enabled?' · indisponible':'')))),h('div',{className:'model-selection-status'},h(ModelStatus,{id:modelId}),h('p',null,modelId==='C8'?'Référence évaluée indépendamment en 2021. Interactions limitées au panel V2 2020.':modelId==='legacy'?'Ancien protocole · catalogue distinct de 201 événements 2020.':'Variante non promue · mêmes 80 événements 2020 que C8.'),models.find(m=>m.id===modelId)?.checkpoint_sha256&&h('code',null,'Checkpoint '+models.find(m=>m.id===modelId).checkpoint_sha256.slice(0,16)+'…')),modelError&&h('p',{role:'alert'},modelError),...models.filter(m=>!m.enabled).map(m=>h('p',{key:m.id,className:'caption',role:'status'},m.label+' — '+m.reason))),
      loading&&h('p',{role:'status',className:'loading-note'},'Vérification du checkpoint, des sources et du catalogue…'),
      h('div',{className:'prediction-form'},
        h('div',{className:'form-field event-field'},h('label',{htmlFor:'forecast-event'},'Événement de validation 2020'),h('div',{className:'event-select-row'},h('select',{id:'forecast-event',value:eventId,disabled:loading||busy,onChange:e=>chooseEvent(e.target.value)},
          h('option',{value:''},'Sélectionner un événement'),...events.map(e=>h('option',{key:e.event_id,value:e.event_id},`${e.event_id} · ${e.scope==='calibration32'?'réglage (32)':e.scope==='observed_confirmation169'?'confirmation observée (169)':'validation V2 (80)'}`))),
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
    apiBase==='/lab'&&h('section',{className:'panel observation-history'},h('h2',null,`Observer le passé · ${eventId}`),h('p',null,'Les observations J−2, J−1 et J alimentent le modèle sélectionné avec huit canaux par jour. La vérité J+1 se consulte séparément après l’inférence.'),observationError?h('p',{role:'alert'},observationError):!observations?h('p',{role:'status'},'Lecture des trois observations passées…'):h(React.Fragment,null,h(SceneTimeline,{observations,result,truth,onReveal:reveal,revealing}),h('details',null,h('summary',null,'Voir les trois observations côte à côte'),h('div',{className:'prediction-maps history-maps'},...observations.observations.map(o=>h(FireMap,{key:o.date,title:'Observation · '+o.date,grid:o.fire,bounds:activeBounds(observations.observations.map(v=>v.fire)),metadata:{...observations,observations:undefined,layer_date:o.date},exportPrefix:eventId.replace('/','__')+'_'+o.date})))))),
    result&&h('section',{className:'panel reveal-control'},h('div',null,h('h2',null,truth?'Vérité J+1 révélée':'Confronter la prévision à J+1'),h('p',null,truth?'Les erreurs utilisent l’intersection des masques valides.':'La cible sera consultée dans une requête indépendante de l’inférence.')),truth?h('span',{className:'truth-revealed',role:'status'},h(Icon,{name:'check',size:16}),'Cible consultée · comparaison disponible'):h('button',{className:'button-primary',onClick:reveal,disabled:busy||revealing},revealing?'Lecture de la vérité…':'Révéler la vérité J+1')),
    result?h(Fade,{key:result.request_id||eventId+day,trigger:'mount',className:'forecast-reveal'},h(ForecastView,{result,truth})):h('section',{className:'studio-empty','aria-label':'Cartes en attente'},h('h2',null,'Observer J. Prédire J+1. Examiner les erreurs.'),h('p',null,'Lancez une prédiction, puis révélez la vérité pour mesurer ses limites.')),
    h('dialog',{ref:picker,className:'event-picker','aria-labelledby':'event-picker-title',onClose:()=>{setPickerOpen(false);browseButton.current?.focus();}},
      h('div',{className:'dialog-heading'},h('h2',{id:'event-picker-title'},'Explorer les événements'),h('button',{className:'icon-button','aria-label':'Fermer les événements',onClick:()=>picker.current?.close()},h(Icon,{name:'close'}))),
      h('label',{className:'sr-only',htmlFor:'event-query'},'Rechercher par identifiant'),h('input',{id:'event-query',type:'search',autoFocus:true,placeholder:'Rechercher par identifiant…',value:query,onChange:e=>setQuery(e.target.value)}),
      h('p',{className:'caption',role:'status'},`${filtered.length} événements · ${apiBase==='/lab'?'validation 2020 uniquement':'catalogue du snapshot historique'}`),
      pickerOpen&&(filtered.length?h(AnimatedList,{key:query,theme:'light',style:{'--al-ink':'var(--text)','--al-muted':'var(--text-secondary)','--al-tile':'var(--surface)','--al-tile-edge':'var(--line)'},items:filtered.map(e=>({id:e.event_id,title:e.event_id,description:e.scope==='calibration32'?'Réglage · 32 événements':e.scope==='observed_confirmation169'?'Confirmation déjà observée · 169 événements':'Validation V2 · 80 événements',meta:e.forecast_dates.length+' dates',icon:h(Icon,{name:'flame'})})),initialSelectedIndex:filtered.findIndex(e=>e.event_id===eventId),selectOnHover:false,animation:motion?'fade':'none',stagger:.035,frame:false,width:'100%',maxHeight:380,showGradients:false,accentColor:'#9A3412',onItemSelect:item=>chooseEvent(item.id)}):h('p',null,'Aucun événement ne correspond à cette recherche.'))));
}
