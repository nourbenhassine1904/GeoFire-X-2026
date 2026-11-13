import React,{useContext,useEffect,useMemo,useRef,useState} from 'react';
import {imagePixels,comparisonLayers,requestForecast,forecastCSV,downloadText} from './prediction_view.mjs';
import {Icon,Badge,MotionContext,Fade} from './studio_ui.mjs';
import Stepper,{Step} from './reactbits/Stepper.jsx';
import AnimatedList from './reactbits/AnimatedList.jsx';
const h=React.createElement;
const n=(v,d=2)=>Number(v).toLocaleString('fr-FR',{maximumFractionDigits:d});

export function FireMap({title,grid,kind='binary',exportPrefix='geofirex'}){
  const canvas=useRef(null),viewer=useRef(null),zoomBox=useRef(null),[image,setImage]=useState(null),[zoom,setZoom]=useState(1);
  useEffect(()=>{
    if(!grid?.length)return;
    const {width,height,pixels}=imagePixels(grid,kind),context=canvas.current.getContext('2d');
    canvas.current.width=width;canvas.current.height=height;
    const data=context.createImageData(width,height);data.data.set(pixels);context.putImageData(data,0,0);
  },[grid,kind]);
  useEffect(()=>{const dialog=viewer.current;if(!dialog)return;if(image&&!dialog.open)dialog.showModal();if(!image&&dialog.open)dialog.close();},[image]);
  useEffect(()=>{const box=zoomBox.current;if(box){box.scrollLeft=(box.scrollWidth-box.clientWidth)/2;box.scrollTop=(box.scrollHeight-box.clientHeight)/2;}},[zoom,image]);
  const legend=kind==='errors'?[['#30bb80','TP · feu détecté'],['#f25252','FP · fausse alerte'],['#3f8efc','FN · feu manqué'],['#14272d','TN · absence prévue'],['#55646c','Exclu']]:[['#ffb846','Feu'],['#14272d','Absence'],['#55646c','Invalide']];
  const sparse=useMemo(()=>{if(kind!=='binary'||!grid?.length)return null;let valid=0,fire=0;for(const row of grid)for(const v of row){if(v!==null){valid++;if(v===1)fire++;}}return valid&&fire/valid<=.01?{valid,fire}:null;},[grid,kind]);
  const legendView=()=>kind==='probability'?h('div',{className:'map-legend'},h('span',null,'0'),h('span',{className:'probability-scale'}),h('span',null,'1'),h('span',null,'Gris : invalide')):
    h('div',{className:'map-legend'},...legend.map(([color,label])=>h('span',{key:label},h('i',{style:{background:color}}),label)));
  return h('figure',{className:'prediction-map'},h('figcaption',null,title),
    h('div',{className:'map-actions'},h('button',{onClick:()=>{setZoom(1);setImage(canvas.current.toDataURL('image/png'));},type:'button'},'Agrandir'),
      h('button',{type:'button',onClick:()=>{const a=document.createElement('a');a.href=canvas.current.toDataURL('image/png');a.download=exportPrefix+'_'+title.normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-zA-Z0-9]+/g,'_')+'.png';a.click();}},'Exporter PNG')),
    h('canvas',{ref:canvas,role:'img','aria-label':title,style:{width:'100%',imageRendering:'pixelated'}}),
    legendView(),
    sparse&&h('p',{className:'sparse-map-note'},`${n(sparse.fire,0)} pixel(s) feu sur ${n(sparse.valid,0)} pixels valides de cet aperçu (≤ 1 %). Une carte peu colorée est un résultat réel ; le zoom et les exports conservent ces valeurs.`),
    h('dialog',{ref:viewer,className:'map-dialog','aria-label':title,onClose:()=>setImage(null)},image&&h(React.Fragment,null,
      h('div',{className:'dialog-heading'},h('strong',null,title),h('button',{onClick:()=>setImage(null),autoFocus:true},'Fermer')),
      h('label',{className:'map-zoom'},`Zoom ×${zoom} `,h('input',{type:'range',min:1,max:6,step:.5,value:zoom,'aria-label':'Zoom de la carte',onChange:e=>setZoom(Number(e.target.value))}),h('span',null,'Faire défiler pour explorer la carte.')),
      h('div',{ref:zoomBox,className:'map-zoom-viewport'},h('img',{src:image,alt:title,style:{imageRendering:'pixelated',width:`${100*zoom}%`,height:`${65*zoom}vh`,maxWidth:'none'}})),legendView())));
}

export function ForecastView({result,truth}){
  const layers=comparisonLayers(result,truth);
  const newPrediction=result.new_predicted_fire||result.predicted_fire.map((row,y)=>row.map((v,x)=>v===null?null:Number(v===1&&result.observed_fire[y][x]===0)));
  const prefix=result.event_id.replace('/','__')+'_'+result.target_date+'_'+result.model.replace(/[^a-zA-Z0-9]+/g,'_');
  const map=props=>h(FireMap,{...props,exportPrefix:prefix});
  return h('section',{className:'panel'},h('h2',null,`Prévision calculée · ${result.target_date}`),
    h('p',{className:'lab-status'},`${result.model} · seuil figé ${result.threshold}${result.device?' · '+result.device:''}${typeof result.elapsed_seconds==='number'?' · '+n(result.elapsed_seconds)+' s':''}${result.tiles!==undefined?' · '+n(result.tiles,0)+' tuiles':''}`),
    h('p',null,`Entrées : ${(result.input_dates||[]).join(', ')}. ${result.target_pixels_read===false?'Aucun pixel de J+1 lu pour cette prédiction.':''}`),
    h('p',null,`Grille complète : ${result.full_shape.join(' × ')} pixels. Aperçu : un pixel sur ${result.preview_stride} par axe ; toutes les tuiles sont inférées avant cet aperçu.`),
    h('p',null,result.scope==='observed_confirmation169'?'Événement du panel169 déjà consulté : démonstration ponctuelle, aucune nouvelle évaluation indépendante.':result.scope==='calibration32'?'Événement des32 de réglage : démonstration sur données de développement.':'Événement historique ; interpréter selon son périmètre de validation.'),
    h('p',{className:'legend'},'Gris : invalide ou exclu. La prédiction utilise les masques des cinq jours disponibles à J. Après lecture séparée de la vérité J+1, les cartes d’erreurs utilisent leur intersection. '+(result.crs?`Référence spatiale TIFF : ${result.crs}.`:'Grille de pixels du compact, sans géoréférencement disponible.')),
    h('div',{className:'lab-exports'},h('strong',null,'Exporter cet aperçu : '),
      h('button',{onClick:()=>downloadText(JSON.stringify({prediction:result,ground_truth:truth,export_scope:'preview grid; pixel coordinates; null invalid; no CRS'},null,2),prefix+'.json','application/json')},'JSON + provenance'),
      h('button',{onClick:()=>downloadText(forecastCSV(result,truth),prefix+'.csv','text/csv;charset=utf-8')},'CSV des pixels'),h('span',null,'PNG disponible sur chaque carte.')),
    h('div',{className:'prediction-maps'},map({title:`Feu observé J · ${result.as_of}`,grid:result.observed_fire}),
      map({title:'Probabilité de feu J+1 · 0 sombre, 1 clair',grid:result.probability,kind:'probability'}),
      map({title:`Prédiction de feu J+1 · seuil ${result.threshold}`,grid:result.predicted_fire}),
      map({title:'Nouveaux pixels prédits · absence à J',grid:newPrediction}),
      layers&&map({title:'Vérité terrain J+1 · lue séparément',grid:layers.ground}),
      layers&&map({title:'Nouveaux feux réels J+1',grid:layers.newTarget}),
      layers&&map({title:'Erreurs globales · TP vert, FP rouge, FN bleu',grid:layers.errors,kind:'errors'}),
      layers&&map({title:'Erreurs nouveaux feux · ancien feu exclu',grid:layers.newErrors,kind:'errors'})),
    h('p',{className:'lab-meta'},`Checkpoint ${result.checkpoint_sha256.slice(0,12)}… · ${n(result.optimizer_updates,0)} updates. Une nouvelle détection ne prouve pas une nouvelle surface brûlée.`),
    h('details',null,h('summary',null,'Identités des données et prétraitement'),h('pre',null,JSON.stringify({checkpoint:result.checkpoint_sha256,request_id:result.request_id,normalizer:result.normalizer_sha256,archive:result.source_archive_sha256,full_shape:result.full_shape,preview_stride:result.preview_stride,validity_dates:result.validity_dates,target_pixels_read:result.target_pixels_read},null,2))));
}

export default function EventForecast({apiBase=''}){
  const [events,setEvents]=useState([]),[eventId,setEventId]=useState(''),[day,setDay]=useState('');
  const [result,setResult]=useState(null),[truth,setTruth]=useState(null),[busy,setBusy]=useState(false),[loading,setLoading]=useState(true);
  const [error,setError]=useState(''),[truthError,setTruthError]=useState('');
  const [catalogueRevision,setCatalogueRevision]=useState(0);
  const request=useRef(null);
  const picker=useRef(null),browseButton=useRef(null),[query,setQuery]=useState(''),[pickerOpen,setPickerOpen]=useState(false);
  const {enabled:motion}=useContext(MotionContext);
  useEffect(()=>{const c=new AbortController();setLoading(true);setError('');
    fetch(`${apiBase}/events`,{signal:c.signal}).then(async r=>{const body=await r.json();if(!r.ok)throw new Error(body.detail||'Catalogue indisponible');setEvents(body);
      const initial=body.find(e=>e.event_id==='2020/fire_24332880')||body[0];
      if(initial){setEventId(initial.event_id);setDay(initial.forecast_dates[Math.floor(initial.forecast_dates.length/2)]||'');}
    }).catch(e=>{if(e.name!=='AbortError')setError(e.message);}).finally(()=>{if(!c.signal.aborted)setLoading(false);});
    return()=>{c.abort();request.current?.abort();};
  },[apiBase,catalogueRevision]);
  const selected=events.find(e=>e.event_id===eventId);
  const dateIndex=selected?.forecast_dates.indexOf(day)??-1;
  const filtered=events.filter(e=>e.event_id.toLowerCase().includes(query.toLowerCase()));
  const target=day?new Date(new Date(day+'T00:00:00Z').getTime()+86400000).toISOString().slice(0,10):'—';
  const stage=truth?4:result?3:busy?2:1;
  function chooseEvent(id){reset();setEventId(id);const chosen=events.find(e=>e.event_id===id);setDay(chosen?.forecast_dates[Math.floor(chosen.forecast_dates.length/2)]||'');picker.current?.close();}
  function reset(){request.current?.abort();setBusy(false);setResult(null);setTruth(null);setError('');setTruthError('');}
  async function predict(){
    reset();const c=new AbortController();request.current=c;setBusy(true);
    try{
      const response=await requestForecast(apiBase,eventId,day,{signal:c.signal,onPrediction:body=>{
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
    h('div',{className:'studio-contract-strip'},h('span',null,h('span',{className:'status-dot green'}),'PyTorch CPU'),h('span',null,apiBase==='/lab'?'Contexte T3 → J+1':'Contexte du snapshot → J+1'),h('span',null,apiBase==='/lab'?'Validation historique 2020':'Corpus du snapshot historique'),h('span',null,h(Icon,{name:'lock',size:13}),'Test 2021 fermé')),
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
      busy&&h('p',{role:'status',className:'inference-status'},h('span',{className:'inference-spinner','aria-hidden':true}),result?'Prédiction disponible. Lecture séparée de la vérité terrain J+1…':'Lecture des tuiles vérifiées → U-Net → carte prédite. La vérité J+1 sera chargée ensuite.'),
      error&&h('p',{role:'alert',className:'lab-error'},error),error&&!events.length&&h('button',{className:'button-secondary',onClick:()=>setCatalogueRevision(v=>v+1)},'Réessayer le catalogue'),truthError&&h('p',{role:'alert',className:'lab-error'},truthError)),
    result?h(Fade,{key:result.request_id||eventId+day,trigger:'mount',className:'forecast-reveal'},h(ForecastView,{result,truth})):h('section',{className:'studio-empty','aria-label':'Cartes en attente'},h('div',{className:'empty-icon'},h(Icon,{name:'studio',size:34})),h('h2',null,'Un événement. Une date. Les preuves en pixels.'),h('p',null,'Lancez une prédiction pour explorer les observations J, les sorties J+1 et les nouvelles détections.'),h('div',{className:'empty-map-labels'},h('span',null,'Observation J'),h('span',null,'Prédiction J+1'),h('span',null,'Vérité séparée'),h('span',null,'Erreurs TP / FP / FN')),h('p',{className:'caption'},apiBase==='/lab'?'Grille raster sans CRS disponible · aucune surface en hectares déduite':'Grille du snapshot TIFF · référence spatiale selon les fichiers')),
    h('dialog',{ref:picker,className:'event-picker','aria-labelledby':'event-picker-title',onClose:()=>{setPickerOpen(false);browseButton.current?.focus();}},
      h('div',{className:'dialog-heading'},h('h2',{id:'event-picker-title'},'Explorer les événements'),h('button',{className:'icon-button','aria-label':'Fermer les événements',onClick:()=>picker.current?.close()},h(Icon,{name:'close'}))),
      h('label',{className:'sr-only',htmlFor:'event-query'},'Rechercher par identifiant'),h('input',{id:'event-query',type:'search',autoFocus:true,placeholder:'Rechercher par identifiant…',value:query,onChange:e=>setQuery(e.target.value)}),
      h('p',{className:'caption',role:'status'},`${filtered.length} événements · ${apiBase==='/lab'?'validation historique 2020 uniquement':'catalogue du snapshot historique'}`),
      pickerOpen&&(filtered.length?h(AnimatedList,{key:query,theme:'light',style:{'--al-ink':'var(--text)','--al-muted':'var(--text-secondary)','--al-tile':'var(--surface)','--al-tile-edge':'var(--line)'},items:filtered.map(e=>({id:e.event_id,title:e.event_id,description:e.scope==='calibration32'?'Réglage · 32 événements':e.scope==='observed_confirmation169'?'Confirmation déjà observée · 169 événements':`Snapshot historique · ${e.split||'périmètre du snapshot'}`,meta:e.forecast_dates.length+' dates',icon:h(Icon,{name:'flame'})})),initialSelectedIndex:filtered.findIndex(e=>e.event_id===eventId),selectOnHover:false,animation:motion?'fade':'none',stagger:.035,frame:false,width:'100%',maxHeight:380,showGradients:false,accentColor:'#9A3412',onItemSelect:item=>chooseEvent(item.id)}):h('p',null,'Aucun événement ne correspond à cette recherche.'))));
}
