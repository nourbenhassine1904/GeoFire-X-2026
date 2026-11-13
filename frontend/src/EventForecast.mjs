import React,{useEffect,useRef,useState} from 'react';
import {imagePixels,comparisonLayers,requestForecast,forecastCSV,downloadText} from './prediction_view.mjs';
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
  const legendView=()=>kind==='probability'?h('div',{className:'map-legend'},h('span',null,'0'),h('span',{className:'probability-scale'}),h('span',null,'1'),h('span',null,'Gris : invalide')):
    h('div',{className:'map-legend'},...legend.map(([color,label])=>h('span',{key:label},h('i',{style:{background:color}}),label)));
  return h('figure',{className:'prediction-map'},h('figcaption',null,title),
    h('div',{className:'map-actions'},h('button',{onClick:()=>{setZoom(1);setImage(canvas.current.toDataURL('image/png'));},type:'button'},'Agrandir'),
      h('button',{type:'button',onClick:()=>{const a=document.createElement('a');a.href=canvas.current.toDataURL('image/png');a.download=exportPrefix+'_'+title.normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-zA-Z0-9]+/g,'_')+'.png';a.click();}},'Exporter PNG')),
    h('canvas',{ref:canvas,role:'img','aria-label':title,style:{width:'100%',imageRendering:'pixelated'}}),
    legendView(),
    h('dialog',{ref:viewer,className:'map-dialog',onClose:()=>setImage(null)},image&&h(React.Fragment,null,
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
  const request=useRef(null);
  useEffect(()=>{const c=new AbortController();setLoading(true);
    fetch(`${apiBase}/events`,{signal:c.signal}).then(async r=>{const body=await r.json();if(!r.ok)throw new Error(body.detail||'Catalogue indisponible');setEvents(body);
      const initial=body.find(e=>e.event_id==='2020/fire_24332880')||body[0];
      if(initial){setEventId(initial.event_id);setDay(initial.forecast_dates[Math.floor(initial.forecast_dates.length/2)]||'');}
    }).catch(e=>{if(e.name!=='AbortError')setError(e.message);}).finally(()=>{if(!c.signal.aborted)setLoading(false);});
    return()=>{c.abort();request.current?.abort();};
  },[apiBase]);
  const selected=events.find(e=>e.event_id===eventId);
  const dateIndex=selected?.forecast_dates.indexOf(day)??-1;
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
  return h('main',{className:'science'},h('header',{className:'science-header'},h('span',{className:'eyebrow'},'Inférence réelle à la demande · PyTorch CPU'),
    h('h1',null,apiBase==='/lab'?'Prediction Lab':'Prévision historique TIFF'),h('p',null,apiBase==='/lab'?'Modèle principal : U-Net poids10 · seuil0,78 · événements historiques2020 · test2021 verrouillé':'Snapshot historique vérifié · modèle et seuil propres au snapshot · test2021 verrouillé')),
    h('section',{className:'panel'},h('h2',null,'Choisir un événement et prévoir J+1'),
      loading&&h('p',{role:'status'},'Vérification du checkpoint, des sources et du catalogue…'),
      h('label',null,'Événement ',h('select',{value:eventId,disabled:loading,onChange:e=>{reset();setEventId(e.target.value);const chosen=events.find(v=>v.event_id===e.target.value);setDay(chosen?.forecast_dates[Math.floor(chosen.forecast_dates.length/2)]||'');}},
        h('option',{value:''},'Sélectionner un événement'),...events.map(e=>h('option',{key:e.event_id,value:e.event_id},`${e.event_id} · ${e.scope==='calibration32'?'réglage (32)':e.scope==='observed_confirmation169'?'confirmation déjà observée (169)':e.split}`)))),
      h('label',{style:{marginLeft:'1rem'}},'Date J ',h('select',{value:day,disabled:loading,onChange:e=>{reset();setDay(e.target.value);}},
        h('option',{value:''},'Choisir une date'),...(selected?.forecast_dates||[]).map(d=>h('option',{key:d,value:d},d)))),
      h('button',{onClick:predict,disabled:loading||busy||!day||!eventId,style:{margin:'1rem'}},busy?'Inférence en cours…':'Prédire J+1'),
      h('div',{className:'temporal-nav'},h('button',{disabled:busy||dateIndex<=0,onClick:()=>{reset();setDay(selected.forecast_dates[dateIndex-1]);}},'← Date précédente'),
        h('span',null,dateIndex>=0?`${dateIndex+1} / ${selected.forecast_dates.length} dates disponibles`:''),
        h('button',{disabled:busy||dateIndex<0||dateIndex>=((selected?.forecast_dates.length||0)-1),onClick:()=>{reset();setDay(selected.forecast_dates[dateIndex+1]);}},'Date suivante →'),
        h('span',null,'Puis cliquer sur « Prédire J+1 ».')),
      busy&&h('p',{role:'status'},'Lecture des tuiles vérifiées → U-Net → carte prédite. La vérité J+1 sera chargée ensuite.'),
      error&&h('p',{role:'alert',className:'lab-error'},error),truthError&&h('p',{role:'alert',className:'lab-error'},truthError)),
    result&&h(ForecastView,{result,truth}));
}
