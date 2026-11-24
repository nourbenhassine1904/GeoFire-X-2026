import React,{useEffect,useMemo,useRef,useState} from 'react';
import {imagePixels,comparisonLayers,activeBounds,cropGrid,previewMetrics,forecastCSV,downloadText} from './prediction_view.mjs';
const h=React.createElement;
const num=(v,d=2)=>v===null||v===undefined?'—':Number(v).toLocaleString('fr-FR',{maximumFractionDigits:d});
const percent=v=>v===null?'—':num(100*v)+' %';
export const metricHelp={TP:'Pixels de feu correctement détectés.',FP:'Pixels annoncés en feu, absents de l’observation.',FN:'Pixels de feu observés mais manqués.',Rappel:'Part des pixels de feu réels retrouvés. Plus haut = davantage de détections.',Dice:'Recouvrement entre prédiction et observation, de 0 à 100 %.',Précision:'Part des pixels annoncés qui sont réellement en feu.','AP exacte':'Classement des scores sur tous les seuils, avec gestion des ex æquo.',Brier:'Erreur quadratique des scores ; plus bas = mieux. Ne prouve pas la calibration.'};

export function FireMap({title,grid,kind='binary',exportPrefix='geofirex',bounds=null,metadata={}}){
 const canvas=useRef(null),viewer=useRef(null),[open,setOpen]=useState(false),[zoom,setZoom]=useState(1),[fullscreen,setFullscreen]=useState(false),[screenError,setScreenError]=useState('');
 const shown=useMemo(()=>cropGrid(grid,bounds),[grid,bounds]);
 useEffect(()=>{if(!shown?.length)return;const {width,height,pixels}=imagePixels(shown,kind),c=canvas.current,ctx=c.getContext('2d');c.width=width;c.height=height;const data=ctx.createImageData(width,height);data.data.set(pixels);ctx.putImageData(data,0,0);},[shown,kind]);
 useEffect(()=>{if(open&&!viewer.current.open)viewer.current.showModal();},[open]);
 useEffect(()=>{const changed=()=>setFullscreen(Boolean(document.fullscreenElement));document.addEventListener('fullscreenchange',changed);return()=>document.removeEventListener('fullscreenchange',changed);},[]);
 if(!grid?.length)return null;
 const name=exportPrefix+'_'+title.normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-zA-Z0-9]+/g,'_');
 const legend=kind==='errors'?[['#30bb80','TP · détecté'],['#f25252','FP · fausse alerte'],['#3f8efc','FN · manqué'],['#14272d','Absence correcte · TN'],['#55646c','Invalide / hors domaine']]:[['#ffb846','Feu actif'],['#14272d','Absence'],['#55646c','Invalide']];
 const legendView=()=>h('div',{className:'map-legend'},...(kind==='probability'?[h('span',{key:'scale'},'Score 0 ',h('i',{className:'probability-scale'}),' 1 · non calibré'),h('span',{key:'null'},'Gris : invalide')]:legend.map(([color,label])=>h('span',{key:label},h('i',{style:{background:color}}),label))));
 function exportPNG(){const a=document.createElement('a');a.href=canvas.current.toDataURL('image/png');a.download=name+'.png';a.click();}
 function exportMetadata(){downloadText(JSON.stringify({...metadata,layer:title,kind,viewport:bounds||{top:0,left:0,height:grid.length,width:grid[0].length},png_pixels:'one image pixel per displayed raster sample; no interpolation; no CRS',legend:kind==='errors'?{0:'TN',1:'TP',2:'FP',3:'FN',null:'excluded'}:kind==='probability'?'continuous score 0..1; invalid grey':{0:'absence',1:'active_fire',null:'invalid'}},null,2),name+'.metadata.json','application/json');}
 return h('figure',{className:'prediction-map propagation-map'},h('figcaption',null,title),
  h('div',{className:'map-actions'},h('button',{type:'button',onClick:()=>{setZoom(1);setOpen(true);}},'Agrandir'),h('button',{type:'button',onClick:exportPNG},'Exporter PNG'),h('button',{type:'button',onClick:exportMetadata},'Métadonnées JSON')),
  h('p',{className:'pixel-coordinates'},`Pixels · ligne ${(metadata.tile_origin?.[0]||0)+(bounds?.top||0)*(metadata.preview_stride||1)}, colonne ${(metadata.tile_origin?.[1]||0)+(bounds?.left||0)*(metadata.preview_stride||1)} · pas ${metadata.preview_stride||1}`),h('canvas',{ref:canvas,role:'img','aria-label':title,style:{width:'100%',height:'auto',aspectRatio:`${shown[0].length} / ${shown.length}`,imageRendering:'pixelated'}}),legendView(),
  h('dialog',{ref:viewer,className:'map-dialog','aria-label':title,onClose:()=>{setOpen(false);if(document.fullscreenElement)document.exitFullscreen?.();}},open&&h(React.Fragment,null,
   h('div',{className:'dialog-heading'},h('strong',null,title),h('button',{onClick:()=>{viewer.current.close();setOpen(false);}},'Fermer')),
   h('div',{className:'map-actions'},h('label',null,`Zoom ×${zoom} `,h('input',{type:'range',min:1,max:8,step:.5,value:zoom,'aria-label':'Zoom de la carte',onChange:e=>setZoom(Number(e.target.value))})),h('button',{onClick:async()=>{setScreenError('');try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();else throw Error('Utilisez l’agrandissement du navigateur.');}catch(e){setScreenError('Plein écran indisponible : '+e.message);}}},fullscreen?'Quitter le plein écran':'Plein écran')),
   screenError&&h('p',{role:'alert'},screenError),
   h('div',{className:'map-zoom-viewport'},h('img',{src:canvas.current?.toDataURL('image/png'),alt:title,style:{imageRendering:'pixelated',width:`${100*zoom}%`,height:'auto',maxWidth:'none'}})),legendView())));
}

export function SceneMetrics({metrics,scope}){
 if(!metrics)return null;
 return h('section',{className:'scene-metrics','aria-label':'Métriques calculées de la scène'},h('h3',null,'Résultats de cette scène'),h('p',{className:'caption'},scope+' · aucune métrique agrégée remplacée.'),
  h('div',{className:'scene-score-grid'},...Object.entries({TP:metrics.tp,FP:metrics.fp,FN:metrics.fn,Rappel:percent(metrics.recall),Dice:percent(metrics.dice),Précision:percent(metrics.precision),'AP exacte':percent(metrics.AP),Brier:num(metrics.Brier,6)}).map(([label,value])=>h('div',{key:label,className:['TP','FP','FN'].includes(label)?'score-'+label:''},h('span',null,label),h('strong',null,typeof value==='number'?num(value,0):value),h('small',null,metricHelp[label])))));
}

export function ForecastView({result,truth}){
 const [domain,setDomain]=useState('new'),[focus,setFocus]=useState(true),[all,setAll]=useState(false),[pair,setPair]=useState(false);
 const layers=comparisonLayers(result,truth),newOnly=domain==='new';
 const newPrediction=result.new_predicted_fire||result.predicted_fire.map((row,y)=>row.map((v,x)=>v===null?null:Number(v===1&&result.observed_fire[y][x]===0)));
 const bounds=useMemo(()=>activeBounds([result.observed_fire,result.predicted_fire]),[result]);
 const metrics=result.scene_metrics?.[newOnly?'new_fire':'global']||previewMetrics(result,truth,newOnly);
 const scope=result.scene_metrics?'Scène complète sauvegardée, masque valide commun':result.tile_origin?'Tuile diagnostique uniquement, pas la scène complète':result.preview_stride===1?'Scène complète, intersection des masques J et J+1':'Aperçu échantillonné uniquement, pas la scène complète';
 const prefix=result.event_id.replace('/','__')+'_'+result.target_date+'_'+result.model.replace(/[^a-zA-Z0-9]+/g,'_');
 const metadata={event_id:result.event_id,as_of:result.as_of,target_date:result.target_date,model:result.model,checkpoint_sha256:result.checkpoint_sha256,source_archive_sha256:result.source_archive_sha256,saved_member_sha256:result.saved_member_sha256,threshold:result.threshold,full_shape:result.full_shape,preview_stride:result.preview_stride,tile_origin:result.tile_origin||[0,0],crs:result.crs,transform:result.transform,provenance:result.provenance||'CPU interactive historical inference',domain,metrics_scope:scope};
 const map=props=>h(FireMap,{...props,key:props.title,exportPrefix:prefix,bounds:focus?bounds:null,metadata});
 return h('section',{className:'panel propagation-result'},h('h2',null,`${result.provenance?.startsWith('Saved')?'Sortie sauvegardée':'Prévision calculée'} · ${result.event_id} · ${result.target_date}`),
  h('p',{className:'lab-status'},`${result.model} · seuil figé ${result.threshold} · ${result.device||'lecture seule'}${result.elapsed_seconds!==undefined?' · '+num(result.elapsed_seconds)+' s':''}`),
  h('p',null,`Entrées : ${(result.input_dates||[]).join(', ')}. ${result.target_pixels_read===false?'Aucun pixel de J+1 lu pour cette prédiction.':''}`),
  h('p',{className:'caption'},`Grille ${result.full_shape.join(' × ')} pixels · aperçu : un pixel sur ${result.preview_stride} par axe. Grille de pixels sans géoréférencement disponible (CRS/affine absents).`),
  h('p',{className:'caption'},result.scope?.includes('V2')?'Validation 2020 V2 · panel de 80 déjà observé · variante expérimentale.':result.scope==='calibration32'?'Historique : événement des 32 de réglage.':'Historique : panel 169 déjà observé, distinct de V2.'),
  h('div',{className:'view-controls'},h('div',{role:'group','aria-label':'Domaine de visualisation'},h('button',{'aria-pressed':newOnly,onClick:()=>setDomain('new')},'Nouveaux feux'),h('button',{'aria-pressed':!newOnly,onClick:()=>setDomain('global')},'Feu global')),
   h('button',{'aria-pressed':focus,onClick:()=>setFocus(!focus)},focus?'Afficher toute la grille':'Zoom zone active'),h('button',{'aria-pressed':all,onClick:()=>setAll(!all)},all?'Vue essentielle':'Toutes les couches'),truth&&h('button',{'aria-pressed':pair,onClick:()=>setPair(v=>!v)},pair?'Vue des couches':'Prévision / vérité côte à côte')),
  h('p',{className:'caption'},focus?`Zoom commun aux cartes : ligne ${bounds.top}, colonne ${bounds.left}, ${bounds.height} × ${bounds.width} échantillons. Cadre fondé sur J et la prédiction ; inchangé à la révélation.`:'Vue complète. Aucun étirement ni interpolation des pixels.'),
  truth&&h(SceneMetrics,{metrics,scope:scope+(newOnly?' · nouveaux feux (absence à J)':' · feu global')}),
  h('div',{className:'lab-exports'},h('button',{onClick:()=>downloadText(JSON.stringify({metadata,prediction:result,ground_truth:truth,scene_metrics:truth?metrics:null,export_scope:'complete preview grid; uncropped; pixel coordinates'},null,2),prefix+'.json','application/json')},'JSON + provenance'),h('button',{onClick:()=>downloadText(forecastCSV(result,truth),prefix+'.csv','text/csv;charset=utf-8')},'CSV des pixels')),
  pair&&truth?h('div',{className:'prediction-maps comparison-pair'},map({title:newOnly?'Nouveaux feux prédits J+1':'Prévision globale J+1',grid:newOnly?newPrediction:result.predicted_fire}),map({title:newOnly?'Nouveaux feux observés J+1':'Vérité observée J+1',grid:newOnly?layers.newTarget:layers.ground})):h('div',{className:'prediction-maps'},map({title:`Feu observé J · ${result.as_of}`,grid:result.observed_fire}),map({title:'Probabilité de feu J+1',grid:result.probability,kind:'probability'}),
   map({title:newOnly?'Nouveaux pixels prédits J+1':`Prédiction globale J+1 · seuil ${result.threshold}`,grid:newOnly?newPrediction:result.predicted_fire}),
   layers&&map({title:newOnly?'Erreurs nouveaux feux · TP / FP / FN':'Erreurs globales · TP / FP / FN',grid:newOnly?layers.newErrors:layers.errors,kind:'errors'}),
   all&&map({title:newOnly?'Prédiction globale J+1':'Nouveaux pixels prédits J+1',grid:newOnly?result.predicted_fire:newPrediction}),
   all&&layers&&map({title:'Vérité terrain J+1 · lecture séparée',grid:layers.ground}),all&&layers&&map({title:'Nouveaux feux réels J+1',grid:layers.newTarget}),
   all&&layers&&map({title:newOnly?'Erreurs globales':'Erreurs nouveaux feux',grid:newOnly?layers.errors:layers.newErrors,kind:'errors'})),
  !truth&&h('p',{className:'reveal-note'},'Prédiction affichée. La vérité J+1 reste masquée jusqu’à votre action.'),
  h('p',{className:'lab-meta'},`Checkpoint ${result.checkpoint_sha256.slice(0,12)}… · ${num(result.optimizer_updates,0)} updates. Une nouvelle détection active ne prouve pas une ignition ou une surface brûlée.`),
  h('details',null,h('summary',null,'Identités et provenance'),h('pre',null,JSON.stringify({...metadata,input_dates:result.input_dates,validity_dates:result.validity_dates,target_pixels_read:result.target_pixels_read,metrics:truth?metrics:null},null,2))));
}
