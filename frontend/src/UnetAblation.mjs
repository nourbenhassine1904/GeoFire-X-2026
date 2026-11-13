import React,{useEffect,useState} from 'react';
const h=React.createElement;
const num=(v,digits=6)=>v===null||v===undefined?'—':Number(v).toLocaleString('fr-FR',{maximumFractionDigits:digits});
const names={B1:'CNN historique · 0,95',B2:'U-Net historique · 0,95',B2_w10:'U-Net poids 10 · 0,78'};
const colors={B1:'#ffad66',B2:'#66d6cb',B2_w10:'#d0a6ff'};
function useJson(url){
  const [state,set]=useState({data:null,error:''});
  useEffect(()=>{const controller=new AbortController();
    fetch(url,{signal:controller.signal}).then(async r=>{if(!r.ok)throw new Error(`Résultat indisponible (${r.status})`);set({data:await r.json(),error:''});}).catch(e=>{if(e.name!=='AbortError')set({data:null,error:e.message});});
    return()=>controller.abort();},[url]);return state;
}
function Table({columns,rows}){return h('div',{className:'table-scroll'},h('table',null,
  h('thead',null,h('tr',null,...columns.map(([key,title])=>h('th',{key,scope:'col'},title)))),
  h('tbody',null,...rows.map((r,i)=>h('tr',{key:i},...columns.map(([key,title,format])=>h('td',{key},format?format(r[key],r):String(r[key]??'—'))))))));}
function EventDistribution({confirmed}){
  const [metric,setMetric]=useState('dice');
  const series={...confirmed.historical_per_event,B2_w10:confirmed.per_event};
  const all=Object.values(series).flat().map(r=>r[metric]).filter(v=>v!==null);
  const max=metric==='brier_score'?Math.max(...all,.001):1;
  const x=v=>45+(v/max)*620,y=v=>195-v*170;
  return h('section',{className:'panel'},h('h2',null,'Distribution sur les 169 événements'),
    h('label',null,'Métrique ',h('select',{value:metric,onChange:e=>setMetric(e.target.value)},...['dice','iou','precision','recall','average_precision','brier_score'].map(k=>h('option',{key:k,value:k},k)))),
    h('svg',{viewBox:'0 0 700 235',className:'distribution',role:'img','aria-label':`Distribution par événement ${metric}`},
      ...[0,.25,.5,.75,1].flatMap(v=>[h('line',{key:'line'+v,x1:45,x2:665,y1:y(v),y2:y(v),stroke:'#365053'}),h('text',{key:'label'+v,x:0,y:y(v)+4,fontSize:12,fill:'#b4cac5'},`${v*100}%`),h('text',{key:'value'+v,x:x(v*max),y:224,textAnchor:'middle',fontSize:12,fill:'#b4cac5'},num(v*max,3))]),
      ...Object.entries(series).map(([key,rows])=>{const values=rows.map(r=>r[metric]).filter(v=>v!==null).sort((a,b)=>a-b);return h('polyline',{key,points:values.map((v,i)=>`${x(v)},${y((i+1)/values.length)}`).join(' '),fill:'none',stroke:colors[key],strokeWidth:2.5});})),
    h('p',{className:'legend'},'Orange : CNN historique · Turquoise : U-Net historique · Violet : U-Net poids 10. Chaque événement a le même poids ; valeurs indéfinies exclues.'),
    h('details',null,h('summary',null,'Métriques par événement · poids 10'),h('div',{className:'event-table'},h(Table,{rows:confirmed.per_event,columns:[['event_id','Événement'],['dice','Dice',v=>num(v)],['iou','IoU',v=>num(v)],['precision','Précision',v=>num(v)],['recall','Rappel',v=>num(v)],['average_precision','AP',v=>num(v)],['brier_score','Brier',v=>num(v)],['tp','TP',v=>num(v,0)],['fp','FP',v=>num(v,0)],['fn','FN',v=>num(v,0)]]}))));
}
function Maps({confirmed,historical}){
  const [choice,setChoice]=useState('');const cases=confirmed.maps;
  const chosen=cases.find(c=>c.event_id===choice)||cases[0];
  if(!chosen)return null;
  const old=(historical||[]).filter(c=>c.event_id===chosen.event_id&&c.target_date===chosen.target_date);
  const display=[...old,chosen];
  const uri=(c,file)=>(c.model==='B2_w10'?'/science/unet-assets/':'/science/assets/')+c.directory+'/'+file;
  return h('section',{className:'panel'},h('h2',null,'Cartes de confirmation · mêmes événements réels'),
    h('label',null,'Événement ',h('select',{value:chosen.event_id,onChange:e=>setChoice(e.target.value)},...cases.map(c=>h('option',{key:c.event_id,value:c.event_id},`${c.event_id} · ${c.target_date}`)))),
    h('p',null,'Trois événements choisis par métadonnées, date médiane et toutes les tuiles. Grille de pixels ; invalides transparents. Aucune sélection de cas selon le score.'),
    ...display.map(c=>h('div',{key:c.model,className:'model-maps'},h('h3',null,names[c.model]),
      h('div',{className:'map-grid'},...[['target','Vérité terrain J+1'],['probability','Probabilités'],['segmentation',`Segmentation · ${num(c.threshold,2)}`],['errors','TP vert · FP rouge · FN bleu']].map(([key,title])=>h('figure',{key},h('figcaption',null,title),h('img',{src:uri(c,c.layers[key]),alt:`${title}, ${c.model}, ${c.event_id}`,loading:'lazy'})))),
      h('a',{href:uri(c,c.npz),download:true},'Tableaux réels NPZ : probabilités, cible, masque, segmentation et erreurs'))));
}
export default function UnetAblation(){
  const [revision,setRevision]=useState(0);
  const state=useJson(`/science/unet-ablation?revision=${revision}`);
  const oldMaps=useJson('/science/maps?phase=holdout169');
  useEffect(()=>{if(!state.data||state.data.confirmation?.status==='COMPLETE')return;
    const timer=setTimeout(()=>setRevision(r=>r+1),30000);return()=>clearTimeout(timer);
  },[state.data]);
  if(state.error)return h('main',{className:'science'},h('p',{role:'alert'},state.error));
  if(!state.data)return h('main',{className:'science'},h('p',{role:'status'},'Lecture des résultats vérifiés…'));
  const data=state.data;
  if(data.status!=='VERIFIED')return h('main',{className:'science'},h('h1',null,'Ablation U-Net'),h('p',null,'Résultats vérifiés non disponibles. Les résultats historiques restent accessibles.'));
  const c=data.confirmation,done=c.status==='COMPLETE';
  const rows=data.variants.map(v=>({...v,...v.metrics}));
  return h('main',{className:'science'},h('header',{className:'science-header'},h('span',{className:'eyebrow'},'Trois entraînements indépendants · vérification des checkpoints'),h('h1',null,'U-Net : du réglage à la confirmation'),h('p',null,'Seed 42 · 4 époques · 23 404 updates · 250 événements train · même état initial et même protocole')),
    h('section',{className:'panel'},h('h2',null,'Décision figée sur les 32 événements de réglage'),
      h(Table,{rows,columns:[['pos_weight','Poids positif'],['threshold','Seuil'],['dice','Dice micro',v=>num(v)],['iou','IoU',v=>num(v)],['precision','Précision',v=>num(v)],['recall','Rappel',v=>num(v)],['average_precision','AP',v=>num(v)],['brier_score','Brier ↓',v=>num(v)]]}),
      h('p',null,`Poids 10 retenu selon la règle Dice micro : Δ Dice = +${num(data.delta_dice_10_vs_30)} face au poids 30. Le poids 30 conserve la meilleure AP. Écart très faible, une seule seed : aucune supériorité statistique démontrée.`)),
    h('aside',{className:'boundary-note'},h('strong',null,'Modèle et seuil verrouillés : U-Net poids 10 · 0,78'),h('p',null,'Le seuil 0,78 est intérieur à la grille, contrairement à la borne historique 0,95. Les 169 événements ne choisissent ni le poids ni le seuil. Aucun essai supplémentaire et aucun pixel 2021.')),
    !done?h('section',{className:'panel'},h('h2',null,'Confirmation CPU en cours'),h('p',{role:'status'},`${c.events_completed||0} / 169 événements inférés. Le calcul exact de l’AP peut continuer après la fin de l’inférence. Aucun score partiel présenté comme confirmé.`)):
    h(React.Fragment,null,
      h('section',{className:'panel'},h('span',{className:'eyebrow'},'Une confirmation · CPU · aucun entraînement'),h('h2',null,'169 événements · seuils fixés, même domaine'),
        h(Table,{rows:c.comparison,columns:[['model','Modèle',v=>names[v]],['threshold','Seuil'],['dice','Dice',v=>num(v)],['iou','IoU',v=>num(v)],['precision','Précision',v=>num(v)],['recall','Rappel',v=>num(v)],['average_precision','AP',v=>num(v)],['brier_score','Brier ↓',v=>num(v)],['tp','TP',v=>num(v,0)],['fp','FP',v=>num(v,0)],['fn','FN',v=>num(v,0)]]}),
        h('p',null,`${num(c.comparison[2].valid_pixels,0)} pixels valides · ${num(c.comparison[2].tiles,0)} tuiles. AP et Brier issus des probabilités originales. La confirmation décrit le candidat retenu ; elle ne re-sélectionne pas le gagnant.`)),
      h(EventDistribution,{confirmed:c}),
      h('section',{className:'panel'},h('h2',null,'Nouveaux pixels de feu à J+1'),
        h('p',null,'Le rappel des nouveaux pixels de feu baisse avec ce point de fonctionnement. Le modèle retenu ne domine pas les références sur tous les critères.'),
        h(Table,{rows:[...(c.historical_new_fire||[]).map(r=>({model:r.model,recall:r.new_fire_recall,dice:r.new_fire_dice,precision:r.new_fire_precision})),{model:'B2_w10',...c.new_fire.new_fire_candidates}],columns:[['model','Nouveaux pixels · modèle',v=>names[v]],['dice','Dice',v=>num(v)],['precision','Précision',v=>num(v)],['recall','Rappel',v=>num(v)]]}),
        h(Table,{rows:[{domain:'Pas de détection à J',...c.new_fire.new_fire_candidates},{domain:'Déjà détecté à J',...c.new_fire.observed_fire_candidates}],columns:[['domain','Sous-domaine'],['positive_pixels','Feux cibles',v=>num(v,0)],['dice','Dice',v=>num(v)],['precision','Précision',v=>num(v)],['recall','Rappel',v=>num(v)],['tp','TP',v=>num(v,0)],['fp','FP',v=>num(v,0)],['fn','FN',v=>num(v,0)]]}),h('p',null,'Les deux sous-domaines partitionnent les pixels valides. Une détection nouvelle ne démontre pas une ignition physique.')),
      h(Maps,{confirmed:c,historical:oldMaps.data})),
    h('details',{className:'panel'},h('summary',null,'Preuves et identités de protocole'),
      h('p',null,'Empreintes des trois checkpoints vérifiées, quatre époques et curseur final contrôlés ; métriques et sommes des comptages par événement recalculées depuis les sorties enregistrées, sans réentraînement.'),
      h('pre',null,JSON.stringify({selection:data.selected,provenance:data.provenance,...(done?{confirmation:c.execution,lock_sha256:c.confirmation_lock_sha256}:{confirmation_status:c.status})},null,2))),
    h('p',{className:'legend'},'Les modèles historiques et leurs scores sont conservés. Cette vue présente les résultats et les cartes enregistrées ; elle ne contourne pas le module de prévision historique.'));
}
