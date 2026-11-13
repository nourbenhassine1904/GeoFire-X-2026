import React,{useEffect,useState} from 'react';
const h=React.createElement,n=(v,d=6)=>Number(v).toLocaleString('fr-FR',{maximumFractionDigits:d});
export function ComparisonView({data}){
  const s=data.selection,ref=s.baseline32;
  const metrics=[['Dice global',ref.full.dice,s.full.dice],['Dice nouveaux feux',ref.new_fire.dice,s.new_fire.dice],
    ['Rappel nouveaux feux',ref.new_fire.recall,s.new_fire.recall],['Précision nouveaux feux',ref.new_fire.precision,s.new_fire.precision],
    ['Précision globale',ref.full.precision,s.full.precision],['AP globale',ref.full.average_precision,s.full.average_precision]];
  const rows=[['Principal · poids10',.78,ref.full,ref.new_fire],['Expérience · tête nouveaux feux',s.threshold,s.full,s.new_fire]];
  return h('section',{className:'panel'},h('h2',null,'Expérience de propagation terminée · comparaison sur les32'),
    h('p',{className:'boundary-note'},'Critères d’acceptation : FALSE. Le modèle principal reste U-Net poids10 au seuil0,78. Aucune promotion automatique.'),
    h('p',null,`${data.training.epochs} époques · ${n(data.training.updates,0)} updates · ${data.training.gpu} · ${n(data.training.seconds/60,2)} min. Même périmètre de32 événements et masques ; seuils enregistrés, aucun nouveau réglage.`),
    h('div',{className:'tradeoff-bars'},...metrics.map(([label,a,b])=>h('div',{key:label},h('strong',null,label),
      ...[['Référence',a,'#66d6cb'],['Expérience',b,'#d0a6ff']].map(([name,v,color])=>h('div',{key:name,className:'tradeoff-row'},h('span',null,name),h('div',{className:'tradeoff-track'},h('div',{style:{width:`${100*v}%`,background:color}})),h('span',null,n(v))))))),
    h('p',null,`Nouveaux feux : +${n(data.delta.new_fire.tp,0)} vrais positifs, +${n(data.delta.new_fire.fp,0)} faux positifs. Le rappel augmente de ${n(100*data.delta.new_fire.recall,2)} points ; la précision baisse de ${n(-100*data.delta.new_fire.precision,2)} points. Le Dice global baisse de ${n(-data.delta.full.dice)} et l’AP de ${n(-data.delta.full.average_precision)}.`),
    h('div',{className:'table-scroll'},h('table',null,h('thead',null,h('tr',null,...['Modèle','Seuil','Dice global','AP','Brier','Précision globale','Rappel global','FP globaux','Dice nouveaux','Précision nouveaux','Rappel nouveaux','FP nouveaux'].map(k=>h('th',{key:k},k)))),
      h('tbody',null,...rows.map(([name,t,f,newfire])=>h('tr',{key:name},...[name,t,f.dice,f.average_precision,f.brier_score,f.precision,f.recall,f.fp,newfire.dice,newfire.precision,newfire.recall,newfire.fp].map((v,i)=>h('td',{key:i},typeof v==='number'?n(v):v))))))),
    h('p',{className:'legend'},'Les sorties fournies contiennent le checkpoint et les métriques agrégées, sans courbes complètes, métriques par événement ni cartes de cette expérience. Ces éléments ne sont pas inventés et l’évaluation n’est pas relancée.'),
    h('details',null,h('summary',null,'Provenance vérifiée et limites'),h('pre',null,JSON.stringify({provenance:data.provenance,limitations:data.limitations},null,2))));
}
export default function PropagationComparison(){
  const [data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{const c=new AbortController();fetch('/science/propagation-comparison',{signal:c.signal}).then(async r=>{if(!r.ok)throw new Error(`Comparaison indisponible (${r.status})`);setData(await r.json());}).catch(e=>{if(e.name!=='AbortError')setError(e.message);});return()=>c.abort();},[]);
  return error?h('p',{role:'alert'},error):data?h(ComparisonView,{data}):h('p',{role:'status'},'Lecture de l’expérience terminée…');
}
