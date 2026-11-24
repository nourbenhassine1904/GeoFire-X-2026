import React,{useState} from 'react';
import {h,PageHeader,Badge,SourceState,useResource,number,downloadJSON,StatCard,Icon,WelcomeTitle,FireEmblem} from './studio_ui.mjs';
import {ForecastView} from './EventForecast.mjs';
import {OverviewEvidence,V2MetricCharts,SavedSpatialComparison} from './V2Views.mjs';

const columns=[['recall_new','Rappel nouveaux',true],['dice_new','Dice nouveaux',true],['precision_new','Précision nouveaux',true],['AP_new','AP nouveaux',true],['FP_new','FP nouveaux',false],['dice_global','Dice global',true],['AP_global','AP globale',true],['Brier','Brier',false]];
export function tileCounts(item){
 const {result,truth}=item;let tp=0,fp=0,fn=0,tn=0;
 truth.forEach((row,y)=>row.forEach((target,x)=>{if(target===null||result.observed_fire[y][x]!==0)return;const predicted=result.predicted_fire[y][x]===1;if(predicted){if(target===1)tp++;else fp++;}else if(target===1)fn++;else tn++;}));
 return {tp,fp,fn,tn,recall:tp+fn?tp/(tp+fn):null,precision:tp+fp?tp/(tp+fp):null,dice:2*tp+fp+fn?2*tp/(2*tp+fp+fn):null};
}

export function JuryOverview({navigate}){
 const legacy=useResource('/science/unet-ablation'),lineage=useResource('/lab/lineage'),v2=useResource('/science/v2/comparison');
 const map=legacy.data?.confirmation?.maps?.[0];
 return h('div',{className:'overview page-width'},
  h('section',{className:'overview-hero'},h(FireEmblem,{hero:true}),h('div',{className:'hero-copy'},h(Badge,{tone:'violet'},'GEOFIRE-X · JURY 2 DÉCEMBRE 2026'),h(WelcomeTitle),
   h('p',{className:'hero-description'},'Trois jours d’observation. Une prévision pour demain. Explorez le feu, confrontez les modèles et comprenez leurs résultats sur des scènes réelles.'),
   h('div',{className:'hero-actions'},h('button',{className:'button-primary',onClick:()=>navigate('studio')},'Explorer la propagation'),h('button',{className:'button-secondary',onClick:()=>navigate('comparison')},'Comparer C8 et les variantes')),
   h('p',{className:'hero-trust'},'Validation historique 2020 · test 2021 fermé · aucun entraînement au lancement')),
   h(OverviewEvidence)),
  h('div',{className:'overview-stats'},h(StatCard,{icon:'layers',label:'Train historique servi',value:lineage.data?.roles.train.events,detail:'2018–2019 · protocole historique'}),h(StatCard,{icon:'studio',label:'Validation historique servie',value:lineage.data?.roles.val.events,detail:'2020 · 32 réglage + 169 observés'}),h(StatCard,{icon:'analytics',label:'Validation comparative V2',value:v2.data?.models?.[0]?.events,detail:'2020 · 80 événements · seuil 0,78'}),h(StatCard,{icon:'clock',label:'Mises à jour du modèle servi',value:legacy.data?23404:undefined,detail:'4 époques · ancien protocole'})),
  h('section',{className:'protocol-band'},h('div',null,h(Badge,null,'INTERACTIF HISTORIQUE'),h('h2',null,'250 train · 201 validation'),h('p',null,'Checkpoint U-Net poids 10 · 32 réglage + 169 déjà observés')),h('div',null,h(Badge,{tone:'violet'},'COMPARAISON SCIENTIFIQUE V2'),h('h2',null,'318 train · 80 validation · 156 test réservé'),h('p',null,'53 événements purgés par embargo · test 2021 fermé · métadonnées du split scellé'))),
  ...[legacy,lineage,v2].filter(s=>s.error).map((state,i)=>h(SourceState,{key:i,state})),
  h('section',{className:'panel'},h('h2',null,'Deux protocoles, deux périmètres explicites'),h('p',null,'L’inférence interactive conserve le U-Net poids 10 historique. La comparaison V2 porte sur un autre checkpoint C8 et un panel de 80 événements commun à NEWPOS2, NEWPOS4 et FIRE13. Leurs scores ne sont pas mélangés.'),h('details',null,h('summary',null,'Méthodologie et statut scientifique'),h('p',null,'Les variantes restent expérimentales et non promues. FIRE13 est clos. Le panel 2020 a déjà été observé ; les résultats ne constituent pas un test aveugle.')),h('div',{className:'hero-actions'},h('button',{className:'button-secondary',onClick:()=>navigate('comparison')},'Ouvrir la comparaison V2'),h('button',{className:'button-quiet',onClick:()=>navigate('research')},'Lire les limites scientifiques'))));
}

export default function ScientificComparison(){
 const state=useResource('/science/v2/comparison'),tiles=useResource('/science/v2/diagnostic-tile');
 const [selected,setSelected]=useState('C8');const data=state.data,item=tiles.data?.cases.find(c=>c.model===selected),counts=item?tileCounts(item):null;
 return h('div',{className:'page-width jury-comparison'},h(PageHeader,{kicker:'PREUVES ARCHIVÉES · PROTOCOLE V2',title:'Comparaison officielle V2',description:'Validation 2020 : 80 événements communs, 30 882 tuiles, seuil principal figé à 0,78. Panel déjà observé ; aucune revendication de test aveugle.',action:h(Badge,null,'TEST 2021 FERMÉ')}),
  h(SourceState,{state}),data&&h(React.Fragment,null,
   h('section',{className:'panel'},h('h2',null,'C8, NEWPOS2, NEWPOS4 et FIRE13'),h('p',null,'Un même panel, un seuil commun : explorez les compromis mesurés entre détections et fausses alertes. C8 reste la référence V2 ; les variantes sont expérimentales.'),h('details',null,h('summary',null,'Interprétation scientifique des résultats'),h('p',null,data.conclusion)),
    h('div',{className:'lab-exports'},h('button',{onClick:()=>downloadJSON(data,'geofirex-validation2020-V2-80.json')},'Exporter JSON + provenance'),h('a',{className:'button-secondary',href:'/science/v2/export.csv',download:'geofirex-validation2020-V2-80.csv'},'Exporter CSV')),
    h(V2MetricCharts,{data}),
    h('details',null,h('summary',null,'Tableau détaillé des métriques officielles'),h('div',{className:'table-scroll'},h('table',null,h('caption',null,'Probabilités originales pour AP et Brier ; pourcentages pour les métriques bornées.'),h('thead',null,h('tr',null,h('th',{scope:'col'},'Modèle / statut'),...columns.map(([key,label])=>h('th',{scope:'col',key},label)))),h('tbody',null,...data.models.map(row=>h('tr',{key:row.model},h('th',{scope:'row'},row.model,h('br'),h(Badge,{tone:row.model==='C8'?'violet':''},row.model==='C8'?'Référence V2':'Expérimental · non promu')),...columns.map(([key,,percent])=>h('td',{key,className:'mono'},number(percent?row[key]*100:row[key],key==='FP_new'?0:percent?4:6)+(percent?' %':''))))))))),
    h('p',{className:'caption'},'AP : average precision exacte, et non aire trapézoïdale. Un rappel accru peut s’accompagner de nombreux faux positifs. Les sorties sigmoid ne sont pas des probabilités calibrées.'),
    h('details',null,h('summary',null,'Critères scientifiques figés et décisions'),h('pre',null,JSON.stringify({criteria:data.acceptance,decisions:data.models.filter(r=>r.scientific_gates).map(r=>({model:r.model,gates:r.scientific_gates,promotion_authorized:false}))},null,2))),
    h('details',null,h('summary',null,'Traçabilité des archives, checkpoints et domaine'),h('pre',null,JSON.stringify({sources:data.sources,domain:data.evaluation_domain_sha256,acceptance_sha256:data.acceptance_sha256},null,2)))),
   h('section',{className:'panel'},h('h2',null,'Ne pas confondre avec l’inférence interactive'),h('p',null,data.live_model.label+'. Checkpoint distinct de la référence C8 V2. Le Prediction Studio ne change pas de modèle après consultation de ce tableau.'))),
  h(SavedSpatialComparison),
  h(SourceState,{state:tiles}),item&&h('details',{className:'panel'},h('summary',null,'Diagnostic existant : une tuile commune aux quatre modèles'),h(React.Fragment,null,
   h('section',{className:'panel'},h('h2',null,'Même tuile, mêmes observations et cible'),h('p',null,'Diagnostic ponctuel : première clé de validation dans l’ordre lexicographique, choisie sans regarder les scores ni les erreurs. Cette tuile n’est pas représentative des 80 événements. Le calcul CPU ponctuel C8 est celui sauvegardé lors de la livraison précédente ; aucun calcul C8 supplémentaire n’est déclenché ici.'),
    h('label',{htmlFor:'v2-model'},'Résultat à afficher '),h('select',{id:'v2-model',value:selected,onChange:e=>setSelected(e.target.value)},...tiles.data.cases.map(c=>h('option',{key:c.model,value:c.model},c.model+(c.model==='C8'?' · référence V2':' · expérimental')))),
    h('p',null,`Origine de la tuile : ligne ${item.result.tile_origin[0]}, colonne ${item.result.tile_origin[1]}. ${item.result.provenance}. Grille de pixels, sans CRS.`),
    h('p',{className:'mono'},`Nouveaux feux sur cette tuile uniquement : TP ${counts.tp} · FP ${counts.fp} · FN ${counts.fn} · TN ${counts.tn}.`)),
   h(ForecastView,{result:item.result,truth:item.truth}))));
}
