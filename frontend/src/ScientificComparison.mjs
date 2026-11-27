import React,{useState} from 'react';
import {h,PageHeader,Badge,SourceState,useResource,number,downloadJSON,StatCard,Icon,WelcomeTitle,FireEmblem} from './studio_ui.mjs';
import {ForecastView} from './EventForecast.mjs';
import {V2MetricCharts,SavedSpatialComparison} from './V2Views.mjs';
import LiveComparison from './LiveComparison.mjs';

const gateNames={AP_new:'AP nouveaux feux',AP_pooled:'AP globale',Dice_pooled:'Dice global',recall_new:'rappel nouveaux feux',Dice_new:'Dice nouveaux feux',FP_new:'plafond de faux positifs',macro_Dice:'Dice moyen par événement',Brier:'Brier'};
function ExperimentGuide({data}){return h('section',{className:'experiment-guide'},h('h2',null,'Une architecture, quatre expériences V2'),
 h('dl',{className:'definition-grid'},...[
 ['C8 V2','U-Net à 8 canaux par jour, trois jours empilés. Perte : 0,5 × BCE pondérée (positifs ×10) + 0,5 × Dice. Référence préenregistrée, seule évaluée indépendamment sur 2021.'],
 ['NEWPOS2','Mêmes entrées, architecture, budget et seuil. Les termes BCE des pixels positifs absents du feu à J sont multipliés par 2 : poids effectif 20 contre 10 pour le feu persistant. Le terme Dice reste inchangé.'],
 ['NEWPOS4','Même changement ciblé, avec un facteur 4 : poids effectif 40 pour les nouveaux pixels positifs. Pénaliser davantage les feux manqués peut améliorer le rappel et augmenter les fausses alertes.'],
 ['FIRE13','Même famille U-Net, 13 canaux par jour : C8 + distance au feu plafonnée à 8 pixels, densité 7 × 7, apparition, changement et support valide. Moments dérivés figés sur train. Pas de météo future ajoutée.']
 ].map(([title,text])=>h('div',{key:title},h('dt',null,title),h('dd',null,text)))),
 h('h3',null,'Pourquoi conserver C8 ?'),h('p',null,'Une hausse isolée du rappel ne suffit pas. Les critères préenregistrés imposent aussi AP, Dice, faux positifs et Brier sur le même domaine. Aucun candidat ne satisfait leur ensemble ; aucune promotion n’est autorisée.'),
 h('ul',null,...data.models.filter(r=>r.scientific_gates).map(r=>h('li',{key:r.model},h('strong',null,r.model+' : '),'critères non satisfaits : '+Object.entries(r.scientific_gates).filter(([,ok])=>!ok).map(([key])=>gateNames[key]||key).join(', ')+'.'))),
 h('p',{className:'scientific-note'},'Les variantes ont été étudiées sur les 80 événements 2020 déjà observés. Elles n’ont pas été testées indépendamment sur 2021. Le test C8 2021 ne permet pas de leur attribuer ses scores.'));}

const columns=[['recall_new','Rappel nouveaux',true],['dice_new','Dice nouveaux',true],['precision_new','Précision nouveaux',true],['AP_new','AP nouveaux',true],['FP_new','FP nouveaux',false],['dice_global','Dice global',true],['AP_global','AP globale',true],['Brier','Brier',false]];
export function tileCounts(item){
 const {result,truth}=item;let tp=0,fp=0,fn=0,tn=0;
 truth.forEach((row,y)=>row.forEach((target,x)=>{if(target===null||result.observed_fire[y][x]!==0)return;const predicted=result.predicted_fire[y][x]===1;if(predicted){if(target===1)tp++;else fp++;}else if(target===1)fn++;else tn++;}));
 return {tp,fp,fn,tn,recall:tp+fn?tp/(tp+fn):null,precision:tp+fp?tp/(tp+fp):null,dice:2*tp+fp+fn?2*tp/(2*tp+fp+fn):null};
}

export {default as JuryOverview} from './Overview.mjs';

export default function ScientificComparison(){
 const state=useResource('/science/v2/comparison'),tiles=useResource('/science/v2/diagnostic-tile');
 const [selected,setSelected]=useState('C8');const data=state.data,item=tiles.data?.cases.find(c=>c.model===selected),counts=item?tileCounts(item):null;
 return h('div',{className:'page-width comparison-page'},h(PageHeader,{kicker:'PREUVES ARCHIVÉES · PROTOCOLE V2',title:'Comparaison des modèles',description:'Validation 2020 : 80 événements communs, 30 882 tuiles, seuil principal figé à 0,78. Panel déjà observé ; aucune revendication de test aveugle.',action:h(Badge,null,'VALIDATION 2020 · 80 ÉVÉNEMENTS')}),
  h(SourceState,{state}),data&&h(React.Fragment,null,
   h('section',{className:'panel'},h('h2',null,'C8, NEWPOS2, NEWPOS4 et FIRE13'),h('p',null,'Un même panel, un seuil commun : explorez les compromis mesurés entre détections et fausses alertes. C8 reste la référence V2 ; les variantes sont expérimentales.'),h('details',null,h('summary',null,'Interprétation scientifique des résultats'),h('p',null,data.conclusion)),
    h('div',{className:'lab-exports'},h('button',{onClick:()=>downloadJSON(data,'geofirex-validation2020-V2-80.json')},'Exporter JSON + provenance'),h('a',{className:'button-secondary',href:'/science/v2/export.csv',download:'geofirex-validation2020-V2-80.csv'},'Exporter CSV')),
    h(ExperimentGuide,{data}),h(V2MetricCharts,{data}),
    h('details',{open:true},h('summary',null,'Tableau des métriques officielles'),h('div',{className:'table-scroll'},h('table',{className:'science-table'},h('caption',null,'Probabilités originales pour AP et Brier ; pourcentages pour les métriques bornées.'),h('thead',null,h('tr',null,h('th',{scope:'col'},'Modèle / statut'),...columns.map(([key,label])=>h('th',{scope:'col',key},label)))),h('tbody',null,...data.models.map(row=>h('tr',{key:row.model},h('th',{scope:'row'},row.model,h('br'),h(Badge,{tone:row.model==='C8'?'violet':''},row.model==='C8'?'Référence V2':'Expérimental · non promu')),...columns.map(([key,,percent])=>h('td',{key,className:'mono'},number(percent?row[key]*100:row[key],key==='FP_new'?0:percent?4:6)+(percent?' %':''))))))))),
    h('p',{className:'caption'},'AP : average precision exacte, et non aire trapézoïdale. Un rappel accru peut s’accompagner de nombreux faux positifs. Les sorties sigmoid ne sont pas des probabilités calibrées.'),
    h('details',null,h('summary',null,'Critères scientifiques figés et décisions'),h('pre',null,JSON.stringify({criteria:data.acceptance,decisions:data.models.filter(r=>r.scientific_gates).map(r=>({model:r.model,gates:r.scientific_gates,promotion_authorized:false}))},null,2))),
    h('details',null,h('summary',null,'Traçabilité des archives, checkpoints et domaine'),h('pre',null,JSON.stringify({sources:data.sources,domain:data.evaluation_domain_sha256,acceptance_sha256:data.acceptance_sha256},null,2)))),
   h('p',{className:'scientific-note'},'Les variantes privilégient différents compromis entre rappel et fausses alertes. Aucune ne satisfait tous les critères de promotion. Les résultats du test indépendant C8 2021 figurent exclusivement dans Performances.')),
  h(LiveComparison),h(SavedSpatialComparison),
  h(SourceState,{state:tiles}),item&&h('details',{className:'panel'},h('summary',null,'Diagnostic existant : une tuile commune aux quatre modèles'),h(React.Fragment,null,
   h('section',{className:'panel'},h('h2',null,'Même tuile, mêmes observations et cible'),h('p',null,'Diagnostic ponctuel : première clé de validation dans l’ordre lexicographique, choisie sans regarder les scores ni les erreurs. Cette tuile n’est pas représentative des 80 événements. Le calcul CPU ponctuel C8 est celui sauvegardé lors de la livraison précédente ; aucun calcul C8 supplémentaire n’est déclenché ici.'),
    h('label',{htmlFor:'v2-model'},'Résultat à afficher '),h('select',{id:'v2-model',value:selected,onChange:e=>setSelected(e.target.value)},...tiles.data.cases.map(c=>h('option',{key:c.model,value:c.model},c.model+(c.model==='C8'?' · référence V2':' · expérimental')))),
    h('p',null,`Origine de la tuile : ligne ${item.result.tile_origin[0]}, colonne ${item.result.tile_origin[1]}. ${item.result.provenance}. Grille de pixels, sans CRS.`),
    h('p',{className:'mono'},`Nouveaux feux sur cette tuile uniquement : TP ${counts.tp} · FP ${counts.fp} · FN ${counts.fn} · TN ${counts.tn}.`)),
   h(ForecastView,{result:item.result,truth:item.truth}))));
}
