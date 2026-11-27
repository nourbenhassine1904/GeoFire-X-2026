import React from 'react';
import {h,Icon,number} from './studio_ui.mjs';

export const metricDefinitions=[
 ['AP','Classe les scores sur tous les seuils. AP exacte avec gestion des ex æquo ; indépendante du seuil 0,78.'],
 ['Dice','Mesure le recouvrement des pixels prédits et observés au seuil 0,78.'],
 ['IoU','Intersection divisée par l’union des deux segmentations.'],
 ['Précision','Part des pixels annoncés en feu qui sont réellement observés en feu.'],
 ['Rappel','Part des pixels de feu observés retrouvée par le modèle.'],
 ['Brier','Erreur quadratique des scores : plus faible est préférable. Ne démontre pas une calibration.'],
 ['Faux positifs','Pixels annoncés en feu et absents de la cible, au seuil 0,78.']
];
export function MetricGuide(){return h('details',{className:'metric-guide'},h('summary',null,h(Icon,{name:'info',size:17}),'Comprendre les métriques'),
 h('dl',{className:'definition-grid'},...metricDefinitions.map(([name,definition])=>h('div',{key:name},h('dt',null,name),h('dd',null,definition)))),
 h('p',{className:'threshold-explainer'},'Une AP positive peut coexister avec un Dice nul : les scores peuvent classer correctement certains pixels sans qu’aucun dépasse 0,78. Changer le seuil modifierait Dice, précision et rappel, mais pas AP. Le seuil reste figé.'));}

export function ProcessFlow({compact=false}){return h('ol',{className:'process-flow'+(compact?' compact':''),'aria-label':'Des observations à la prédiction'},
 ...[['data','Observations','J−2 · J−1 · J'],['layers','Préparation','Normalisation · tuiles'],['code','U-Net C8','3 jours × 8 canaux'],['studio','Prédiction J+1','Scores · seuil 0,78']].map(([icon,title,text])=>h('li',{key:title},h(Icon,{name:icon,size:21}),h('strong',null,title),h('span',null,text))));}

export function MetricTiles({metrics,compact=false}){return h('div',{className:'metric-tiles'+(compact?' compact':''),role:'group','aria-label':'Indicateurs scientifiques'},
 ...[['average_precision','AP exacte'],['dice','Dice'],['recall','Rappel']].map(([key,label])=>h('div',{key},h('span',null,label),h('strong',null,number(metrics?.[key],3)),h('small',null,key==='average_precision'?'Tous les seuils':'Seuil 0,78'))));}

export function ModelStatus({id}){const reference=id==='C8';return h('span',{className:'model-status '+(reference?'reference':id==='legacy'?'historical':'experimental')},reference?'Référence C8 V2':id==='legacy'?'Ancien protocole':id==='NEWPOS4'?'Expérimental · haute sensibilité':'Expérimental');}
