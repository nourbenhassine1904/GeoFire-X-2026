import React,{useState} from 'react';
import {h,Icon,Fade,WelcomeTitle,AuroraBackdrop,StatCard,PageHeader,Badge,useResource,SourceState,number,downloadJSON} from './studio_ui.mjs';
import {Results,BigData} from './ScienceResults.mjs';
import UnetAblation from './UnetAblation.mjs';
import Propagation from './Propagation.mjs';

export function Overview({navigate}){
 const science=useResource('/science/summary'),unet=useResource('/science/unet-ablation'),lineage=useResource('/lab/lineage'),big=useResource('/science/big-data');
 const s=science.data,u=unet.data,l=lineage.data,b=big.data;
 const candidate=u?.confirmation?.comparison?.find(r=>r.model==='B2_w10'),inventory=b?.tables.annual.reduce((a,r)=>a+r.events,0);
 const map=u?.confirmation?.maps?.find(m=>m.event_id==='2020/fire_24332880')||u?.confirmation?.maps?.[0];
 const epoch=s?.provenance.checkpoints.B2.completed_epochs,updates=s?.provenance.checkpoints.B2.updates;
 return h('div',{className:'overview page-width'},
  h('section',{className:'overview-hero'},h(AuroraBackdrop),h('div',{className:'hero-copy'},h(Badge,{tone:'violet'},h('span',{className:'badge-dot'}),'GEOFIRE-X · RESEARCH STUDIO'),h(WelcomeTitle),
   h('p',{className:'hero-description'},'Des données satellitaires aux prédictions de détection active du feu. Un espace pour explorer, comparer et comprendre les résultats — avec leurs limites.'),
   h('div',{className:'hero-actions'},h('button',{className:'button-primary',onClick:()=>navigate('studio')},'Ouvrir Prediction Studio',h(Icon,{name:'arrow'})),h('button',{className:'button-quiet',onClick:()=>navigate('analytics')},'Explorer les résultats',h(Icon,{name:'chevron',size:15}))),
   h('div',{className:'hero-trust'},h(Icon,{name:'shield',size:15}),'U-Net poids 10 · seuil figé 0,78 · test 2021 verrouillé')),
   h('div',{className:'hero-raster'},h('div',{className:'raster-toolbar'},h('span',null,h('span',{className:'status-dot amber'}),'PRÉDICTION ENREGISTRÉE'),h('span',{className:'mono'},'J+1')),
    h('div',{className:'raster-preview'},map?h('img',{src:'/science/unet-assets/'+map.directory+'/'+map.layers.segmentation,alt:`Segmentation U-Net enregistrée, ${map.event_id}, cible ${map.target_date}` }):h('div',{className:'raster-loading',role:'status'},'Lecture de la carte scientifique…'),h('span',{className:'raster-corner top-left'}),h('span',{className:'raster-corner bottom-right'})),
    h('div',{className:'raster-caption'},h('div',null,h('strong',null,map?.event_id||'Validation historique 2020'),h('span',null,map?`Cible ${map.target_date} · ${map.shape.join(' × ')} pixels`:'Résultat sauvegardé, sans nouvelle inférence')),h(Badge,null,'SANS CRS')))),
  h('div',{className:'section-heading'},h('h2',null,'La recherche, en chiffres'),h('span',null,'Métadonnées vérifiées · aucun calcul relancé')),
  h('div',{className:'overview-stats'},h(StatCard,{icon:'layers',label:'Événements d’entraînement',value:l?.roles.train.events,detail:'Corpus train · 2018–2019'}),h(StatCard,{icon:'studio',label:'Événements de validation',value:l?.roles.val.events,detail:'2020 · 32 réglage + 169 observés'}),h(StatCard,{icon:'data',label:'Événements inventoriés',value:inventory,detail:'607 officiels · métadonnées seulement en 2021'}),h(StatCard,{icon:'analytics',label:'Mises à jour par modèle',value:updates,detail:epoch?`${epoch} époques réalisées · seed 42`:'Lecture du protocole enregistré'})),
  ...[science,unet,lineage,big].filter(v=>v.error).map((state,i)=>h(SourceState,{key:i,state})),
  h(Fade,null,h('div',{className:'overview-grid'},
   h('section',{className:'feature-panel model-snapshot'},h('div',{className:'section-heading'},h('h2',null,'Le modèle retenu'),h(Badge,{tone:'violet'},'U-NET · POIDS 10')),
    h('p',null,'Sélection sur 32 événements, puis confirmation sur les 169 événements désormais observés.'),h('div',{className:'snapshot-metrics'},h('div',null,h('span',null,'Dice · validation 169'),h('strong',null,number(candidate?.dice,4))),h('div',null,h('span',null,'Average precision'),h('strong',null,number(candidate?.average_precision,4)))),
    h('div',{className:'model-comparison-mini'},...(u?.confirmation?.comparison||[]).map(r=>h('div',{key:r.model},h('span',null,r.model==='B2_w10'?'U-Net · poids 10':r.model==='B1'?'CNN · historique':'U-Net · historique'),h('div',{className:'mini-track'},h('span',{style:{width:`${r.dice*100}%`},className:r.model==='B2_w10'?'selected':''})),h('span',{className:'mono'},number(r.dice,3))))),
    h('p',{className:'caption'},'Dice micro, axe 0–1 · seuils propres aux modèles. Le candidat ne domine pas tous les critères.'),h('button',{className:'text-link',onClick:()=>navigate('analytics')},'Examiner les métriques et ablations',h(Icon,{name:'arrow',size:16}))),
   h('section',{className:'feature-panel data-snapshot'},h('div',{className:'section-heading'},h('h2',null,'Des données aux preuves'),h(Icon,{name:'data'})),
    h('div',{className:'pipeline-compact'},...['Inventaire','Spark SQL','Parquet','Prediction Lab'].map((v,i)=>h('div',{key:v},h('span',{className:'pipeline-number'},String(i+1).padStart(2,'0')),h('strong',null,v),i<3&&h(Icon,{name:'chevron',size:14})))),
    h('div',{className:'data-facts'},h('div',null,h('strong',null,number(l?.partitions)),h('span',null,'partitions compactes')),h('div',null,h('strong',null,l?number(l.source_bytes/1e9,2)+' Go':'—'),h('span',null,'sources compactes')),h('div',null,h('strong',null,number(b?.parquet_files.length)),h('span',null,'agrégats Parquet'))),
    h('p',{className:'caption'},'Spark exécuté en local[2]. Les vues lisent des sorties enregistrées, sans démarrer de job.'),h('button',{className:'text-link',onClick:()=>navigate('data')},'Entrer dans l’observatoire',h(Icon,{name:'arrow',size:16}))))),
  h('section',{className:'research-banner'},h('div',{className:'research-banner-icon'},h(Icon,{name:'research',size:24})),h('div',null,h('h2',null,'La clarté scientifique fait partie du résultat.'),h('p',null,'Grilles de pixels, détection active quotidienne, contexte T3. Ni hectares inventés, ni météo absente, ni test 2021 ouvert.')),h('button',{className:'button-secondary',onClick:()=>navigate('research')},'Lire la méthodologie',h(Icon,{name:'arrow',size:16})))
 );
}

const analysisTabs=[['unet','Modèle principal'],['historical','CNN / U-Net'],['propagation','Propagation'],['architectures','Architectures']];
export function ModelAnalytics(){
 const [tab,setTab]=useState('unet');
 function onKey(e){const i=analysisTabs.findIndex(t=>t[0]===tab);let next;if(e.key==='ArrowRight')next=(i+1)%analysisTabs.length;if(e.key==='ArrowLeft')next=(i-1+analysisTabs.length)%analysisTabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=analysisTabs.length-1;if(next!==undefined){e.preventDefault();setTab(analysisTabs[next][0]);document.getElementById('analytics-tab-'+analysisTabs[next][0])?.focus();}}
 return h('div',{className:'analytics-page page-width'},h(PageHeader,{kicker:'COMPARER · MESURER · INTERPRÉTER',title:'Model Analytics',description:'Des résultats vérifiés, leurs compromis et les décisions expérimentales qui les accompagnent.',action:h(Badge,{tone:'violet'},'PRINCIPAL · U-NET POIDS 10')}),
  h('div',{className:'studio-tabs',role:'tablist','aria-label':'Analyses des modèles',onKeyDown:onKey},...analysisTabs.map(([key,title])=>h('button',{key,id:'analytics-tab-'+key,role:'tab','aria-selected':tab===key,'aria-controls':'analytics-panel',tabIndex:tab===key?0:-1,onClick:()=>setTab(key)},title))),
  h('div',{role:'tabpanel',id:'analytics-panel','aria-labelledby':'analytics-tab-'+tab,tabIndex:0},h(Fade,{key:tab,trigger:'mount'},tab==='unet'?h(UnetAblation):tab==='historical'?h(Results):tab==='propagation'?h(Propagation):h(Architectures))));
}
export function Architectures(){return h(Fade,null,h('div',{className:'architecture-grid'},
 ...[['B1 · CNN','Référence convolutive','Convolutions 2D','Contexte T3 empilé','Tête de segmentation J+1'],['B2 · U-Net','Modèle principal · poids 10','Encodeur multirésolution','Connexions de saut','Décodeur + logits J+1']].map(([name,detail,...steps])=>h('section',{className:'feature-panel',key:name},h(Badge,{tone:name.startsWith('B2')?'violet':''},detail),h('h2',null,name),h('div',{className:'architecture-flow'},...steps.map((s,i)=>h(React.Fragment,{key:s},h('div',null,h(Icon,{name:i===1?'layers':'code'}),s),i<steps.length-1&&h('span',{'aria-hidden':true},'↓')))),h('p',null,'Entrées compactes : 8 canaux. Les dates du contexte et les masques sont contrôlés ; la cible J+1 reste séparée de l’inférence.')))),
 h('section',{className:'panel'},h('h2',null,'Des architectures comparables, des rôles différents'),h('p',null,'Les schémas ci-dessus résument le flux fonctionnel, sans inventer de nombres de paramètres. Le code des architectures reste dans src/geofirex/benchmark_models.py et unet.py. La tête spécialisée de propagation constitue une expérience comparative non retenue.'),h('p',null,'Le seuil décide de la segmentation. Il ne change ni les sorties sigmoid, ni l’AP, ni le Brier. Une sortie sigmoid sous loss pondérée ne constitue pas une probabilité d’incendie calibrée.')));}

export function DataObservatory(){return h('div',{className:'page-width data-page'},h(PageHeader,{kicker:'INVENTORIER · TRANSFORMER · TRACER',title:'Big Data Observatory',description:'Du corpus officiel aux tuiles du modèle : explorer les volumes, les requêtes et les preuves matérialisées.',action:h(Badge,null,'SPARK SQL → PARQUET')}),h(BigData));}

export function Research(){
 const science=useResource('/science/summary'),lineage=useResource('/lab/lineage'),unet=useResource('/science/unet-ablation');
 return h('div',{className:'page-width research-page'},h(PageHeader,{kicker:'PROTOCOLES · PROVENANCE · REPRODUCTIBILITÉ',title:'Research',description:'Comprendre ce qui est mesuré, ce qui est prédit et ce que les résultats permettent réellement de conclure.',action:h(Badge,null,h(Icon,{name:'lock',size:13}),'TEST 2021 VERROUILLÉ')}),
 h(Fade,null,h('section',{className:'feature-panel research-principle'},h('span',{className:'eyebrow'},'LE CONTRAT SCIENTIFIQUE'),h('h2',null,'Prédire une détection future. Préserver la séparation temporelle.'),h('p',null,'Le modèle observe J−2, J−1 et J pour prédire la détection active à J+1. Les cinq jours disponibles à J définissent la validité des entrées. La vérité terrain J+1 n’est lue que dans une requête distincte, après la prédiction.'),h('div',{className:'temporal-contract'},...['J−4','J−3','J−2','J−1','J','J+1'].map((v,i)=>h('div',{key:v,className:i===5?'future':i>=2?'model-input':''},h('strong',null,v),h('span',null,i===5?'Cible séparée':i>=2?'Entrée du modèle':'Masque de validité')))))),
 h('div',{className:'research-grid'},...[
  ['01','Données et séparation','607 événements inventoriés ; 250 événements d’entraînement (2018–2019) et 201 de validation (2020) dans le compact. L’inventaire 2021 est constitué de métadonnées : aucun pixel de test n’est consulté.'],
  ['02','Sélection expérimentale','Les 32 événements de réglage choisissent le poids et le seuil. La confirmation porte sur 169 événements désormais observés : ce panel ne constitue plus un test aveugle pour des décisions ultérieures.'],
  ['03','Métriques et déséquilibre','Dice et IoU évaluent les segmentations. Précision, rappel et faux positifs explicitent leurs compromis. AP et Brier portent sur les scores originaux et restent indépendants du seuil.'],
  ['04','Reproductibilité','Même seed 42 et même budget pour les ablations : quatre époques, 23 404 mises à jour par entraînement. Checkpoints, archives, normaliseur train-only et index sont liés par SHA-256.']
 ].map(([id,title,text])=>h(Fade,{key:id,className:'feature-panel'},h('section',null,h('span',{className:'research-number mono'},id),h('h2',null,title),h('p',null,text))))),
 h('section',{className:'panel limitations'},h('h2',null,'Lire les résultats à leur juste échelle'),h('ul',null,
  h('li',null,'Détection active quotidienne ≠ surface brûlée cumulée. Un nouveau pixel ne prouve pas une ignition ou une propagation physique.'),
  h('li',null,'Grille raster compacte sans CRS ni affine disponibles : aucune carte géographique ou surface en hectares n’est déduite.'),
  h('li',null,'Les huit canaux compacts ne contiennent ni vent, ni humidité, ni météo prévue.'),
  h('li',null,'Une seule seed et des écarts d’ablation faibles : aucune supériorité statistique n’est affirmée.'),
  h('li',null,'La tête spécialisée de propagation n’a pas satisfait les critères d’acceptation. U-Net poids 10, seuil 0,78, reste le modèle principal.'),
  h('li',null,'Cette plateforme présente des expériences de recherche historiques ; elle ne constitue pas un système opérationnel d’alerte.'))),
 h('section',{className:'panel'},h('div',{className:'section-heading'},h('h2',null,'Retrouver et exporter les preuves'),h('button',{className:'button-secondary',disabled:!science.data||!lineage.data||!unet.data,onClick:()=>downloadJSON({scientific_provenance:science.data.provenance,compact:lineage.data,selection:unet.data.selected,ablation_provenance:unet.data.provenance},'geofirex-research-provenance.json')},h(Icon,{name:'download',size:15}),'Exporter la provenance')),
  h('p',null,'Les données ci-dessous sont lues depuis les API scientifiques existantes. L’export ne contient aucune nouvelle évaluation.'),
  ...[science,lineage,unet].map((state,i)=>h(SourceState,{key:i,state})),
  h('details',null,h('summary',null,'Identités du modèle principal et des données'),h('pre',null,JSON.stringify({selection:unet.data?.selected,compact:lineage.data,archives:science.data?.provenance},null,2))),
  h('div',{className:'source-list'},...['reports/unet_confirmation_20261112_v1/RESULTATS_CONFIRMES.md','reports/calibration_20261112_v1/DIAGNOSTIC.md','reports/propagation_20261112_v1/EXPERIENCE_ET_COMMANDES.md','docs/GITHUB_PRESERVATION_20261113.md'].map(path=>h('a',{key:path,href:'https://github.com/nourbenhassine1904/GeoFire-X-2026/blob/main/'+path,target:'_blank',rel:'noopener noreferrer'},h(Icon,{name:'research',size:16}),path,h(Icon,{name:'arrow',size:14}))))));
}
