import React,{useState} from 'react';
import {h,PageHeader,Badge,Icon,Fade,SourceState,useResource,downloadJSON} from './studio_ui.mjs';
import {ProcessFlow,MetricGuide} from './ScientificUI.mjs';

export const rawBands=[
 ['M11','Réflectance VIIRS'],['I2','Réflectance VIIRS'],['I1','Réflectance VIIRS'],
 ['NDVI_last','Végétation · dernière observation'],['EVI2_last','Végétation · dernière observation'],
 ['total precipitation','Précipitations observées'],['wind speed','Vitesse du vent observée'],
 ['wind direction','Direction du vent observée · défaut de quadrant documenté'],
 ['minimum temperature','Température minimale'],['maximum temperature','Température maximale'],
 ['energy release component','Indicateur de combustibilité'],['specific humidity','Humidité spécifique'],
 ['slope','Pente · utilisée par C8'],['aspect','Orientation · utilisée par C8'],['elevation','Altitude · utilisée par C8'],
 ['pdsi','Indice de sécheresse'],['LC_Type1','Occupation des sols · classe annuelle'],
 ['total_precipitation_surface_sum','Prévision GFS de précipitations'],['forecast wind speed','Prévision GFS de vitesse du vent'],
 ['forecast wind direction','Prévision GFS de direction du vent'],['forecast temperature','Prévision GFS de température'],
 ['forecast specific humidity','Prévision GFS d’humidité spécifique'],['active fire','Détection active VIIRS · utilisée par C8']
];
export const c8Channels=[
 ['slope_z','Pente','Centrage-réduction train-only'],['elevation_z','Altitude','Centrage-réduction train-only'],
 ['slope_missing','Pente manquante','Indicateur binaire'],['elevation_missing','Altitude manquante','Indicateur binaire'],
 ['aspect_sin','Orientation','Sinus de l’angle'],['aspect_cos','Orientation','Cosinus de l’angle'],
 ['aspect_missing','Orientation manquante','Indicateur binaire'],['fire_J','Feu actif du jour','Présence d’une détection non NaN']
];
function FlowCard({title,children,icon='layers'}){return h('section',{className:'method-card'},h(Icon,{name:icon,size:24}),h('h3',null,title),children);}
function BigDataEvidence(){const state=useResource('/science/big-data'),data=state.data;return h(React.Fragment,null,h(SourceState,{state}),data&&h(React.Fragment,null,
 h('p',null,'Exécution historique vérifiée : Spark '+data.master+' ; deux threads locaux, sans cluster distribué ni streaming. Cette vue lit les sorties enregistrées et ne lance aucun job.'),
 h('p',null,data.parquet_files.length+' agrégats Parquet ; durée historique '+data.total_seconds.toLocaleString('fr-FR',{maximumFractionDigits:2})+' s.'),
 h('div',{className:'table-scroll'},h('table',{className:'science-table'},h('caption',null,'Artefacts matérialisés et leurs empreintes'),h('thead',null,h('tr',null,...['Parquet','Lignes','SHA-256'].map(t=>h('th',{key:t,scope:'col'},t)))),h('tbody',null,...data.parquet_files.map(f=>h('tr',{key:f.path},h('th',{scope:'row'},f.path),h('td',null,f.rows),h('td',{className:'mono'},f.sha256)))))),h('pre',null,JSON.stringify({report_sha256:data.report_sha256,source_hashes:data.source_hashes},null,2))));}
export default function Methodology(){
 const provenance=useResource('/science/v2/provenance');
 const [showData,setShowData]=useState(false);
 return h('div',{className:'page-width methodology-page'},h(PageHeader,{kicker:'DONNÉES · TEMPS · REPRODUCTIBILITÉ',title:'Méthodologie et données',description:'Ce que le modèle observe, ce qu’il prédit et les contrôles qui rendent les résultats interprétables.',action:h(Badge,null,'WILDFIRESPREADTS · 2018–2021')}),
  h('nav',{className:'section-index','aria-label':'Sections de la méthodologie'},...[['data-contract','Données'],['temporal-contract','Temporalité'],['c8-contract','Canaux C8'],['unet-contract','Architecture'],['evaluation-contract','Évaluation'],['crisp-contract','CRISP-DM']].map(([id,label])=>h('a',{key:id,href:'#'+id,onClick:e=>{e.preventDefault();document.getElementById(id)?.scrollIntoView({behavior:'auto',block:'start'});}},label))),
  h(Fade,null,h('section',{className:'panel',id:'data-contract'},h('span',{className:'eyebrow'},'LE CONTRAT DE DONNÉES'),h('h2',null,'Prévoir une détection active à J+1'),h('p',null,'WildfireSpreadTS rassemble des séries quotidiennes de feux de 2018 à 2021. Chaque événement contient des fichiers GeoTIFF multibandes datés. La cible est la présence de feu actif le lendemain ; elle ne représente pas une surface brûlée cumulée.'),h(ProcessFlow),
   h('div',{className:'method-card-grid'},h(FlowCard,{title:'23 bandes d’origine',icon:'data'},h('p',null,'Les TIFF bruts combinent réflectance, végétation, météo observée, terrain, occupation des sols, prévisions et détection active. C8 en dérive huit canaux ; il n’utilise pas les 23 bandes.' )),
    h(FlowCard,{title:'Tuiles de 64 × 64',icon:'layers'},h('p',null,'Le compact référence chaque jour et chaque tuile, sans dupliquer les séquences temporelles. Les pixels manquants et le padding sont masqués.')),
    h(FlowCard,{title:'Géométrie explicite',icon:'studio'},h('p',null,'Les TIFF disposent de métadonnées spatiales ; les grilles compactes servies ici ne fournissent pas de CRS/affine exploitable. Les cartes et exports restent en coordonnées de pixels.'))),
   h('details',null,h('summary',null,'Les 23 bandes TIFF et leur rôle'),h('div',{className:'table-scroll'},h('table',{className:'science-table raw-band-table'},h('caption',null,'Ordre original, numérotation TIFF à partir de 1. Source : descriptions GDAL auditées et real_constants.py.'),h('thead',null,h('tr',null,h('th',{scope:'col'},'Bande'),h('th',{scope:'col'},'Nom d’origine'),h('th',{scope:'col'},'Rôle'))),h('tbody',null,...rawBands.map(([name,role],i)=>h('tr',{key:name,className:[12,13,14,22].includes(i)?'used-band':''},h('th',{scope:'row'},i+1),h('td',{className:'mono'},name),h('td',null,role))))))))),
  h('section',{className:'panel',id:'temporal-contract'},h('h2',null,'Cinq jours de validité, trois observations pour prédire'),
   h('div',{className:'temporal-schema',role:'img','aria-label':'J moins 4 et J moins 3 pour les masques ; J moins 2, J moins 1 et J pour les entrées ; J plus 1 séparé'},
    ...['J−4','J−3','J−2','J−1','J','J+1'].map((day,i)=>h('div',{key:day,className:i===5?'target-day':i>=2?'input-day':'mask-day'},h('strong',null,day),h('span',null,i===5?'Cible séparée':i>=2?'Entrée + masque':'Masque de validité')))),
   h('div',{className:'method-card-grid'},h(FlowCard,{title:'Disponible à J',icon:'clock'},h('p',null,'Les entrées sont uniquement J−2/J−1/J. L’intersection des cinq masques passés J−4 à J définit la validité de la prévision. Aucun pixel cible J+1 n’est lu pendant le calcul.')),
    h(FlowCard,{title:'Révélation indépendante',icon:'lock'},h('p',null,'Une requête distincte ouvre ensuite la cible et son masque. Les erreurs utilisent l’intersection de la validité passée et de J+1. Le cadrage des cartes reste fondé sur les observations et la prévision.')),
    h(FlowCard,{title:'Pas de fuite temporelle',icon:'shield'},h('p',null,'Les prévisions GFS sont exclues de C8 : les en-têtes ne suffisent pas à établir l’heure d’émission et l’heure de validité. Les agrégats météo et les bandes supplémentaires demandent une disponibilité à J documentée.'))),
   h('p',{className:'scientific-note'},'Les autres bandes observées ne sont pas toutes déclarées indisponibles : elles ne font pas partie du périmètre conservateur C8. La direction du vent observée est écartée en raison du défaut de quadrant documenté. Aucun vent ou champ météo futur n’est ajouté à l’inférence.')),
  h('section',{className:'panel',id:'c8-contract'},h('h2',null,'Huit canaux C8, dans leur ordre exact'),h('p',null,'Pente et altitude sont centrées et réduites avec le normaliseur ajusté uniquement sur des observations train 2018–2019, sous-ensemble du train V2. L’orientation devient sinus/cosinus et les valeurs manquantes ont leurs indicateurs. Le compact est déjà normalisé : le service ne le normalise pas une seconde fois.'),
   h('div',{className:'channel-grid'},...c8Channels.map(([name,source,transform],i)=>h('article',{key:name},h('span',{className:'channel-index'},'Canal '+i),h('h3',{className:'mono'},name),h('strong',null,source),h('p',null,transform)))),
   h('div',{className:'tensor-contract'},h('span',null,'3 jours'),h('b',null,'×'),h('span',null,'8 canaux'),h('b',null,'×'),h('span',null,'64 × 64 pixels'),h('b',null,'='),h('strong',null,'24 canaux empilés en entrée 2D')),
   h('p',{className:'caption'},'La bande brute active fire contient des codes de détection ; leur amplitude n’est pas une intensité de feu. La convention auditée est la présence d’une valeur non NaN, convertie en masque binaire.')),
  h('section',{className:'panel',id:'unet-contract'},h('div',{className:'section-heading'},h('h2',null,'U-Net : conserver le contexte spatial'),h(Badge,null,'123 977 PARAMÈTRES')),
   h('div',{className:'unet-schema',role:'img','aria-label':'U-Net profondeur 3 : encodeur 8, 16, 32 ; goulot 64 ; décodeur 32, 16, 8 avec connexions de saut'},
    h('div',{className:'unet-path'},h('span',null,'ENTRÉE'),h('strong',null,'24 × 64 × 64'),h('p',null,'Contexte temporel empilé')),
    h('div',{className:'unet-path encoder'},h('span',null,'ENCODEUR'),...['8 · 64×64','16 · 32×32','32 · 16×16'].map(t=>h('strong',{key:t},t))),
    h('div',{className:'unet-path bottleneck'},h('span',null,'GOULOT'),h('strong',null,'64 · 8×8'),h('p',null,'Contexte multirésolution')),
    h('div',{className:'unet-path decoder'},h('span',null,'DÉCODEUR + SAUTS'),...['32 · 16×16','16 · 32×32','8 · 64×64'].map(t=>h('strong',{key:t},t))),
    h('div',{className:'unet-path'},h('span',null,'SORTIE'),h('strong',null,'1 × 64 × 64'),h('p',null,'Logits → sigmoid → seuil'))),
   h('p',null,'StackedUNet / LightUNet, largeur de base 8 et profondeur 3. Les connexions de saut réinjectent les détails de l’encodeur dans le décodeur. Le modèle utilise une interpolation interne bilinéaire ; les cartes binaires affichées sont, elles, rendues pixel par pixel sans lissage.'),
   h('div',{className:'training-facts'},...['Époque fixe 4','28 976 mises à jour','Seed 42','Float32 CPU','Seuil 0,78'].map(t=>h(Badge,{key:t},t))),h('p',{className:'caption'},'Aucun entraînement n’est lancé par cette application. NEWPOS2 et NEWPOS4 partagent architecture et entrées C8, avec des pertes d’entraînement différentes ; elles restent expérimentales. FIRE13 est consulté via ses sorties archivées.'),
   h('h3',null,'Familles d’architectures et expériences entraînées'),
   h('div',{className:'table-scroll'},h('table',{className:'science-table architecture-status'},h('caption',null,'Implémentation et évaluation sont des statuts distincts'),h('thead',null,h('tr',null,...['Famille','Implémentation','Entraînement et évaluation attestés'].map(t=>h('th',{scope:'col',key:t},t)))),h('tbody',null,...[
    ['CNN','B1 ConvLogistic ; prototype SpatialCNN','CNN historique entraîné et évalué selon son ancien protocole. Aucune revendication V2 pour SpatialCNN.'],
    ['U-Net','StackedUNet / LightUNet','U-Net historique et expériences C8, NEWPOS2, NEWPOS4, FIRE13 entraînés et évalués. FIRE13 : 39 canaux empilés et 125 057 paramètres.'],
    ['ConvLSTM','B3, encodeur spatial et portes récurrentes','Architecture implémentée ; aucun résultat ConvLSTM certifié dans les comparaisons V2 présentées. Aucun score inventé.']
   ].map(([family,implementation,status])=>h('tr',{key:family},h('th',{scope:'row'},family),h('td',null,implementation),h('td',null,status))))))),
  h('section',{className:'panel',id:'evaluation-contract'},h('h2',null,'Des populations séparées pour une évaluation traçable'),
   h('div',{className:'split-diagram'},...[[318,'Train V2','Avant la coupure chronologique'],[80,'Validation 2020','Panel déjà observé'],[53,'Événements purgés','Embargo temporel'],[156,'Test indépendant 2021','C8 seulement · terminé']].map(([count,label,detail])=>h('div',{key:label},h('strong',null,count),h('span',null,label),h('small',null,detail)))),
   h('p',null,'La coupure V2 est le 23 août 2020. Les fenêtres temporelles et les événements sous embargo sont contrôlés avant lecture. Le normaliseur n’est pas ajusté sur la validation ni sur le test. Les variantes ne sont pas promues après la comparaison 2020.'),
   h('p',null,'Les nouveaux pixels de feu sont définis par « pixel valide et absence de feu observé à J ». Cette définition permet d’étudier la détection nouvelle, sans affirmer une propagation physique ou une ignition.'),h(MetricGuide)),
  h('section',{className:'panel',id:'crisp-contract'},h('h2',null,'Les six phases CRISP-DM'),h('ol',{className:'crisp-grid'},...[
   ['Compréhension métier','Anticiper une détection active quotidienne à J+1 et expliciter les limites d’usage.'],
   ['Compréhension des données','Auditer les TIFF, bandes, dates, masques et métadonnées du corpus.'],
   ['Préparation','Construire les splits chronologiques, le normaliseur train-only et les tuiles compactes.'],
   ['Modélisation','Comparer les modèles déjà entraînés selon leurs protocoles figés.'],
   ['Évaluation','Validation 2020 puis test indépendant C8 2021 terminé, avec résultats scellés.'],
   ['Déploiement','Application locale CPU, contrôles d’intégrité, exports et procédures de maintenance.']
  ].map(([title,text])=>h('li',{key:title},h('h3',null,title),h('p',null,text))))),
  h('details',{className:'panel big-data-evidence',onToggle:e=>setShowData(e.currentTarget.open)},h('summary',null,'Spark et Parquet : preuves d’exécution locale'),showData&&h(BigDataEvidence)),
  h('details',{className:'panel'},h('summary',null,'Reproductibilité, provenance et limites'),h(SourceState,{state:provenance}),
   h('button',{className:'button-secondary',disabled:!provenance.data,onClick:()=>downloadJSON(provenance.data,'geofirex-V2-provenance.json')},h(Icon,{name:'download'}),'Exporter la provenance'),
   h('div',{className:'method-card-grid'},h(FlowCard,{title:'Identités vérifiées',icon:'shield'},h('p',null,'Checkpoints, sources, archives, index et normaliseur sont reliés par SHA-256. La qualification CPU compare les inférences 2020 aux références archivées. Le test 2021 est consulté exclusivement depuis son archive de résultats.')),
    h(FlowCard,{title:'Limites actuelles',icon:'info'},h('p',null,'Une seule seed, validation déjà observée, rappel faible sur les nouveaux feux, scores sigmoid non calibrés et absence de géoréférencement dans le compact. La plateforme n’est pas un système opérationnel d’alerte.')),
    h(FlowCard,{title:'Perspectives à qualifier',icon:'research'},h('p',null,'Disponibilité temporelle des variables supplémentaires, calibration, robustesse hors domaine et géoréférencement des exports. Ce sont des pistes ; aucune nouvelle expérience ni amélioration mesurée n’est revendiquée.'))),
   h('p',{className:'caption'},'Sources de cette présentation : real_constants.py, feature_profiles.py, baseline_c8.json, protocol_v2.py, benchmark_models.py, unet.py et manifestes scientifiques scellés.'),h('pre',null,JSON.stringify(provenance.data,null,2))));
}
