import React, {useEffect, useState} from 'react';
import {Fade} from './studio_ui.mjs';
const h = React.createElement;
const names = {B1: 'CNN · B1', B2: 'U-Net · B2'};
const colors = {B1: '#B45309', B2: '#0F766E'};
const number = (v, digits = 6) => v === null || v === undefined ? '—' : Number(v).toLocaleString('fr-FR', {maximumFractionDigits: digits});
const pct = v => v === null ? '—' : `${number(v * 100, 2)} %`;
const asset = name => `/science/assets/${name.split('/').map(encodeURIComponent).join('/')}`;
function useData(url) {
  const [state, set] = useState({data: null, error: ''});
  useEffect(() => {
    const controller = new AbortController(); set({data: null, error: ''});
    fetch(url, {signal: controller.signal}).then(async r => {
      if (!r.ok) throw new Error(`Résultat indisponible (${r.status})`);
      set({data: await r.json(), error: ''});
    }).catch(e => {if (e.name !== 'AbortError') set({data: null, error: e.message});});
    return () => controller.abort();
  }, [url]);
  return state;
}
function State({data, error}) {return error ? h('p', {role: 'alert'}, error) : !data ? h('p', {role: 'status'}, 'Lecture des résultats enregistrés…') : null;}
function Table({columns, rows}) {
  return h('div', {className: 'table-scroll'}, h('table', null,
    h('thead', null, h('tr', null, ...columns.map(([key, label]) => h('th', {key, scope: 'col'}, label)))),
    h('tbody', null, ...rows.map((row, i) => h('tr', {key: row.id || i}, ...columns.map(([key, label, format]) =>
      h('td', {key}, format ? format(row[key], row) : String(row[key] ?? '—'))))))));
}
function Provenance({source}) {
  return h('details', {className: 'provenance'}, h('summary', null, 'Provenance et empreintes vérifiées'),
    h('p', null, source.archive), h('dl', null,
      h('dt', null, 'ZIP SHA256'), h('dd', null, source.archive_sha256),
      h('dt', null, 'Verrou des seuils SHA256'), h('dd', null, source.threshold_lock_sha256),
      ...Object.entries(source.checkpoints).flatMap(([m, c]) => [h('dt', {key:m+'k'}, names[m]), h('dd', {key:m+'v'}, `${c.sha256} · ${c.updates} updates`)])));
}
function Distribution({data, metric}) {
  const width=720, height=230, left=48, right=22, top=20, bottom=40;
  const all=Object.values(data.models).flat().map(r=>r[metric]).filter(v=>v!==null);
  const max=metric==='brier_score' ? Math.max(...all, .001) : 1;
  const x=v=>left+v/max*(width-left-right), y=v=>height-bottom-v*(height-top-bottom);
  return h('svg', {viewBox:`0 0 ${width} ${height}`, role:'img', 'aria-label':`Distribution cumulative ${metric} par événement`, className:'distribution'},
    ...[0,.25,.5,.75,1].flatMap(v=>[
      h('line',{key:'l'+v,x1:left,x2:width-right,y1:y(v),y2:y(v),stroke:'var(--line)'}),
      h('text',{key:'t'+v,x:5,y:y(v)+4,fill:'var(--text-secondary)',fontSize:12},`${v*100}%`),
      h('text',{key:'x'+v,x:x(v*max),y:height-10,fill:'var(--text-secondary)',fontSize:12,textAnchor:'middle'},number(v*max,3))]),
    ...['B1','B2'].flatMap(model=>[.5,.95].map(t=>{
      const values=data.models[model].filter(r=>r.threshold===t && r[metric]!==null).map(r=>r[metric]).sort((a,b)=>a-b);
      const points=values.map((v,i)=>`${x(v)},${y((i+1)/values.length)}`).join(' ');
      return h('polyline',{key:model+t,points,fill:'none',stroke:colors[model],strokeWidth:2.5,strokeDasharray:t===.5?'5 5':undefined});
    })));
}
function EventPanel({phase}) {
  const state=useData(`/science/events?phase=${phase}`);
  const [metric,setMetric]=useState('dice'), [sort,setSort]=useState('event_id');
  if (!state.data) return h(State,state);
  const data=state.data;
  const ids=[...new Set(data.models.B1.map(r=>r.event_id))];
  const rows=ids.map(event_id=>{
    const row={event_id};
    for (const m of ['B1','B2']) {
      const base=data.models[m].find(r=>r.event_id===event_id && r.threshold===.5);
      const selected=data.models[m].find(r=>r.event_id===event_id && r.threshold===.95);
      row[m+'base']=base[metric]; row[m+'selected']=selected[metric];
      row[m+'gain']=selected[metric]===null || base[metric]===null ? null : selected[metric]-base[metric];
    }
    return row;
  }).sort((a,b)=>sort==='event_id'?a.event_id.localeCompare(b.event_id):(b[sort]??-Infinity)-(a[sort]??-Infinity));
  return h('section', {className:'panel'},h('h2',null,'Chaque événement compte'),
    h('p',null,`${ids.length} événements complets. Les courbes donnent le même poids à chaque événement ; le tableau principal agrège les pixels.`),
    h('label',null,'Métrique ',h('select',{value:metric,onChange:e=>setMetric(e.target.value)},
      ...['dice','iou','precision','recall','average_precision','brier_score'].map(k=>h('option',{key:k,value:k},k)))),
    h(Distribution,{data,metric}),
    h('p',{className:'legend'},'Orange : CNN · Turquoise : U-Net · Pointillés : 0,50 · Traits pleins : 0,95. Valeurs indéfinies exclues de la CDF.'),
    h('label',null,'Ordre du tableau ',h('select',{value:sort,onChange:e=>setSort(e.target.value)},
      h('option',{value:'event_id'},'Identifiant'),h('option',{value:'B1gain'},'Écart CNN décroissant'),h('option',{value:'B2gain'},'Écart U-Net décroissant'))),
    h('div',{className:'event-table'},h(Table,{rows,columns:[['event_id','Événement'],['B1base','CNN · 0,50',v=>number(v)],['B1selected','CNN · 0,95',v=>number(v)],['B1gain','Δ CNN',v=>number(v)],['B2base','U-Net · 0,50',v=>number(v)],['B2selected','U-Net · 0,95',v=>number(v)],['B2gain','Δ U-Net',v=>number(v)]]})),
    h('details',null,h('summary',null,'Probabilités, fiabilité et erreurs par événement'),
      h('p',null,'AP et Brier sont indépendants du seuil. Les histogrammes sont normalisés séparément pour chaque classe ; le non-feu domine en nombre de pixels.'),
      h('img',{className:'diagnostic-image',src:asset(`${phase}_diagnostics.png`),alt:`Histogrammes feu/non-feu, fiabilité, distributions et faux positifs : ${phase}`})));
}
function MapPanel({phase}) {
  const state=useData(`/science/maps?phase=${phase}`);
  const [chosen,setChosen]=useState('');
  if (!state.data) return h(State,state);
  const cases=state.data, events=[...new Set(cases.map(c=>c.event_id+'|'+c.target_date))];
  const key=events.includes(chosen)?chosen:events[0];
  const selected=cases.filter(c=>c.event_id+'|'+c.target_date===key);
  return h('section',{className:'panel'},h('h2',null,'Cartes réelles · J+1'),
    h('label',null,'Cas illustré ',h('select',{value:key,onChange:e=>setChosen(e.target.value)},...events.map(e=>h('option',{key:e,value:e},e.replace('|',' · '))))),
    h('p',null,'Sélection par métadonnées : événements régulièrement espacés, date cible médiane. Cas illustratifs, sans sélection sur la performance. Pixels invalides transparents.'),
    ...selected.map(c=>h('div',{key:c.model,className:'model-maps'},h('h3',null,`${names[c.model]} · seuil ${number(c.threshold,2)} · ${c.shape.join(' × ')} pixels`),
      h('div',{className:'map-grid'},...[['target','Vérité terrain J+1'],['probability','Probabilité'],['segmentation','Segmentation à 0,95'],['errors','Erreurs']].map(([layer,title])=>
        h('figure',{key:layer},h('figcaption',null,title),h('img',{src:asset(c.directory+'/'+c.layers[layer]),alt:`${title} ${c.model} ${c.event_id}`,loading:'lazy'})))),
      h('p',{className:'legend'},'Erreurs : TP vert · FP rouge · FN bleu · TN sombre. Grille de pixels, sans géoréférencement.'),
      h('a',{href:asset(c.directory+'/'+c.npz),download:true},`Télécharger les tableaux ${c.model} (NPZ)`))));
}
export function Results() {
  const state=useData('/science/summary');
  const [phase,setPhase]=useState('holdout169');
  if (!state.data) return h('section',{className:'science'},h(State,state));
  const data=state.data, rows=data.comparison.filter(r=>r.phase===phase);
  const cards=['B1','B2'].map(model=>{
    const b=rows.find(r=>r.model===model && r.mode==='baseline_05'), c=rows.find(r=>r.model===model && r.mode==='calibrated');
    return h('article',{className:'metric-card',key:model},h('span',{className:'eyebrow'},names[model]),
      h('div',{className:'score'},number(c.dice)),h('p',null,`Dice à 0,95 · ${number(b.dice)} à 0,50`),
      h('strong',null,`+${number(c.dice-b.dice)} Dice · ${pct(1-c.fp/b.fp)} de FP en moins`),
      h('p',null,`Rappel ${number(b.recall,3)} → ${number(c.recall,3)} · AP ${number(c.average_precision)}`));
  });
  return h('section',{className:'science'},
    h('header',{className:'science-header'},h('span',{className:'eyebrow'},'Résultats Kaggle vérifiés · 12 novembre 2026'),
      h('h2',null,'CNN & U-Net, au seuil mesuré'),h('p',null,'250 événements train · 4 époques · 23 404 updates par modèle · T=3 → J+1 · seed 42')),
    h('div',{className:'phase-switch',role:'group','aria-label':'Groupe de validation'},
      ...[['calibration32','32 événements · réglage'],['holdout169','169 événements · confirmation']].map(([k,label])=>h('button',{key:k,'aria-pressed':phase===k,onClick:()=>setPhase(k)},label))),
    h('p',null,phase==='holdout169'?'Seuils fixés sur les 32 événements, puis vérifiés ici sans ajustement. Ce panel désormais observé ne sert pas à multiplier les essais.':'Ces 32 événements servent au réglage et ont déjà été utilisés pendant l’entraînement. Les scores de sélection ne sont pas un test indépendant.'),
    h('div',{className:'metric-grid'},...cards),
    h('aside',{className:'boundary-note'},h('strong',null,'0,95 : borne supérieure atteinte'),h('p',null,'Meilleur seuil parmi ceux testés de 0,10 à 0,95. Un optimum au-delà reste possible, sans être démontré. Le seuil est conservé pour ces résultats ; aucune recherche sur les 169 événements.')),
    h('section',{className:'panel'},h('h2',null,'Comparaison sur le même domaine'),
      h(Table,{rows,columns:[['model','Modèle',v=>names[v]],['threshold','Seuil'],['dice','Dice',v=>number(v)],['iou','IoU',v=>number(v)],['precision','Précision',v=>number(v)],['recall','Rappel',v=>number(v)],['average_precision','AP',v=>number(v)],['brier_score','Brier ↓',v=>number(v)],['tp','TP',v=>number(v,0)],['fp','FP',v=>number(v,0)],['fn','FN',v=>number(v,0)]]}),
      h('p',null,`${number(rows[0].valid_pixels,0)} pixels valides · ${number(rows[0].tiles,0)} tuiles par modèle. AP et Brier ne changent pas avec le seuil. Le meilleur Dice sur les 32 revient au CNN ; la meilleure AP revient au U-Net.`)),
    h(EventPanel,{key:'events'+phase,phase}),h(MapPanel,{key:'maps'+phase,phase}),
    h('section',{className:'panel'},h('h2',null,'Ce que la loss explique — et ce qu’elle ne prouve pas'),
      h('p',null,'Loss actuelle : 0,5 × BCEWithLogits(pos_weight=100) + 0,5 × Dice par image. La prévalence pixel train mesurée est '+pct(data.train.prevalence)+'. Le poids 100 favorise le rappel, mais ce résultat seul ne prouve pas qu’il soit excessif.'),
      h('p',null,'Les scores sigmoid sont produits sous loss pondérée : ils ne sont pas des probabilités d’incendie calibrées. Le Brier des deux modèles dépasse ici celui d’une probabilité constante égale à la prévalence train ; le classement AP et la calibration probabiliste sont deux propriétés distinctes.'),
      h('p',null,'La suite de cette expérience est présentée dans l’onglet Ablation U-Net : trois poids vérifiés, puis une confirmation du seul candidat retenu sur les 32. Les scores historiques affichés ici restent inchangés. Test 2021 verrouillé.'),
      h('details',null,h('summary',null,'Nouveaux pixels de feu à J+1'),h(Table,{rows:data.diagnosis.rows.filter(r=>r.phase===phase),columns:[['model','Modèle',v=>names[v]],['new_fire_recall','Rappel nouveaux feux',v=>number(v)],['new_fire_dice','Dice nouveaux feux',v=>number(v)],['new_fire_fp','FP hors feu J',v=>number(v,0)],['observed_fire_fp','FP sur feu J',v=>number(v,0)]]}),h('p',null,'Nouveau pixel = absence de détection à J, présence à J+1. Cela ne prouve pas une ignition physique.'))),
    h(Provenance,{source:data.provenance}));
}
export function BigData() {
  const state=useData('/science/big-data');
  const lineage=useData('/lab/lineage');
  if (!state.data) return h('section',{className:'science'},h(State,state));
  const data=state.data;
  return h('section',{className:'science'},h('header',{className:'science-header'},h('span',{className:'eyebrow'},'Spark SQL → Parquet · résultats existants'),h('h2',null,'Inventaire, volumes et provenance'),
    h('p',null,`Spark ${data.spark_version} · ${data.master} · exécution locale à deux threads · aucun job relancé`)),
    h(Fade,null,h('section',{className:'panel'},h('h2',null,'Le parcours des données'),
      h('div',{className:'pipeline-diagram','aria-label':'Pipeline exécuté puis lecture des sorties'},...[
        ['01','Inventaire',`${number(data.tables.annual.reduce((n,r)=>n+r.observations,0),0)} observations indexées`],
        ['02','Spark SQL',`Spark ${data.spark_version} · ${data.master}`],
        ['03','Parquet',`${data.parquet_files.length} agrégats matérialisés`],
        ['04','FastAPI → React','Lecture des résultats et de leurs SHA-256']
      ].map(([step,title,detail])=>h('div',{className:'pipeline-node',key:step},h('span',{className:'eyebrow'},step),h('strong',null,title),h('small',null,detail)))),
      h('p',null,'Spark local[2] réellement exécuté : deux threads locaux, sans cluster distribué ni streaming. La démonstration lit les sorties déjà calculées ; aucun calcul Spark n’est déclenché par l’interface.'))),
    lineage.data&&h('section',{className:'panel'},h('h2',null,'Du corpus compact au Prediction Lab'),
      h('p',null,`${lineage.data.partitions} partitions existantes · ${lineage.data.archive_count} archives événementielles · ${number(lineage.data.source_bytes/1e9,2)} Go de fichiers sources compacts. Aucune reconstruction ni republication.`),
      h(Table,{rows:Object.entries(lineage.data.roles).map(([split,r])=>({split,...r})),columns:[['split','Périmètre'],['events','Événements'],['tiles','Tuiles',v=>number(v,0)]]}),
      h('p',null,'Parquet et index temporel → références aux tuiles NPZ déjà normalisées → sélection d’un événement et d’une date → lecture des cinq jours disponibles → U-Net T3 → cartes React. La vérité terrain J+1 est une lecture séparée.'),
      h('p',null,'Huit canaux réellement disponibles : pente, altitude, leurs valeurs manquantes, aspect sinus/cosinus/manquant et détection active du feu. Pas de vent ni de météo prévue dans ce compact.'),
      h('details',null,h('summary',null,'Identités du compact et du normaliseur train-only'),h('pre',null,JSON.stringify(lineage.data,null,2)))),
    lineage.error&&h('p',{role:'alert'},'Traçabilité du Prediction Lab indisponible ; les résultats Spark restent affichés.'),
    h('div',{className:'metric-grid'},h('article',{className:'metric-card'},h('span',null,'Temps total historique'),h('div',{className:'score'},`${number(data.total_seconds,2)} s`),h('p',null,`JVM ${number(data.jvm_start_seconds,2)} s · traitement ${number(data.processing_seconds,2)} s`)),
      h('article',{className:'metric-card'},h('span',null,'Inventaire de métadonnées'),h('div',{className:'score'},`${number(data.tables.annual.reduce((n,r)=>n+r.events,0),0)} événements`),h('p',null,`${number(data.tables.annual.reduce((n,r)=>n+r.observations,0),0)} observations indexées · 2021 : inventaire seulement`))),
    h(Fade,null,h('section',{className:'panel'},h('h2',null,'Inventaire annuel'),h(InventoryChart,{rows:data.tables.annual}),h(Table,{rows:data.tables.annual,columns:[['year','Année'],['split','Périmètre',v=>v==='test'?'Test verrouillé · métadonnées uniquement':v],['events','Événements',v=>number(v,0)],['observations','Observations',v=>number(v,0)]]}))),
    h('section',{className:'panel'},h('h2',null,'Fenêtres temporelles alignées'),h(Table,{rows:data.tables.windows,columns:[['split','Split'],['context','Jours de contexte'],['events','Événements',v=>number(v,0)],['windows','Fenêtres',v=>number(v,0)]]})),
    h('section',{className:'panel'},h('h2',null,'Jointure des fenêtres T=3 avec leurs cibles'),h(Table,{rows:data.tables.window_join,columns:[['split','Split'],['matched_targets','Cibles appariées',v=>number(v,0)]]}),h('p',null,'Résultat SQL déjà calculé : chaque fenêtre T=3 est reliée à une observation cible par événement et date. Le tableau expose les nombres réellement matérialisés en Parquet.')),
    h('section',{className:'panel'},h('h2',null,'Qualité déjà calculée · observations train/validation'),h(Table,{rows:data.tables.quality,columns:[['split','Split'],['observations','Observations',v=>number(v,0)],['positive_pixels','Pixels feu',v=>number(v,0)],['valid_pixels','Pixels valides',v=>number(v,0)],['invalid_pixels','Pixels invalides',v=>number(v,0)],['no_detection_days','Jours sans détection',v=>number(v,0)]]}),
      h('p',null,`${data.duplicate_keys.length} clé dupliquée · ${data.tables.annual_leakage[0].violations} violation annuelle. Ces comptes d’observations ne sont pas les pixels des fenêtres de calibration.`)),
    ...data.caveats.map((c,i)=>h('p',{key:i,className:'legend'},c)),
    h('details',{className:'panel'},h('summary',null,'Requêtes SQL, Parquet et SHA256'),
      ...Object.entries(data.queries).map(([k,q])=>h('div',{key:k},h('h3',null,`${k} · ${number(data.query_seconds[k],3)} s`),h('pre',null,q))),
      h(Table,{rows:data.parquet_files,columns:[['path','Parquet existant'],['rows','Lignes'],['bytes','Octets',v=>number(v,0)],['sha256','SHA256']]}),
      h('p',null,data.provenance),h('pre',null,JSON.stringify({report_sha256:data.report_sha256,source_hashes:data.source_hashes},null,2))));
}

function InventoryChart({rows}){
 const totals=Object.entries(rows.reduce((acc,r)=>({...acc,[r.split]:(acc[r.split]||0)+r.events}),{})).map(([split,events])=>({year:split,split,events}));
 const chart=(values,label)=>h('div',{className:'annual-chart',role:'img','aria-label':label+': '+values.map(r=>`${r.year} ${r.split} ${r.events} événements`).join('; ')},...values.map(r=>h('div',{className:'annual-row',key:r.year},h('span',null,r.year,h('small',null,r.split==='test'?'test · métadonnées':r.split)),h('div',{className:'annual-track','aria-hidden':true},h('div',{className:'annual-bar','data-split':r.split,style:{width:(r.events/250*100)+'%'}})),h('strong',null,number(r.events,0)))));
 return h(React.Fragment,null,chart(rows,'Événements par année et split, axe 0 à 250'),h('details',null,h('summary',null,'Répartition par split'),chart(totals,'Total par split, axe 0 à 250')),h('p',{className:'caption'},'Longueur proportionnelle au nombre d’événements · axe commun 0–250. Source : agrégat Spark annual.parquet. Le test 2021 reste un inventaire de métadonnées ; aucun pixel lu.'));
}
