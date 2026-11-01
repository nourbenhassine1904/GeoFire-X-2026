import React, {useEffect, useRef, useState} from 'react';
const h = React.createElement;

function FireMap({title, grid, probability = false}) {
  const canvas = useRef(null);
  useEffect(() => {
    if (!grid?.length) return;
    const height = grid.length, width = grid[0].length;
    const context = canvas.current.getContext('2d');
    canvas.current.width = width; canvas.current.height = height;
    const pixels = context.createImageData(width, height);
    grid.forEach((row, y) => row.forEach((value, x) => {
      const offset = (y * width + x) * 4;
      const color = value === null ? [220, 226, 232] : probability
        ? [Math.round(255 * value), Math.round(90 * value), Math.round(90 * (1 - value))]
        : value ? [242, 95, 40] : [20, 39, 45];
      pixels.data.set([...color, 255], offset);
    }));
    context.putImageData(pixels, 0, 0);
  }, [grid, probability]);
  return h('figure', null, h('figcaption', null, title),
    h('canvas', {ref: canvas, role: 'img', 'aria-label': title,
      style: {width: '100%', imageRendering: 'pixelated'}}));
}

export default function EventForecast({apiBase = ''}) {
  const [events, setEvents] = useState([]), [eventId, setEventId] = useState('');
  const [date, setDate] = useState(''), [result, setResult] = useState(null);
  const [truth, setTruth] = useState(null), [truthError, setTruthError] = useState('');
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  const request = useRef(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/events`, {signal: controller.signal}).then(async response => {
      if (!response.ok) throw new Error('Catalogue indisponible');
      setEvents(await response.json());
    }).catch(error => { if (error.name !== 'AbortError') setError(error.message); });
    return () => {controller.abort(); request.current?.abort();};
  }, [apiBase]);
  const selected = events.find(event => event.event_id === eventId);
  function reset() {request.current?.abort(); setBusy(false); setResult(null); setTruth(null); setError(''); setTruthError('');}
  async function predict() {
    reset(); const controller = new AbortController(); request.current = controller;
    setBusy(true);
    try {
      const query = new URLSearchParams({event_id: eventId, as_of: date, max_side: 256});
      const response = await fetch(`${apiBase}/predict?${query}`, {signal: controller.signal});
      const body = await response.json();
      if (controller.signal.aborted || request.current !== controller) return;
      if (!response.ok) throw new Error(body.detail || 'La prédiction a échoué');
      setResult(body);
      // Historical comparison is a separate read, after inference has finished.
      const actual = await fetch(`${apiBase}/ground-truth?${query}`, {signal: controller.signal});
      const observation = await actual.json();
      if (controller.signal.aborted || request.current !== controller) return;
      if (!actual.ok) {setTruthError(observation.detail || 'Vérité terrain indisponible'); return;}
      if (JSON.stringify(observation.full_shape) !== JSON.stringify(body.full_shape)
          || observation.preview_stride !== body.preview_stride || observation.crs !== body.crs
          || JSON.stringify(observation.transform) !== JSON.stringify(body.transform)) {
        setTruthError('Géométrie de vérité terrain incompatible'); return;
      }
      setTruth(observation.ground_truth.map((row,y) => row.map((v,x) => body.probability[y][x] === null ? null : v)));
    } catch (error) {if (error.name !== 'AbortError') setError(error.message);}
    finally {if (request.current === controller) setBusy(false);}
  }
  return h('main', {style: {maxWidth: 1100, margin: '2rem auto', padding: '1rem', fontFamily: 'system-ui'}},
    h('h1', null, 'GeoFire-X · Prévision de propagation'),
    h('p', null, 'Événements historiques · prévision à J+1 · jeu de test 2021 verrouillé'),
    h('label', null, 'Événement ', h('select', {value: eventId, onChange: e => {
      reset(); setEventId(e.target.value); setDate('');
    }}, h('option', {value: ''}, 'Sélectionner un événement'),
      ...events.map(event => h('option', {key: event.event_id, value: event.event_id}, `${event.event_id} · ${event.split}`)))),
    h('label', {style: {marginLeft: '1rem'}}, 'Date J ', h('select', {value: date, onChange: e => {
      reset(); setDate(e.target.value);
    }}, h('option', {value: ''}, 'Choisir une date'),
      ...(selected?.forecast_dates || []).map(day => h('option', {key: day, value: day}, day)))),
    h('button', {onClick: predict, disabled: busy || !date || !eventId, style: {marginLeft: '1rem'}},
      busy ? 'Calcul en cours…' : 'Prédire J+1'),
    error && h('p', {role: 'alert'}, error),
    result && h(React.Fragment, null,
      h('p', null, `${result.model} · ${result.qualification} · cible ${result.target_date} · seuil ${result.threshold}`),
      h('p', null, result.data_origin === 'official_wildfirespreadts_verified_bytes'
        ? `Données réelles WildfireSpreadTS, octets vérifiés. Qualification : ${result.qualification}.`
        : 'Données personnalisées ou exemples de test : aucune validation scientifique.'),
      h('p', null, result.split === 'train'
        ? 'Événement d’entraînement : démonstration de fonctionnement, sans évaluation hors échantillon.'
        : 'Événement de validation : utilisé pour choisir le checkpoint et le seuil, sans conclusion sur le test final.'),
      h('div', {style: {display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))'}},
        h(FireMap, {title: `Feu observé à J · ${date}`, grid: result.observed_fire}),
        h(FireMap, {title: 'Probabilité de feu à J+1 · 0 sombre, 1 orange', grid: result.probability, probability: true}),
        h(FireMap, {title: 'Prédiction au seuil de validation', grid: result.predicted_fire}),
        truth && h(FireMap, {title: `Vérité terrain · ${result.target_date}`, grid: truth})),
      truthError && h('p', {role: 'alert'}, truthError),
      h('small', null, `Checkpoint ${result.checkpoint_sha256.slice(0, 12)} · ${result.optimizer_updates} mises à jour. Aperçu spatial des prédictions du modèle.`)));
}
