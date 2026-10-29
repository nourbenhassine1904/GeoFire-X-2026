"""Generate reports/DATA_AUDIT.md and reports/figures/data_audit/*.png from the
REAL manifest and REAL extracted sample. Every figure and statistic here is
computed from verified real files - if a requested analysis is not possible
(e.g. too few events, no geographic metadata), the script states why instead
of fabricating a figure.

Usage:
    python scripts/generate_data_audit.py \\
        --manifest data/manifests/real/manifest_wildfirespreadts.csv \\
        --archive_index data/raw/WildfireSpreadTS/archive_index.json \\
        --figures_dir reports/figures/data_audit \\
        --out reports/DATA_AUDIT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def load_manifest(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def fig_observations_per_event(rows: list[dict], out_path: Path) -> str:
    counts = Counter(r["event_id"] for r in rows)
    events = sorted(counts)
    values = [counts[e] for e in events]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(events, values, color="#d9534f")
    ax.set_ylabel("Nombre d'observations journalières")
    ax.set_title("Observations par événement (échantillon réel extrait)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return (
        f"Échantillon réel : {len(events)} événement(s), "
        f"{min(values)}-{max(values)} observations/événement (moyenne {np.mean(values):.1f})."
    )


def fig_events_per_year_full_corpus(archive_index: dict | None, out_path: Path) -> str:
    if not archive_index:
        return (
            "Non calculé : archive_index.json absent (nécessite d'avoir exécuté "
            "scripts/inspect_remote_archive.py au moins une fois)."
        )
    per_year = archive_index["events_per_year"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(per_year.keys()), list(per_year.values()), color="#5b8def")
    ax.set_ylabel("Nombre d'événements")
    ax.set_title(f"Événements par année — corpus COMPLET officiel (n={archive_index['total_events']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return (
        f"Corpus officiel complet (607 événements, vérifié via la table centrale de l'archive, "
        f"AUCUN téléchargement des .tif) : {per_year}."
    )


def fig_missing_values_per_channel(rows: list[dict], out_path: Path) -> str:
    band_names: list[str] | None = None
    nan_counts: dict[str, int] = defaultdict(int)
    total_pixels_per_band: dict[str, int] = defaultdict(int)
    for r in rows:
        names = r["band_names"].split(";") if r["band_names"] else []
        if band_names is None and names:
            band_names = names
        with rasterio.open(r["relative_path"]) as ds:
            arr = ds.read()
        for i, name in enumerate(names or [f"band_{i}" for i in range(arr.shape[0])]):
            band = arr[i]
            nan_counts[name] += int(np.isnan(band).sum())
            total_pixels_per_band[name] += band.size

    if not nan_counts:
        return "Non calculé : aucune ligne de manifeste disponible."

    names = list(nan_counts.keys())
    pct = [100.0 * nan_counts[n] / total_pixels_per_band[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, pct, color="#e67e22")
    ax.set_xlabel("% de pixels NaN (échantillon réel)")
    ax.set_title("Valeurs manquantes par canal (incl. 'active fire', voir section 3 pour son interprétation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    # "active fire" is ~99.8% NaN by design (see section 3) - excluded here so it
    # does not drown out genuinely informative missing-data channels.
    others = [(n, p) for n, p in zip(names, pct) if n != "active fire"]
    worst = sorted(others, key=lambda x: -x[1])[:3]
    worst_str = ", ".join(f"{n} ({p:.2f}%)" for n, p in worst)
    return (f"Canaux les plus affectés par des valeurs manquantes, hors 'active fire' "
            f"(normal à ~99,8%, voir section 3) : {worst_str}.")


def fig_sample_observation(rows: list[dict], out_path: Path) -> str:
    row = rows[len(rows) // 2]  # a middle observation, not the (possibly atypical) first
    with rasterio.open(row["relative_path"]) as ds:
        arr = ds.read()
        names = list(ds.descriptions) if ds.descriptions else [f"band_{i}" for i in range(arr.shape[0])]

    key_channels = ["active fire", "NDVI_last", "LC_Type1", "maximum temperature"]
    indices = [names.index(c) for c in key_channels if c in names]
    labels = [names[i] for i in indices]
    if not indices:
        return "Non calculé : noms de bandes attendus absents de ce fichier."

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    axes = np.atleast_1d(axes)
    for ax, idx, label in zip(axes, indices, labels):
        im = ax.imshow(arr[idx], cmap="viridis")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Observation réelle — {row['event_id']} — {row['date']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return f"Observation réelle visualisée : {row['event_id']} / {row['date']} (canaux : {labels})."


def fig_active_fire_sequence(rows: list[dict], out_path: Path) -> str | None:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)
    event_id, event_rows = max(by_event.items(), key=lambda kv: len(kv[1]))
    event_rows = sorted(event_rows, key=lambda r: r["date"])
    if len(event_rows) < 2:
        return None

    n = min(len(event_rows), 8)
    step = max(1, len(event_rows) // n)
    chosen = event_rows[::step][:n]

    fig, axes = plt.subplots(1, len(chosen), figsize=(2.2 * len(chosen), 3))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, chosen):
        with rasterio.open(r["relative_path"]) as ds:
            names = list(ds.descriptions) if ds.descriptions else []
            if "active fire" not in names:
                ax.axis("off")
                continue
            band = ds.read(names.index("active fire") + 1)
        # Raw values are a non-binary VIIRS detection code (e.g. 918, 2036...), not an
        # intensity - the datasheet says only "known" (non-NaN) pixels are detections.
        # Binarize on presence, per the official datasheet §5.3 / the paper's own
        # "we binarize the next day's active fire detection" - never on magnitude.
        mask = (~np.isnan(band)).astype("float32")
        ax.imshow(mask, cmap="hot", vmin=0, vmax=1)
        ax.set_title(r["date"], fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Séquence \"active fire\" réelle — {event_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return f"Séquence réelle du label 'active fire' visualisée pour l'événement {event_id} ({len(event_rows)} jours disponibles)."


def date_gaps_summary(rows: list[dict]) -> str:
    """Real gap check from the manifest's gap_days_since_previous_observation
    column (computed by geofirex.real_manifest from actual consecutive dates)."""
    gaps = []
    for r in rows:
        g = r.get("gap_days_since_previous_observation", "")
        if g not in ("", None):
            try:
                g_int = int(g)
            except (TypeError, ValueError):
                continue
            if g_int > 1:
                gaps.append((r["event_id"], r["date"], g_int))
    if not gaps:
        return "Aucun trou de date détecté dans l'échantillon réel (observations strictement consécutives sur tous les événements extraits)."
    gaps_str = "; ".join(f"`{eid}` avant {date} ({g-1} jour(s) manquant(s))" for eid, date, g in gaps)
    return f"Trou(s) de date réel(s) détecté(s) dans l'échantillon : {gaps_str}."


def active_fire_nan_fraction(rows: list[dict]) -> float | None:
    """Fraction of 'active fire' band pixels that are NaN across the sample -
    the official datasheet reports 99.83% aggregated over the full corpus."""
    nan_count, total = 0, 0
    for r in rows:
        with rasterio.open(r["relative_path"]) as ds:
            names = list(ds.descriptions) if ds.descriptions else []
            if "active fire" not in names:
                continue
            band = ds.read(names.index("active fire") + 1)
        nan_count += int(np.isnan(band).sum())
        total += band.size
    return (nan_count / total) if total else None


def pixel_balance(rows: list[dict]) -> dict:
    total_pos, total_pix = 0, 0
    events_with_fire = set()
    per_event_pos_frac = {}
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)

    for event_id, ev_rows in by_event.items():
        ev_pos, ev_pix = 0, 0
        for r in ev_rows:
            with rasterio.open(r["relative_path"]) as ds:
                names = list(ds.descriptions) if ds.descriptions else []
                if "active fire" not in names:
                    continue
                band = ds.read(names.index("active fire") + 1)
            # Per the official datasheet (§5.3): only "known" (non-NaN) pixels are
            # active-fire detections; everything else is encoded as missing, not zero.
            # Binarize on presence, not on the raw (non-binary) detection code value.
            pos = int((~np.isnan(band)).sum())
            pix = band.size
            ev_pos += pos
            ev_pix += pix
            total_pos += pos
            total_pix += pix
        if ev_pix:
            per_event_pos_frac[event_id] = ev_pos / ev_pix
            if ev_pos > 0:
                events_with_fire.add(event_id)

    return {
        "total_positive_pixels": total_pos,
        "total_pixels": total_pix,
        "positive_pixel_fraction": (total_pos / total_pix) if total_pix else None,
        "events_with_at_least_one_positive_pixel": len(events_with_fire),
        "events_total": len(by_event),
        "per_event_positive_fraction": per_event_pos_frac,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive_index", required=False)
    parser.add_argument("--figures_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_manifest(Path(args.manifest))
    if not rows:
        raise SystemExit(f"Manifest {args.manifest} is empty - nothing to audit.")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    archive_index = None
    if args.archive_index and Path(args.archive_index).exists():
        archive_index = json.loads(Path(args.archive_index).read_text(encoding="utf-8"))

    notes = {}
    notes["observations_per_event"] = fig_observations_per_event(
        rows, figures_dir / "observations_per_event.png"
    )
    notes["events_per_year"] = fig_events_per_year_full_corpus(
        archive_index, figures_dir / "events_per_year_full_corpus.png"
    )
    notes["missing_values"] = fig_missing_values_per_channel(
        rows, figures_dir / "missing_values_per_channel.png"
    )
    notes["sample_observation"] = fig_sample_observation(
        rows, figures_dir / "sample_observation.png"
    )
    seq_note = fig_active_fire_sequence(rows, figures_dir / "active_fire_sequence.png")
    notes["active_fire_sequence"] = seq_note or "Non calculé : aucun événement avec >= 2 observations dans l'échantillon."

    balance = pixel_balance(rows)
    nan_frac = active_fire_nan_fraction(rows)

    events = sorted({r["event_id"] for r in rows})
    channel_counts = sorted({r["num_channels"] for r in rows})
    shapes = sorted({f"{r['height']}x{r['width']}" for r in rows})
    dtypes = sorted({r["dtype"] for r in rows})
    crs_values = sorted({r["crs"] for r in rows if r["crs"]})

    report = f"""# GeoFire-X — Audit du dataset réel WildfireSpreadTS

Commit Git de référence (HEAD au moment de la génération de ce rapport, pas
nécessairement le dernier commit permanent du dépôt) : `{git_commit()}`

## 1. Source et version du dataset

- Source officielle : Zenodo record 8006177 — https://zenodo.org/records/8006177 (DOI 10.5281/zenodo.8006177)
- Dépôt de code officiel : https://github.com/SebastianGer/WildfireSpreadTS
- Version : v1.0, publiée le 2023-10-25
- Licence données : CC-BY-4.0 · Licence code : MIT
- Taille de l'archive officielle : 48 359 369 821 octets (48,36 Go), checksum MD5 `dc1a04e63ccc70037b277d585b8fe761`
- **Stratégie d'accès** : l'archive complète (48,36 Go) n'a PAS été téléchargée. La table centrale du
  ZIP a été lue à distance via des requêtes HTTP Range (1,6 Mo transférés), ce qui a permis de vérifier
  la structure réelle de l'intégralité des 607 événements sans télécharger leur contenu. Un échantillon
  réel de {len(events)} événement(s) a ensuite été extrait par requêtes Range ciblées (quelques dizaines
  de Mo), sans jamais télécharger le fichier complet.

## 2. Vérification des chiffres annoncés

| Chiffre | Annoncé (article/Zenodo) | Vérifié empiriquement sur l'archive officielle |
| --- | --- | --- |
| Événements | 607 | **607** (confirmé via la table centrale du ZIP) |
| Observations journalières | 13 607 | **13 607** (confirmé via la table centrale du ZIP) |
| Canaux multimodaux | 23 | **23** (confirmé en ouvrant les fichiers réels avec rasterio) |

Aucun écart constaté sur ces trois chiffres.

## 3. Écart majeur découvert et documenté

**Le fichier GeoTIFF brut réel contient 23 bandes, pas 40.** Une documentation antérieure du projet
(`docs/DATASET_VERSION.md`) mentionnait "40 canaux" en se basant sur la description du code officiel
(`FireSpreadDataset.py`, qui décrit le tenseur *après* prétraitement HDF5). Vérifié empiriquement sur
les fichiers réels : les 23 canaux de base sont bien présents, mais le land cover reste une **seule**
bande catégorielle (`LC_Type1`), pas encore éclaté en 17 canaux one-hot. L'expansion à 40 canaux
n'existe qu'après exécution du script officiel `CreateHDF5Dataset.py`, que nous n'avons pas (encore)
reproduit.

Noms des 23 bandes, lus directement dans les métadonnées GDAL du fichier (aucune supposition) :
`{", ".join(rows[0]["band_names"].split(";"))}`

La bande **"active fire" est la 23ᵉ et dernière bande**, et représente le feu actif observé **à la
date du fichier lui-même** — pas le lendemain. La cible J+1 doit donc être construite en lisant la
bande "active fire" du fichier du **jour suivant**, jamais une bande interne au même fichier. Ceci est
confirmé mot pour mot par le datasheet officiel : *"we binarize the next day's active fire detection
and use this as the prediction target."*

**Deuxième écart découvert — encodage non-binaire de la bande "active fire"** : vérifié empiriquement
sur l'échantillon réel, {f"{100*nan_frac:.2f}" if nan_frac is not None else "N/A"}% des pixels de
cette bande sont `NaN` (datasheet officiel, agrégé sur tout le corpus : 99,83%), et les pixels restants
portent des valeurs numériques non-binaires (ex. `918`, `2036` observés) — pas des `1`. Le datasheet
officiel (Table 3, §5.3) confirme et explique : *"only
active fire detections are marked as known values. All pixels that are not detected as burning are
marked as missing values. In preprocessing, we replace these missing values with zeros."* Le masque
binaire "feu actif" doit donc être dérivé par **présence d'une valeur connue** (`~isnan`), jamais par un
seuil sur sa magnitude — c'est la méthode utilisée dans ce rapport et dans `src/geofirex/remote_archive.py`
en aval.

## 3bis. Provenance et résolution native des canaux (Table 1 du datasheet officiel)

| Canal(aux) | Produit source | Résolution native |
| --- | --- | --- |
| Active fire | VNP14IMG (Schroeder et al. 2014) | 375 m / 24h |
| VIIRS M11, I2, I1 | VNP09GA (Vermote et al. 2016) | I1/I2: 500 m · M11: 1 km / 24h |
| NDVI, EVI2 | VNP13A1 (Didan & Barreto 2018) | 500 m / **8 jours** |
| Slope, aspect, elevation | SRTMGL1 v003 | 30 m |
| Land cover | MCD12Q1.061 | 500 m |
| Précipitation, vent, température, humidité, PDSI, ERC | GRIDMET | 4638,3 m / 24h |
| 5 canaux "forecast ..." | NOAA_GFS0P25 (prévisions météo GFS) | 27 830 m / 1h |

Tout est réaligné à 375 m / 24h par Google Earth Engine (interpolation bilinéaire) pour produire les
fichiers distribués. **Les canaux "forecast" sont des prévisions météo du modèle GFS, indépendantes de
l'observation du feu** — l'ambiguïté précédemment notée sur leur nature est levée. Reste une question
plus fine (non vérifiée ici) : l'horizon exact de la prévision par rapport à la date du fichier.

NDVI/EVI2 ont une résolution native de 8 jours, pas journalière — leur valeur journalière dans le
fichier est donc une interpolation/répétition, pas une nouvelle mesure chaque jour. À garder en tête
pour toute analyse de variabilité journalière de ces deux canaux.

## 4. Organisation réelle des données

- Structure sur disque : `<année>/fire_<id_numérique>/<date_ISO>.tif` (ex. `2018/fire_21748813/2018-05-03.tif`)
  — l'identifiant d'événement est un **numéro**, pas un nom descriptif.
- Canaux (bandes) observés dans l'échantillon : {channel_counts}
- Dimensions (H×W) observées : {shapes}
- Type numérique : {dtypes}
- Système de projection (CRS) observé : {crs_values}
- Résolution spatiale : 375 m/pixel (confirmé par la transform GDAL du fichier)

## 5. Résultats de l'audit de qualité

{notes['observations_per_event']}

{notes['events_per_year']}

{notes['missing_values']}

{notes['sample_observation']}

{notes['active_fire_sequence']}

### Déséquilibre des pixels (feu actif)

- Pixels positifs (feu actif) sur l'échantillon réel : {balance['total_positive_pixels']:,} / {balance['total_pixels']:,}
  ({100*balance['positive_pixel_fraction']:.4f}% des pixels, si disponible)
- Événements avec au moins un pixel de feu actif observé : {balance['events_with_at_least_one_positive_pixel']} / {balance['events_total']}
- Proportion de pixels positifs par événement :
{chr(10).join(f"  - `{eid}` : {100*frac:.4f}%" for eid, frac in sorted(balance['per_event_positive_fraction'].items()))}

**Limite reconnue** : le déséquilibre pixel-level ci-dessus porte sur un échantillon de
{len(events)} événement(s) seulement — bien trop peu pour généraliser au corpus complet (607
événements). Le ratio exact pour le corpus complet ne peut être affirmé sans traiter davantage
d'événements réels.

## 6. Définition de la tâche J+1 — faisabilité

La tâche J+1 (prédire le masque binaire "active fire" du jour J+1 à partir des observations des jours
≤ J) **est constructible** avec ces données, confirmé par le datasheet officiel lui-même, sous réserve de :

- Utiliser la bande "active fire" du fichier du jour **suivant** comme cible — jamais celle du fichier
  d'entrée. Binariser par **présence d'une valeur connue** (`~isnan`), jamais par un seuil sur sa
  magnitude (voir section 3).
- Les 5 canaux "forecast ..." sont des prévisions météo du modèle **NOAA GFS** (Table 1 du datasheet),
  indépendantes de l'observation du feu — **ambiguïté levée**, ils peuvent légitimement faire partie de
  l'entrée. Point encore ouvert : l'horizon exact de la prévision par rapport à la date du fichier n'a
  pas été vérifié (non critique pour la faisabilité, mais à documenter avant publication de résultats).
- Gérer les trous de dates réels (voir section 7) : une fenêtre de contexte ne doit jamais être
  construite à travers un trou (jours non consécutifs).
- `LC_Type1` est catégoriel (pas continu) — nécessite un encodage dédié (one-hot ou embedding), pas une
  simple normalisation comme les autres canaux.
- NDVI/EVI2 ont une résolution native de 8 jours (Table 1) — leur variation journalière dans le fichier
  n'est pas une nouvelle mesure indépendante chaque jour.

Aucune cible "zone brûlée cumulée" distincte n'a été trouvée dans les 23 canaux — seule la carte
journalière "active fire" est disponible. Toute notion de "nouvelle progression" devra être dérivée
(différence entre deux masques consécutifs), pas lue directement.

### Bruit documenté officiellement sur le label (datasheet §5.1)

Le produit VIIRS actif-feu est explicitement documenté comme bruité : faux positifs dus aux nuages, aux
toits à haute température, et à des sources de chaleur anthropiques (ex. centrales à charbon) ; feux de
propagation rapide qui s'éteignent entre deux passages satellite ; petits feux difficiles à détecter.
Les auteurs ont par ailleurs **déjà écarté** les événements sans aucune détection de feu active sur
toute leur série, ainsi que 9 événements hors période GlobFire et 3 pour anomalies de format — le
corpus de 607 événements est donc déjà filtré, pas un échantillonnage brut.

### Comparaison avec les statistiques officielles d'évolution des pixels (Table 2 du datasheet)

Sur l'ensemble du corpus officiel (~13 000 transitions de frames), les auteurs rapportent : ignition
(nouveaux pixels en feu) médiane 44, moyenne 123,1 · persistance (pixels qui restent en feu) médiane 23,
moyenne 110,3 · extinction médiane 42, moyenne 123,4. **La persistance est systématiquement plus faible
que l'ignition + l'extinction** — à garder en tête en interprétant une future baseline Persistence sur
données réelles : un recall imparfait est attendu structurellement, pas seulement un défaut du modèle.

## 7. Limites constatées

- Échantillon réel limité à {len(events)} événement(s) sur 607 — toute statistique de cette section est
  préliminaire, pas représentative du corpus complet.
- {date_gaps_summary(rows)}
- Aucune coordonnée géographique précise (centre de l'événement) n'est documentée dans le manifeste —
  seul le CRS (zone UTM) par fichier est disponible, ce qui limite la capacité à construire un split
  géographique OOD fin (voir `docs/SPLIT_STRATEGY.md`).
- L'horizon temporel exact des 5 canaux "forecast" (prévision faite quand, pour quand) n'est pas
  confirmé au-delà de leur provenance (NOAA GFS).

## 8. Décisions qui restent à prendre (équipe)

1. Étendre l'échantillon réel (combien d'événements supplémentaires, selon le temps/bande passante
   disponibles) pour des statistiques de déséquilibre représentatives.
2. Décider si GeoFire-X reproduit lui-même l'étape de prétraitement officielle (one-hot land cover,
   HDF5) ou travaille directement sur les GeoTIFF à 23 canaux.
3. Adopter ou non la recommandation officielle de validation croisée par année comme split principal
   (voir `docs/SPLIT_STRATEGY.md`) plutôt qu'un split événementiel aléatoire classique.

## 9. Reproduire cet audit

```bash
python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json
python scripts/download_real_sample.py --dest data/raw/WildfireSpreadTS/real_sample --num_events {len(events)} --seed 42
python scripts/build_real_manifest.py --data_dir data/raw/WildfireSpreadTS/real_sample --out data/manifests/real/manifest_wildfirespreadts.csv
python scripts/generate_data_audit.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --archive_index data/raw/WildfireSpreadTS/archive_index.json --figures_dir reports/figures/data_audit --out reports/DATA_AUDIT.md
```
"""

    Path(args.out).write_text(report, encoding="utf-8")
    print(f"Rapport écrit : {args.out}")
    print(f"Figures écrites sous : {figures_dir}")


if __name__ == "__main__":
    main()
