# WildfireSpreadTS — Dataset Version

## Role in GeoFire-X

Primary dataset for next-day active-fire forecasting.

## Source

Official dataset source: https://zenodo.org/records/8006177
Official repository: https://github.com/SebastianGer/WildfireSpreadTS

## Version

Dataset release: v1.0 (Zenodo record 8006177)
Publication date: 2023-10-25
DOI: 10.5281/zenodo.8006177
Repository commit: not pinned yet — pin to a specific commit SHA before relying on
`src/preprocess/CreateHDF5Dataset.py` or `src/dataloader/FireSpreadDataset.py`.
Archive checked (metadata + central directory + partial extraction, see Phase 2
section below): 2026-10-10. Full 48.36 GB archive: not downloaded.

## Expected Content (as described by the paper / the official code's post-preprocessing format)

Expected wildfire events: 607 (United States, January 2018 – October 2021)
Expected daily observations: 13,607 daily GeoTIFF observations across all events
Expected channels: 23 base channels + land cover expanded to 17 one-hot channels
  during preprocessing = 40 channels in the HDF5/tensor form produced by the
  official `CreateHDF5Dataset.py` / consumed by `FireSpreadDataset`.
Target: binary next-day active-fire mask
Spatial resolution: 375 m (active-fire product resolution)

## Verified Content — Phase 2 real-data audit (2026-10-10)

Checked directly against the official Zenodo record and a real extracted sample
(see `reports/DATA_AUDIT.md` for the full audit). **Do not assume the raw files
match the post-preprocessing figures above without checking.**

- Events: **607** — confirmed by reading the ZIP's central directory remotely
  (no .tif content downloaded, ~1.6 MB transferred).
- Daily observations: **13,607** — confirmed the same way.
- **Raw per-day GeoTIFF channel count: 23, not 40.** The land cover channel
  (`LC_Type1`) is a single categorical band in the raw files; the 17-way one-hot
  expansion only happens in the official HDF5 preprocessing step
  (`CreateHDF5Dataset.py`), which GeoFire-X has not run yet. Treat any "40
  channels" reference elsewhere in this repo as describing the *preprocessed*
  tensor format, never the raw Zenodo files.
- Band names, read directly from each real file's GDAL band descriptions (ground
  truth, not inferred): `M11, I2, I1, NDVI_last, EVI2_last, total precipitation,
  wind speed, wind direction, minimum temperature, maximum temperature, energy
  release component, specific humidity, slope, aspect, elevation, pdsi,
  LC_Type1, total_precipitation_surface_sum, forecast wind speed, forecast wind
  direction, forecast temperature, forecast specific humidity, active fire`.
- **"active fire" is band 23 (the last band)** and represents fire observed on
  the file's own date — the J+1 target must be read from the *next* day's file,
  never from within the same file.
- On-disk layout confirmed: `<year>/fire_<numeric_id>/<date>.tif` — the event
  "name" is a numeric id, not a descriptive name.
- CRS/UTM zone and image dimensions vary per event (observed `EPSG:32610`,
  338×297 px on one real event) — the fixed 375 m/pixel resolution was
  confirmed via the file's own GDAL transform.
- Role of the 4 "forecast ..." channels relative to causality (are they
  forecasts made *as of* the file's date, safe to use as input?) is **not yet
  confirmed** against `WildfireSpreadTS_Documentation.pdf` — open question, see
  `reports/DATA_AUDIT.md` §6/§8.

## Downloadable Files (verified via the Zenodo API, not fully fetched)

| File | Size (exact, bytes) | MD5 |
| --- | --- | --- |
| WildfireSpreadTS.zip | 48,359,369,821 (48.36 GB) | `dc1a04e63ccc70037b277d585b8fe761` |
| WildfireSpreadTS_Documentation.pdf | 7,841,847 (7.8 MB) | `1abb9a9975a0dae1dbd0324a782d00dc` |

`WildfireSpreadTS_Documentation.pdf` has been downloaded (its size matches exactly).
`WildfireSpreadTS.zip` has not — see `reports/DATA_AUDIT.md` §1 for the remote
Range-request strategy used instead of a full download.

## Licensing

- Dataset (Zenodo): CC-BY-4.0
- Official code repository (GitHub): MIT — different license, cite both separately.

## Local Storage

data/raw/WildfireSpreadTS/

Status as of this document:
- `archive/WildfireSpreadTS_Documentation.pdf` — official documentation, fetched in full (7.8 MB).
- `archive_index.json` — full central-directory index of the official archive (607 events,
  13,607 files), built with zero .tif content downloaded.
- `real_sample/` — a real, deterministic sample of events extracted directly from the
  official archive via targeted HTTP Range requests (seed=42, see
  `scripts/download_real_sample.py`). Only the bytes of the sampled events were
  transferred — the full 48.36 GB archive was never downloaded. Event count and exact
  byte count transferred are reported in `reports/DATA_AUDIT.md`.
- The full 48.36 GB archive itself: not downloaded. Downloading it in full requires an
  explicit team decision (disk space, bandwidth, time) — not taken automatically.

Development before this phase used synthetic GeoTIFFs generated by
`src/geofirex/synthetic.py` (40 channels, matching the *preprocessed* tensor
format, not the raw 23-channel files above) so the pipeline could be built and
tested before any real data was available. That synthetic pipeline is kept
unchanged and independent — see `data/manifests/demo/` / `splits/demo/` below.

A frozen example manifest and event split, produced by this exact synthetic
pipeline with a fixed seed, are committed for reference at:

- `data/manifests/demo/manifest_synthetic.csv`
- `splits/demo/event_split_synthetic.csv`

Both describe **synthetic data only** — never the real WildfireSpreadTS dataset.
The `_synthetic` suffix and `demo/` subfolder are deliberate to avoid any
confusion with manifests/splits that will later be generated from real data
(which must use different, non-`demo` paths, e.g. `data/manifests/manifest.csv`).

## Raw Data Policy

- Raw data must remain unchanged.
- Raw data must never be committed to Git.
- All transformations must be reproducible.
- Derived files must be stored in data/interim/ or data/processed/.

## Known Issues

- Dataset class bug fixed upstream at commit `ab3c8f35c5ec8c52c306a4488eaeb71a5a13d0de`
  — performance differs slightly before/after this fix. Pin and record the exact
  commit used once real data is integrated.
- Angle-feature bug (reported upstream in early 2026): wind/forecast direction angle
  features are only transformed with sine; both sine and cosine are needed to avoid
  losing angular information. Apply the cos transform ourselves if using angle
  channels, until upstream fixes it.
- No official train/val/test split is provided by the dataset or the official
  repository — GeoFire-X defines its own event-level split (see `splits/`).
