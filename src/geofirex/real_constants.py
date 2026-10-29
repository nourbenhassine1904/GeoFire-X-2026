"""Constants describing the REAL, raw WildfireSpreadTS GeoTIFF layout.

Deliberately separate from constants.py, which describes the SYNTHETIC
pipeline's 40-channel post-preprocessing format. Mixing the two would silently
break the synthetic fixtures/tests (Phase 1) and misrepresent the raw data
(Phase 2 audit finding: raw files have 23 bands, not 40 - see
docs/DATASET_VERSION.md and reports/DATA_AUDIT.md §3).

Band order and names verified empirically against the real sample (GDAL band
descriptions embedded in the files themselves) and cross-checked against the
official datasheet (WildfireSpreadTS_Documentation.pdf, "What data does each
instance consist of?"). Both sources agree on this exact order.
"""

from __future__ import annotations

RAW_NUM_CHANNELS = 23

# Index (0-based) -> official meaning. Source: real GDAL band descriptions +
# datasheet §1.2, cross-verified, see reports/DATA_AUDIT.md §3bis for provenance
# (satellite product, native resolution) of each channel.
RAW_BAND_NAMES: tuple[str, ...] = (
    "M11",  # VIIRS surface reflectance, band M11 (1 km native)
    "I2",  # VIIRS surface reflectance, band I2 (500 m native)
    "I1",  # VIIRS surface reflectance, band I1 (500 m native)
    "NDVI_last",  # vegetation index, 500 m / 8-day native resolution
    "EVI2_last",  # vegetation index, 500 m / 8-day native resolution
    "total precipitation",  # GRIDMET, 24h
    "wind speed",  # GRIDMET, 24h
    "wind direction",  # GRIDMET, 24h
    "minimum temperature",  # GRIDMET, 24h
    "maximum temperature",  # GRIDMET, 24h
    "energy release component",  # GRIDMET, 24h
    "specific humidity",  # GRIDMET, 24h
    "slope",  # SRTM, static
    "aspect",  # SRTM, static
    "elevation",  # SRTM, static
    "pdsi",  # Palmer Drought Severity Index, GRIDMET
    "LC_Type1",  # land cover CLASS id (categorical, NOT one-hot in raw files)
    "total_precipitation_surface_sum",  # NOAA GFS forecast
    "forecast wind speed",  # NOAA GFS forecast
    "forecast wind direction",  # NOAA GFS forecast
    "forecast temperature",  # NOAA GFS forecast
    "forecast specific humidity",  # NOAA GFS forecast
    "active fire",  # VIIRS VNP14IMG detection code; NaN = no detection (see below)
)

assert len(RAW_BAND_NAMES) == RAW_NUM_CHANNELS

# 0-based index of each channel referenced by name elsewhere in the real pipeline.
RAW_ACTIVE_FIRE_CHANNEL = RAW_BAND_NAMES.index("active fire")  # 22
RAW_LAND_COVER_CHANNEL = RAW_BAND_NAMES.index("LC_Type1")  # 16

# Official binarization convention (datasheet §5.3, verified empirically on the
# real sample - see reports/DATA_AUDIT.md §3): the raw "active fire" band is
# NaN for every pixel with no detection and a non-binary VIIRS detection code
# (e.g. 918, 2036) for the rest. The binary fire/no-fire mask used everywhere
# in GeoFire-X is `~isnan(raw_active_fire_band)` - NEVER a threshold on the
# raw value's magnitude, which is not an intensity.
