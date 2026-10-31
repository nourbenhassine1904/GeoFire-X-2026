"""Explicit opt-in ablations; historical baseline channels stay unchanged.

Forecast bands need issue-time/valid-time provenance absent from TIFF headers.
The legacy observed wind direction has a documented quadrant bug, so it is
excluded. An availability declaration is required for observational extensions.
"""
import json
from pathlib import Path

from geofirex.corpus import sha256_file


PROFILES = {
    "conservative_fire_terrain": {"continuous": [12, 14], "landcover": False},
    "terrain_vegetation_v1": {"continuous": [0, 1, 2, 3, 4, 12, 14], "landcover": True},
    "multimodal_observed_j_v1": {"continuous": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15],
                                 "landcover": True},
}


def feature_profile(config):
    policy = config["feature_policy"]
    profile = PROFILES[policy]
    if policy != "conservative_fire_terrain":
        path = config.get("feature_availability")
        if not path:
            raise ValueError("Multimodal policy requires reviewed feature_availability evidence")
        declaration = json.loads(Path(path).read_text(encoding="utf-8"))
        bands = set(profile["continuous"]) | {13, 16, 22}
        if (declaration.get("policy") != policy or declaration.get("decision_time") != "end_of_day_J"
                or not bands <= set(declaration.get("available_by_J_bands", []))
                or not declaration.get("evidence") or declaration.get("reviewed") is not True):
            raise ValueError("Availability by J not established for every selected band")
    return profile


def availability_hash(config):
    return sha256_file(config["feature_availability"]) if config.get("feature_availability") else None
