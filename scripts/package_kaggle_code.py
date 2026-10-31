"""Build a small code-only Kaggle attachment; never includes local raw data/secrets."""
import argparse
import json
import zipfile
from pathlib import Path

from geofirex.corpus import json_new, sha256_file


def package(destination):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination)
    files = [root/"pyproject.toml", root/"README.md", root/"docs/KAGGLE_CLOUD_SCALE.md",
             root/"data/manifests/real/corpus_zip_index_v1.json"]
    for folder, extensions in (("src", {".py"}), ("scripts", {".py"}),
                               ("tests", {".py"}), ("configs", {".yaml", ".json"}),
                               ("splits/benchmark/annual_v1", {".csv", ".json"})):
        files += [p for p in (root/folder).rglob("*")
                  if p.is_file() and p.suffix in extensions and "__pycache__" not in p.parts]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as out, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(root/name, "GeoFire-X-2026/"+name)
        z.writestr("GeoFire-X-2026/code_manifest.json", json.dumps(hashes, indent=2))
    json_new(destination.with_suffix(".sha256.json"), {"sha256": sha256_file(destination), "files": len(hashes)})
    return {"archive": str(destination.resolve()), "files": len(hashes), "bytes": destination.stat().st_size}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    print(json.dumps(package(p.parse_args().out), indent=2))
