"""Build the enriched manifest for the real WildfireSpreadTS sample.

Usage:
    python scripts/build_real_manifest.py \\
        --data_dir data/raw/WildfireSpreadTS/real_sample \\
        --out data/manifests/real/manifest_wildfirespreadts.csv
"""

from __future__ import annotations

if __name__ == "__main__":
    from geofirex.real_manifest import main

    main()
