"""Generate a small synthetic WildfireSpreadTS-layout dataset for development.

Usage:
    python scripts/generate_synthetic.py --data_dir data/raw/synthetic --num_events 6
"""

from __future__ import annotations

if __name__ == "__main__":
    from geofirex.synthetic import main

    main()
