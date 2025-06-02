"""
Generate tidy Parquet files from the raw Club Brugge tracking deliverables.

Usage
-----
Run from the project root *after* installing the package in editable mode:

    python -m pip install -e .
    python scripts/generate_parquet.py
"""

from pathlib import Path

from clubbrugge.io import (
    load_metadata,
    load_tracking_data,
    write_parquet,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]         # project root
RAW = ROOT / "SourceFiles"                         # raw JSON / JSONL live here
OUT = ROOT / "data"                                # destination for Parquet

# guarantee target directory exists
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # --- metadata ----------------------------------------------------------- #
    meta_df = load_metadata(RAW / "metadata.json")
    write_parquet(meta_df, OUT / "metadata.parquet")

    # --- tracking ----------------------------------------------------------- #
    track_df = load_tracking_data(RAW / "tracking-produced.jsonl")
    write_parquet(track_df, OUT / "tracking.parquet")

    print("✓ Parquet files written")


if __name__ == "__main__":  # so the module can be imported without side-effects
    main()
