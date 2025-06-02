"""
End-to-end validation of Club Brugge loaders and Parquet outputs.
Run with:  pytest -q
"""
from pathlib import Path
import json                              # <-- NEW
import pandas as pd
from clubbrugge.io import load_metadata, load_tracking_data

ROOT = Path(__file__).resolve().parents[1]       # project root
RAW_DIR = ROOT / "SourceFiles"                   # raw JSON / JSONL
PARQUET_DIR = ROOT / "data"                      # generated Parquet files


# --------------------------------------------------------------------------- #
# Loader-level checks (raw → DataFrame)
# --------------------------------------------------------------------------- #
def test_metadata_loader_schema():
    df = load_metadata(RAW_DIR / "metadata.json")
    assert list(df.columns) == [
        "team_id",
        "player_id",
        "player_name",
        "player_position",
        "player_number",
    ]
    assert not df.empty


def test_tracking_loader_schema():
    df = load_tracking_data(RAW_DIR / "tracking-produced.jsonl", chunk_size=50_000)
    assert list(df.columns) == [
        "period_id",
        "frame_index",
        "game_clock",
        "wall_clock",
        "player_id",
        "player_number",
        "speed",
        "x",
        "y",
        "z",
    ]
    assert df["player_id"].nunique() >= 22


# --------------------------------------------------------------------------- #
# Parquet-level checks (I/O, dtypes, plausibility)
# --------------------------------------------------------------------------- #
def test_parquet_round_trip_schema():
    meta_df = pd.read_parquet(PARQUET_DIR / "metadata.parquet")
    assert list(meta_df.columns) == [
        "team_id",
        "player_id",
        "player_name",
        "player_position",
        "player_number",
    ]

    track_df = pd.read_parquet(PARQUET_DIR / "tracking.parquet")
    assert list(track_df.columns) == [
        "period_id",
        "frame_index",
        "game_clock",
        "wall_clock",
        "player_id",
        "player_number",
        "speed",
        "x",
        "y",
        "z",
    ]


def test_dtypes_and_missing_values():
    df = pd.read_parquet(PARQUET_DIR / "tracking.parquet")

    assert str(df.dtypes["period_id"]) == "int8"
    assert str(df.dtypes["frame_index"]) == "int32"
    assert str(df.dtypes["player_number"]) == "int8"

    assert df[["frame_index", "player_id", "speed"]].isna().sum().sum() == 0
    assert df[["x", "y"]].isna().mean().max() < 0.01


def test_row_count_and_ranges():
    df = pd.read_parquet(PARQUET_DIR / "tracking.parquet")

    # ---- row / frame sanity
    n_frames = df["frame_index"].nunique()
    n_players = df["player_id"].nunique()
    rows_per_frame = len(df) / n_frames
    assert 0.7 * n_players <= rows_per_frame <= 1.3 * n_players

    # ---- pitch-bounds sanity  --------------------------------------------
    meta = json.load((RAW_DIR / "metadata.json").open("r"))
    half_len = meta["data"]["pitchLength"] / 2          # e.g. 52.5 m
    half_wid = meta["data"]["pitchWidth"] / 2           # e.g. 34.0 m

    buffer = 3.0                                        # allow small over-runs
    in_x = df["x"].between(-(half_len + buffer), half_len + buffer)
    in_y = df["y"].between(-(half_wid + buffer), half_wid + buffer)

    # require ≥ 99.5 % of samples on (or just off) the grass
    assert in_x.mean() > 0.995
    assert in_y.mean() > 0.995

    # non-negative speeds
    assert (df["speed"] >= 0).all()
