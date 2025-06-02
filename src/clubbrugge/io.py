from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Loading functions
# --------------------------------------------------------------------------- #
def load_metadata(metadata_file: str | Path) -> pd.DataFrame:
    """
    Load Second Spectrum match metadata and return one row per player.

    Parameters
    ----------
    metadata_file
        Path to `metadata.json`.

    Returns
    -------
    pd.DataFrame
        Columns: ``team_id, player_id, player_name, player_position, player_number``.
    """
    with Path(metadata_file).open("r", encoding="utf-8") as f:
        meta = json.load(f)

    teams = [meta["data"]["homeTeam"], meta["data"]["awayTeam"]]
    records: list[dict] = []
    for team in teams:
        for p in team["players"]:
            records.append(
                {
                    "team_id": team["id"],
                    "player_id": p["id"],
                    "player_name": p["name"],
                    "player_position": p["position"],
                    "player_number": int(p["number"]),
                }
            )

    df = pd.DataFrame.from_records(records)
    df["player_number"] = df["player_number"].astype("int16")
    return df


def load_tracking_data(tracking_data_file: str | Path, *, chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Load newline-delimited tracking messages and explode to *player-frame* level.

    Parameters
    ----------
    tracking_data_file
        Path to `tracking-produced.jsonl`.
    chunk_size
        Lines to buffer before concatenating to keep RAM usage predictable.

    Returns
    -------
    pd.DataFrame
        Columns: ``period_id, frame_index, game_clock, wall_clock,
                  player_id, player_number, speed, x, y, z``.
    """
    dfs: list[pd.DataFrame] = []
    batch: list[dict] = []

    with Path(tracking_data_file).open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="reading tracks"):
            obj = json.loads(line)

            # skip non-tracking messages such as the "periodEnd" signal
            if "period" not in obj:
                continue

            period = obj["period"]
            frame = obj["frameIdx"]
            clock = obj["gameClock"]
            epoch_ms = obj["wallClock"]

            for side in ("homePlayers", "awayPlayers"):
                for pl in obj[side]:
                    xyz: Iterable[float] = pl["xyz"] or (np.nan, np.nan, np.nan)
                    batch.append(
                        {
                            "period_id": period,
                            "frame_index": frame,
                            "game_clock": clock,
                            "wall_clock": epoch_ms,
                            "player_id": pl["playerId"],
                            "player_number": pl["number"],
                            "speed": pl["speed"],
                            "x": xyz[0],
                            "y": xyz[1],
                            "z": xyz[2],
                        }
                    )

            # flush to DataFrame every chunk_size lines
            if len(batch) >= chunk_size:
                dfs.append(pd.DataFrame.from_records(batch))
                batch.clear()

    # final flush
    if batch:
        dfs.append(pd.DataFrame.from_records(batch))

    df = (
        pd.concat(dfs, ignore_index=True)
        .astype(
            {
                "period_id": "int8",
                "frame_index": "int32",
                "player_number": "int8",
            }
        )
        .sort_values(["period_id", "frame_index", "player_id"])
        .reset_index(drop=True)
    )
    return df


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame to Parquet using the 'pyarrow' engine. :contentReference[oaicite:11]{index=11}
    """
    df.to_parquet(path, engine="pyarrow", index=False)
