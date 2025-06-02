from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helper constants
# --------------------------------------------------------------------------- #
FPS = 25.0                # frames per second in the Second Spectrum feed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def total_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Metres covered by each player across the entire DataFrame.

    Parameters
    ----------
    df
        Tracking DataFrame with columns ``player_id, x, y`` and
        sorted by ``period_id, frame_index, player_id``.

    Returns
    -------
    pd.DataFrame
        Columns: ``player_id, distance_m``.
    """
    diffs = (
        df.groupby("player_id")[["x", "y"]]
        .diff()                   # step-to-step delta
        .fillna(0.0)
    )
    step_dist = np.hypot(diffs["x"], diffs["y"])
    out = (
        step_dist.groupby(df["player_id"])
        .sum()
        .rename("distance_m")
        .reset_index()
    )
    return out


def distance_by_speed_band(
    df: pd.DataFrame,
    bands: Iterable[tuple[float, float]],
) -> pd.DataFrame:
    """
    Split distance covered into user-defined absolute speed zones.

    Parameters
    ----------
    df
        Tracking DataFrame containing ``speed`` in m/s.
    bands
        Iterable of ``(lower, upper]`` tuples in m/s.  Example:
        ``[(0, 4), (4, 5.5), (5.5, 7), (7, np.inf)]``.

    Returns
    -------
    pd.DataFrame
        ``player_id`` plus one column per band, named ``"0-4"`` etc.
    """
    dt = 1.0 / FPS
    out_parts = []
    for lo, hi in bands:
        mask = (df["speed"] > lo) & (df["speed"] <= hi)
        # distance = speed × time
        dist = (df["speed"] * dt).where(mask, 0.0)
        col = f"{lo}-{'' if np.isinf(hi) else hi}"
        out_parts.append(
            dist.groupby(df["player_id"]).sum().rename(col)
        )

    out = pd.concat(out_parts, axis=1).reset_index()
    return out


def count_high_speed_accel(
    df: pd.DataFrame,
    *,
    min_speed: float = 5.5,      # m/s  ≈ 19.8 km/h
    min_accel: float = 3.0,      # m/s²
) -> pd.DataFrame:
    """
    Count explosive acceleration events per player.

    An event is logged when:
      1. `speed` ≥ `min_speed`
      2. instantaneous accel (`Δspeed / Δt`) ≥ `min_accel`
      3. we count the *rising edge* only (no double-count).

    Returns
    -------
    pd.DataFrame with ``player_id, n_accel``.
    """
    dt = 1.0 / FPS
    speed = df["speed"]
    accel = speed.groupby(df["player_id"]).diff() / dt
    qualifying = (
        (speed >= min_speed)
        & (accel >= min_accel)
    )

    # rising edges: True where current row qualifies and previous row didn't
    rising = qualifying & (~qualifying.groupby(df["player_id"]).shift(fill_value=False))

    out = (
        rising.groupby(df["player_id"])
        .sum()
        .astype("int16")
        .rename("n_accel")
        .reset_index()
    )
    return out
