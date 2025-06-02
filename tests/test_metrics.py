import numpy as np
import pandas as pd
from clubbrugge.metrics import (
    total_distance,
    distance_by_speed_band,
    count_high_speed_accel,
)

# create a tiny synthetic 25-Hz clip for two players, 10 frames
FPS = 25.0
dt = 1.0 / FPS
frames = 10
players = [101, 202]

data = []
for pid in players:
    for f in range(frames):
        # move 1 m horizontally each frame => 25 m s⁻¹ constant speed
        x = f * 1.0
        y = 0.0
        data.append(
            dict(
                period_id=1,
                frame_index=f,
                player_id=pid,
                speed=1.0 / dt,       # 25 m/s
                x=x,
                y=y,
            )
        )

df_test = pd.DataFrame(data)

def test_total_distance_simple():
    res = total_distance(df_test)
    # 9 steps * 1 m per step = 9 m
    assert np.isclose(res.loc[res["player_id"] == 101, "distance_m"].item(), 9.0)

def test_speed_band_split():
    bands = [(0, 4), (4, np.inf)]
    res = distance_by_speed_band(df_test, bands)

    # all distance should appear in the second band (open-ended upper bound)
    assert res.loc[res["player_id"] == 202, "0-4"].item() == 0.0

    # find whatever name the function created for the (4, inf] band
    col = [c for c in res.columns if c.startswith("4-")][0]
    assert res.loc[res["player_id"] == 202, col].item() > 0.0


def test_high_speed_accel():
    # synthetic clip uses constant speed → should yield zero events
    res = count_high_speed_accel(df_test, min_speed=5.5, min_accel=3.0)
    assert res["n_accel"].sum() == 0
