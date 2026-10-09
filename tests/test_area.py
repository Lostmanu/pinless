"""The automatic area is where you run most: a holiday far away must not pull the poster off your city."""
import datetime as dt

import numpy as np

from pinless import area
from pinless.ingest import Activity


def _loop(lat, lon, r_deg, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return list(zip(lon + r_deg * np.cos(t), lat + r_deg * np.sin(t)))


def test_auto_bbox_ignores_a_far_away_holiday():
    acts = [Activity(id=str(i), type="Run", start=dt.datetime(2025, 1, 1), lonlat=_loop(40.41, -3.70, 0.02 + 0.001 * i),
                     source="x") for i in range(30)]
    acts += [Activity(id=f"h{i}", type="Run", start=dt.datetime(2025, 8, 1), lonlat=_loop(39.47, -0.33, 0.01), source="y")
             for i in range(4)]
    s, w, n, e = area.auto_bbox(acts)
    assert s < 40.41 < n and w < -3.70 < e
    assert e < -1.0, "the Valencia holiday must not stretch the area east"
    assert (n - s) < 0.3 and (e - w) < 0.3
