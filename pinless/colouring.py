"""How the routes are coloured. The line weight always grows with how often a street was run; the colour can say more:

- heat: how often (the default), the same scale as the weight;
- single: one ink, only the weight changes;
- month: when in the year (local month of each run), a seasons palette;
- pace: how fast each run was (its average pace: robust to GPS noise);
- elevation: how high each point is (from the file's altitude).
"""
from __future__ import annotations

import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import LineString

from .localtime import local_start

MODES = ("heat", "single", "month", "pace", "elevation")
# January blue, spring green, summer gold, autumn red, December violet: one way through the year, so no two months
# share a colour; readable on light and dark paper
SEASONS = LinearSegmentedColormap.from_list("seasons", ["#3d6fb6", "#2e9d8f", "#7bb342", "#f2b134", "#e8590c",
                                                        "#b5332e", "#7b4b9e"])


def raw_values(mode, act_idx, segments, activities, tracks_xy):
    """One raw number per drawn segment (month 0-11, pace in s/km, elevation in m), or None for heat and single.
    act_idx: the activity of each segment (index into activities and tracks_xy). NaN where it is unknown."""
    n = len(segments)
    if mode in ("heat", "single") or n == 0:
        return None
    if mode == "month":
        month = np.array([(t.month - 1) if (t := local_start(a)) else np.nan for a in activities], dtype=float)
        return month[act_idx]
    if mode == "pace":
        pace = np.full(len(activities), np.nan)
        for i, (a, xy) in enumerate(zip(activities, tracks_xy)):
            km = float(np.hypot(*np.diff(np.asarray(xy), axis=0).T).sum()) / 1000 if len(xy) > 1 else 0.0
            if a.times and km > 0.5:
                pace[i] = (a.times[-1] - a.times[0]) / km
        return pace[act_idx]
    if mode == "elevation":
        out = np.full(n, np.nan)
        for i in np.unique(act_idx):
            a, xy = activities[i], np.asarray(tracks_xy[i])
            if not a.ele or len(xy) < 2:
                continue
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
            line = LineString(xy)
            sel = np.nonzero(act_idx == i)[0]
            mids = shapely.points(segments[sel].mean(axis=1))  # pieces lie on their own track: an exact position
            out[sel] = np.interp(shapely.line_locate_point(line, mids), cum, np.asarray(a.ele, dtype=float))
        return out
    raise ValueError(f"unknown colouring {mode!r}")


def normalise(mode, raws):
    """[0, 1] colour values for each map's raw values (one shared scale for an atlas), and the legend's numbers.
    Unknown values sit in the middle of the scale."""
    if raws is None or all(r is None for r in raws):
        return [None] * (len(raws) if raws is not None else 1), {"mode": mode}
    if mode == "month":
        return [np.nan_to_num(r / 11.0, nan=0.5) for r in raws], {"mode": mode}
    every = np.concatenate([r[np.isfinite(r)] for r in raws if r is not None and np.isfinite(r).any()] or [[0.0]])
    lo, hi = np.percentile(every, [3, 97]) if len(every) else (0.0, 1.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    vals = []
    for r in raws:
        v = np.clip((r - lo) / (hi - lo), 0, 1)
        if mode == "pace":  # faster runs take the strong end of the palette
            v = 1 - v
        vals.append(np.nan_to_num(v, nan=0.5))
    return vals, {"mode": mode, "lo": float(lo), "hi": float(hi)}


def fmt_pace(sec_per_km):
    m, s = divmod(int(round(sec_per_km)), 60)
    return f"{m}:{s:02d}/km"
