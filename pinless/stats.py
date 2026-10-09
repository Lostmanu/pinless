"""Bragging stats from the activities themselves (computed on the FULL tracks, before privacy cuts; only aggregates
are printed, never coordinates)."""
from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree


def _cum(xy):
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def best_effort(xy, secs, dist_m):
    """Fastest time (s) to cover dist_m inside one activity (two-pointer over cumulative distance)."""
    if secs is None or len(xy) < 2:
        return None
    s = _cum(xy)
    t = np.asarray(secs, dtype=float)
    if s[-1] < dist_m:
        return None
    best, j = None, 0
    for i in range(len(s)):
        while j < len(s) and s[j] - s[i] < dist_m:
            j += 1
        if j >= len(s):
            break
        d = t[j] - t[i]
        if d > 0 and (best is None or d < best):
            best = d
    return best


def street_counts(tracks_xy, edges_gdf, step_m=40.0, tol_m=14.0, groups=None):
    """How many different ACTIVITIES ran each named street (nearest named edge within tol_m, sampled every step_m).
    `groups` gives the activity id of each element of tracks_xy (privacy cuts split one activity into pieces);
    a street counts once per activity."""
    named = edges_gdf[edges_gdf["name"].notna()] if "name" in edges_gdf.columns else edges_gdf.iloc[0:0]
    geoms = list(named.geometry.values)
    names = [n[0] if isinstance(n, list) else n for n in named["name"].values]
    tree = STRtree(geoms)
    per_act = {}
    for gi, xy in enumerate(tracks_xy):
        if len(xy) < 2:
            continue
        s = _cum(xy)
        seen = per_act.setdefault(groups[gi] if groups is not None else gi, set())
        for d in np.arange(0, s[-1], step_m):
            k = min(np.searchsorted(s, d), len(xy) - 1)
            p = Point(xy[k])
            hit = {names[int(h)] for h in tree.query_nearest(p, max_distance=tol_m)}
            if len(hit) == 1:  # equidistant streets (a sample on a shared corner) say nothing: skip, whatever the edge order
                seen.add(hit.pop())
    per_street = Counter()
    for seen in per_act.values():
        per_street.update(seen)
    return per_street


def longest_streak(dates):
    days = sorted(set(dates))
    best = cur = 1 if days else 0
    for a, b in zip(days, days[1:]):
        cur = cur + 1 if (b - a).days == 1 else 1
        best = max(best, cur)
    return best


def compute(activities, tracks_xy, edges_gdf=None, private_pieces_xy=None, piece_groups=None, sensitive_xy=(),
            fav_min_dist_m=1000.0):
    """activities: selected ingest.Activity objects; tracks_xy: their projected FULL tracks (same order).
    Street stats use ONLY the privacy-cut pieces (private_pieces_xy), never the full tracks: the most-run street of a
    full track is usually the runner's own street."""
    km = [float(_cum(xy)[-1]) / 1000 for xy in tracks_xy]
    secs_total = sum((a.times[-1] - a.times[0]) for a in activities if a.times)
    b5 = [best_effort(xy, a.times, 5000) for a, xy in zip(activities, tracks_xy)]
    b10 = [best_effort(xy, a.times, 10000) for a, xy in zip(activities, tracks_xy)]
    b5 = min([x for x in b5 if x], default=None)
    b10 = min([x for x in b10 if x], default=None)
    dates = [a.start.date() for a in activities if a.start]
    early = sum(1 for a in activities if a.start and a.start.hour < 8)
    out = {
        "n": len(activities), "km": sum(km), "hours": secs_total / 3600, "longest_km": max(km) if km else 0,
        "best5k_s": b5, "best10k_s": b10, "active_days": len(set(dates)), "streak": longest_streak(dates),
        "early_runs": early,
    }
    if edges_gdf is not None and private_pieces_xy is not None:
        sc = street_counts(private_pieces_xy, edges_gdf, groups=piece_groups)
        out["streets"] = len(sc)
        # the favourite street must not point at a sensitive place: skip streets within fav_min_dist_m of any zone
        far = set()
        if len(sensitive_xy):
            named = edges_gdf[edges_gdf["name"].notna()]
            for nm, geom in zip(named["name"].values, named.geometry.values):
                nm = nm[0] if isinstance(nm, list) else nm
                if any(geom.distance(Point(c)) < fav_min_dist_m for c in sensitive_xy):
                    far.add(nm)  # «far» = forbidden: too close to a sensitive place
        ok = [(nm, c) for nm, c in sc.most_common() if nm not in far]
        out["fav_street"], out["fav_street_n"] = ok[0] if ok else ("", 0)
    return out


def fmt_time(s):
    if not s:
        return "—"
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
