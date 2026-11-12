"""Bragging stats from the activities themselves (computed on the FULL tracks, before privacy cuts; only aggregates
are printed, never coordinates)."""
from __future__ import annotations

import bisect
import datetime as dt
import math
from collections import Counter

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

from .ingest import distance_scale
from .localtime import local_start


def _cum(xy):
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


FOOT = {"Run", "TrailRun", "VirtualRun", "Walk", "Hike", "Snowshoe"}
GATE = {True: (2.0, 6.5), False: (3.0, 25.0)}  # on foot or not: (times the median speed, and at least m/s)
WANDER_M = 5.0  # a fix wanders this far round where it is, whatever the time between two
TRUST = 0.9  # the jumps of the fixes are a few percent of a track: a gate that drops more is fighting a broken clock


def measured_along(xy, times=None, kind=None):
    """Metres along a track at each of its fixes, counting only the fixes that can be reached from the last one
    counted at the activity's own pace: at most GATE times its median speed, and never less than GATE's floor, plus
    WANDER_M. A watch among tall buildings jumps 20 to 60 m away and back within seconds; summed along the track,
    those jumps were metres run, and the kilometres with them printed faster than they were (km 10 and 12 of a half
    marathon in Madrid, by 10 and 5 s). A fix in between is placed by its order between the two counted round it.
    The time to reach a fix adds only the steps of the clock that go forward: a clock that jumps back once (an hour
    of summer time written wrong) would otherwise hold every fix after it out of reach for that long. Without a
    clock, or when the gate would keep less than TRUST of the sum of the steps: the sum of the steps."""
    xy = np.asarray(xy, dtype=float)
    raw = _cum(xy)
    if times is None or len(times) != len(xy) or len(xy) < 3:
        return raw
    step, dt = np.hypot(*np.diff(xy, axis=0).T), np.diff(np.asarray(times, dtype=float))
    speed = step[dt > 0] / dt[dt > 0]
    moving = speed[speed > 0.5]
    if not len(moving):
        return raw
    times_median, floor = GATE[kind in FOOT]
    vmax = max(times_median * float(np.median(moving)), floor)
    clock = np.concatenate([[0.0], np.cumsum(np.maximum(dt, 0.0))])
    keep = [0]
    for i in range(1, len(xy)):
        j = keep[-1]
        if math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]) <= vmax * (clock[i] - clock[j]) + WANDER_M:
            keep.append(i)
    keep = np.asarray(keep)
    gated = np.interp(np.arange(len(xy)), keep, _cum(xy[keep]))
    return gated if gated[-1] >= TRUST * raw[-1] else raw


def best_effort(xy, secs, dist_m, cum=None):
    """Fastest time (s) to cover dist_m inside one activity (two-pointer over cumulative distance: cum, by default
    the sum of the steps)."""
    if secs is None or len(xy) < 2:
        return None
    s = _cum(xy) if cum is None else np.asarray(cum, dtype=float)
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
            fav_min_dist_m=1000.0, keep_streets=False):
    """activities: selected ingest.Activity objects; tracks_xy: their projected FULL tracks (same order).
    Street stats use ONLY the privacy-cut pieces (private_pieces_xy), never the full tracks: the most-run street of a
    full track is usually the runner's own street. keep_streets: also return the set of street names run
    ("street_names"), in memory only, to count the streets two people both ran."""
    # distances as Strava recorded them where the export says (ingest.distance_scale): a sum of the steps between fixes
    # adds their noise, a few percent on a watch in a city
    cums = [measured_along(xy, a.times, a.type) for a, xy in zip(activities, tracks_xy)]  # no jumps of the fixes
    own = [float(c[-1]) for c in cums]
    scale = [distance_scale(a, m) for a, m in zip(activities, own)]
    km = [m * f / 1000 for m, f in zip(own, scale)]
    secs_total = sum((a.times[-1] - a.times[0]) for a in activities if a.times)
    b5 = [best_effort(xy, a.times, 5000 / f, c) for a, xy, f, c in zip(activities, tracks_xy, scale, cums)]
    b10 = [best_effort(xy, a.times, 10000 / f, c) for a, xy, f, c in zip(activities, tracks_xy, scale, cums)]
    b5 = min([x for x in b5 if x], default=None)
    b10 = min([x for x in b10 if x], default=None)
    local = [local_start(a) for a in activities]  # files store UTC: days and hours are counted where you ran
    dates = [t.date() for t in local if t]
    early = sum(1 for t in local if t and t.hour < 8)
    out = {
        "n": len(activities), "km": sum(km), "hours": secs_total / 3600, "longest_km": max(km) if km else 0,
        "best5k_s": b5, "best10k_s": b10, "active_days": len(set(dates)), "streak": longest_streak(dates),
        "early_runs": early,
    }
    if edges_gdf is not None and private_pieces_xy is not None:
        sc = street_counts(private_pieces_xy, edges_gdf, groups=piece_groups)
        out["streets"] = len(sc)
        if keep_streets:
            out["street_names"] = set(sc)
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


def _clock(act, xy):
    """(absolute seconds, x, y, cumulative metres) with a strictly increasing clock, or None without a clock.
    A naive start is taken as UTC, as localtime does. Repeated stamps (a paused watch) are dropped."""
    if (act.start is None or act.times is None or len(act.times) != len(xy) or len(xy) < 2
            or any(t is None for t in act.times)):
        return None
    start = act.start if act.start.tzinfo else act.start.replace(tzinfo=dt.timezone.utc)
    t = np.maximum.accumulate(start.timestamp() + np.asarray(act.times, dtype=float))
    keep = np.concatenate([[True], np.diff(t) > 0])
    if keep.sum() < 2:
        return None
    xy = np.asarray(xy, dtype=float)
    return t[keep], xy[keep, 0], xy[keep, 1], _cum(xy)[keep]


def together(acts_a, tracks_a, acts_b, tracks_b, step_s=30.0, min_s=600.0, min_frac=0.5, med_m=75.0, near_m=150.0,
             near_frac=0.8):
    """Runs two people ran side by side, from their clocks and FULL tracks (in memory only): an activity of each that
    overlap in time for at least max(min_s, min_frac of the shorter one), with a median gap of med_m or less and
    at least near_frac of the shared time within near_m. Matched one to one, longest overlap first.
    Returns {"runs", "km" (the mean of the two along-track distances over the shared time), "known" (both have
    some clock), "no_clock" (activities without a clock: (first person's, second person's))}."""
    ca = [_clock(a, xy) for a, xy in zip(acts_a, tracks_a)]
    cb = [_clock(b, xy) for b, xy in zip(acts_b, tracks_b)]
    sa = [distance_scale(a, float(_cum(np.asarray(xy, dtype=float))[-1])) for a, xy in zip(acts_a, tracks_a)]
    sb = [distance_scale(b, float(_cum(np.asarray(xy, dtype=float))[-1])) for b, xy in zip(acts_b, tracks_b)]
    out = {"runs": 0, "km": 0.0, "known": any(c is not None for c in ca) and any(c is not None for c in cb),
           "no_clock": (sum(c is None for c in ca), sum(c is None for c in cb))}
    bs = sorted((c[0][0], j) for j, c in enumerate(cb) if c is not None)
    if not out["known"] or not bs:
        return out
    starts = [s for s, _ in bs]
    longest_b = max(cb[j][0][-1] - cb[j][0][0] for _, j in bs)
    pairs = []
    for i, a in enumerate(ca):
        if a is None:
            continue
        a0, a1 = a[0][0], a[0][-1]
        for _, j in bs[bisect.bisect_left(starts, a0 - longest_b):bisect.bisect_left(starts, a1)]:
            b = cb[j]
            lo, hi = max(a0, b[0][0]), min(a1, b[0][-1])
            ov = hi - lo
            if ov < max(min_s, min_frac * min(a1 - a0, b[0][-1] - b[0][0])):
                continue
            t = np.append(np.arange(lo, hi, step_s), hi)
            d = np.hypot(np.interp(t, a[0], a[1]) - np.interp(t, b[0], b[1]),
                         np.interp(t, a[0], a[2]) - np.interp(t, b[0], b[2]))
            if np.median(d) <= med_m and np.mean(d <= near_m) >= near_frac:
                along = [(np.interp(hi, c[0], c[3]) - np.interp(lo, c[0], c[3])) * f
                         for c, f in ((a, sa[i]), (b, sb[j]))]
                pairs.append((ov, i, j, float(np.mean(along))))
    used_a, used_b = set(), set()
    for ov, i, j, m in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out["runs"] += 1
        out["km"] += m / 1000
    return out


def fmt_time(s):
    if not s:
        return "—"
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
