"""Layered, fail-closed privacy for activity tracks (projected metres).

1. Trim the first and last `trim_m` metres of EVERY activity.
2. Suggest sensitive zones from ALL persistent start/end clusters (not only the most frequent one) and add user zones.
3. Zones are irregular polygons with a random offset (fresh per export, never stored), so a hole does not point at its centre.
4. Remove every point inside any zone (not only the ends): tracks that pass near home are cut too.
5. Leak check before export: no vertex or segment may touch a zone and nothing may come within `margin_m` of a
   cluster centre. If it fails, nothing is exported.
No address and no zone coordinate is written anywhere.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union


class PrivacyError(RuntimeError):
    pass


@dataclass
class PrivacyReport:
    activities_in: int = 0
    zones_suggested: int = 0
    zones_user: int = 0
    km_in: float = 0.0
    km_out: float = 0.0
    leak_check: str = "not run"
    notes: list = field(default_factory=list)
    centres: list = field(default_factory=list, repr=False)  # in memory only: never printed or written


def trim_ends(xy: np.ndarray, trim_m: float) -> np.ndarray:
    if len(xy) < 2:
        return xy
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 2 * trim_m:
        return xy[:0]
    keep = (s >= trim_m) & (s <= total - trim_m)
    return xy[keep]


def _clusters(points: np.ndarray, eps: float, min_pts: int):
    """Tiny DBSCAN-like grouping on a grid (no extra dependency). Returns list of (centre_xy, n)."""
    if len(points) == 0:
        return []
    cell = eps
    keys = np.floor(points / cell).astype(int)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    seen, out = set(), []
    for k in list(buckets):
        if k in seen:
            continue
        # flood fill over neighbouring non-empty cells
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.extend(buckets[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in buckets and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(comp) >= min_pts:
            out.append((points[comp].mean(axis=0), len(comp)))
    return out


def _irregular_zone(centre, radius, rng: random.Random, n=14):
    off = rng.uniform(0, 0.35 * radius)
    ang = rng.uniform(0, 2 * math.pi)
    cx, cy = centre[0] + off * math.cos(ang), centre[1] + off * math.sin(ang)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rng.uniform(-0.15, 0.15)
        r = radius * rng.uniform(0.85, 1.25)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return Polygon(pts).buffer(0)


def apply(tracks_xy: list, trim_m=300.0, zone_r=500.0, eps=180.0, min_share=0.04, user_zones_xy=(), margin_m=200.0,
          min_pts_floor=3):
    """tracks_xy: list of (N,2) arrays in metres. Returns (list of LineString pieces, report). Raises PrivacyError.
    min_pts_floor: fewest start/end points that make a zone; 2 means a single loop run already marks its start."""
    rep = PrivacyReport(activities_in=len(tracks_xy))
    rep.km_in = sum(float(np.hypot(*np.diff(t, axis=0).T).sum()) for t in tracks_xy if len(t) > 1) / 1000
    ends = np.array([p for t in tracks_xy if len(t) > 1 for p in (t[0], t[-1])])
    min_pts = max(min_pts_floor, int(min_share * len(tracks_xy)))
    found = _clusters(ends, eps, min_pts)
    rng = random.SystemRandom()  # fresh randomness every export; never stored
    centres = [c for c, _ in found]
    zones = [_irregular_zone(c, zone_r, rng) for c in centres]
    for (x, y, r) in user_zones_xy:
        zones.append(_irregular_zone((x, y), r, rng))
        centres.append(np.array([x, y]))
    rep.zones_suggested, rep.zones_user = len(found), len(user_zones_xy)
    rep.centres = [np.asarray(c) for c in centres]
    if not zones:
        rep.notes.append("no persistent start/end cluster found: only end trimming applied")
    forbidden = unary_union(zones) if zones else None
    pieces = []  # (activity_index, LineString)
    for i, t in enumerate(tracks_xy):
        t2 = trim_ends(t, trim_m)
        if len(t2) < 2:
            continue
        g = LineString(t2)
        if forbidden is not None:
            g = g.difference(forbidden)
        if g.is_empty:
            continue
        parts = [g] if isinstance(g, LineString) else list(getattr(g, "geoms", []))
        pieces.extend((i, p) for p in parts if isinstance(p, LineString) and p.length > 30)
    # leak check: fail closed
    if forbidden is not None:
        union = MultiLineString([p for _, p in pieces]) if pieces else None
        if union is not None and union.intersects(forbidden.buffer(-0.5)):
            raise PrivacyError("leak check failed: a track still touches a sensitive zone")
        for c in centres:
            if union is not None and union.distance(Point(c)) < margin_m:
                raise PrivacyError("leak check failed: a track comes too close to a sensitive point")
    rep.km_out = sum(p.length for _, p in pieces) / 1000
    rep.leak_check = "passed" if forbidden is not None else "no zones"
    return pieces, rep


def clip_to_extent(pieces, extent, margin_frac=0.0):
    """Only what is drawn is written: cut the pieces to the map box, so a PDF or SVG holds no geometry outside the
    visible map, where a clip path would hide it but not remove it. The renderers call it themselves."""
    x0, x1, y0, y1 = extent
    mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
    frame = Polygon([(x0 - mx, y0 - my), (x1 + mx, y0 - my), (x1 + mx, y1 + my), (x0 - mx, y1 + my)])
    out = []
    for i, line in pieces:
        cut = line.intersection(frame)
        for part in getattr(cut, "geoms", [cut]):
            if isinstance(part, LineString) and not part.is_empty and part.length > 0:
                out.append((i, part))
    return out
