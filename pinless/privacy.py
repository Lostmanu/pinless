"""Layered, fail-closed privacy for activity tracks (projected metres).

1. Trim the first and last `trim_m` metres of EVERY activity.
2. Suggest sensitive zones from ALL persistent start/end clusters (not only the most frequent one) and add user zones.
3. Zones are irregular polygons with a random offset (fresh per export, never stored), so a hole does not point at its centre.
4. Remove every point inside any zone (not only the ends): tracks that pass near home are cut too.
5. Leak check before export: no vertex or segment may touch a zone and nothing may come within `margin_m` of a
   cluster centre. If it fails, nothing is exported.
A poster for two (pair_places) learns each person's places with their own threshold, adds the places where their
ends meet, and cuts BOTH people's lines with one forbidden area drawn once, one zone per place.
No address and no zone coordinate is written anywhere.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import shapely
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


def _clusters(points: np.ndarray, eps: float, min_pts: int, accept=None):
    """Tiny DBSCAN-like grouping on a grid (no extra dependency). Returns list of (centre_xy, n).
    accept(indices) -> bool decides whether a group of points is a place (default: at least min_pts of them)."""
    ok = accept or (lambda idx: len(idx) >= min_pts)
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
        stack, comp, cells = [k], [], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.extend(buckets[c])
            cells.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in buckets and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if ok(comp):
            out.append((points[comp].mean(axis=0), len(comp)))
            # a chain of cells (home, a few starts on the way, a busy park) can drag that mean far from any of them:
            # every busy neighbourhood of 3 x 3 cells is a centre too, so home keeps a zone of its own even when its
            # starts straddle a cell edge
            block = {c: [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in buckets.get((c[0] + dx, c[1] + dy), ())]
                     for c in cells}
            for c in sorted(cells, key=lambda c: (-len(block[c]), c)):
                if ok(block[c]):
                    centre = points[block[c]].mean(axis=0)
                    if all(np.hypot(*(centre - o)) > eps for o, _ in out):
                        out.append((centre, len(block[c])))
            _uncovered_places(points, buckets, cells, eps, ok, out)
    return out


def _uncovered_places(points, buckets, cells, eps, ok, out):
    """A smaller place next to a bigger one (a gym 450 m from home): its only block also holds the home's ends, so that
    block's mean lies at home and is skipped. Look again at the ends farther than eps from every centre found so far:
    any block of them that is a place on its own gets a centre too (appended to out), until none is left."""
    tried = set()
    while True:
        centres = np.array([o for o, _ in out])
        left = {}
        for c in cells:
            idx = np.asarray(buckets[c])
            gap = np.hypot(*(points[idx][:, None, :] - centres[None, :, :]).transpose(2, 0, 1)).min(axis=1)
            if (gap > eps).any():
                left[c] = idx[gap > eps].tolist()
        if not left:
            return
        block = {c: [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in left.get((c[0] + dx, c[1] + dy), ())]
                 for c in cells}
        for c in sorted(cells, key=lambda c: (-len(block[c]), c)):
            key = (c, tuple(block[c]))
            if not block[c] or key in tried:
                continue
            tried.add(key)
            if ok(block[c]):
                centre = points[block[c]].mean(axis=0)
                if all(np.hypot(*(centre - o)) > eps for o, _ in out):
                    out.append((centre, len(block[c])))
                    break
        else:
            return


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


def _outside(xy, forbidden):
    """The runs of a track outside the forbidden area, in track order, as coordinate arrays. Cut segment by segment:
    an overlay of the whole line would also split it wherever it crosses itself, and the short bits between two
    crossings (a street run out and back) would then be dropped as stubs."""
    segs = shapely.linestrings(np.stack([xy[:-1], xy[1:]], axis=1))
    hit = shapely.intersects(segs, forbidden)
    runs, cur = [], []

    def flush():
        if len(cur) > 1:
            runs.append(np.asarray(cur))
        cur.clear()

    for k in range(len(segs)):
        a, b = xy[k], xy[k + 1]
        if not hit[k]:
            if not cur:
                cur.append(a)
            cur.append(b)
            continue
        rest = segs[k].difference(forbidden)
        parts = [np.asarray(p.coords) for p in getattr(rest, "geoms", [rest]) if not p.is_empty and p.length > 0]
        parts = [p if np.hypot(*(p[0] - a)) <= np.hypot(*(p[-1] - a)) else p[::-1] for p in parts]
        for p in sorted(parts, key=lambda p: np.hypot(*(p[0] - a))):
            if not (cur and np.hypot(*(p[0] - cur[-1])) < 1e-6):
                flush()
                cur.append(p[0])
            cur.append(p[-1])
        if not (cur and np.hypot(*(cur[-1] - b)) < 1e-6):
            flush()
    flush()
    return runs


def find_places(tracks_xy, eps=180.0, min_share=0.04, min_pts_floor=3, n_runs=None):
    """The persistent start/end clusters of the tracks: [(centre_xy, n), ...] (in memory only).
    n_runs: how many runs the threshold is a share of (default: all the tracks; a poster for two leaves out the
    activities found in both exports, so copies of the other person's runs never raise it)."""
    ends = np.array([p for t in tracks_xy if len(t) > 1 for p in (t[0], t[-1])])
    return _clusters(ends, eps, _threshold(len(tracks_xy) if n_runs is None else n_runs, min_share, min_pts_floor))


def _threshold(n_runs, min_share, min_pts_floor):
    return max(min_pts_floor, int(min_share * n_runs))


def pooled_places(groups, eps=180.0, min_share=0.04, min_pts_floor=2, n_runs=None):
    """Places where the ends of several people's tracks meet: a cluster of everyone's start/end points is a place when
    a/tA + b/tB + ... >= 1, where a is how many of them are the first person's and tA that person's own threshold
    (as in find_places). It never accepts less evidence than one full personal threshold, and it catches a place
    where each person ends too few times to make it a place alone (a union of the tracks would need tA + tB).
    groups: one list of tracks per person; n_runs: as in find_places, one per person. [(centre_xy, n), ...], in memory
    only."""
    ends, labels = [], []
    for g, tracks in enumerate(groups):
        for t in tracks:
            if len(t) > 1:
                ends += [t[0], t[-1]]
                labels += [g, g]
    if not ends:
        return []
    labels = np.asarray(labels)
    n_runs = [len(tracks) for tracks in groups] if n_runs is None else n_runs
    thr = [_threshold(n, min_share, min_pts_floor) for n in n_runs]

    def accept(idx):
        n = np.bincount(labels[idx], minlength=len(groups))
        return sum(Fraction(int(n[g]), thr[g]) for g in range(len(groups))) >= 1

    return _clusters(np.asarray(ends, dtype=float), eps, 0, accept)


def merge_places(centres, eps=180.0, zone_r=500.0, grow=2.5):
    """One zone per place: centres closer than eps (single linkage) become one zone at their mean, with radius
    zone_r + grow * (the farthest member's distance from the mean). [(x, y, r), ...] in order of each group's first
    member. Why 2.5: _irregular_zone keeps at least 0.4411 r from its own centre (an offset of 0.35 r and its widest
    edge at 0.7911 r), so a member d from the mean keeps 0.4411 (zone_r + grow d) - d >= 0.4411 zone_r when
    grow >= 1 / 0.4411; with 2.0 it would fall under the 200 m leak margin once d > 174 m."""
    pts = [np.asarray(c, dtype=float) for c in centres]
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(*(pts[i] - pts[j])) < eps:
                a, b = find(i), find(j)
                parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(groups, key=lambda r: min(groups[r])):
        members = np.array([pts[i] for i in groups[root]])
        mean = members.mean(axis=0)
        d_max = float(np.hypot(*(members - mean).T).max())
        out.append((float(mean[0]), float(mean[1]), float(zone_r + grow * d_max)))
    return out


def pair_places(groups, zone_r=500.0, eps=180.0, min_share=0.04, min_pts_floor=2, n_runs=None):
    """The private places of a poster for several people: each person's own places (their own threshold, so a quiet
    partner's home or gym is found), the places where their ends meet (pooled_places), one zone per place
    (merge_places). n_runs: each person's number of runs for their threshold, without the activities found in both
    exports (default: len of each group). Returns (zones [(x, y, r)], witnesses [every original centre, for the leak
    check], counts). Nothing is written anywhere."""
    n_runs = [len(g) for g in groups] if n_runs is None else list(n_runs)
    per = [find_places(g, eps, min_share, min_pts_floor, n) for g, n in zip(groups, n_runs)]
    pooled = pooled_places(groups, eps, min_share, min_pts_floor, n_runs)
    own = [c for found in per for c, _ in found]
    centres = own + [c for c, _ in pooled]
    zones = merge_places(centres, eps, zone_r)
    counts = {"per": [len(found) for found in per],
              "pooled_new": sum(1 for c, _ in pooled if all(np.hypot(*(c - o)) > eps for o in own)),
              "zones": len(zones)}
    # how many of the places counted above (each person's, and the new shared ones) were joined into another's zone
    counts["merged"] = sum(counts["per"]) + counts["pooled_new"] - counts["zones"]
    return zones, [np.asarray(c) for c in centres], counts


def apply(tracks_xy: list, trim_m=300.0, zone_r=500.0, eps=180.0, min_share=0.04, user_zones_xy=(), margin_m=200.0,
          min_pts_floor=3, places_xy=None, witnesses_xy=(), rng=None):
    """tracks_xy: list of (N,2) arrays in metres. Returns (list of LineString pieces, report). Raises PrivacyError.
    min_pts_floor: fewest start/end points that make a zone; 2 means a single loop run already marks its start.
    places_xy: the private places as (x, y, r), instead of learning them from these tracks (pair_places), drawn as
    one forbidden area that cuts every track; witnesses_xy: points that no line may come within margin_m of.
    rng: for tests only; by default the zones are fresh for every export and never stored."""
    rep = PrivacyReport(activities_in=len(tracks_xy))
    rep.km_in = sum(float(np.hypot(*np.diff(t, axis=0).T).sum()) for t in tracks_xy if len(t) > 1) / 1000
    rng = rng or random.SystemRandom()  # fresh randomness every export; never stored
    if places_xy is None:
        found = find_places(tracks_xy, eps, min_share, min_pts_floor)
        centres = [c for c, _ in found]
        zones = [_irregular_zone(c, zone_r, rng) for c in centres]
        rep.zones_suggested = len(found)
    else:
        centres = [np.array([x, y], dtype=float) for x, y, _ in places_xy]
        zones = [_irregular_zone((x, y), r, rng) for x, y, r in places_xy]
        rep.zones_suggested = len(places_xy)
    for (x, y, r) in user_zones_xy:
        zones.append(_irregular_zone((x, y), r, rng))
        centres.append(np.array([x, y]))
    centres += [np.asarray(w, dtype=float) for w in witnesses_xy]
    rep.zones_user = len(user_zones_xy)
    rep.centres = [np.asarray(c) for c in centres]
    if not zones:
        rep.notes.append("no persistent start/end cluster found: only end trimming applied")
    forbidden = unary_union(zones) if zones else None
    if forbidden is not None:
        shapely.prepare(forbidden)
    pieces = []  # (activity_index, LineString)
    for i, t in enumerate(tracks_xy):
        t2 = trim_ends(t, trim_m)
        if len(t2) < 2:
            continue
        parts = [t2] if forbidden is None else _outside(t2, forbidden)
        pieces.extend((i, p) for p in map(LineString, parts) if p.length > 30)
    # leak check: fail closed
    if forbidden is not None or centres:
        union = MultiLineString([p for _, p in pieces]) if pieces else None
        if union is not None and forbidden is not None and union.intersects(forbidden.buffer(-0.5)):
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
    out = []
    for i, line in pieces:
        # cut by the rectangle alone, without noding the line: a course that crosses itself (a stop where the fix
        # wandered for minutes crosses itself thousands of times) keeps its own vertices, one piece where it stays
        # in the map, instead of breaking at every crossing
        cut = shapely.clip_by_rect(line, x0 - mx, y0 - my, x1 + mx, y1 + my)
        for part in getattr(cut, "geoms", [cut]):
            if isinstance(part, LineString) and not part.is_empty and part.length > 0:
                out.append((i, part))
    return out
