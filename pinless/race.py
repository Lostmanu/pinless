"""--layout race: one activity as a finisher print, with the privacy rules a single course needs.

What is drawn is ONE uninterrupted stretch of the run, and every number on the print (distance, time, pace, profile,
climb, splits) describes that stretch, or, for a public course nothing was cut from, the whole course.

- Private places: the start/end clusters learned from every other activity of the same type (all years); the start
  and end points of other activities near this run's own start (end) when two or more activities put them there (a
  new home, a holiday flat: too few runs to be learned from years of history); and every --zone.
- A run STARTS (ENDS) at a private place when a point of its first (last) trim metres of movement (privacy.trim_span,
  and two thirds of them in a straight line from the start), the part never drawn, comes within 1.5 zone radii of one.
  It is refused by default: one course drawn from there points at it. With
  allow_private_ends the print keeps the longest stretch that stays farther than a random 1.6-2.6 radii from those
  places (always beyond the reach of any zone, wherever the run comes back to them), shortened by a random 200-800 m
  at each end that a place lies beyond, and numbers only that stretch.
- A course that only passes a private place after its first trim metres is a passing course: it is drawn whole,
  trimmed at both ends, since a gap in a single public course would point at the very place it hides. A --zone is
  always kept away from, as above.
- Stray GPS fixes at the ends are dropped first (ingest.clean), for this run and for the others. The city in the
  title is looked up from the middle of the drawn stretch.
- A loop from a place no other run starts from (a first run from a hotel) looks like a race start: it is drawn whole,
  and the log says so.
- The title is the city unless the user gives one: an activity name can say where a run starts.
"""
from __future__ import annotations

import datetime as dt
import random
import re

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import substring

from . import area, ingest, privacy, render, render_race, stats
from . import relief
from .localtime import local_start
from .render_race import along

REACH = 1.6  # an irregular zone reaches at most 1.25 r around a centre offset by up to 0.35 r (privacy._irregular_zone)
KEEP = (1.6, 2.6)  # keep-away radius around a place, in zone radii, drawn once per export
EXTRA = (200.0, 800.0)  # random extra cut (m) at each end of the stretch that a private place lies beyond
MIN_M = 1000.0  # shortest stretch worth a print
DENSE_M = 10.0  # the track is checked every 10 m, not only at its own (possibly sparse) points
LOOP_M = 250.0  # a run that ends this close to its start is a loop


class PrivateEnds(Exception):
    """The run starts or ends at a private place and allow_private_ends was not given."""


class AcrossDateline(ValueError):
    """The map crosses the 180th meridian: one box for OpenStreetMap would wrap round the globe."""


def _km(a):
    xy = np.asarray(a.lonlat, dtype=float)
    if len(xy) < 2:
        return 0.0
    lat = np.radians(xy[:, 1])
    dx = np.radians(np.diff(xy[:, 0])) * np.cos(lat[:-1])
    return float(6371.0 * np.hypot(dx, np.diff(lat)).sum())


def pick(activities, how="longest"):
    """longest (default), latest, a date YYYY-MM-DD (the longest run that local day) or an activity id."""
    acts = [a for a in activities if len(a.lonlat) > 1]
    if not acts:
        return None
    if how == "longest":
        return max(acts, key=_km)
    if how == "latest":
        return max(acts, key=lambda a: local_start(a) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", how):
        day = dt.date.fromisoformat(how)  # the CLI has checked it is a real date
        same = [a for a in acts if (t := local_start(a)) is not None and t.date() == day]
        return max(same, key=_km) if same else None
    return next((a for a in acts if a.id == how), None)


def _windows(dd, span):
    """(first, last): the points of the dense track before and after the stretch left by the trim (span:
    privacy.trim_span, in metres along the track), the parts never drawn."""
    lo, hi = span
    return dd <= lo, dd >= hi


def _places(dense, dd, others, span, zone_r, user):
    """(private places (x, y, r) that matter to this run, how many were learned from the others).

    Learned: the start/end clusters of all the other activities. Local: when the start or end points of two or more
    other activities lie within 1.5 r of this run's untrimmed start (end), every one of those points is a place
    (a new home or a holiday flat, too new to be learned from years of history). Then every --zone."""
    learned = [(float(c[0]), float(c[1]), float(zone_r)) for c, _ in privacy.find_places(others)]
    places = learned + list(user)
    tracks = [t for t in others if len(t) > 1]
    if tracks:
        ends = np.array([p for t in tracks for p in (t[0], t[-1])])
        owner = np.repeat(np.arange(len(tracks)), 2)
        pts = shapely.points(ends)
        for window in _windows(dd, span):
            w = dense[window]
            near = shapely.distance(pts, LineString(w) if len(w) > 1 else Point(w[0])) <= 1.5 * zone_r
            if len(set(owner[near].tolist())) >= 2:
                places += [(float(x), float(y), float(zone_r)) for x, y in ends[near]]
    return places, len(learned)


def _private_ends(dense, dd, places, span):
    """(head, tail): the private places where the track starts and ends: a point of what the trim cuts from its start
    (end), the part never drawn, within 1.5 r of one. Not only the first fix: a watch that finds the satellites a little
    down the street, or a loop round the block, still starts at home. dense: the track every 10 m."""
    head, tail = [], []
    first, last = _windows(dd, span)
    for x, y, r in places:
        near = np.hypot(dense[:, 0] - x, dense[:, 1] - y) <= 1.5 * r
        if near[first].any():
            head.append((x, y, r))
        if near[last].any():
            tail.append((x, y, r))
    return head, tail


def _runs(ok):
    """Maximal runs of True in a boolean array, as (first index, last index)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2] - 1))


def own_ends(track, a):
    """--private-ends: the activity's first and last fix as private places of the usual radius, (x, y, r) in the
    track's own metres. A single file that starts at home has no other activities to learn that home from, and
    asking for a --zone there means typing the coordinates of the door; this takes them from the track, in memory."""
    if not getattr(a, "private_ends", False) or len(track) < 2:
        return []
    return [(float(track[i][0]), float(track[i][1]), float(a.zone_radius)) for i in (0, -1)]


def cut(track, others, trim_m, zone_r, user_zones=(), allow_private_ends=False, rng=None):
    """Privacy for one end-cleaned track. others: every OTHER activity of its type, projected and cleaned.
    Returns (pieces, report, places learned from the others, (start_m, end_m)): at most one piece, and the
    along-track stretch the numbers describe. Raises PrivateEnds, or privacy.PrivacyError when the result breaks its
    own guarantee."""
    xy = np.asarray(track, dtype=float)
    d = along(xy)
    dense = np.asarray(shapely.segmentize(LineString(xy), DENSE_M).coords)
    dd = along(dense)
    user = [tuple(map(float, z)) for z in user_zones]
    # the trim in metres of movement (privacy.trim_span): a watch left running at the door does not use it up; when
    # nothing is left, the whole track counts as its two ends and nothing is drawn
    span = privacy.trim_span(xy, trim_m) or (float(dd[-1]) / 2, float(dd[-1]) / 2)
    places, learned = _places(dense, dd, others, span, zone_r, user)
    head, tail = _private_ends(dense, dd, places, span)
    if (head or tail) and not allow_private_ends:
        raise PrivateEnds("this run starts or ends at a private place (where your other runs start or end: home, "
                          "work, a regular meeting point) or at a --zone")
    keep_off = list(dict.fromkeys(user + head + tail))
    rng = rng or random.SystemRandom()  # fresh every export: no drawn end may sit a fixed distance from a place
    keep = rng.uniform(*KEEP)
    far = np.ones(len(dense), dtype=bool)
    for x, y, r in keep_off:
        far &= np.hypot(dense[:, 0] - x, dense[:, 1] - y) > r * keep
    if far.all():  # a public course: drawn whole (trimmed at both ends) and numbered whole
        lo, hi, numbers = span[0], span[1], (0.0, float(d[-1]))
        if hi - lo < MIN_M:
            return [], privacy.PrivacyReport(activities_in=1), learned, (0.0, 0.0)
    else:
        ok = far & (dd >= span[0]) & (dd <= span[1])
        best = None
        for i, j in _runs(ok):
            a, b = float(dd[i]), float(dd[j])
            if not far[:i].all():  # a place lies back along the track from this start: a random stretch more
                a += rng.uniform(*EXTRA)
            if not far[j + 1:].all():
                b -= rng.uniform(*EXTRA)
            if b - a >= MIN_M and (best is None or b - a > best[1] - best[0]):
                best = (a, b)
        if best is None:
            return [], privacy.PrivacyReport(activities_in=1), learned, (0.0, 0.0)
        lo, hi = best
        numbers = best
    line = substring(LineString(xy), lo, hi)
    # leak check, fail closed: the stretch stays beyond the reach of any zone around every place kept away from (a
    # stronger test than privacy.apply's, whose overlay would also split a course that crosses itself)
    if abs(line.length - (hi - lo)) > 1.0 or any(line.distance(Point(x, y)) < REACH * r - 1.0 for x, y, r in keep_off):
        raise privacy.PrivacyError("leak check failed: the race stretch is not clear of a private place")
    rep = privacy.PrivacyReport(activities_in=1, zones_user=len(keep_off), km_in=float(d[-1]) / 1000,
                                km_out=line.length / 1000, leak_check="passed" if keep_off else "no zones")
    return [(0, line)], rep, learned, numbers


def distance_scale(act, total_m, official_km=None, log=print):
    """The share of the track's own length (total_m, as projected) that the course really measured: the official
    distance given with --distance, else the distance Strava recorded (ingest.recorded_distance), else 1. Distance,
    pace, splits and the profile's kilometres are scaled by it; nothing that is drawn. A watch in a city adds a few
    percent to the sum of the steps between its fixes (a half marathon in Madrid: 21.9 km summed)."""
    if total_m <= 0:
        return 1.0
    if official_km:
        log(f"[distance] {official_km:g} km, as given with --distance (the fixes sum {total_m / 1000:.2f} km)")
        return official_km * 1000 / total_m
    scale = ingest.distance_scale(act, total_m)
    if scale != 1.0:
        log(f"[distance] {total_m * scale / 1000:.2f} km, as Strava recorded it (the fixes sum "
            f"{total_m / 1000:.2f} km)")
    else:
        log("[distance] the sum of the steps between the fixes: GPS noise can add a few percent (3-4 % for a watch "
            "in a city); for a race, give its official distance with --distance (21.0975 for a half marathon)")
    return scale


def frame_course(pieces, aspect, pad_frac=0.08, min_m=2000.0):
    """The map extent (x0, x1, y0, y1) around the drawn pieces: their box padded by pad_frac of its longer side, at
    least min_m wide and high, grown to the map's aspect (width / height)."""
    pts = np.concatenate([np.asarray(p.coords) for _, p in pieces])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad = pad_frac * max(x1 - x0, y1 - y0, min_m)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0 + 2 * pad, min_m), max(y1 - y0 + 2 * pad, min_m / aspect)
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def osm_box(extent, crs):
    """(south, west, north, east) in degrees for the OpenStreetMap download: the map box plus about 200 m, so the
    streets reach its edges."""
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    lons, lats = inv.transform([extent[0], extent[1]], [extent[2], extent[3]])
    if max(lons) - min(lons) > 180:
        raise AcrossDateline("this map crosses the 180th meridian, which Pinless cannot map yet (one box for "
                             "OpenStreetMap would wrap round the globe)")
    return (min(lats) - 0.002, min(lons) - 0.002, max(lats) + 0.002, max(lons) + 0.002)


def run(sel, pool, a, log=print):
    """sel: activities after the type/year filters; pool: every activity of that type, all years (for the places)."""
    chosen = pick(sel, a.activity)
    if chosen is None:
        log(f"no activity matches --activity {a.activity}")
        return 2
    race = ingest.clean(chosen)
    if race is None:
        log("this activity has only stray GPS fixes")
        return 2
    lon, lat = np.median(np.asarray(race.lonlat, dtype=float), axis=0)
    crs = render.utm_crs(lon, lat)
    tr = Transformer.from_crs(4326, crs, always_xy=True)

    def project(act):
        arr = np.asarray(act.lonlat, dtype=float)
        return np.column_stack(tr.transform(arr[:, 0], arr[:, 1]))

    zones = []
    for z in a.zone:
        zla, zlo, zr = map(float, z.split(","))
        zx, zy = tr.transform(zlo, zla)
        zones.append((zx, zy, zr))
    track = project(race)
    zones += own_ends(track, a)
    allow = a.allow_private_ends or getattr(a, "private_ends", False)
    others = [project(c) for x in pool if x is not chosen for c in [ingest.clean(x)] if c and len(c.lonlat) > 1]
    try:
        pieces, rep, learned, stretch = cut(track, others, a.trim, a.zone_radius, zones, allow_private_ends=allow)
    except PrivateEnds as ex:
        log(f"[privacy] STOP: {ex}. One course drawn from there would point at it. Pick another run, or pass "
            f"--allow-private-ends to draw it without that end (and a random stretch more), with its numbers "
            f"limited to what is drawn.")
        return 2
    except privacy.PrivacyError as ex:
        log(f"[privacy] STOP: {ex}. Nothing was exported.")
        return 3
    if not pieces:
        log("this activity has too little left after the privacy cuts (it stays near a private place)")
        return 2
    total = float(along(track)[-1])
    whole = stretch == (0.0, total)
    log(f"[privacy] private places learned from {len(others)} other activities: {learned} · km "
        f"{total / 1000:.1f} -> drawn {rep.km_out:.1f}"
        f"{' (a public course: numbers for all of it)' if whole else ' (numbers for the drawn stretch only)'} · "
        f"leak check {rep.leak_check}")
    if whole and np.hypot(*(track[0] - track[-1])) < LOOP_M:
        log("[privacy] this run starts and ends at the same place, and no other run of yours starts there: it is "
            "drawn as a race start. If it is not one (a hotel, a friend's home), add --private-ends.")

    measured = float(stats.measured_along(track, race.times, race.type)[-1])  # without the fixes' jumps
    scale = distance_scale(race, measured, getattr(a, "distance", None), log)
    extent = frame_course(pieces, render_race.map_aspect(a.size))
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    try:
        big = osm_box(extent, crs)
    except AcrossDateline as ex:
        log(f"[osm] STOP: {ex}. Nothing was sent or exported.")
        return 2
    try:
        basemap = render.load_basemap(area.fetch_basemap(big, log=log), big, crs)
    except RuntimeError as ex:
        log(f"[osm] STOP: {ex}. Run the same command again later.")
        return 4
    relief_ = relief.prepare_many([(extent, crs, render.SIZES[a.size][0] * (1 - 36 / 297))],
                                  log=log)[0] if getattr(a, "relief", "off") == "shade" else None
    mid_lon, mid_lat = inv.transform(*np.median(np.asarray(pieces[0][1].coords), axis=0))
    city, country = area.place_names(round(float(mid_lat), 2), round(float(mid_lon), 2), lang=a.lang)
    if race.name and not a.title:
        log(f"[race] title: {city or 'none'}. The activity's own name is not printed unless you pass it with --title "
            f"(names often say where a run starts)")
    title = a.title or city
    mode = a.color_by if a.color_by in ("pace", "elevation", "single") else "pace"
    info = render_race.render(race, track, pieces, basemap, extent, a.out, style=a.style or "huella", size=a.size,
                              dpi=a.dpi, title=title, city=", ".join(v for v in (city, country) if v), name=a.name,
                              date=local_start(race), lang=a.lang, color_by=mode, pdf_path=a.pdf, svg_path=a.svg,
                              preview_path=a.preview, times=race.times, ele=race.ele, stretch=stretch,
                              relief=relief_, scale=scale)
    if info["legend"]["mode"] != mode:
        log(f"[colour] this activity has no {mode} data: drawn in one colour")
    log(f"[render] {a.out} · {title} · {info['km']:.2f} km")
    return 0
