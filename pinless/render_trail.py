"""Trail: one outdoor route as a print. The route on a topographic map (terrain relief, paths and forests, named
summits), its elevation profile with the high point marked, and the route's numbers: length, elevation gain, high
point, route type and time.

As on the race print, the caller passes the privacy-cut pieces and the along-track stretch the numbers may describe
(race.cut): every number, the profile and the route type describe that stretch. No start or finish pin is drawn. The
date of the activity is printed: a print of one day out.
"""
from __future__ import annotations

import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.patches import PathPatch, Polygon as MPolygon
from shapely.geometry import LineString, box as _box

from . import privacy
from .atlas import _Page as Page
from .colouring import fmt_pace
from .render import MM, SIZES
from .render_editorial import DISPLAY, REGULAR, STYLES, TEXT, _line_path, draw_basemap, draw_relief, draw_routes
from .render_editorial import footer_credits, place_text, svg_rc
from .render_race import MONTHS, along, climb, pace_along, profile, restrict, vertex_positions
from .render_sport import _num

LEFT, RIGHT = 18.0, 279.0  # the A3 layout grid (mm), scaled by k for other paper sizes
MAP_TOP, MAP_H = 90.0, 246.0
PROFILE_TOP, PROFILE_H = 344.0, 22.0
NUMBERS_TOP = 376.0
LOOP_M = 250.0  # a stretch whose ends are this close is a loop (unless its second half retraces the first)
RETRACE_M, RETRACE_SHARE = 40.0, 0.70  # out and back: this share of the second half within this of the first
PAUSE_S, PAUSE_MPS, WINDOW_S = 120.0, 0.3, 60.0  # a pause: longer than PAUSE_S below PAUSE_MPS, over WINDOW_S
MAX_PEAKS = 6
PEAK_PT = 5.4  # summit labels (pt x k)
CASING_PT = 3.4  # the course's casing in the paper colour (pt x k), as on the race print
PT_MM = 25.4 / 72
ROUTE_TYPES = {"loop": ("LOOP", "CIRCULAR"), "out_and_back": ("OUT & BACK", "IDA Y VUELTA"),
               "point_to_point": ("POINT TO POINT", "LINEAL")}


def map_aspect(size="A3"):
    """Width / height of the map: the same on every paper size (the layout grid scales)."""
    return (RIGHT - LEFT) / MAP_H


# --- numbers -----------------------------------------------------------------------------------------------------------

def _resample(xy, step):
    """Points every `step` metres along the track (and its last point), with their along-track distances."""
    xy = np.asarray(xy, dtype=float)
    d = along(xy)
    at = np.append(np.arange(0.0, d[-1], step), d[-1])
    return np.column_stack([np.interp(at, d, xy[:, 0]), np.interp(at, d, xy[:, 1])]), at


def route_type(xy, step=10.0):
    """loop, out_and_back or point_to_point, for the numbered stretch.

    Out and back when at least RETRACE_SHARE of the second half (by distance, every `step` metres) lies within
    RETRACE_M of the first half's line; otherwise a loop when the ends are within LOOP_M; otherwise point to point."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2 or along(xy)[-1] <= 0:
        return "point_to_point"
    pts, at = _resample(xy, step)
    half = at[-1] / 2
    first, second = pts[at <= half], pts[at > half]
    if len(first) >= 2 and len(second):
        near = shapely.distance(shapely.points(second), LineString(first)) <= RETRACE_M
        if near.mean() >= RETRACE_SHARE:
            return "out_and_back"
    if math.hypot(*(xy[0] - xy[-1])) <= LOOP_M:
        return "loop"
    return "point_to_point"


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def moving_time(xy, times):
    """(seconds, paused): the time without the pauses, or the elapsed time when there were none; (None, False)
    without a working clock.

    A pause is a stretch longer than PAUSE_S where the speed, measured as the straight-line distance covered over
    WINDOW_S around each moment (GPS jitter while standing adds up along the track, not across it), stays under
    PAUSE_MPS. A watch that stopped recording while you stood (auto-pause) leaves one long step that is slow too."""
    if times is None or len(times) < 2:
        return None, False
    t = np.maximum.accumulate(np.asarray(times, dtype=float))
    xy = np.asarray(xy, dtype=float)
    total = float(t[-1] - t[0])
    if not total > 0:
        return None, False
    keep = np.concatenate([[True], np.diff(t) > 0])  # np.interp needs increasing times: drop repeated stamps
    tt, px, py = t[keep], xy[keep, 0], xy[keep, 1]
    mid = (t[:-1] + t[1:]) / 2
    lo = np.clip(mid - WINDOW_S / 2, t[0], t[-1])
    hi = np.clip(mid + WINDOW_S / 2, t[0], t[-1])
    moved = np.hypot(np.interp(hi, tt, px) - np.interp(lo, tt, px), np.interp(hi, tt, py) - np.interp(lo, tt, py))
    slow = moved < PAUSE_MPS * np.maximum(hi - lo, 1e-9)
    steps = np.diff(t)
    paused = sum(float(steps[i:j].sum()) for i, j in _runs(slow) if steps[i:j].sum() > PAUSE_S)
    return total - paused, paused > 0


def high_point(xy, ele):
    """(along-track metres, metres above sea level) of the highest point of the smoothed profile."""
    d, e = profile(xy, ele)
    i = int(np.argmax(e))
    return float(d[i]), float(e[i])


def fmt_hours(secs):
    """4:12 (hours and minutes): a day out is not timed to the second."""
    m = int(round(secs / 60))
    return f"{m // 60}:{m % 60:02d}"


_ELE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(m|metres|meters|ft|feet|')?")


def parse_ele(value):
    """The height of an OSM summit in metres: "2428", "2428.5", "2428 m" or "7234 ft"; None for anything else."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    m = _ELE.fullmatch(str(value).strip().lower())
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    if m.group(2) in ("ft", "feet", "'"):
        v *= 0.3048
    return v if -500 <= v <= 9000 else None


# --- the map -----------------------------------------------------------------------------------------------------------

def _towards(a, b, t):
    """The colour t of the way from a to b (t > 1 goes past b), clipped to the sRGB cube."""
    a, b = np.asarray(to_rgb(a)), np.asarray(to_rgb(b))
    return to_hex(np.clip(a + (b - a) * t, 0, 1))


def trail_style(S):
    """The style of the trail map, derived from S (so a style file works too; STYLES is not changed): paths and
    tracks, the network that matters off the streets, drawn in the side-street colour and wider than on a city
    poster; woods and meadows a step stronger."""
    out = dict(S)
    streets = dict(S["street"])
    minor_colour, minor_w = S["street"]["minor"]
    streets["path"] = (minor_colour, max(minor_w, S["street"]["path"][1]) * 1.6)
    out["street"] = streets
    out["green"] = _towards(S["bg"], S["green"], 1.8)
    return out


def _grid_xy(extent, x, y):
    """Map coordinates to the A3 layout grid (mm, y down)."""
    x0, x1, y0, y1 = extent
    return (LEFT + (np.asarray(x) - x0) / (x1 - x0) * (RIGHT - LEFT),
            MAP_TOP + (y1 - np.asarray(y)) / (y1 - y0) * MAP_H)


TRI_W, TRI_UP, TRI_DOWN = 1.9, 1.1, 0.55  # the summit triangle (grid mm): width, apex above and base below the point
EDGE_MM, GAP_MM, ROUTE_GAP_MM = 1.2, 0.8, 0.5


def peak_label(name, ele):
    return f"{name.strip().upper()} {ele:.0f} m" if ele is not None else name.strip().upper()


def place_peaks(peaks, extent, k, course, max_n=MAX_PEAKS):
    """The summits to draw: [(gx, gy, text, (x0, y0, x1, y1) label box, (x, y) label top-left)] in the layout grid.

    peaks: (x, y, name, ele) in map units. Only named summits inside the map, highest first (unknown heights last),
    at most max_n. Each label goes right of its triangle, else left, above or below; a summit whose label would
    leave the map, touch the course's casing or another summit's label or triangle is skipped, so nothing outside
    the visible map is drawn. course: the drawn course lines in map units."""
    cand = []
    for x, y, name, ele in peaks:
        if not isinstance(name, str) or not name.strip():
            continue
        gx, gy = (float(v) for v in _grid_xy(extent, x, y))
        if not (LEFT + EDGE_MM + TRI_W <= gx <= RIGHT - EDGE_MM - TRI_W
                and MAP_TOP + EDGE_MM + TRI_UP <= gy <= MAP_TOP + MAP_H - EDGE_MM - TRI_DOWN):
            continue
        cand.append((ele is None, -(ele or 0.0), name.strip(), gx, gy, ele))
    cand.sort(key=lambda c: c[:3])
    frame = _box(LEFT + EDGE_MM, MAP_TOP + EDGE_MM, RIGHT - EDGE_MM, MAP_TOP + MAP_H - EDGE_MM)
    lines = [np.column_stack(_grid_xy(extent, *np.asarray(c, dtype=float).T)) for c in course if len(c) > 1]
    route = (shapely.union_all([LineString(c) for c in lines]).buffer(CASING_PT * PT_MM / 2 + ROUTE_GAP_MM)
             if lines else None)
    marks, placed = [], []  # the labels and triangles drawn so far, padded
    for _, _, name, gx, gy, ele in cand:
        if len(placed) >= max_n:
            break
        text = peak_label(name, ele)
        _, _, w, h = place_text(text, 0, 0, k, PEAK_PT, TEXT)
        tri = _box(gx - TRI_W / 2, gy - TRI_UP, gx + TRI_W / 2, gy + TRI_DOWN)
        if any(tri.intersects(b) for b in marks):  # under another summit's label or triangle (the course may run
            continue                              # over a summit: its triangle is drawn on top)
        cy = gy - (TRI_UP - TRI_DOWN) / 2
        options = ((gx + TRI_W / 2 + 0.7, cy - h / 2), (gx - TRI_W / 2 - 0.7 - w, cy - h / 2),
                   (gx - w / 2, gy - TRI_UP - 0.7 - h), (gx - w / 2, gy + TRI_DOWN + 0.7))
        for lx, ly in options:
            label = _box(lx, ly, lx + w, ly + h)
            if (frame.contains(label) and (route is None or not label.intersects(route))
                    and not any(label.intersects(b) for b in marks)):
                placed.append((gx, gy, text, label.bounds, (lx, ly)))
                marks += [label.buffer(GAP_MM, join_style="mitre"), tri.buffer(GAP_MM / 2, join_style="mitre")]
                break
    return placed


# --- the page ----------------------------------------------------------------------------------------------------------

def render(activity, track_xy, pieces, basemap, extent, out_path, style="huella", size="A3", dpi=300, title="",
           place="", name="", date=None, lang="en", color_by="single", pdf_path=None, svg_path=None, preview_path=None,
           times=None, ele=None, stretch=None, relief=None, peaks=()):
    """activity: the route; track_xy: its (end-cleaned) projected track, with times and ele aligned to it (default:
    the activity's); pieces: the privacy-cut pieces (index, LineString), drawn; stretch: (start_m, end_m) along the
    track that the numbers describe (default: all of it); place: the subline (region, country); date: the local
    datetime of the activity; relief: a relief.Relief for the map; peaks: (x, y, name, ele) of OSM summits in map
    units, of which place_peaks draws up to six inside the map."""
    S = STYLES[style]
    es = lang == "es"
    wmm, hmm = SIZES[size]
    k = wmm / 297.0
    fig = plt.figure(figsize=(wmm * MM, hmm * MM), dpi=dpi, facecolor=S["bg"])
    page = Page(fig, wmm, hmm, k, S)
    T, rule = page.text, page.rule

    xy, times, ele = restrict(track_xy, activity.times if times is None else times,
                              activity.ele if ele is None else ele, stretch)
    dist = along(xy)
    secs, paused = moving_time(xy, times) if times else (None, False)

    # header
    T(18, 13, "P I N L E S S", pt=8.4)
    T(279, 13, "RUTA" if es else "TRAIL", pt=6.6, align="right", color=S["sub"])
    head = (title or ("RUTA" if es else "TRAIL")).upper()
    *_, title_h = place_text(head, 18, 22, k, 235, DISPLAY, max_w=212, height=40, clear=(15, 36, 17.0))
    T(18, 22 + max(0.0, 40 - title_h) / 2, head, pt=235, font=DISPLAY, max_w=212, height=40, clear=(15, 36, 17.0))
    if date is not None:
        T(279, 24, str(date.day), pt=43, font=REGULAR, color=S["accent"], align="right", max_w=39)
        T(279, 44, f"{MONTHS['es' if es else 'en'][date.month - 1]} {date.year}", pt=6.6, align="right",
          color=S["sub"])
    T(18, 70, place.upper(), pt=7.2, color=S["sub"], max_w=150)
    T(279, 69.5, name.upper(), pt=13.2, align="right", max_w=105)
    rule(80)

    # the map: relief, paths and woods, the course, the summits
    TS = trail_style(S)
    m, map_w, map_h = LEFT * k, (RIGHT - LEFT) * k, MAP_H * k
    ax = fig.add_axes([m / wmm, 1 - (MAP_TOP * k + map_h) / hmm, map_w / wmm, map_h / hmm], zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    draw_basemap(ax, basemap, extent, TS, k, map_w)
    drawn = draw_relief(ax, relief, extent, TS, basemap[2], k, map_w) if relief is not None else False
    cut = privacy.clip_to_extent(pieces, extent)
    coords = [c for c in (np.asarray(p.coords) for _, p in cut) if len(c) > 1]
    segments = np.concatenate([np.stack((c[:-1], c[1:]), axis=1) for c in coords]) if coords else np.empty((0, 2, 2))
    heat = LinearSegmentedColormap.from_list("trail", S["heat"])
    cmap, values, legend = ListedColormap([S["accent"]]), None, {"mode": "single"}
    if len(segments) and color_by in ("pace", "elevation"):
        pts = list(map(tuple, xy))
        pairs = {pq: i for i, pq in enumerate(zip(pts[:-1], pts[1:]))}
        at = np.concatenate([(lambda p: (p[:-1] + p[1:]) / 2)(vertex_positions(c, xy, dist, pairs)) for c in coords])
        if color_by == "pace" and times and secs:
            v = np.interp(at, dist, pace_along(dist, times))
            lo, hi = np.percentile(v, [5, 95])
            values, cmap = 1 - np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1), heat  # faster = stronger
            legend = {"mode": "pace", "lo": lo, "hi": hi}
        elif color_by == "elevation" and ele:
            v = np.interp(at, dist, profile(xy, ele)[1])
            lo, hi = float(v.min()), float(v.max())
            values, cmap = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1), heat
            legend = {"mode": "elevation", "lo": lo, "hi": hi}
    if values is None:
        values = np.zeros(len(segments))
    if len(segments):  # a casing in the paper colour lifts the course off the paths
        ax.add_patch(PathPatch(_line_path(list(segments)), fill=False, edgecolor=S["bg"], linewidth=CASING_PT * k,
                               capstyle="round", joinstyle="round", zorder=4.8))
    draw_routes(ax, segments, np.full(len(segments), 30.0), 3.0, cmap, k, base=1.7, gain=0.0,
                glow=S.get("glow", 0.0), values=values)
    summits = place_peaks(peaks, extent, k, coords)
    for gx, gy, text, _, (lx, ly) in summits:
        tri = [((gx - TRI_W / 2) * k, (gy + TRI_DOWN) * k), ((gx + TRI_W / 2) * k, (gy + TRI_DOWN) * k),
               (gx * k, (gy - TRI_UP) * k)]
        page.ax.add_patch(MPolygon(tri, closed=True, facecolor=S["sub"], edgecolor="none", lw=0))
        T(lx, ly, text, pt=PEAK_PT, color=S["sub"])

    # elevation profile under the map, its highest point marked
    hi_point = None
    if ele:
        d, e = profile(xy, ele)
        span = max(e.max() - e.min(), 10.0)
        base, top = e.min() - 0.15 * span, e.max() + 0.30 * span
        pax = fig.add_axes([LEFT * k / wmm, 1 - (PROFILE_TOP + PROFILE_H) * k / hmm, (RIGHT - LEFT) * k / wmm,
                            PROFILE_H * k / hmm], zorder=2)
        pax.set_facecolor("none")
        fill, line = (S["accent"], S["accent"]) if legend["mode"] == "single" else (S["heat"][0], S["heat"][3])
        pax.fill_between(d / 1000, base, e, color=fill, alpha=0.22 if legend["mode"] == "single" else 0.55,
                         linewidth=0)
        pax.plot(d / 1000, e, color=line, linewidth=0.9 * k)
        pax.set_xlim(0, d[-1] / 1000)
        pax.set_ylim(base, top)
        pax.axis("off")
        hd, he = high_point(xy, ele)
        hi_point = he
        gx = LEFT + hd / d[-1] * (RIGHT - LEFT)
        gy = PROFILE_TOP + (top - he) / (top - base) * PROFILE_H
        page.ax.add_patch(plt.Circle((gx * k, gy * k), 0.55 * k, color=S["ink"], lw=0, zorder=3))
        label = f"{he:.0f} m"
        _, _, lw_, lh_ = place_text(label, 0, 0, k, 5.6, TEXT)
        T(min(max(gx, LEFT + lw_ / 2), RIGHT - lw_ / 2), gy - 1.1 - lh_, label, pt=5.6, align="center")
        step = 5 if d[-1] > 12000 else 1
        for km_mark in range(0, int(d[-1] // 1000) + 1, step):
            T(LEFT + km_mark * 1000 / d[-1] * (RIGHT - LEFT), PROFILE_TOP + PROFILE_H + 2, str(km_mark), pt=4.4,
              color=S["sub"], align="center")
        T(18, PROFILE_TOP - 3.2, "PERFIL" if es else "ELEVATION", pt=5.6, color=S["sub"])
        T(279, PROFILE_TOP - 3.2, f"{e.min():.0f} – {e.max():.0f} m", pt=5.6, color=S["sub"], align="right")

    # numbers
    ytop = NUMBERS_TOP
    rule(ytop - 3)
    km = dist[-1] / 1000
    kind = route_type(xy)
    cells = [(_num(km, lang, 1), "km", "DISTANCIA" if es else "LENGTH"),
             (f"+{climb(profile(xy, ele)[1]):.0f}" if ele else "—", "m", "DESNIVEL POSITIVO" if es else "ELEVATION GAIN"),
             (f"{hi_point:.0f}" if hi_point is not None else "—", "m", "COTA MÁXIMA" if es else "HIGH POINT"),
             (ROUTE_TYPES[kind][1 if es else 0], "", "TIPO DE RUTA" if es else "ROUTE TYPE"),
             (fmt_hours(secs) if secs else "—", "h" if secs else "",
              ("TIEMPO EN MOVIMIENTO" if es else "MOVING TIME") if paused else ("TIEMPO" if es else "TIME"))]
    col = (RIGHT - LEFT) / len(cells)
    for i, (value, unit, label) in enumerate(cells):
        x = LEFT + i * col
        unit_w = place_text(unit, 0, 0, k, 9, REGULAR)[2] + 1.5 if unit else 0.0
        w_, _ = T(x, ytop + 1, value, pt=40, font=DISPLAY, max_w=col - 6 - unit_w, height=11)
        if unit:
            T(x + w_ + 1.5, ytop + 6.5, unit, pt=9, font=REGULAR, color=S["accent"])
        T(x, ytop + 15, label, pt=6.0, color=S["sub"], max_w=col - 4)

    # footer: what the line colour means, attribution
    rule(398)
    T(18, 404, "CADA RUTA DEJA HUELLA." if es else "EVERY ROUTE LEAVES A TRACE.", pt=6.8)
    if legend["mode"] in ("pace", "elevation"):
        T(212, 404, ("RITMO" if es else "PACE") if legend["mode"] == "pace" else ("ALTITUD" if es else "ELEVATION"),
          pt=5.8, color=S["sub"], align="right")
        for i, v in enumerate(np.linspace(0, 1, 6)):
            page.ax.plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [405 * k, 405 * k], color=heat(v), lw=1.3 * k,
                         solid_capstyle="round")
        lo_t = fmt_pace(legend["hi"]) if legend["mode"] == "pace" else f"{legend['lo']:.0f} m"
        hi_t = fmt_pace(legend["lo"]) if legend["mode"] == "pace" else f"{legend['hi']:.0f} m"
        T(218, 407.6, lo_t, pt=4.6, color=S["sub"])
        T(268, 407.6, hi_t, pt=4.6, color=S["sub"], align="right")
    footer_credits(T, S, lang, 412, 200, 238, relief.dataset if drawn else None)

    # No timestamps, activity ids, source paths or author metadata, as on the other prints.
    fig.savefig(out_path, dpi=dpi, facecolor=S["bg"], metadata={"Software": "Pinless"})
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=S["bg"],
                    metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None,
                              "Title": None, "Author": None, "Subject": None, "Keywords": None})
    if svg_path:
        with matplotlib.rc_context(svg_rc("pinless-trail")):
            fig.savefig(svg_path, facecolor=S["bg"], metadata={"Date": None, "Creator": "Pinless"})
    if preview_path:
        fig.savefig(preview_path, dpi=1600 * 25.4 / hmm, facecolor=S["bg"], pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return {"km": km, "secs": secs, "paused": paused, "legend": legend, "route_type": kind, "high": hi_point,
            "summits": summits, "relief": drawn}
