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
import pandas as pd
import shapely
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.patches import PathPatch, Polygon as MPolygon
from shapely.geometry import LineString, box as _box

from . import landcover, privacy
from .atlas import _Page as Page
from .colouring import fmt_pace
from .relief import raster_shape as relief_raster_shape
from .render import MAJOR, MINOR, MM, SIZES, _hw
from .render_editorial import DISPLAY, PRINT_TOL_MM, REGULAR, STYLES, TEXT, _clip_simplify, _keep_alpha, _line_path
from .render_editorial import _lstar, _parts, _poly_path, _relief_alpha, draw_basemap, draw_relief, draw_routes
from .render_editorial import footer_credits
from .render_editorial import place_text, svg_rc
from .render_race import MONTHS, along, climb, pace_along, profile, restrict, vertex_positions
from .render_sport import _num

LEFT, RIGHT = 18.0, 279.0  # the A3 layout grid (mm), scaled by k for other paper sizes
MAP_TOP, MAP_H = 90.0, 246.0
BARE_MAP_H = 274.0  # without heights there is no profile: the map takes its band, down to 8 mm above the numbers
PROFILE_TOP, PROFILE_H = 344.0, 22.0
NUMBERS_TOP = 376.0
LOOP_M = 250.0  # a stretch whose ends are this close is a loop (unless its second half retraces the first)
RETRACE_M, RETRACE_SHARE = 40.0, 0.70  # out and back: this share of the second half within this of the first
PAUSE_S, PAUSE_MPS, WINDOW_S = 120.0, 0.3, 60.0  # a pause: longer than PAUSE_S below PAUSE_MPS, over WINDOW_S
SMOOTH_S, STEP_S = 20.0, 5.0  # the track along which a pause must be slow too: averaged, then sampled
FLAT_M = 20.0  # a stretch that climbs less than this from its lowest to its highest point is not coloured by height
PROFILE_SPAN_M = 50.0  # the profile band shows at least this many metres of height
MAX_PEAKS = 6
PEAK_PT = 5.4  # summit labels (pt x k)
HALO_PT = 1.2  # their halo in the paper colour, all round (pt x k)
CASING_PT = 3.4  # the course's casing in the paper colour (pt x k), as on the race print
PT_MM = 25.4 / 72
ROUTE_TYPES = {"loop": ("LOOP", "CIRCULAR"), "out_and_back": ("OUT & BACK", "IDA Y VUELTA"),
               "point_to_point": ("POINT TO POINT", "LINEAL")}


def map_aspect(size="A3", profile=True):
    """Width / height of the map: the same on every paper size (the layout grid scales); taller for a route without
    heights, which has no profile under it."""
    return (RIGHT - LEFT) / (MAP_H if profile else BARE_MAP_H)


def _fit(extent, aspect):
    """The extent grown about its centre to the aspect (width / height) of the map box."""
    x0, x1, y0, y1 = extent
    w, h = x1 - x0, y1 - y0
    if abs(w / h - aspect) <= 1e-6 * aspect:
        return extent
    w, h = max(w, h * aspect), max(h, w / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


# --- numbers ----------------------------------------------------------------------------------------------------------

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
        line = LineString(first)
        shapely.prepare(line)  # indexed: a long-distance trail of 800 km is 40,000 points against 40,000 more
        near = shapely.dwithin(line, shapely.points(second), RETRACE_M)
        if near.mean() >= RETRACE_SHARE:
            return "out_and_back"
    if math.hypot(*(xy[0] - xy[-1])) <= LOOP_M:
        return "loop"
    return "point_to_point"


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def paused_steps(xy, times, ele=None):
    """A mask over the steps of the track (len(xy) - 1): True for the steps inside a pause; None without a working
    clock.

    A pause is a stretch longer than PAUSE_S where the speed, measured two ways over WINDOW_S around each moment,
    stays under PAUSE_MPS:
    - the straight-line distance covered (GPS jitter while standing adds up along the track, not across it);
    - the distance along the track once it is averaged over SMOOTH_S and sampled every STEP_S, which takes the jitter
      out: a steady climb on tight hairpins folds back on itself within a minute and covers little ground, but it
      keeps moving along its path, so it is not a pause.
    With elevations (ele, aligned with xy) both count the climb too: a steep scramble covers little ground, but it is
    not a pause. A watch that stopped recording while you stood (auto-pause) leaves one long step that is slow too."""
    if times is None or len(times) < 2 or len(times) != len(xy):
        return None
    t = np.maximum.accumulate(np.asarray(times, dtype=float))
    xy = np.asarray(xy, dtype=float)
    if not t[-1] - t[0] > 0:
        return None
    z = profile(xy, ele)[1] if ele is not None and len(ele) == len(xy) else np.zeros(len(xy))
    keep = np.concatenate([[True], np.diff(t) > 0])  # np.interp needs increasing times: drop repeated stamps
    tt, pts = t[keep], np.column_stack([xy[keep], z[keep]])
    mid = (t[:-1] + t[1:]) / 2
    lo = np.clip(mid - WINDOW_S / 2, t[0], t[-1])
    hi = np.clip(mid + WINDOW_S / 2, t[0], t[-1])
    span = np.maximum(hi - lo, 1e-9)
    net = np.sqrt(sum((np.interp(hi, tt, v) - np.interp(lo, tt, v)) ** 2 for v in pts.T))
    # the track averaged over SMOOTH_S (exactly, from the running integral of the positions), every STEP_S
    grid = np.append(np.arange(tt[0], tt[-1], STEP_S), tt[-1])
    a = np.clip(grid - SMOOTH_S / 2, tt[0], tt[-1])
    b = np.clip(grid + SMOOTH_S / 2, tt[0], tt[-1])
    smooth = []
    for v in pts.T:
        integral = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(tt))])
        smooth.append((np.interp(b, tt, integral) - np.interp(a, tt, integral)) / np.maximum(b - a, 1e-9))
    path = np.concatenate([[0.0], np.cumsum(np.sqrt(sum(np.diff(v) ** 2 for v in smooth)))])
    on_path = np.interp(hi, grid, path) - np.interp(lo, grid, path)
    slow = (net < PAUSE_MPS * span) & (on_path < PAUSE_MPS * span)
    steps = np.diff(t)
    out = np.zeros(len(steps), dtype=bool)
    for i, j in _runs(slow):
        if steps[i:j].sum() > PAUSE_S:
            out[i:j] = True
    return out


def moving_time(xy, times, ele=None):
    """(seconds, paused): the time without the pauses (paused_steps), or the elapsed time when there were none;
    (None, False) without a working clock."""
    mask = paused_steps(xy, times, ele)
    if mask is None:
        return None, False
    steps = np.diff(np.maximum.accumulate(np.asarray(times, dtype=float)))
    paused = float(steps[mask].sum())
    return float(steps.sum()) - paused, paused > 0


def moving_pace(dist, times, pauses=None, window=250.0):
    """Pace (s/km) at every point, over `window` metres of the track covered while moving: the steps of a pause
    (pauses, a mask over the steps) count neither their time nor the metres the fix wandered while standing."""
    t = np.maximum.accumulate(np.asarray(times, dtype=float))
    dd, dt_ = np.diff(np.asarray(dist, dtype=float)), np.diff(t)
    if pauses is not None:
        dd, dt_ = np.where(pauses, 0.0, dd), np.where(pauses, 0.0, dt_)
    return pace_along(np.concatenate([[0.0], np.cumsum(dd)]), np.concatenate([[0.0], np.cumsum(dt_)]), window)


def weighted_percentiles(values, weights, qs):
    """The percentiles qs (0-100) of values, each counted by its weight; plain percentiles when no weight is left."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if not w.sum() > 0:
        return [float(x) for x in np.percentile(v, qs)]
    order = np.argsort(v)
    cum = np.cumsum(w[order])
    return [float(v[order][min(int(np.searchsorted(cum, q / 100 * cum[-1])), len(v) - 1)]) for q in qs]


KM_MARKS = 12  # at most this many kilometre steps under the profile


def km_step(metres):
    """The step of the kilometre marks under the profile: 1, 2, 5, 10, 20, 50... km, the first that leaves at most
    KM_MARKS steps, so a long-distance trail of 800 km gets marks that do not run into each other."""
    step = 1
    while metres / 1000 / step > KM_MARKS:
        step = step * 5 // 2 if str(step)[0] == "2" else step * 2
    return step


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


# --- the map ----------------------------------------------------------------------------------------------------------

def _towards(a, b, t):
    """The colour t of the way from a to b (t > 1 goes past b), clipped to the sRGB cube."""
    a, b = np.asarray(to_rgb(a)), np.asarray(to_rgb(b))
    return to_hex(np.clip(a + (b - a) * t, 0, 1))


WOODS = 1.25  # woods and meadows: a step further from the paper than on a city poster; the relief carries the rest
TRAIL_INK = 0.55  # paths and tracks: this far from the paper towards the small-text colour
TRAIL_W = 0.34  # their width (pt x k): about twice a side street's on a city poster
FOOTPATH_DASH = (0, (5.0, 2.6))  # footpaths dashed (in line widths), as on a topographic map; tracks solid
TRACKS = {"track"}
# the walking network of the hills, in the trail ink; a town's footways (sidewalks), pedestrian streets, steps and
# cycleways stay in the street colours of every print, or every street of a village would be two dashed lines
TRAILS = {"path", "track", "bridleway"}
# roads over the relief, in an ink of their own between the paper and S["ink"]: their L* contrast against the paper is
# this multiple of the trail ink's, held under this share of the course's (the course leads) but never under the
# trail ink's; then their width (pt x k). Major roads get a casing in the paper colour.
ROADS = {"major": (1.5, 0.75, 0.56), "minor": (1.15, 0.6, 0.36)}
ROAD_CASING_PT = 0.3  # the major roads' casing, beyond the line (pt x k)


def trail_style(S):
    """The style of the trail map, derived from S (so a style file works too; STYLES is not changed): woods and
    meadows a step stronger, the paths and tracks in their own ink (trail_network), the roads in inks of their own
    that read at least as strongly (ROADS, trail_roads), and a stronger relief than on a city poster (trail_relief),
    since the course is one bold line."""
    out = dict(S)
    out["green"] = _towards(S["bg"], S["green"], WOODS)
    out["trail"] = _towards(S["bg"], S["sub"], TRAIL_INK)
    trail_l = abs(_lstar(to_rgb(out["trail"])) - _lstar(to_rgb(S["bg"])))
    course_l = abs(_lstar(to_rgb(S["accent"])) - _lstar(to_rgb(S["bg"])))
    for cls, (times, share, _) in ROADS.items():
        target = max(trail_l, min(times * trail_l, share * course_l))
        out[f"road_{cls}"] = _ink_at(S["bg"], (S["ink"], S["sub"]), target)
    out["relief"] = trail_relief(S)
    return out


def _ink_at(bg, inks, target):
    """The colour on the way from bg to the first of inks that gets there (else black or white, whichever lies
    farther from bg) whose L* differs from bg's by target; the end colour itself if none gets there. A style file
    with a faint ink still gets roads that read."""
    l0 = _lstar(to_rgb(bg))

    def gap(c):
        return abs(_lstar(to_rgb(c)) - l0)

    ends = [*inks, "#000000" if l0 >= 50 else "#ffffff"]
    ink = next((c for c in ends if gap(c) >= target), max(ends, key=gap))
    if gap(ink) <= target:
        return to_hex(to_rgb(ink))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(_towards(bg, ink, mid)) < target else (lo, mid)
    return _towards(bg, ink, hi)


def trail_relief(S, ground=None):
    """((shadow rgb, alpha), (light rgb, alpha)), as render_editorial.relief_colours, set against the course's own
    colour (the accent, one bold line) instead of the faintest heat ink: light paper darkens by up to min(13, 0.35
    route) L* and lightens by min(4, 0.2 route); dark paper darkens by min(8, 0.5 L*bg, 0.35 route) and lightens by
    min(8, 0.3 route). The light side is held to what the map under it can bear, as on the other prints. Over the
    land cover (ground: its mean colour) the same changes of L* are measured on that colour instead of the paper.
    A style that sets "relief" keeps it."""
    if "relief" in S:
        return S["relief"]
    bg = to_rgb(S["bg"])
    lb = _lstar(bg)
    route = abs(lb - _lstar(to_rgb(S["accent"])))
    if lb >= 50:
        shadow, ts, light, tl = S["ink"], min(13, 0.35 * route), "#ffffff", min(4, 0.2 * route)
    else:
        shadow, ts, light, tl = "#000000", min(8, 0.5 * lb, 0.35 * route), S["ink"], min(8, 0.3 * route)
    base = bg if ground is None else ground
    a_light = min(_relief_alpha(base, to_rgb(light), tl, 0.6), _keep_alpha(S, to_rgb(light), 0.6))
    return ((to_rgb(shadow), _relief_alpha(base, to_rgb(shadow), ts, 0.5)), (to_rgb(light), a_light))


# --- the ground: the land cover under the map -------------------------------------------------------------------------

GROUND_Z = 0.5  # the land cover: over the paper, under the water (2), the relief (3.5) and everything else
SIGNIFICANT = 0.05  # a ground colour that covers this share of the map sets how strong paths and roads must be
# a course whose accent lies closer than this (CIEDE2000) to the ground under most of it takes a route ink instead:
# the forest style's deep green on tree cover (24) and the sunset orange on bare land (25) sink into it; every other
# accent of the built-in styles lies 30 or more from every ground colour of its palette
COURSE_DE = 30.0
FOOT_STEP, FOOT_LIFT = 3.0, 2.0  # three credit lines: their spacing (A3-grid mm), and how far the line above rises
ROUTE_INKS = {False: (("deep red", "#a8231c"), ("deep blue", "#1f4f96"), ("black", "#141414")),  # light paper
              True: (("red", "#ff6b57"), ("blue", "#7cc3ff"), ("white", "#ffffff"))}  # dark paper


def draw_ground(ax, ground, extent, S, map_w_mm):
    """The land cover as ONE image of exactly the map box (interpolation 'none': a PDF or SVG embeds that array as
    it is), clipped to the box, at the bottom of the map. Returns its colours: {class: rgb} of the classes drawn,
    their shares of the map and the mean colour."""
    if ground.classes.shape != relief_raster_shape(map_w_mm, extent):
        raise ValueError("the land cover raster does not match the map")
    rgb = landcover.ground_rgb(ground.classes, S, ground.dry)
    x0, x1, y0, y1 = extent
    im = ax.imshow(rgb, extent=extent, origin="upper", interpolation="none", zorder=GROUND_Z)
    im.set_clip_path(PathPatch(_poly_path([_box(x0, y0, x1, y1)]), transform=ax.transData))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    shares = landcover.shares(ground.classes)
    colours = landcover.palette(S, ground.dry)
    return {"colours": {k: colours[k] for k in shares}, "shares": shares,
            "mean": tuple(float(v) for v in rgb.reshape(-1, 3).mean(axis=0) / 255)}


def ground_style(TS, S, painted):
    """The trail style over the land cover: the relief measured on the ground's mean colour, and the paths and roads
    re-inked along the way from the paper to the style's inks so they keep, against the darkest ground colour that
    covers SIGNIFICANT of the map (the lightest on dark paper), the contrast they have against the paper."""
    out = dict(TS)
    out["relief"] = trail_relief(S, painted["mean"])
    dark = _lstar(to_rgb(S["bg"])) < 50
    big = [c for k, c in painted["colours"].items() if painted["shares"][k] >= SIGNIFICANT] or [painted["mean"]]
    ref = (max if dark else min)(big, key=_lstar)
    l_bg = _lstar(to_rgb(S["bg"]))
    shift = abs(l_bg - _lstar(ref))
    for key in ("trail", "road_minor", "road_major"):
        gap = abs(_lstar(to_rgb(TS[key])) - l_bg)
        out[key] = _ink_at(S["bg"], (S["sub"], S["ink"]), shift + gap)
    return out


def ground_under(classes, extent, segments):
    """{class: share of the course's length} of the land cover under the drawn course (segments: (n, 2, 2) in map
    units), sampled about once per pixel of the raster."""
    if not len(segments):
        return {}
    H, W = classes.shape
    x0, x1, y0, y1 = extent
    a, b = segments[:, 0], segments[:, 1]
    length = np.hypot(*(b - a).T)
    n = np.maximum(1, np.ceil(length / ((x1 - x0) / W))).astype(int)
    seg = np.repeat(np.arange(len(segments)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[seg]
    p = a[seg] + (b - a)[seg] * t[:, None]
    col = np.clip(((p[:, 0] - x0) / (x1 - x0) * W).astype(int), 0, W - 1)
    row = np.clip(((y1 - p[:, 1]) / (y1 - y0) * H).astype(int), 0, H - 1)
    weight, cls = (length / n)[seg], classes[row, col]
    total = weight.sum()
    if not total > 0:
        return {}
    return {int(c): float(weight[cls == c].sum() / total) for c in np.unique(cls)}


def course_ink(S, colours, under):
    """(ink, note): the style's accent, unless it lies closer than COURSE_DE (CIEDE2000) to the ground colour under
    most of the course (it would sink into it); then the first route ink of ROUTE_INKS that lies at least COURSE_DE
    from that colour (else the one that lies farthest), and a note for the log that says why (without any place)."""
    if not under:
        return S["accent"], None
    dominant = max(sorted(under), key=lambda c: under[c])
    ground = colours[dominant]
    d = landcover.delta_e(S["accent"], ground)
    if d >= COURSE_DE:
        return S["accent"], None
    inks = ROUTE_INKS[_lstar(to_rgb(S["bg"])) < 50]
    name, ink = next((t for t in inks if landcover.delta_e(t[1], ground) >= COURSE_DE),
                     max(inks, key=lambda t: landcover.delta_e(t[1], ground)))
    return ink, (f"the course is drawn in {name}: the style's accent would sink into the {landcover.NAMES[dominant]} "
                 f"under most of it (colour difference {d:.0f}, below {COURSE_DE:.0f})")


def split_network(basemap):
    """(the basemap with only its town footways, the roads, the paths and tracks). The roads (draw_basemap's major
    and minor classes) are drawn by trail_roads and the walking network of the hills (TRAILS) by trail_network, both
    over the relief; footways, pedestrian streets, steps and cycleways go to draw_basemap, as on every print."""
    edges, green, water = basemap
    hw = edges["highway"].map(_hw)
    roads = hw.isin(MAJOR | MINOR).to_numpy(dtype=bool)
    trails = hw.isin(TRAILS).to_numpy(dtype=bool)
    return (edges[~roads & ~trails], green, water), edges[roads], edges[trails]


def _network_path(edges, extent, tol):
    """One compound path of the edges, each street once whatever its direction, cut to the map and simplified to
    what paper shows (as draw_basemap); None when nothing is left."""
    if not len(edges):
        return None
    geoms = np.asarray(edges.geometry.values, dtype=object)
    edges = edges[~pd.Series(shapely.to_wkb(shapely.normalize(geoms))).duplicated().to_numpy()]
    return _line_path(p for g in _clip_simplify(edges.geometry, extent, tol) for p in _parts(g, ("LineString",)))


def trail_network(ax, trails, extent, S, k, map_w_mm):
    """Paths and tracks over the relief: tracks solid, footpaths dashed, one compound path each, in S["trail"]."""
    if not len(trails):
        return
    tol = (extent[1] - extent[0]) / map_w_mm * PRINT_TOL_MM
    track = trails["highway"].map(_hw).isin(TRACKS).to_numpy()
    for mask, style in ((track, "solid"), (~track, FOOTPATH_DASH)):
        path = _network_path(trails[mask], extent, tol)
        if path is not None:
            ax.add_patch(PathPatch(path, fill=False, edgecolor=S["trail"], linewidth=TRAIL_W * k, linestyle=style,
                                   capstyle="butt", joinstyle="round", zorder=3.6))


def trail_roads(ax, roads, extent, S, k, map_w_mm):
    """Roads over the relief and the paths, in their own ink (trail_style), so a paved road reads at least as
    strongly as a footpath: minor roads as a line, major roads as a line on a casing in the paper colour that cuts
    the paths crossing them, as on a topographic map."""
    if not len(roads):
        return
    tol = (extent[1] - extent[0]) / map_w_mm * PRINT_TOL_MM
    major = roads["highway"].map(_hw).isin(MAJOR).to_numpy()
    w_minor, w_major = ROADS["minor"][2] * k, ROADS["major"][2] * k
    minor_path, major_path = _network_path(roads[~major], extent, tol), _network_path(roads[major], extent, tol)
    for path, colour, width, z in ((minor_path, S["road_minor"], w_minor, 3.61),
                                   (major_path, S["bg"], w_major + 2 * ROAD_CASING_PT * k, 3.62),
                                   (major_path, S["road_major"], w_major, 3.63)):
        if path is not None:
            ax.add_patch(PathPatch(path, fill=False, edgecolor=colour, linewidth=width, capstyle="round",
                                   joinstyle="round", zorder=z))


def _grid_xy(extent, x, y, map_h=MAP_H):
    """Map coordinates to the A3 layout grid (mm, y down)."""
    x0, x1, y0, y1 = extent
    return (LEFT + (np.asarray(x) - x0) / (x1 - x0) * (RIGHT - LEFT),
            MAP_TOP + (y1 - np.asarray(y)) / (y1 - y0) * map_h)


TRI_W, TRI_UP, TRI_DOWN = 1.9, 1.1, 0.55  # the summit triangle (grid mm): width, apex above and base below the point
EDGE_MM, GAP_MM, ROUTE_GAP_MM = 1.2, 0.8, 0.5
ROUTE_CLEAR_MM = CASING_PT * PT_MM / 2 + ROUTE_GAP_MM  # a summit label keeps this far from the course's centre line


def _baseline(value, k, **kw):
    """How far below its top (A3-grid mm) place_text sets the baseline of a text."""
    return float(place_text(value, 0, 0, k, **kw)[1].transform((0, 0))[1]) / k


def peak_label(name, ele):
    return f"{name.strip().upper()} {ele:.0f} m" if ele is not None else name.strip().upper()


def place_peaks(peaks, extent, k, course, max_n=MAX_PEAKS, map_h=MAP_H):
    """The summits to draw: [(gx, gy, text, (x0, y0, x1, y1) label box, (x, y) label top-left)] in the layout grid.

    peaks: (x, y, name, ele) in map units. Only named summits inside the map, highest first (unknown heights last),
    at most max_n. Each label goes right of its triangle, else left, above, below or at a corner; a summit whose
    label would leave the map, touch the course's casing or another summit's label or triangle wherever it goes is
    skipped, so nothing outside the visible map is drawn. course: the drawn course lines in map units."""
    cand = []
    for x, y, name, ele in peaks:
        if not isinstance(name, str) or not name.strip():
            continue
        gx, gy = (float(v) for v in _grid_xy(extent, x, y, map_h))
        if not (LEFT + EDGE_MM + TRI_W <= gx <= RIGHT - EDGE_MM - TRI_W
                and MAP_TOP + EDGE_MM + TRI_UP <= gy <= MAP_TOP + map_h - EDGE_MM - TRI_DOWN):
            continue
        cand.append((ele is None, -(ele or 0.0), name.strip(), gx, gy, ele))
    if not cand:
        return []
    cand.sort(key=lambda c: c[:3])
    frame = _box(LEFT + EDGE_MM, MAP_TOP + EDGE_MM, RIGHT - EDGE_MM, MAP_TOP + map_h - EDGE_MM)
    lines = [np.column_stack(_grid_xy(extent, *np.asarray(c, dtype=float).T, map_h)) for c in course if len(c) > 1]
    # a label keeps clear of the course by distance, not by a buffer of it: a stop where the fix wandered for twenty
    # minutes crosses itself hundreds of times, and buffering that takes minutes
    route = shapely.MultiLineString(lines) if lines else None
    if route is not None:
        shapely.prepare(route)
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
        right, left = gx + TRI_W / 2 + 0.7, gx - TRI_W / 2 - 0.7 - w
        up, down = gy - TRI_UP - 0.7 - h, gy + TRI_DOWN + 0.7
        options = ((right, cy - h / 2), (left, cy - h / 2), (gx - w / 2, up), (gx - w / 2, down),  # beside, then
                   (right, up), (right, down), (left, up), (left, down))  # diagonal: a summit on the course
        for lx, ly in options:
            label = _box(lx, ly, lx + w, ly + h)
            if (frame.contains(label) and (route is None or not shapely.dwithin(route, label, ROUTE_CLEAR_MM))
                    and not any(label.intersects(b) for b in marks)):
                placed.append((gx, gy, text, label.bounds, (lx, ly)))
                marks += [label.buffer(GAP_MM, join_style="mitre"), tri.buffer(GAP_MM / 2, join_style="mitre")]
                break
    return placed


def draw_summits(ax, summits, S, k):
    """The placed summits (place_peaks) on the page axes (millimetres, y down): a triangle and the label in the sub
    colour, both with a thin halo in the paper colour, so a path or a contour that runs under a name does not run
    through its letters, as on a topographic map."""
    halo = [patheffects.withStroke(linewidth=HALO_PT * k, foreground=S["bg"])]
    for gx, gy, text, _, (lx, ly) in summits:
        tri = [((gx - TRI_W / 2) * k, (gy + TRI_DOWN) * k), ((gx + TRI_W / 2) * k, (gy + TRI_DOWN) * k),
               (gx * k, (gy - TRI_UP) * k)]
        ax.add_patch(MPolygon(tri, closed=True, facecolor=S["sub"], edgecolor="none", lw=0, path_effects=halo))
        path, trans, _, _ = place_text(text, lx, ly, k, PEAK_PT, TEXT)
        ax.add_patch(PathPatch(path, transform=trans + ax.transData, color=S["sub"], lw=0, snap=False,
                               path_effects=halo))


# --- the page ---------------------------------------------------------------------------------------------------------

def render(activity, track_xy, pieces, basemap, extent, out_path, style="huella", size="A3", dpi=300, title="",
           place="", name="", date=None, lang="en", color_by="single", pdf_path=None, svg_path=None, preview_path=None,
           times=None, ele=None, stretch=None, relief=None, peaks=(), ground=None, log=None):
    """activity: the route; track_xy: its (end-cleaned) projected track, with times and ele aligned to it (default:
    the activity's); pieces: the privacy-cut pieces (index, LineString), drawn; stretch: (start_m, end_m) along the
    track that the numbers describe (default: all of it); place: the subline (region, country); date: the local
    datetime of the activity; relief: a relief.Relief for the map; peaks: (x, y, name, ele) of OSM summits in map
    units, of which place_peaks draws up to six inside the map; ground: a landcover.LandCover for the map (the
    colours of the land under it); log: where to say which ink the course took over the ground, and why."""
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
    secs, paused = moving_time(xy, times, ele) if times else (None, False)
    pauses = paused_steps(xy, times, ele) if times else None

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

    # the map: woods, water and a town's footways, the relief, the paths and tracks, the roads, the course, the
    # summits
    TS = trail_style(S)
    grid_h = MAP_H if ele else BARE_MAP_H  # no heights, no profile: the map takes its band
    extent = _fit(extent, map_aspect(size, profile=bool(ele)))  # trail.run frames it so already
    m, map_w, map_h = LEFT * k, (RIGHT - LEFT) * k, grid_h * k
    ax = fig.add_axes([m / wmm, 1 - (MAP_TOP * k + map_h) / hmm, map_w / wmm, map_h / hmm], zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    streets, roads, trails = split_network(basemap)
    painted = draw_ground(ax, ground, extent, S, map_w) if ground is not None else None
    if painted is not None:  # the land cover already says where the woods and meadows are
        TS = ground_style(TS, S, painted)
        streets = (streets[0], streets[1].iloc[:0], streets[2])
    draw_basemap(ax, streets, extent, TS, k, map_w)
    drawn = draw_relief(ax, relief, extent, TS, basemap[2], k, map_w) if relief is not None else False
    trail_network(ax, trails, extent, TS, k, map_w)
    trail_roads(ax, roads, extent, TS, k, map_w)
    cut = privacy.clip_to_extent(pieces, extent)
    coords = [c for c in (np.asarray(p.coords) for _, p in cut) if len(c) > 1]
    segments = np.concatenate([np.stack((c[:-1], c[1:]), axis=1) for c in coords]) if coords else np.empty((0, 2, 2))
    heat = LinearSegmentedColormap.from_list("trail", S["heat"][1:])  # the faintest ink would sink into the relief
    ink, why = S["accent"], None
    if painted is not None:
        ink, why = course_ink(S, painted["colours"], ground_under(ground.classes, extent, segments))
    cmap, values, legend = ListedColormap([ink]), None, {"mode": "single"}
    if len(segments) and color_by in ("pace", "elevation"):
        pts = list(map(tuple, xy))
        pairs = {pq: i for i, pq in enumerate(zip(pts[:-1], pts[1:]))}
        at = np.concatenate([(lambda p: (p[:-1] + p[1:]) / 2)(vertex_positions(c, xy, dist, pairs)) for c in coords])
        if color_by == "pace" and times and secs:
            # the pace while moving: a stop takes the pace of the walk around it, and the legend's ends are the
            # fastest and slowest 5 % of the drawn walking, by length (the fixes of a stop crowd the course)
            v = np.interp(at, dist, moving_pace(dist, times, pauses))
            step = np.clip(np.searchsorted(dist, at, side="right") - 1, 0, len(dist) - 2)
            moving = ~pauses[step] if pauses is not None else np.ones(len(at), dtype=bool)
            lo, hi = weighted_percentiles(v, np.hypot(*(segments[:, 1] - segments[:, 0]).T) * moving, (5, 95))
            values, cmap = 1 - np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1), heat  # faster = stronger
            legend = {"mode": "pace", "lo": lo, "hi": hi}
        elif color_by == "elevation" and ele:
            e = profile(xy, ele)[1]  # the key's ends are the profile's: the lowest and highest of the stretch
            lo, hi = float(e.min()), float(e.max())
            if hi - lo < FLAT_M:  # a flat route: colouring it would colour the altimeter's noise
                legend = {"mode": "single", "flat": True}
            else:
                values, cmap = np.clip((np.interp(at, dist, e) - lo) / (hi - lo), 0, 1), heat
                legend = {"mode": "elevation", "lo": lo, "hi": hi}
    if values is None:
        values = np.zeros(len(segments))
    if len(segments):  # a casing in the paper colour lifts the course off the paths
        ax.add_patch(PathPatch(_line_path(list(segments)), fill=False, edgecolor=S["bg"], linewidth=CASING_PT * k,
                               capstyle="round", joinstyle="round", zorder=4.8))
    draw_routes(ax, segments, np.full(len(segments), 30.0), 3.0, cmap, k, base=1.7, gain=0.0,
                glow=S.get("glow", 0.0), values=values)
    summits = place_peaks(peaks, extent, k, coords, map_h=grid_h)
    draw_summits(page.ax, summits, S, k)

    # elevation profile under the map, its highest point marked
    hi_point = None
    if ele:
        d, e = profile(xy, ele)
        span = max(e.max() - e.min(), PROFILE_SPAN_M)  # a flat route stays flat, not a sawtooth of altimeter noise
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
        step = km_step(d[-1])
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
             (f"+{climb(profile(xy, ele)[1]):.0f}" if ele else "—", "m" if ele else "",
              "DESNIVEL POSITIVO" if es else "ELEVATION GAIN"),
             (f"{hi_point:.0f}" if hi_point is not None else "—", "m" if hi_point is not None else "",
              "COTA MÁXIMA" if es else "HIGH POINT"),
             (ROUTE_TYPES[kind][1 if es else 0], "", "TIPO DE RUTA" if es else "ROUTE TYPE"),
             (fmt_hours(secs) if secs else "—", "h" if secs else "",
              ("TIEMPO EN MOVIMIENTO" if es else "MOVING TIME") if paused else ("TIEMPO" if es else "TIME"))]
    col = (RIGHT - LEFT) / len(cells)
    base = _baseline("0", k, pt=40, font=DISPLAY, height=11)
    for i, (value, unit, label) in enumerate(cells):
        x = LEFT + i * col
        unit_w = place_text(unit, 0, 0, k, 9, REGULAR)[2] + 1.5 if unit else 0.0
        fit = dict(pt=40, font=DISPLAY, max_w=col - 9 - unit_w, height=11)
        # a value shrunk to its column (POINT TO POINT) sits on the numbers' baseline, not under their top
        w_, _ = T(x, ytop + 1 + base - _baseline(value, k, **fit), value, **fit)
        if unit:
            T(x + w_ + 1.5, ytop + 6.5, unit, pt=9, font=REGULAR, color=S["accent"])
        T(x, ytop + 15, label, pt=6.0, color=S["sub"], max_w=col - 4)

    # footer: what the line colour means, attribution (three credit lines sit closer, and the line above them rises)
    rule(398)
    three = drawn and painted is not None
    top = 404 - (FOOT_LIFT if three else 0.0)
    T(18, top, "CADA RUTA DEJA HUELLA." if es else "EVERY ROUTE LEAVES A TRACE.", pt=6.8)
    if legend["mode"] in ("pace", "elevation"):
        T(212, top, ("RITMO" if es else "PACE") if legend["mode"] == "pace" else ("ALTITUD" if es else "ELEVATION"),
          pt=5.8, color=S["sub"], align="right")
        for i, v in enumerate(np.linspace(0, 1, 6)):
            page.ax.plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [(top + 1) * k, (top + 1) * k], color=heat(v),
                         lw=1.3 * k, solid_capstyle="round")
        lo_t = fmt_pace(legend["hi"]) if legend["mode"] == "pace" else f"{legend['lo']:.0f} m"
        hi_t = fmt_pace(legend["lo"]) if legend["mode"] == "pace" else f"{legend['hi']:.0f} m"
        T(218, top + 3.6, lo_t, pt=4.6, color=S["sub"])
        T(268, top + 3.6, hi_t, pt=4.6, color=S["sub"], align="right")
    footer_credits(T, S, lang, 412, 200, 238, relief.dataset if drawn else None, ground=painted is not None,
                   step=FOOT_STEP if three else 3.4)
    if why and legend["mode"] == "single" and log is not None:
        log(f"[ground] {why}")

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
            "summits": summits, "relief": drawn, "ground": painted is not None,
            "course_ink": ink if legend["mode"] == "single" else None}
