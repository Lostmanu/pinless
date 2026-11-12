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
import contourpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.patches import PathPatch, Polygon as MPolygon, Rectangle
from matplotlib.path import Path as MPath
from PIL import Image, ImageFilter
from shapely.geometry import LineString, box as _box

from . import landcover, privacy
from .atlas import _Page as Page
from .colouring import fmt_pace
from .relief import raster_shape as relief_raster_shape
from .stats import measured_along
from .render import MAJOR, MINOR, MM, SIZES, _hw
from .render_editorial import DISPLAY, PRINT_TOL_MM, REGULAR, STYLES, TEXT, _clip_simplify, _keep_alpha, _line_path
from .render_editorial import RAMP_FULL, RAMP_GAMMA, _lstar, _parts, _poly_path, _relief_alpha, draw_basemap
from .render_editorial import draw_relief, draw_routes
from .render_editorial import baseline_of, footer_credits, land_clip_path
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
PEAK_PT = 6.2  # summit labels (pt x k)
HALO_PT = 1.9  # their halo in the paper colour, all round (pt x k)
COURSE_PT = 2.3  # the course (pt x k)
CASING_PT = 4.3  # its casing in the paper colour (pt x k)
PT_MM = 25.4 / 72
PROFILE_FILL = (0.42, 0.04)  # the profile's gradient: this share of the way from the paper to the line, top to bottom
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
    """((shadow rgb, alpha), (light rgb, alpha)), as render_editorial.relief_colours, for the trail map without the
    land cover (over it the relief is baked into the ground: bake), set against the course's own colour (the accent,
    one bold line) instead of the faintest heat ink: light paper darkens by up to min(18, 0.6 route) L* and lightens
    towards a warm white by min(10, 0.35 route); dark paper darkens by min(16, 0.6 L*bg, 0.6 route) and lightens by
    min(16, 0.35 route). The light side is held to what the map under it can bear, as on the other prints. ground: a
    colour to measure the changes of L* on instead of the paper. A style that sets "relief" keeps it."""
    if "relief" in S:
        return S["relief"]
    bg = to_rgb(S["bg"])
    lb = _lstar(bg)
    route = abs(lb - _lstar(to_rgb(S["accent"])))
    if lb >= 50:
        shadow, ts, a_s, light, tl = S["ink"], min(18, 0.6 * route), 0.60, "#fff6e0", min(10, 0.35 * route)
    else:
        shadow, ts, a_s, light, tl = "#000000", min(16, 0.6 * lb, 0.6 * route), 0.80, S["ink"], min(16, 0.35 * route)
    base = bg if ground is None else ground
    a_light = min(_relief_alpha(base, to_rgb(light), tl, 0.6), _keep_alpha(S, to_rgb(light), 0.6))
    return ((to_rgb(shadow), _relief_alpha(base, to_rgb(shadow), ts, a_s)), (to_rgb(light), a_light))


# --- the ground: the land cover under the map -------------------------------------------------------------------------

GROUND_Z = 0.5  # the land cover: over the paper, under the water (2), the relief (3.5) and everything else
FRAME_PT, FRAME_Z = 0.45, 4.7  # the hairline round a map with the land cover: the page's rules, under the course
SIGNIFICANT = 0.05  # a ground colour that covers this share of the map sets how strong paths and roads must be
# a course whose accent lies closer than COURSE_DE (CIEDE2000) to the ground under SINK_SHARE or more of its length
# takes a route ink instead. Among the built-in styles: the forest style's deep green on tree cover (24 in a green
# land, 27 at dryness 0.5, 30 in a dry one) and on mangroves (19), and the sunset orange on the bare land of a dry
# land (22 at dryness 1, 28 at 0.5; 30 or more below 0.3) sink into it; every other accent lies 30 or more from every
# ground colour of its palette, on light and dark paper
COURSE_DE = 30.0
SINK_SHARE = 0.15
UNDER_SHARE = 0.05  # a route ink keeps COURSE_DE from every ground colour under this share of the course or more
FOOT_STEP, FOOT_LIFT = 3.0, 2.0  # three credit lines: their spacing (A3-grid mm), and how far the line above rises
ROUTE_INKS = {False: (("deep red", "#a8231c"), ("deep blue", "#1f4f96"), ("black", "#141414")),  # light paper
              True: (("red", "#ff6b57"), ("blue", "#7cc3ff"), ("white", "#ffffff"))}  # dark paper


# The relief baked into the land cover (bake): at full shadow a colour keeps 1 - ks * k of each channel, where k
# runs from K_COOL in a green land (red falls fastest: a cool shadow) to K_WARM in a dry one (blue falls fastest: a
# burnt, warm shadow); at full light it moves kl of the way to BAKE_LIGHT, a warm white. (ks, kl) on light and dark
# paper. A multiply keeps the hue of the ground: a forest in shadow is a deep green, never slate, pale rock grey green
BAKE = {False: (0.58, 0.32), True: (0.62, 0.24)}
K_COOL, K_WARM, BAKE_LIGHT = (1.0, 0.96, 0.86), (0.86, 1.0, 1.06), (1.0, 0.97, 0.88)
BUILT_FEATHER_MM = 1.2  # the smooth shade fades in over built-up land over this much paper
REF_PCT = (15, 85)  # paths keep their contrast on the darkest 15 % of the baked ground (lightest on dark paper)


def built_weight(classes, map_w_mm):
    """float32 (H, W) in [0, 1]: where the land cover's built-up land lies, grown by a pixel and feathered over
    BUILT_FEATHER_MM of paper, the weight of the smooth shade (the city ground filter) against the crisp one, so a
    town's roofs do not print as speckle; None when the map has no built-up land."""
    mask = np.asarray(classes) == landcover.BUILT
    if not mask.any():
        return None
    px = mask.shape[1] / map_w_mm  # raster pixels per mm of paper
    im = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
    w = np.asarray(im.filter(ImageFilter.GaussianBlur(BUILT_FEATHER_MM * px)), np.float32) / 255
    return np.clip(w * 1.5, 0, 1).astype(np.float32)


HOLD_SAMPLE = 250_000  # the dark paper's light side is measured on about this many pixels of the map


def _dark_hold(rgb, shadowed, lift, keep):
    """The share (0 to 1) of the light side's lift that a dark map can take: the largest at which the mean L* of
    its land stays at most that of the unbaked ground (rgb), from the shadows (shadowed) and the full lift. Measured on
    a regular sample of HOLD_SAMPLE pixels or so."""
    H, W = rgb.shape[:2]
    step = max(1, math.ceil(math.sqrt(H * W / HOLD_SAMPLE)))
    land = np.ones((H, W), bool) if keep is None else ~np.asarray(keep, dtype=bool)
    land = land[::step, ::step]
    if not land.any():
        return 1.0
    a, b, d = (v[::step, ::step][land] for v in (rgb, shadowed, lift))
    l0 = float(lstar_image(a * 255).mean())

    def mean_at(t):
        return float(lstar_image(np.clip(b + d * t, 0, 1) * 255).mean())

    if mean_at(1.0) <= l0:
        return 1.0
    lo, hi = 0.0, 1.0  # the shadows alone only darken: mean_at(0) <= l0
    for _ in range(16):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_at(mid) <= l0 else (lo, mid)
    return lo


def bake(rgb, s, dry, dark, keep=None):
    """The relief baked into the ground colours (rgb, float in [0, 1], (H, W, 3)) by the shade s ((H, W) in [-1,
    1]): shadows multiply each channel by 1 - ks * |s| * k (k from K_COOL to K_WARM with the dryness), light moves
    kl * s of the way to BAKE_LIGHT (BAKE[dark] = (ks, kl)). On dark paper the light side gives back at most what the
    shadows took (_dark_hold): a map whose slopes mostly face the light (a valley side lit from the north-west) does
    not print lighter than its ground, so the poster stays dark. keep: a mask of pixels left exactly as they were (the
    water). Returns uint8 (H, W, 3)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)[..., None]
    k = np.asarray(K_COOL) + (np.asarray(K_WARM) - np.asarray(K_COOL)) * float(dry)
    ks, kl = BAKE[bool(dark)]
    out = rgb * (1 - ks * np.maximum(-s, 0) * k)
    lift = (np.asarray(BAKE_LIGHT) - out) * kl * np.maximum(s, 0)
    if dark:
        lift *= _dark_hold(rgb, out, lift, keep)
    out = np.clip(np.round((out + lift) * 255), 0, 255).astype(np.uint8)
    if keep is not None:
        out[keep] = np.clip(np.round(rgb[keep] * 255), 0, 255).astype(np.uint8)
    return out


def lstar_image(rgb8):
    """CIE L* of every pixel of a uint8 (..., 3) sRGB image."""
    c = np.asarray(rgb8, dtype=np.float64) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(y > (6 / 29) ** 3, 116 * np.cbrt(y) - 16, y * (29 / 3) ** 3)


def draw_ground(ax, ground, extent, S, map_w_mm, shade=None):
    """The land cover as ONE image of exactly the map box (interpolation 'none': a PDF or SVG embeds that array as
    it is), clipped to the box, at the bottom of the map, its classes calmed (landcover.ground_palette) and, with a
    shade, the relief baked into it (bake; the water untouched). Returns its colours: {class: rgb} of the classes
    drawn, their shares of the map, the mean colour of the unbaked ground, whether the relief was baked in, and then
    ref_l, the L* that paths must keep their contrast against: the REF_PCT percentile of the printed land (None
    when nothing was baked)."""
    if ground.classes.shape != relief_raster_shape(map_w_mm, extent):
        raise ValueError("the land cover raster does not match the map")
    rgb = landcover.ground_rgb(ground.classes, S, ground.dry, px_mm=map_w_mm / ground.classes.shape[1])
    dark = _lstar(to_rgb(S["bg"])) < 50
    water = np.asarray(ground.classes) == landcover.WATER
    printed, ref_l = rgb, None
    if shade is not None:
        if np.shape(shade) != ground.classes.shape:
            raise ValueError("the relief raster does not match the map")
        printed = bake(rgb.astype(np.float64) / 255, shade, ground.dry, dark, keep=water)
        land = ~water
        if land.any():
            ref_l = float(np.percentile(lstar_image(printed[land]), REF_PCT[dark]))
    x0, x1, y0, y1 = extent
    im = ax.imshow(printed, extent=extent, origin="upper", interpolation="none", zorder=GROUND_Z)
    im.set_clip_path(PathPatch(_poly_path([_box(x0, y0, x1, y1)]), transform=ax.transData))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    shares = landcover.shares(ground.classes)
    land = landcover.land_lstar(S, shares, ground.dry)
    colours = landcover.ground_palette(ground.classes, S, ground.dry)
    return {"colours": {k: colours[k] for k in shares}, "shares": shares, "land": land,
            "mean": tuple(float(v) for v in rgb.reshape(-1, 3).mean(axis=0) / 255), "baked": shade is not None,
            "ref_l": ref_l}


def frame_map(ax, S, k):
    """A hairline round the map box in the colour and weight of the page's rules: over the land cover the map is a
    coloured rectangle, and snow or pale rock at its edge would melt into the paper without one."""
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=S["rule"],
                           linewidth=FRAME_PT * k, joinstyle="miter", zorder=FRAME_Z, clip_on=False))


INK_KEYS = ("trail", "road_minor", "road_major")  # the inks drawn over the relief: paths and tracks, then the roads


def shadow_inks(TS, S, ref_l, streets=False):
    """The paths and the roads of the trail style (with streets, a town's footways and steps too: the city poster's
    street colours, which draw_basemap uses) re-inked for a ground that darkens to the L* ref_l (lightens to it, on dark
    paper): each moves along the way from the paper to the style's inks until its L* contrast against the paper is
    the one it had plus the ground's shift, so against that ground it keeps the contrast it has against the paper.
    None goes past the page's own ink (S["ink"]), where a deep shadow would push the roads to pure black: the paths
    come first and go as far as they need, up to the ink; what is stronger than them (the roads) keeps its order
    between them and the ink, its distances compressed by one factor, so a minor road is never fainter than a path nor
    a major road than a minor one. None ends fainter than it was. A ground lighter than the paper (darker, on dark
    paper) changes nothing."""
    out = dict(TS)
    l_bg = _lstar(to_rgb(S["bg"]))
    dark = l_bg < 50
    shift = max(0.0, (ref_l - l_bg) if dark else (l_bg - ref_l))

    def gap(colour):
        return abs(_lstar(to_rgb(colour)) - l_bg)

    keys = [(key, TS[key]) for key in INK_KEYS]
    if streets:
        keys += [(("street", cls), colour) for cls, (colour, _) in TS["street"].items()]
    want = {key: shift + gap(colour) for key, colour in keys}
    cap = gap(S["ink"])
    paths, top = want["trail"], max(want.values())
    first = min(paths, cap)  # where the paths go
    squeeze = min(1.0, (cap - first) / (top - paths)) if top > paths else 1.0

    def target(key):
        w = want[key]
        return min(w, cap) if w <= paths else first + (w - paths) * squeeze

    def ink(key, colour):
        return _ink_at(S["bg"], (S["sub"], S["ink"]), max(gap(colour), target(key)))

    for key in INK_KEYS:
        out[key] = ink(key, TS[key])
    if streets:
        out["street"] = {cls: (ink(("street", cls), colour), width) for cls, (colour, width) in TS["street"].items()}
    return out


def ground_style(TS, S, painted):
    """The trail style over the land cover: its water (landcover.water), the relief measured on the ground's mean
    colour, and the paths, the roads and a town's footways and steps re-inked (shadow_inks) for the darkest ground
    colour that covers SIGNIFICANT of the map (the lightest on dark paper). With the relief baked into the ground,
    the reference is also the deep shadow (the light faces on dark paper): painted["ref_l"], when it is darker
    (lighter)."""
    out = dict(TS)
    out["relief"] = trail_relief(S, painted["mean"])
    land = painted.get("land")
    out["water"], out["water_line"] = (to_hex(landcover.water(S, land=land)),
                                       to_hex(landcover.water(S, line=True, land=land)))
    dark = _lstar(to_rgb(S["bg"])) < 50
    big = [c for k, c in painted["colours"].items() if painted["shares"][k] >= SIGNIFICANT] or [painted["mean"]]
    ref_l = (max if dark else min)(_lstar(c) for c in big)
    if painted.get("ref_l") is not None:
        ref_l = (max if dark else min)(ref_l, painted["ref_l"])
    return shadow_inks(out, S, ref_l, streets=True)


def overlay_ref_l(S, colours, shade):
    """The L* that paths over the drawn relief (the trail map without the land cover) must keep their contrast
    against: the paper under the overlay's shadow (colours, as relief_colours) at the REF_PCT[0] percentile of the
    shade, the deep shadow; on dark paper under its light at the REF_PCT[1] percentile, the bright faces. The paper's
    own L* when the map has no such side."""
    bg = np.asarray(to_rgb(S["bg"]))
    dark = _lstar(bg) < 50
    s = float(np.percentile(shade, REF_PCT[dark]))
    s = max(s, 0.0) if dark else min(s, 0.0)
    (colour, alpha) = colours[1] if dark else colours[0]
    a = alpha * min(1.0, abs(s) / RAMP_FULL) ** RAMP_GAMMA
    return _lstar((1 - a) * bg + a * np.asarray(colour, dtype=float))


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
    """(ink, note): the style's accent, unless it lies closer than COURSE_DE (CIEDE2000) to the ground colours under
    SINK_SHARE or more of the course's length (it would sink into them there); then the first route ink of
    ROUTE_INKS that lies at least COURSE_DE from every ground colour under UNDER_SHARE or more of the course (else the
    one whose nearest such colour lies farthest), and a note for the log that says why (without any place).
    under: {class: share of the course's length} (ground_under)."""
    if not under:
        return S["accent"], None
    gap = {c: landcover.delta_e(S["accent"], colours[c]) for c in sorted(under)}
    sink = {c: under[c] for c in gap if gap[c] < COURSE_DE}
    if sum(sink.values()) < SINK_SHARE:
        return S["accent"], None
    worst = max(sorted(sink), key=lambda c: sink[c])  # the ground it would sink into along most of the course
    seen = [colours[c] for c in gap if under[c] >= UNDER_SHARE] or [colours[worst]]

    def nearest(ink):
        return min(landcover.delta_e(ink, g) for g in seen)

    inks = ROUTE_INKS[_lstar(to_rgb(S["bg"])) < 50]
    name, ink = next((t for t in inks if nearest(t[1]) >= COURSE_DE), max(inks, key=lambda t: nearest(t[1])))
    share = sum(sink.values())
    where = "most of it" if share > 0.5 else f"{round(share * 100)} % of it"
    return ink, (f"the course is drawn in {name}: the style's accent would sink into the {landcover.NAMES[worst]} "
                 f"under {where} (colour difference {gap[worst]:.0f}, below {COURSE_DE:.0f})")


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


# --- contour lines ----------------------------------------------------------------------------------------------------

# The interval: the national maps' rule (at least the scale's denominator over CONTOUR_SCALE: 10 m at 1:25,000, 20 m
# at 1:50,000), coarser where the steep ground (the 90th percentile of the slope) would set the lines closer than
# CONTOUR_MIN_GAP_MM on paper. Every CONTOUR_INDEX-th line is an index line. Closed loops shorter than
# CONTOUR_MIN_LOOP_MM of paper are bumps of the data and go; lines are simplified to CONTOUR_SIMPLIFY_MM of paper.
CONTOUR_STEPS = (10, 20, 50, 100, 200, 500, 1000, 2000)
CONTOUR_SCALE = 3500
CONTOUR_MIN_GAP_MM = 0.42
CONTOUR_INDEX = 5
CONTOUR_SIMPLIFY_MM = 0.05
CONTOUR_MIN_LOOP_MM = 2.0
CONTOUR_PT = (0.20, 0.42)  # minor and index lines (pt x k)
CONTOUR_INK = {False: "#7a4a22", True: "#f0d7b4"}  # over the land cover, light and dark paper; without it S["sub"]
CONTOUR_ALPHA = {False: (0.30, 0.55), True: (0.16, 0.32)}  # minor and index lines, light and dark paper
# without the land cover the lines are in S["sub"], the colour the paths are inked towards (TRAIL_INK of the way): an
# index line stays at about half the paths' way from the paper, so the lines read as terrain and the paths as paths
CONTOUR_ALPHA_PLAIN = {False: (0.15, 0.28), True: (0.12, 0.24)}
CONTOUR_Z = 3.55  # over the ground (0.5) and the water (2), under the paths (3.6)


def contour_step(z, extent, map_w_mm, land=None):
    """The contour interval (m) of a map: the first of CONTOUR_STEPS at or above the scale's rule whose lines on the
    steep ground stay CONTOUR_MIN_GAP_MM apart on paper, else the last. land: a mask of the pixels to measure the
    slope on (all of them when None or empty)."""
    x0, x1, y0, y1 = extent
    H, W = np.shape(z)
    m_mm = (x1 - x0) / map_w_mm  # metres of the map per mm of paper
    gy, gx = np.gradient(np.asarray(z, dtype=np.float64), (y1 - y0) / H, (x1 - x0) / W)
    g = np.hypot(gx, gy)
    g = g[land] if land is not None and np.any(land) else g.ravel()
    p90 = float(np.percentile(g, 90))
    floor = m_mm * 1000 / CONTOUR_SCALE
    for step in CONTOUR_STEPS:
        if step >= 0.98 * floor and step / max(p90, 1e-6) / m_mm >= CONTOUR_MIN_GAP_MM:
            return step
    return CONTOUR_STEPS[-1]


def contour_lines(z, extent, step, map_w_mm, mask=None):
    """[(level, (n, 2) array in map units)]: the contour lines of the raster z (row 0 north) at every multiple of
    step but 0 (the sea's own level: the coast is the water's edge, drawn with the water; land below the sea, by the
    Dead Sea, keeps its lines), traced between the pixel centres (so every vertex lies inside the map), never through
    a pixel of mask (True where no line may run), without the short closed loops, simplified to what paper shows."""
    x0, x1, y0, y1 = extent
    H, W = np.shape(z)
    xs = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
    ys = y0 + (np.arange(H) + 0.5) * (y1 - y0) / H
    zz = np.asarray(z, dtype=np.float64)[::-1]  # south up, as ys
    if mask is not None and np.any(mask):
        zz = np.ma.masked_array(zz, mask=np.asarray(mask)[::-1])
    gen = contourpy.contour_generator(xs, ys, zz, line_type=contourpy.LineType.Separate)
    m_mm = (x1 - x0) / map_w_mm
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    out = []
    for level in np.arange(math.ceil(lo / step) * step, hi + step, step):
        if abs(level) < 1e-9 or level > hi:
            continue
        for line in gen.lines(float(level)):
            if len(line) < 3:
                continue
            ls = LineString(line)
            if np.allclose(line[0], line[-1]) and ls.length < CONTOUR_MIN_LOOP_MM * m_mm:
                continue
            c = np.asarray(shapely.simplify(ls, CONTOUR_SIMPLIFY_MM * m_mm).coords)
            if len(c) >= 2:
                out.append((float(level), c))
    return out


def _grow(mask):
    """The mask grown by one pixel (3 x 3)."""
    p = np.pad(np.asarray(mask, dtype=bool), 1)
    H, W = np.shape(mask)
    return np.any([p[i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=0)


def draw_contours(ax, z, extent, S, k, map_w_mm, water, classes=None):
    """Contour lines of the heights z (the map raster), as two compound paths (minor and index lines) under the paths
    and over the ground, clipped to the map minus the drawn water and, with the land cover (classes), kept a pixel
    off its water. In a brown ink over the land cover, else the style's sub colour, fainter than the paths
    (CONTOUR_ALPHA_PLAIN). Returns {"step", "lines", "vertices"}, or None when nothing of the map is land."""
    clip = land_clip_path(extent, water, map_w_mm)
    if clip is None:
        return None
    water_px = None if classes is None else np.asarray(classes) == landcover.WATER
    step = contour_step(z, extent, map_w_mm, None if water_px is None else ~water_px)
    lines = contour_lines(z, extent, step, map_w_mm, None if water_px is None else _grow(water_px))
    dark = _lstar(to_rgb(S["bg"])) < 50
    ink, alphas = (CONTOUR_INK[dark], CONTOUR_ALPHA[dark]) if classes is not None else (S["sub"],
                                                                                        CONTOUR_ALPHA_PLAIN[dark])
    for index in (False, True):
        runs = [c for level, c in lines if (round(level / step) % CONTOUR_INDEX == 0) == index]
        path = _line_path(runs)
        if path is None:
            continue
        patch = PathPatch(path, fill=False, edgecolor=ink, alpha=alphas[index],
                          linewidth=CONTOUR_PT[index] * k, capstyle="round", joinstyle="round", zorder=CONTOUR_Z)
        ax.add_patch(patch)
        patch.set_clip_path(PathPatch(clip, transform=ax.transData))
    return {"step": step, "lines": len(lines), "vertices": int(sum(len(c) for _, c in lines))}


def _grid_xy(extent, x, y, map_h=MAP_H):
    """Map coordinates to the A3 layout grid (mm, y down)."""
    x0, x1, y0, y1 = extent
    return (LEFT + (np.asarray(x) - x0) / (x1 - x0) * (RIGHT - LEFT),
            MAP_TOP + (y1 - np.asarray(y)) / (y1 - y0) * map_h)


TRI_W, TRI_UP, TRI_DOWN = 2.4, 1.4, 0.7  # the summit triangle (grid mm): width, apex above and base below the point
EDGE_MM, GAP_MM, ROUTE_GAP_MM = 1.2, 0.8, 0.5
ROUTE_CLEAR_MM = CASING_PT * PT_MM / 2 + ROUTE_GAP_MM  # a summit label keeps this far from the course's centre line


SAME_PEAK_M = 300.0  # two summits that share a base name this close are one mountain: only the higher is named
_SUBNAME = re.compile(r"\s+[-\u2013\u2014/]\s+")  # "Aiguilles Crochues - Sommet S", "Cima Grande - Grosse Zinne"


def peak_label(name, ele):
    return f"{name.strip().upper()} {ele:.0f} m" if ele is not None else name.strip().upper()


def base_name(name):
    """The part of a summit's name before a spaced dash or slash: the mountain of a named sub-summit (Aiguilles
    Crochues - Sommet S), or the first language of a name in two (Cima Grande - Grosse Zinne)."""
    return _SUBNAME.split(name.strip(), maxsplit=1)[0].strip()


def place_peaks(peaks, extent, k, course, max_n=MAX_PEAKS, map_h=MAP_H):
    """The summits to draw: [(gx, gy, text, (x0, y0, x1, y1) label box, (x, y) label top-left)] in the layout grid.

    peaks: (x, y, name, ele) in map units. Only named summits inside the map, highest first (unknown heights last),
    at most max_n. A summit within SAME_PEAK_M of a higher one of the same base name (base_name) is that mountain's
    other top: it is not named, and the higher one is named by the base name alone. Each label goes right of its
    triangle, else left, above, below or at a corner, at the first of those places that hides no summit still to come
    (else the one whose first hidden summit comes latest: the lowest); a summit whose label would leave the map,
    touch the course's casing or another summit's label or triangle wherever it goes is skipped, so nothing outside
    the visible map is drawn. course: the drawn course lines in map units."""
    cand = []
    for x, y, name, ele in peaks:
        if not isinstance(name, str) or not name.strip():
            continue
        gx, gy = (float(v) for v in _grid_xy(extent, x, y, map_h))
        if not (LEFT + EDGE_MM + TRI_W <= gx <= RIGHT - EDGE_MM - TRI_W
                and MAP_TOP + EDGE_MM + TRI_UP <= gy <= MAP_TOP + map_h - EDGE_MM - TRI_DOWN):
            continue
        cand.append((ele is None, -(ele or 0.0), name.strip(), gx, gy, ele, float(x), float(y)))
    if not cand:
        return []
    cand.sort(key=lambda c: c[:3])
    names, kept = [c[2] for c in cand], []
    for i, c in enumerate(cand):  # the other tops of a mountain go; its highest top takes the mountain's name
        base = base_name(c[2]).casefold()
        twin = next((j for j in kept if base_name(cand[j][2]).casefold() == base
                     and math.hypot(cand[j][6] - c[6], cand[j][7] - c[7]) <= SAME_PEAK_M), None)
        if twin is None:
            kept.append(i)
        else:
            names[twin] = base_name(cand[twin][2])
    cand = [cand[i][:2] + (names[i],) + cand[i][3:6] for i in kept]
    frame = _box(LEFT + EDGE_MM, MAP_TOP + EDGE_MM, RIGHT - EDGE_MM, MAP_TOP + map_h - EDGE_MM)
    lines = [np.column_stack(_grid_xy(extent, *np.asarray(c, dtype=float).T, map_h)) for c in course if len(c) > 1]
    # a label keeps clear of the course by distance, not by a buffer of it: a stop where the fix wandered for twenty
    # minutes crosses itself hundreds of times, and buffering that takes minutes
    route = shapely.MultiLineString(lines) if lines else None
    if route is not None:
        shapely.prepare(route)
    tris = [_box(gx - TRI_W / 2, gy - TRI_UP, gx + TRI_W / 2, gy + TRI_DOWN) for *_, gx, gy, _ in cand]
    others = shapely.STRtree(tris)
    marks, placed = [], []  # the labels and triangles drawn so far, padded
    for i, (_, _, name, gx, gy, ele) in enumerate(cand):
        if len(placed) >= max_n:
            break
        text = peak_label(name, ele)
        _, _, w, h = place_text(text, 0, 0, k, PEAK_PT, TEXT)
        tri = tris[i]
        if any(tri.intersects(b) for b in marks):  # under another summit's label or triangle (the course may run
            continue                              # over a summit: its triangle is drawn on top)
        cy = gy - (TRI_UP - TRI_DOWN) / 2
        right, left = gx + TRI_W / 2 + 0.7, gx - TRI_W / 2 - 0.7 - w
        up, down = gy - TRI_UP - 0.7 - h, gy + TRI_DOWN + 0.7
        options = ((right, cy - h / 2), (left, cy - h / 2), (gx - w / 2, up), (gx - w / 2, down),  # beside, then
                   (right, up), (right, down), (left, up), (left, down))  # diagonal: a summit on the course
        best = None  # (the first summit still to come that the label hides, or len(cand) for none; x; y; label)
        for lx, ly in options:
            label = _box(lx, ly, lx + w, ly + h)
            if not (frame.contains(label) and (route is None or not shapely.dwithin(route, label, ROUTE_CLEAR_MM))
                    and not any(label.intersects(b) for b in marks)):
                continue
            padded = label.buffer(GAP_MM, join_style="mitre")
            hides = min((int(j) for j in others.query(padded, predicate="intersects") if j > i), default=len(cand))
            if best is None or hides > best[0]:
                best = (hides, lx, ly, label)
            if hides == len(cand):
                break
        if best is not None:
            _, lx, ly, label = best
            placed.append((gx, gy, text, label.bounds, (lx, ly)))
            marks += [label.buffer(GAP_MM, join_style="mitre"), tri.buffer(GAP_MM / 2, join_style="mitre")]
    return placed


def draw_summits(ax, summits, S, k):
    """The placed summits (place_peaks) on the page axes (millimetres, y down): a triangle and the label in the ink,
    both with a halo in the paper colour, so a path, a contour or a shadow that runs under a name does not run
    through its letters, as on a topographic map. The halo's corners are round: a mitred stroke sends spikes out of
    the sharp corners of letters (the M, the V)."""
    halo = [patheffects.withStroke(linewidth=HALO_PT * k, foreground=S["bg"], joinstyle="round")]
    for gx, gy, text, _, (lx, ly) in summits:
        tri = [((gx - TRI_W / 2) * k, (gy + TRI_DOWN) * k), ((gx + TRI_W / 2) * k, (gy + TRI_DOWN) * k),
               (gx * k, (gy - TRI_UP) * k)]
        ax.add_patch(MPolygon(tri, closed=True, facecolor=S["ink"], edgecolor="none", lw=0, path_effects=halo))
        path, trans, _, _ = place_text(text, lx, ly, k, PEAK_PT, TEXT)
        ax.add_patch(PathPatch(path, transform=trans + ax.transData, color=S["ink"], lw=0, snap=False,
                               path_effects=halo))


# --- the page ---------------------------------------------------------------------------------------------------------

UNIT_PT, UNIT_GAP_MM = 11, 1.2  # the units beside the numbers (pt x k, A3-grid mm), on the numbers' baseline


def trail_surface(relief, ground, map_w_mm):
    """(shade, heights) of the trail map, or (None, None) without relief: the crisp shade and heights of the surface,
    and where the land cover has built-up land (built_weight), those of the city ground filter fading in, so a town's
    roofs print neither as speckle nor as contour rings. Without the land cover (--ground style) or without the smooth
    surface: relief.shade and relief.z."""
    if relief is None:
        return None, None
    shade, z = relief.shade, getattr(relief, "z", None)
    smooth = getattr(relief, "smooth", None)
    w = built_weight(ground.classes, map_w_mm) if ground is not None and smooth is not None else None
    if w is None or w.shape != shade.shape:
        return shade, z
    shade = (shade * (1 - w) + smooth * w).astype(np.float32)
    if z is not None and getattr(relief, "z_smooth", None) is not None:
        z = (z * (1 - w) + relief.z_smooth * w).astype(np.float32)
    return shade, z


def draw_profile_fill(pax, x, e, base, line, paper):
    """The area under the profile as ONE opaque image of 256 x 1 pixels, a gradient pre-composited on the paper from
    PROFILE_FILL[0] of the way to the line colour at the profile's top to PROFILE_FILL[1] at its foot, clipped to the
    area under the line. Opaque: a PDF holds it without a soft mask."""
    t = np.linspace(*PROFILE_FILL, 256)[:, None, None]  # row 0 at the top
    p, c = np.asarray(to_rgb(paper)), np.asarray(to_rgb(line))
    img = np.clip(np.round((p + (c - p) * t) * 255), 0, 255).astype(np.uint8)
    top = float(np.max(e))
    im = pax.imshow(img, extent=(float(x[0]), float(x[-1]), base, top), aspect="auto", origin="upper",
                    interpolation="none", zorder=1)
    verts = np.concatenate([[[x[0], base]], np.column_stack([x, e]), [[x[-1], base], [x[0], base]]])
    codes = np.full(len(verts), MPath.LINETO, dtype=np.uint8)
    codes[0], codes[-1] = MPath.MOVETO, MPath.CLOSEPOLY
    im.set_clip_path(PathPatch(MPath(verts, codes), transform=pax.transData))
    return im


def render(activity, track_xy, pieces, basemap, extent, out_path, style="huella", size="A3", dpi=300, title="",
           place="", name="", date=None, lang="en", color_by="single", pdf_path=None, svg_path=None, preview_path=None,
           times=None, ele=None, stretch=None, relief=None, peaks=(), ground=None, log=None, scale=1.0):
    """activity: the route; track_xy: its (end-cleaned) projected track, with times and ele aligned to it (default:
    the activity's); pieces: the privacy-cut pieces (index, LineString), drawn; stretch: (start_m, end_m) along the
    track that the numbers describe (default: all of it); place: the subline (region, country); date: the local
    datetime of the activity; relief: a relief.Relief for the map; peaks: (x, y, name, ele) of OSM summits in map
    units, of which place_peaks draws up to six inside the map; ground: a landcover.LandCover for the map (the
    colours of the land under it); log: where to say which ink the course took over the ground, and why; scale: the
    share of the track's own length the route really measured (race.distance_scale), for its length, pace and the
    profile's km, never its drawing."""
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
    real = measured_along(xy, times, activity.type) * scale  # its numbers' metres; the drawing keeps dist
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
    # the crisp shade and heights of the surface, and over a town (the land cover's built-up land, feathered) those
    # of the city ground filter, where the roofs would print as speckle
    shade, z = trail_surface(relief, ground, map_w)
    painted = draw_ground(ax, ground, extent, S, map_w, shade=shade) if ground is not None else None
    if painted is not None:  # the land cover already says where the woods and meadows are
        frame_map(ax, S, k)
        TS = ground_style(TS, S, painted)
        streets = (streets[0], streets[1].iloc[:0], streets[2])
    elif shade is not None:  # the relief drawn over the paper: the paths and roads over it keep their contrast in its
        TS = shadow_inks(TS, S, overlay_ref_l(S, TS["relief"], shade))  # shadow (the town's footways lie under it)
    draw_basemap(ax, streets, extent, TS, k, map_w)
    if painted is not None and painted["baked"]:
        drawn = True  # the relief is in the ground's colours
    else:
        drawn = draw_relief(ax, relief, extent, TS, basemap[2], k, map_w, shade=shade) if relief is not None else False
    contours = None
    if z is not None:
        contours = draw_contours(ax, z, extent, TS, k, map_w, basemap[2],
                                 classes=ground.classes if painted is not None else None)
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
            v = np.interp(at, dist, moving_pace(real, times, pauses))
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
    draw_routes(ax, segments, np.full(len(segments), 30.0), 3.0, cmap, k, base=COURSE_PT, gain=0.0,
                glow=S.get("glow", 0.0), values=values)
    summits = place_peaks(peaks, extent, k, coords, map_h=grid_h)
    draw_summits(page.ax, summits, S, k)

    # elevation profile under the map: the course's line over a gradient from it to the paper, the high point marked
    hi_point = None
    if ele:
        _, e = profile(xy, ele)
        d = real
        span = max(e.max() - e.min(), PROFILE_SPAN_M)  # a flat route stays flat, not a sawtooth of altimeter noise
        base, top = e.min() - 0.08 * span, e.max() + 0.24 * span
        pax = fig.add_axes([LEFT * k / wmm, 1 - (PROFILE_TOP + PROFILE_H) * k / hmm, (RIGHT - LEFT) * k / wmm,
                            PROFILE_H * k / hmm], zorder=2)
        pax.set_facecolor("none")
        line = ink if legend["mode"] == "single" else S["heat"][3]
        draw_profile_fill(pax, d / 1000, e, base, line, S["bg"])
        pax.plot(d / 1000, e, color=line, linewidth=1.1 * k, solid_capstyle="round", zorder=2)
        pax.set_xlim(0, d[-1] / 1000)
        pax.set_ylim(base, top)
        pax.axis("off")
        hd, he = high_point(xy, ele)
        hd = float(np.interp(hd, dist, real))
        hi_point = he
        gx = LEFT + hd / d[-1] * (RIGHT - LEFT)
        gy = PROFILE_TOP + (top - he) / (top - base) * PROFILE_H
        page.ax.add_patch(plt.Circle((gx * k, gy * k), 0.75 * k, color=S["ink"], lw=0, zorder=3))
        label = f"{he:.0f} m"
        _, _, lw_, lh_ = place_text(label, 0, 0, k, 6.0, TEXT)
        T(min(max(gx, LEFT + lw_ / 2), RIGHT - lw_ / 2), gy - 1.3 - lh_, label, pt=6.0, align="center")
        step = km_step(d[-1])
        for km_mark in range(0, int(d[-1] // 1000) + 1, step):
            T(LEFT + km_mark * 1000 / d[-1] * (RIGHT - LEFT), PROFILE_TOP + PROFILE_H + 2, str(km_mark), pt=4.4,
              color=S["sub"], align="center")
        T(18, PROFILE_TOP - 3.2, "PERFIL" if es else "ELEVATION", pt=5.6, color=S["sub"])
        T(279, PROFILE_TOP - 3.2, f"{e.min():.0f} – {e.max():.0f} m", pt=5.6, color=S["sub"], align="right")

    # numbers
    ytop = NUMBERS_TOP
    rule(ytop - 3)
    km = real[-1] / 1000
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
    base = baseline_of("0", k, pt=40, font=DISPLAY, height=11)
    for i, (value, unit, label) in enumerate(cells):
        x = LEFT + i * col
        unit_w = place_text(unit, 0, 0, k, UNIT_PT, REGULAR)[2] + UNIT_GAP_MM if unit else 0.0
        fit = dict(pt=40, font=DISPLAY, max_w=col - 9 - unit_w, height=11)
        # a value shrunk to its column (POINT TO POINT) sits on the numbers' baseline, not under their top
        w_, _ = T(x, ytop + 1 + base - baseline_of(value, k, **fit), value, **fit)
        if unit:  # on the numbers' baseline
            T(x + w_ + UNIT_GAP_MM, ytop + 1 + base - baseline_of(unit, k, pt=UNIT_PT, font=REGULAR), unit,
              pt=UNIT_PT, font=REGULAR, color=S["accent"])
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
            "course_ink": ink if legend["mode"] == "single" else None,
            "contours": contours["step"] if contours else None}
