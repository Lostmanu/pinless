"""Huella: a personal running archive, composed as an A-series art print.

All paths come from privacy-cut geometry. Density changes ink colour and stroke
weight only; no invented routes, snapping, endpoint markers or soft glow.
The PDF/SVG use vector streets, routes and outlined, bundled OFL typography.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from shapely.geometry import box as _box
from shapely.geometry.polygon import orient

from . import colouring, landcover, privacy
from . import relief as _relief
from .graphs import draw_band
from .render import MAJOR, MINOR, MM, SIZES, _hw, heat_raster
from .render_sport import _num
from .stats import fmt_time

FONT_DIR = Path(__file__).resolve().parent / "fonts"  # bundled OFL fonts (also in assets/fonts with their README)
if not FONT_DIR.exists():
    FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
DISPLAY = FontProperties(fname=str(FONT_DIR / "BarlowCondensed-ExtraBold.ttf"))
REGULAR = FontProperties(fname=str(FONT_DIR / "BarlowCondensed-Regular.ttf"))
TEXT = FontProperties(fname=str(FONT_DIR / "Manrope-Medium.ttf"))

STYLES = {
    "huella": {
        "bg": "#f5f0e6", "ink": "#222820", "sub": "#736f65", "accent": "#b33b25",
        "green": "#edf0e4", "water": "#e2e9e2", "water_line": "#cedcd3", "rule": "#bdb7aa",
        "street": {"major": ("#cdc8bc", 0.50), "minor": ("#d7d1c4", 0.32), "path": ("#dbd4c6", 0.24)},
        "heat": ["#cb694e", "#bd5036", "#a63724", "#782b20", "#46231d"],
    },
    "carbon": {
        "bg": "#131e19", "ink": "#f2eddf", "sub": "#b0b5a5", "accent": "#dfa575",
        "green": "#1a261f", "water": "#1f3028", "water_line": "#2c4137", "rule": "#586055",
        "street": {"major": ("#394a3b", 0.50), "minor": ("#2f4133", 0.32), "path": ("#2b4131", 0.24)},
        "heat": ["#bc875d", "#d19e6d", "#e0b582", "#eed0a8", "#fff2d7"],
    },
    "midnight": {
        "bg": "#0f1626", "ink": "#eef2f7", "sub": "#8a96ab", "accent": "#5fd4ff",
        "green": "#121c2f", "water": "#16223a", "water_line": "#22324d", "rule": "#2c3852",
        "street": {"major": ("#303d59", 0.50), "minor": ("#263149", 0.32), "path": ("#222d42", 0.24)},
        "heat": ["#1f6f9c", "#2aa3d6", "#5fd4ff", "#b7f0ff", "#ffffff"], "glow": 0.16,
    },
    "blueprint": {
        "bg": "#1d4a86", "ink": "#f3f7fc", "sub": "#b4c8e3", "accent": "#ffd45c",
        "green": "#204f8d", "water": "#1a4479", "water_line": "#3c6aa6", "rule": "#5d87bd",
        "street": {"major": ("#5681b7", 0.50), "minor": ("#4976ac", 0.32), "path": ("#426fa7", 0.24)},
        "heat": ["#a9c8ee", "#d3e4f8", "#ffffff", "#ffe9a8", "#ffd45c"],
    },
    "ink": {
        "bg": "#fbfaf7", "ink": "#111111", "sub": "#6b6b6b", "accent": "#111111",
        "green": "#f1f1ec", "water": "#e7eaec", "water_line": "#d6dadd", "rule": "#c9c9c4",
        "street": {"major": ("#cdccc6", 0.50), "minor": ("#d7d6d0", 0.32), "path": ("#deddd7", 0.24)},
        "heat": ["#9c9c9c", "#6e6e6e", "#444444", "#222222", "#000000"],
    },
    "graphite": {
        "bg": "#222324", "ink": "#f1f1ef", "sub": "#9b9d9b", "accent": "#ff6b3d",
        "green": "#26282a", "water": "#2b2f33", "water_line": "#383d42", "rule": "#4a4c4e",
        "street": {"major": ("#404345", 0.50), "minor": ("#343739", 0.32), "path": ("#313437", 0.24)},
        "heat": ["#6a6c6e", "#9a9c9e", "#c9cbcc", "#ebecec", "#ffffff"], "glow": 0.10,
    },
    "neon": {
        "bg": "#0c0812", "ink": "#f6f0ff", "sub": "#9c8fb3", "accent": "#ff3d9a",
        "green": "#120d1a", "water": "#140f20", "water_line": "#221a33", "rule": "#33284a",
        "street": {"major": ("#2e253d", 0.50), "minor": ("#241d32", 0.32), "path": ("#201a2d", 0.24)},
        "heat": ["#6a1fa0", "#b5179e", "#f72585", "#ff8c42", "#ffe066"], "glow": 0.20,
    },
    "lavender": {
        "bg": "#efebf5", "ink": "#2a2140", "sub": "#7d7395", "accent": "#6741d9",
        "green": "#e8e4ef", "water": "#e2e2f0", "water_line": "#d0d0e4", "rule": "#cfc8dc",
        "street": {"major": ("#d8d0e3", 0.50), "minor": ("#e0d9eb", 0.32), "path": ("#e6e0ed", 0.24)},
        "heat": ["#c0b3e8", "#9b87d8", "#7655c7", "#5233a3", "#2f1b66"],
    },
    "sunset": {
        "bg": "#fbf1e6", "ink": "#2a1d2e", "sub": "#8a7468", "accent": "#e8590c",
        "green": "#f4ecdc", "water": "#ece6e4", "water_line": "#ddd2cf", "rule": "#d8c8b8",
        "street": {"major": ("#e5d9c9", 0.50), "minor": ("#e9dfd1", 0.32), "path": ("#ebe3d8", 0.24)},
        "heat": ["#ffb35c", "#ff8a2b", "#f25d1d", "#c2255c", "#5f3dc4"],
    },
    "forest": {
        "bg": "#eef1e8", "ink": "#1f2d1f", "sub": "#6d7a68", "accent": "#2f6b3a",
        "green": "#e1e8d6", "water": "#dde7e4", "water_line": "#c7d6d1", "rule": "#c3cbbb",
        "street": {"major": ("#d0d7c7", 0.50), "minor": ("#dbe1d2", 0.32), "path": ("#e1e6d9", 0.24)},
        "heat": ["#a3c39b", "#6fa065", "#3f7a45", "#25522c", "#13301a"],
    },
    "blush": {
        "bg": "#f8ece9", "ink": "#3a1f24", "sub": "#9a7b7f", "accent": "#c2185b",
        "green": "#f2e6e1", "water": "#ece4e6", "water_line": "#dfd2d5", "rule": "#dccbc8",
        "street": {"major": ("#e7d4d1", 0.50), "minor": ("#ecddda", 0.32), "path": ("#eee2df", 0.24)},
        "heat": ["#f2a7b8", "#e06688", "#c2185b", "#880e4f", "#4a0629"],
    },
    "ocean": {
        "bg": "#edf4f7", "ink": "#0f2a3d", "sub": "#6b8595", "accent": "#0b6e99",
        "green": "#e4eeee", "water": "#d3e6ee", "water_line": "#bcd6e2", "rule": "#bfd0d9",
        "street": {"major": ("#d1dfe6", 0.50), "minor": ("#d7e4ea", 0.32), "path": ("#dde7ec", 0.24)},
        "heat": ["#8fd0e8", "#3fa9d1", "#0b6e99", "#0a4a6e", "#062a40"],
    },
}
# A poster for two: five inks per style, in the order first person, mostly first, both, mostly second, second.
# "both" is the style's ink (the darkest line on light paper, the brightest on dark paper) and the first person
# always takes the warm ink. Each "mostly" ink is a deeper (light paper) or quieter (dark paper) tone of its person's
# ink, at least 15 L* from "both": a thin line of it must not print as the shared street's black or white. Checked
# in CIEDE2000, also under simulated colour blindness (tests/test_couple_poster.py).
WHO = {
    "huella": ["#c4462c", "#94011a", "#222820", "#384a5d", "#2b6b8f"],
    "carbon": ["#e58a4e", "#e3baab", "#f2eddf", "#b4c5cc", "#7cc4d6"],
    "midnight": ["#ffb45c", "#d8c4a7", "#eef2f7", "#adc2d9", "#4fc3ff"],
    "blueprint": ["#ffd45c", "#d9c0a2", "#f3f7fc", "#fdadd3", "#ff7ad9"],
    "ink": ["#d1462f", "#7a0116", "#111111", "#042e6e", "#1f5fa8"],
    "graphite": ["#ff7a45", "#edbaaf", "#f1f1ef", "#b5c3dd", "#59b8ff"],
    "neon": ["#ffb020", "#d9c49b", "#f6f0ff", "#a8c4d3", "#3ad6ff"],
    "lavender": ["#b8641a", "#7f2d01", "#2a2140", "#642c8a", "#5b3fd1"],
    "sunset": ["#c9510c", "#871300", "#2a1d2e", "#393594", "#3c5fc4"],
    "forest": ["#9e7020", "#713f00", "#1f2d1f", "#444a6b", "#2c5f9e"],
    "blush": ["#d4583b", "#97081b", "#3a1f24", "#27487f", "#3b6fa8"],
    "ocean": ["#c55a22", "#901a05", "#0f2a3d", "#005988", "#1a86b8"],
}
for _name, _inks in WHO.items():
    STYLES[_name]["who"] = _inks


LEVELS = 32  # route colour and weight steps: finer than the eye separates on paper, one path per step in a PDF/SVG
PRINT_TOL_MM = 0.04  # geometry is simplified to this, on paper: far below what a print shows
SHARED_MM = 0.5  # two lines this close on paper read as one street
SMOOTH_MM = 1.5  # a change of whose street it is, shorter than this along a line, is a fleck: it takes its neighbours'
COUPLE_BASE = 0.62  # the thinnest couple stroke (pt x k): a hairline keeps its hue
ROUTE_GAIN = 0.42  # how much a route thickens from its rarest to its commonest weight (pt x k); the key uses it too


class Couple(NamedTuple):
    """What a poster for two draws besides the pieces: the two names, whose each piece's activity is (owner[i] is 0
    for the first person, 1 for the second), each person's stats.compute numbers, stats.together and how many
    streets both ran (None when the basemap names none)."""
    names: tuple
    owner: np.ndarray
    stats: tuple
    together: dict
    streets_both: int | None


def svg_rc(salt):
    """The settings of every SVG export, whatever the user's matplotlibrc says: a fixed hash salt, text as paths,
    and images inside the file. With svg.image_inline off, matplotlib would write each image as a PNG next to the
    SVG and put that file's local path (user name included) into the SVG."""
    return {"svg.hashsalt": salt, "svg.fonttype": "path", "svg.image_inline": True}


def _clip_simplify(geoms, extent, tol):
    """Cut to the map box and simplify to tol (map units): nothing outside the map is written, nothing visible lost."""
    x0, x1, y0, y1 = extent
    g = shapely.simplify(shapely.clip_by_rect(np.asarray(list(geoms), dtype=object), x0, y0, x1, y1), tol)
    return [x for x in g if x is not None and not x.is_empty]


def _parts(g, kinds):
    for p in getattr(g, "geoms", [g]):
        if p.geom_type in kinds and not p.is_empty:
            yield p
        elif p.geom_type.startswith(("Multi", "Geometry")):
            yield from _parts(p, kinds)


def _line_path(lines):
    """One compound path for many polylines (arrays or LineStrings): one element in a PDF/SVG instead of one each."""
    verts, codes = [], []
    for a in lines:
        a = np.asarray(a.coords if hasattr(a, "coords") else a)[:, :2]
        if len(a) < 2:
            continue
        c = np.full(len(a), MPath.LINETO, dtype=np.uint8)
        c[0] = MPath.MOVETO
        verts.append(a)
        codes.append(c)
    return MPath(np.concatenate(verts), np.concatenate(codes)) if verts else None


def _poly_path(polys):
    """One compound filled path; outer rings counterclockwise and holes clockwise, so holes stay empty."""
    verts, codes = [], []
    for poly in polys:
        poly = orient(poly, 1.0)
        for ring in [poly.exterior, *poly.interiors]:
            a = np.asarray(ring.coords)[:, :2]
            if len(a) < 4:
                continue
            c = np.full(len(a), MPath.LINETO, dtype=np.uint8)
            c[0], c[-1] = MPath.MOVETO, MPath.CLOSEPOLY
            verts.append(a)
            codes.append(c)
    return MPath(np.concatenate(verts), np.concatenate(codes)) if verts else None


def draw_basemap(ax, basemap, extent, S, k, map_w_mm, sw=1.0):
    """Green, water (sea included) and streets, as one compound path per kind.

    OSM walk graphs hold every street twice (one per direction), so each geometry is drawn once whatever its
    direction. Everything is cut to the map and simplified to PRINT_TOL_MM on paper."""
    edges, green, water = basemap
    tol = (extent[1] - extent[0]) / map_w_mm * PRINT_TOL_MM
    for layer, color, z in ((green, S["green"], 1), (water, S["water"], 2)):
        polys = [p for g in _clip_simplify(layer.geometry[layer.geometry.type.isin(["Polygon", "MultiPolygon"])],
                                           extent, tol) for p in _parts(g, ("Polygon",))]
        path = _poly_path(polys)
        if path is not None:
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", lw=0, zorder=z))
    wl = water.geometry[water.geometry.type.isin(["LineString", "MultiLineString"])]
    path = _line_path(p for g in _clip_simplify(wl, extent, tol) for p in _parts(g, ("LineString",)))
    if path is not None:
        ax.add_patch(PathPatch(path, fill=False, edgecolor=S["water_line"], linewidth=0.7 * k * sw, zorder=2))
    geoms = np.asarray(edges.geometry.values, dtype=object)
    keep = ~pd.Series(shapely.to_wkb(shapely.normalize(geoms))).duplicated().to_numpy()
    edges = edges[keep]
    hw = edges["highway"].map(_hw)
    for cls, mask in (("path", ~hw.isin(MAJOR | MINOR)), ("minor", hw.isin(MINOR)), ("major", hw.isin(MAJOR))):
        color, width = S["street"][cls]
        path = _line_path(p for g in _clip_simplify(edges[mask].geometry, extent, tol)
                          for p in _parts(g, ("LineString",)))
        if path is not None:
            ax.add_patch(PathPatch(path, fill=False, edgecolor=color, linewidth=width * k * sw, capstyle="round",
                                   joinstyle="round", zorder=3))


def _lstar(rgb):
    """CIE L* of an sRGB colour (components in [0, 1])."""
    c = np.asarray(rgb, dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Y = float(lin @ [0.2126, 0.7152, 0.0722])
    f = Y ** (1 / 3) if Y > (6 / 29) ** 3 else Y / (3 * (6 / 29) ** 2) + 4 / 29
    return 116 * f - 16


def _relief_alpha(bg, colour, target, cap):
    """The alpha at which colour over bg (sRGB compositing) changes L* by target (bisection), or cap if it never
    gets there."""
    bg, colour = np.asarray(bg, dtype=float), np.asarray(colour, dtype=float)
    l0 = _lstar(bg)

    def d(a):
        return abs(_lstar((1 - a) * bg + a * colour) - l0)

    if d(cap) <= target:
        return cap
    lo, hi = 0.0, cap
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if d(mid) < target else (lo, mid)
    return lo


RELIEF_KEEP = 0.70  # under full light, every street, park and water line keeps this share of its contrast


def _keep_alpha(S, colour, cap):
    """The largest alpha (up to cap) at which colour laid over the map leaves every basemap colour drawn under the
    relief (the three street classes, parks, water lines) at least RELIEF_KEEP of its L* contrast with the paper."""
    bg, colour = np.asarray(to_rgb(S["bg"])), np.asarray(colour, dtype=float)
    under = [np.asarray(to_rgb(c)) for c in [S["street"][k][0] for k in ("major", "minor", "path")]
             + [S["green"], S["water_line"]]]
    under = [(u, abs(_lstar(u) - _lstar(bg))) for u in under]

    def kept(a):
        top = _lstar((1 - a) * bg + a * colour)
        return min(abs(_lstar((1 - a) * u + a * colour) - top) / d0 for u, d0 in under if d0 > 0)

    if kept(cap) >= RELIEF_KEEP:
        return cap
    lo, hi = 0.0, cap
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kept(mid) >= RELIEF_KEEP else (lo, mid)
    return lo


def relief_colours(S):
    """((shadow rgb, alpha), (light rgb, alpha)) of the relief in a style, calibrated so the relief stays quieter
    than the faintest route ink: in L*, light styles darken by at most min(16, 0.5 route) and lighten by min(4, 0.2
    route); dark styles darken by min(8, 0.6 L*bg, 0.5 route) and lighten by min(12, 0.35 route), where route is the
    L* contrast of the first route colour. No shadow goes past an alpha of 0.60, no light past 0.70.

    The light side is also held to what the map under it can bear: over light paper a white veil lifts the paper by
    only a few L* steps while it fades the streets and parks under it, so its alpha is capped where they keep
    RELIEF_KEEP of their contrast. S["relief"] overrides it all."""
    if "relief" in S:
        return S["relief"]
    bg = to_rgb(S["bg"])
    lb = _lstar(bg)
    route = abs(lb - _lstar(to_rgb(S["heat"][0])))
    if lb >= 50:
        shadow, ts, light, tl = S["ink"], min(16, 0.5 * route), "#ffffff", min(4, 0.2 * route)
    else:
        shadow, ts, light, tl = "#000000", min(8, 0.6 * lb, 0.5 * route), S["ink"], min(12, 0.35 * route)
    a_light = min(_relief_alpha(bg, to_rgb(light), tl, 0.70), _keep_alpha(S, to_rgb(light), 0.70))
    return ((to_rgb(shadow), _relief_alpha(bg, to_rgb(shadow), ts, 0.60)), (to_rgb(light), a_light))


RAMP_FULL, RAMP_GAMMA = _relief.RAMP_FULL, _relief.RAMP_GAMMA


def relief_rgba(shade, colours):
    """uint8 (H, W, 4): the shadow colour where the shade is negative, the light colour elsewhere, with alpha
    the side's alpha * min(1, |shade| / RAMP_FULL) ** RAMP_GAMMA. Flat ground (0) is exactly transparent."""
    (shadow, a_s), (light, a_l) = colours
    s = np.asarray(shade, dtype=np.float32)
    neg = s < 0
    rgba = np.empty(s.shape + (4,), np.uint8)
    rgba[..., :3] = np.where(neg[..., None], np.round(np.asarray(shadow) * 255), np.round(np.asarray(light) * 255))
    ramp = np.minimum(1.0, np.abs(s) / RAMP_FULL) ** RAMP_GAMMA
    rgba[..., 3] = np.round(255 * np.where(neg, a_s, a_l) * ramp)
    return rgba


def land_clip_path(extent, water, map_w_mm):
    """The map box minus the water polygons that draw_basemap draws (made valid), as a Path; None when nothing of
    the map is land. The relief and the contour lines are clipped to it, so they stay off lakes and the sea."""
    x0, x1, y0, y1 = extent
    clip = _box(x0, y0, x1, y1)
    if water is not None and len(water):
        tol = (x1 - x0) / map_w_mm * PRINT_TOL_MM  # as draw_basemap: the clip follows the drawn water exactly
        polys = [p for g in _clip_simplify(water.geometry[water.geometry.type.isin(["Polygon", "MultiPolygon"])],
                                           extent, tol) for p in _parts(g, ("Polygon",))]
        if polys:  # made valid first: a self-crossing lake that draw_basemap fills would stop GEOS here
            clip = shapely.difference(clip, shapely.union_all(shapely.make_valid(np.asarray(polys, dtype=object))))
    return _poly_path(list(_parts(clip, ("Polygon",)))) if not clip.is_empty else None


def draw_relief(ax, relief, extent, S, water, k, map_w_mm, shade=None):
    """The relief as ONE image of exactly the map box (interpolation 'none': a PDF or SVG embeds that array as it
    is), between the streets (zorder 3) and the routes (4.8 and up), clipped off the water polygons that are drawn.
    shade: the raster to draw; by default the smooth shade over the city ground filter (relief.smooth), where a
    town's roofs do not print, else relief.shade. Returns whether anything was drawn."""
    if shade is None:
        shade = relief.smooth if getattr(relief, "smooth", None) is not None else relief.shade
    if np.shape(shade) != _relief.raster_shape(map_w_mm, extent):
        raise ValueError("the relief raster does not match the map")
    x0, x1, y0, y1 = extent
    path = land_clip_path(extent, water, map_w_mm)
    if path is None:
        return False
    im = ax.imshow(relief_rgba(shade, relief_colours(S)), extent=extent, origin="upper",
                   interpolation="none", zorder=3.5)
    im.set_clip_path(PathPatch(path, transform=ax.transData))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return True


def footer_credits(text, S, lang, y, osm_max_w, relief_max_w, dataset=None, ground=False, step=3.4):
    """The credit lines at the foot of a print: OpenStreetMap at y; with relief, the Copernicus notice takes that
    line and OpenStreetMap moves up one, so the bottom margin never changes. With the land cover of the trail print
    (ground), its line goes between them. The bottom line may reach relief_max_w, the others osm_max_w."""
    lines = (["© OpenStreetMap contributors · ODbL · openstreetmap.org/copyright"]
             + ([landcover.notice(lang)] if ground else []) + ([_relief.notice(dataset, lang)] if dataset else []))
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        text(18, y - (len(lines) - 1 - i) * step, line, pt=6.2, color=S["sub"],
             max_w=relief_max_w if last and len(lines) > 1 else osm_max_w)


def draw_routes(ax, segments, counts, cap, cmap, k, base=0.55, gain=ROUTE_GAIN, scale=1.0, glow=0.0, values=None):
    """Routes weighted by how often they were run and coloured by `values` in [0, 1] (default: the same frequency),
    in LEVELS steps, thinnest first. Consecutive segments of a step are joined back into polylines; the geometry
    itself is untouched. glow > 0 lays a wide, faint stroke of the same colour under each step (vector, so the PDF
    stays sharp): for dark styles."""
    if not len(segments):
        return
    strength = np.clip(np.log1p(counts) / math.log1p(cap), 0, 1)
    if values is None:  # heat: weight and colour tell the same thing
        wl = cl = np.rint(strength * (LEVELS - 1)).astype(int)
    else:  # weight in 8 steps, colour in LEVELS steps
        wl = (np.rint(strength * 7) * (LEVELS - 1) / 7).astype(int)
        cl = np.rint(np.clip(values, 0, 1) * (LEVELS - 1)).astype(int)
    groups = Counter(zip(wl.tolist(), cl.tolist()))
    # thinnest first; within a weight, the colour most segments share is drawn last, on top of the rarer ones
    order = sorted(groups) if values is None else sorted(groups, key=lambda g: (g[0], groups[g], g[1]))
    for w_lv, c_lv in order:
        segs = segments[(wl == w_lv) & (cl == c_lv)]
        breaks = np.nonzero(np.any(segs[1:, 0] != segs[:-1, 1], axis=1))[0] + 1
        runs = [np.vstack([r[0, 0], r[:, 1]]) for r in np.split(segs, breaks)]
        w, c = w_lv / (LEVELS - 1), cmap(c_lv / (LEVELS - 1))
        path, width = _line_path(runs), (base + gain * w) * k * scale
        if glow:
            ax.add_patch(PathPatch(path, fill=False, edgecolor=c, alpha=glow, linewidth=width * 4.0,
                                   capstyle="round", joinstyle="round", zorder=4.9))
        ax.add_patch(PathPatch(path, fill=False, edgecolor=c, linewidth=width, capstyle="round",
                               joinstyle="round", zorder=5))


def route_colours(color_by, raws, S):
    """(colour values per map, legend, colormap) for colouring.MODES; raws: one colouring.raw_values per map.
    A mode with no data at all (no clock, no altitude) falls back to one ink; the legend says so ("fallback")."""
    ink = LinearSegmentedColormap.from_list("route", S["heat"])
    if color_by == "heat":
        return [None] * len(raws), {"mode": "heat"}, ink
    known = any(r is not None and np.isfinite(r).any() for r in raws)
    if color_by == "single" or not known:
        legend = {"mode": "single"} if color_by == "single" else {"mode": "single", "fallback": color_by}
        return [None if r is None else np.full(len(r), 0.85) for r in raws], legend, ink
    cmap = colouring.SEASONS if color_by == "month" else ink
    vals, legend = colouring.normalise(color_by, raws)
    return vals, legend, cmap


def draw_legend(plot, text, k, S, cmap, legend, lang, y, names=None):
    """The key at the bottom right: what the line colour means. plot/text draw in A3-grid millimetres.
    names: the two names of a poster for two (mode "who")."""
    es = lang == "es"
    mode = legend.get("mode", "heat")
    if mode == "who":  # weight: how often either ran a street; colour: whose street it is, in five steps
        text(150, y, "REPETICIÓN" if es else "REPETITION", pt=5.8, color=S["sub"], align="right")
        for i, v in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
            x = 153 + i * 6.6
            plot([x * k, (x + 5) * k], [(y + 1) * k, (y + 1) * k], color=S["sub"], lw=(COUPLE_BASE + ROUTE_GAIN * v) * k,
                 solid_capstyle="round")
        text(212, y, "QUIÉN" if es else "WHO", pt=5.8, color=S["sub"], align="right")
        for i, colour in enumerate(S["who"]):
            x = 218 + i * 10.2
            plot([x * k, (x + 8) * k], [(y + 1) * k, (y + 1) * k], color=colour, lw=1.6 * k, solid_capstyle="butt")
        first, second = (n.strip().upper() for n in names)
        text(218, y + 3.6, first, pt=4.6, color=S["sub"], max_w=16)
        text(242.4, y + 3.6, "LOS DOS" if es else "BOTH", pt=4.6, color=S["sub"], align="center", max_w=16)
        text(266.8, y + 3.6, second, pt=4.6, color=S["sub"], align="right", max_w=16)
        return
    names = {"heat": ("REPETICIÓN", "REPETITION"), "single": ("REPETICIÓN", "REPETITION"), "month": ("MES", "MONTH"),
             "pace": ("RITMO", "PACE"), "elevation": ("ALTITUD", "ELEVATION")}
    text(212, y, names[mode][0 if es else 1], pt=5.8, color=S["sub"], align="right")
    if mode in ("heat", "single"):
        for i, v in enumerate(np.linspace(0, 1, 6)):
            plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [(y + 1) * k, (y + 1) * k],
                 color=cmap(v if mode == "heat" else 0.85), lw=(0.55 + ROUTE_GAIN * v) * k, solid_capstyle="round")
        text(279, y - 1, "+", pt=8, align="right", color=S["sub"])
    elif mode == "month":
        letters = "EFMAMJJASOND" if es else "JFMAMJJASOND"
        for i in range(12):
            x = 218 + i * 5.1
            plot([x * k, (x + 3.8) * k], [(y + 1) * k, (y + 1) * k], color=cmap(i / 11), lw=1.6 * k,
                 solid_capstyle="round")
            text(x + 1.9, y + 2.6, letters[i], pt=4.2, color=S["sub"], align="center")
    else:
        for i, v in enumerate(np.linspace(0, 1, 6)):
            plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [(y + 1) * k, (y + 1) * k], color=cmap(v), lw=1.3 * k,
                 solid_capstyle="round")
        if mode == "pace":  # colour 0 is the slow end
            lo, hi = colouring.fmt_pace(legend["hi"]), colouring.fmt_pace(legend["lo"])
        else:
            lo, hi = f"{legend['lo']:.0f} m", f"{legend['hi']:.0f} m"
        text(218, y + 2.6, lo, pt=4.6, color=S["sub"])
        text(268, y + 2.6, hi, pt=4.6, color=S["sub"], align="right")


def _outline(path, trans, step):
    """The glyph outlines after trans, sampled every `step` along each edge (corners alone miss a slanted accent)."""
    out = []
    for poly in path.to_polygons(transform=trans):
        seg = np.diff(poly, axis=0)
        n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step)).astype(int)
        for a, d, m in zip(poly[:-1], seg, n):
            out.append(a + d * (np.arange(m)[:, None] / m))
    return np.concatenate(out) if out else np.empty((0, 2))


def place_text(value, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
    """Path, transform and size (A3-grid millimetres) of a text whose top-left is (x, y), page y pointing down.

    The box runs from the capital height down to the lowest descender (a comma still counts) and ignores what rises
    above the capitals, so CÁDIZ gets the same letters as MADRID and its accent sticks out on top. clear=(x0, x1,
    y_min): nothing of the text may rise above y_min between x0 and x1 (an accent under the PINLESS label); only then
    is the whole box, accent included, fitted to the height instead."""
    path = TextPath((0, 0), str(value), size=pt * k, prop=font)
    b = path.get_extents()
    bottom = min(b.y0, 0.0)

    def fit(top):
        scale = 25.4 / 72.0
        if height is not None and top > bottom:
            scale = min(scale, height * k / (top - bottom))
        if max_w is not None and b.width * scale > max_w * k:
            scale = max_w * k / b.width
        width = b.width * scale
        anchor = x * k - (width if align == "right" else width / 2 if align == "center" else 0)
        return Affine2D().translate(-b.x0, -top).scale(scale, -scale).translate(anchor, y * k), width, scale

    top = TextPath((0, 0), "H", size=pt * k, prop=font).get_extents().y1
    trans, width, scale = fit(top)
    if clear is not None and b.y1 > top:
        pts = _outline(path, trans, 0.3 * k)
        under = (pts[:, 0] >= clear[0] * k) & (pts[:, 0] <= clear[1] * k)
        if under.any() and pts[under, 1].min() < clear[2] * k:
            top = b.y1
            trans, width, scale = fit(top)
    return path, trans, width / k, (top - bottom) * scale / k


def place_runs(parts, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
    """A line of text in several colours, set like one string: parts is [(text, colour), ...]. The scale and the
    transform are those of the joined string (place_text), so max_w, height and clear apply to the whole line; each
    part starts where the text before it ends (the advance of that prefix, measured against an "H").
    Returns ([(path, transform, colour), ...], width in mm)."""
    texts = [str(t) for t, _ in parts]
    _, trans, width, _ = place_text("".join(texts), x, y, k, pt, font, align, max_w, height, clear)
    h_right = TextPath((0, 0), "H", size=pt * k, prop=font).get_extents().x1
    out = []
    for i, (t, (_, colour)) in enumerate(zip(texts, parts)):
        if not t.strip():
            continue
        prefix = TextPath((0, 0), "".join(texts[:i]) + "H", size=pt * k, prop=font) if i else None
        adv = prefix.get_extents().x1 - h_right if i else 0.0
        out.append((TextPath((adv, 0), t, size=pt * k, prop=font), trans, colour))
    return out, width


NAMES_ONE_LINE_MM = 35  # a names headline lower than this on one line leaves the 57 mm band half empty


def name_lines(pair, k):
    """The names headline of a poster for two as [(parts, y, place_runs keywords)], in the band from y 24 to 81 that
    the city fills on a single poster. One line, shrunk to fit; or, when that line would be lower than
    NAMES_ONE_LINE_MM, two lines at one size ("LUCÍA &" over "MARCOS") that fill the band, unless that would make the
    letters smaller than three quarters of the one line's. pair: [(first name, ink), (" & ", ink), (second, ink)]."""
    one = dict(pt=235, font=DISPLAY, max_w=215, height=57, clear=(15, 36, 17.0))  # an accent must not touch PINLESS
    h1 = place_text("".join(t for t, _ in pair), 18, 24, k, **one)[3]
    if h1 >= NAMES_ONE_LINE_MM:
        return [(pair, 24, one)]
    top, bottom, gap = [pair[0], (pair[1][0].rstrip(), pair[1][1])], [pair[2]], 4.0
    second = "".join(t for t, _ in bottom)
    cap = TextPath((0, 0), "H", size=235 * k, prop=DISPLAY).get_extents().y1
    b = TextPath((0, 0), second, size=235 * k, prop=DISPLAY).get_extents()
    rise = max(0.0, b.y1 - cap) / (cap - min(b.y0, 0.0))  # accents over the second line, as a share of its height

    def kw(h, first):  # nothing of the first line under PINLESS; the second line's accents get room of their own
        return dict(pt=235, font=DISPLAY, max_w=215, height=h, clear=(15, 36, 17.0) if first else None)

    def fit(parts, h, first):
        return place_text("".join(t for t, _ in parts), 18, 24, k, **kw(h, first))[3]

    # both lines at one size, ending 2 mm above one line would: the city or motto line below needs the air
    h = (55 - gap) / (2 + rise)
    h = min(fit(top, h, True), fit(bottom, h, False))  # a name too wide for the line sets the size of both
    if h < 0.75 * h1:
        return [(pair, 24, one)]
    return [(top, 24, kw(h, True)), (bottom, 24 + h + gap + rise * h, kw(h, False))]


MISSING = fmt_time(None)  # the mark of a number that is not known


def couple_cells(couple, lang):
    """Every string (and the split) of the numbers of a poster for two. A run together counts for each person in
    their own km; with no clock, or no run together, the two cells show total runs and recorded hours instead."""
    es = lang == "es"
    st_a, st_b = couple.stats
    na, nb = (n.strip().upper() for n in couple.names)
    km_a, km_b = st_a["km"], st_b["km"]
    total = km_a + km_b
    wa = float(np.clip(km_a / total * 138.8, 3.0, 135.8)) if total > 0 else 69.4
    ra, rb = int(round(km_a)), int(round(km_b))  # each person's km rounded once: the two printed parts add up

    def runs(n):
        return ("SALIDA" if n == 1 else "SALIDAS") if es else ("RUN" if n == 1 else "RUNS")

    tog = couple.together or {}
    if tog.get("known") and tog.get("runs", 0) > 0:
        cells = [(_num(tog["runs"], lang), "SALIDAS JUNTAS" if es else "RUNS TOGETHER"),
                 (_num(tog["km"], lang), "KM CODO CON CODO" if es else "KM SIDE BY SIDE")]
    else:
        cells = [(_num(st_a["n"] + st_b["n"], lang), "SALIDAS" if es else "RUNS"),
                 (_num(st_a["hours"] + st_b["hours"], lang), "HORAS REGISTRADAS" if es else "RECORDED HOURS")]
    return {
        "total": _num(ra + rb, lang), "wa": wa,
        "first": f"{na}  {_num(ra, lang)} KM · {_num(st_a['n'], lang)} {runs(st_a['n'])}",
        "second": f"{_num(rb, lang)} KM · {_num(st_b['n'], lang)} {runs(st_b['n'])}  {nb}",
        "cells": cells,
        "records": [((fmt_time(st_a.get("best5k_s")), fmt_time(st_b.get("best5k_s"))), "MEJOR 5K" if es else "BEST 5K"),
                    ((fmt_time(st_a.get("best10k_s")), fmt_time(st_b.get("best10k_s"))),
                     "MEJOR 10K" if es else "BEST 10K"),
                    ((_num(st_a["longest_km"], lang, 1), _num(st_b["longest_km"], lang, 1)),
                     "TIRADA MÁS LARGA · KM" if es else "LONGEST RUN · KM")],
        "streets": (MISSING if couple.streets_both is None else _num(couple.streets_both, lang),
                    "CALLES DE LOS DOS" if es else "STREETS YOU BOTH RAN"),
    }


def map_aspect(size="A3", graphs=False):
    wmm, _ = SIZES[size]
    k = wmm / 297
    return (wmm - 36 * k) / ((190 if graphs else 222) * k)  # the graphs band takes 32 mm from the map


def _segments(pieces):
    """The original segments of the pieces, (n, 2, 2), and the activity of each one. The geometry is untouched."""
    segments, ids = [], []
    for act, line in pieces:
        c = np.asarray(line.coords)
        if len(c) > 1:
            segments.append(np.stack((c[:-1], c[1:]), axis=1))
            ids.append(np.full(len(c) - 1, act))
    if not segments:
        return np.empty((0, 2, 2)), np.empty(0, dtype=int)
    return np.concatenate(segments), np.concatenate(ids)


def _probe_px(segments, extent, shape):
    """Pixel rows and columns (n, 3) of three probes per segment: both ends and the midpoint."""
    x0, x1, y0, y1 = extent
    h, w = shape
    probes = np.stack((segments[:, 0], segments.mean(axis=1), segments[:, 1]), axis=1)
    px = np.clip(((probes[:, :, 0] - x0) / (x1 - x0) * w).astype(int), 0, w - 1)
    py = np.clip(((y1 - probes[:, :, 1]) / (y1 - y0) * h).astype(int), 0, h - 1)
    return py, px


def _density_segments(pieces, extent, acc, with_ids=False):
    """Colour original segments from an activity-per-pixel density field.

    No interpolation, simplification or quantization changes the geometry.
    Sampling at both ends and the midpoint gives a stable frequency measure.
    """
    segments, ids = _segments(pieces)
    if not len(segments):
        return (np.empty((0, 2, 2)), np.empty(0)) + ((np.empty(0, dtype=int),) if with_ids else ())
    py, px = _probe_px(segments, extent, acc.shape)
    density = np.median(acc[py, px], axis=1)
    if with_ids:  # the activity of each segment, for colourings that depend on the run
        return segments, np.maximum(density, 1), ids
    return segments, np.maximum(density, 1)


def who_classes(a, b, owner_is_b):
    """Whose street a segment is, from how many of each person's activities pass within SHARED_MM on paper (exact
    integer counts a and b, t = a + b): 0 first person (b under 10 %), 1 mostly first (under 35 %), 2 both (35-65 %),
    3 mostly second (up to 90 %), 4 second. Exactly symmetric: who_classes(b, a) == 4 - who_classes(a, b). With no
    count at all (t = 0) a segment takes its owner's class."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    t = a + b
    c = np.select([10 * b < t, 100 * b < 35 * t, 100 * b <= 65 * t, 10 * b <= 9 * t], [0, 1, 2, 3], 4)
    return np.where(t == 0, np.where(owner_is_b, 4, 0), c)


def smooth_classes(classes, segments, mm_per_unit, min_mm=SMOOTH_MM):
    """Whose street a line is changes along it only where the street does. A stretch of another class shorter than
    min_mm on paper, between two stretches of one class on the same line, takes their class: GPS noise that brings
    a line within reach of another for a few metres, or out of it, would otherwise print as a dashed line. The
    shortest such stretch goes first, until none is left. A line's first and last stretch, and a short stretch
    between two different classes (a real change of street), are kept. segments: (n, 2, 2) in drawing order, where
    a line goes on while each segment starts where the one before it ends."""
    classes = np.array(classes, copy=True)
    if len(classes) < 3:
        return classes
    length = np.hypot(*(segments[:, 1] - segments[:, 0]).T) * mm_per_unit
    breaks = np.nonzero(np.any(segments[1:, 0] != segments[:-1, 1], axis=1))[0] + 1
    for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, len(classes)]):
        cls = classes[lo:hi]
        edges = np.r_[0, np.nonzero(np.diff(cls))[0] + 1, len(cls)]
        if len(edges) < 4:  # fewer than three stretches: nothing lies between two others
            continue
        runs = [[int(cls[a]), a, b, float(length[lo + a:lo + b].sum())] for a, b in zip(edges[:-1], edges[1:])]
        while True:
            short = [(r[3], i) for i, r in enumerate(runs[1:-1], 1)
                     if r[3] < min_mm and runs[i - 1][0] == runs[i + 1][0]]
            if not short:
                break
            _, i = min(short)
            before, after = runs[i - 1], runs[i + 1]
            runs[i - 1:i + 2] = [[before[0], before[1], after[2], before[3] + runs[i][3] + after[3]]]
        for c, a, b, _ in runs:
            cls[a:b] = c
    return classes


def who_values(pieces, owner, extent, W, H, k, width_px):
    """(segments, weight, colour values in [0, 1], classes) for a poster of two. owner[i]: 0 or 1 for activity i.

    a and b are how many of each person's activities pass within SHARED_MM on paper (the same in metres at every
    paper size): each activity is drawn 2 * SHARED_MM wider than its line and counted once per pixel. The colour
    belongs to the street, not the line: both lines of a shared street read the same a and b, so they get the same
    class, and one run of the second person on a street the first ran 20 times takes the first person's ink.
    Weight: a + b, the same measure for one person and for two, so a street run 6 times by one of you weighs what a
    street run 3 + 3 does, even when each run's GPS line lies a few metres from the others."""
    owner = np.asarray(owner)
    segments, ids = _segments(pieces)
    if not len(segments):
        return segments, np.empty(0), np.empty(0), np.empty(0, dtype=int)
    r = max(1, int(round(SHARED_MM * k * 220 / 25.4)))  # 4 px at A3, 7 at 50x70
    reach = width_px + 2 * r  # a line this wide covers every pixel within SHARED_MM of the run
    acc_a = heat_raster([(i, p) for i, p in pieces if owner[i] == 0], extent, W, H, reach)
    acc_b = heat_raster([(i, p) for i, p in pieces if owner[i] == 1], extent, W, H, reach)
    py, px = _probe_px(segments, extent, acc_a.shape)

    def med(img):  # the median of three integer probes is the middle one: an exact integer
        return np.sort(img[py, px].astype(np.int64), axis=1)[:, 1]

    a, b = med(acc_a), med(acc_b)
    is_b = owner[ids] == 1
    weight = np.maximum(1, a + b)
    mm_per_unit = W / 220 * 25.4 / (extent[1] - extent[0])  # the density raster has 220 px per inch of paper
    classes = smooth_classes(who_classes(a, b, is_b), segments, mm_per_unit)
    return segments, weight, (classes + 0.5) / 5, classes


def render(pieces, basemap, extent, stats, out_path, style="huella", size="A3", dpi=300,
           name="", city="", year="", lang="es", pdf_path=None, svg_path=None,
           headline="city", preview_path=None, color_by="heat", activities=None, tracks_xy=None, graphs=False,
           relief=None, couple=None):
    """activities and tracks_xy (the runs the pieces come from, in the same order) are needed by colourings other
    than heat and single. relief: a relief.Relief for this map (relief.prepare_many), drawn under the routes. couple:
    a Couple, for a poster of two people (pieces of both, owner[i] says whose); then the colour only says whose street
    it is, color_by is ignored, stats may be None and graphs is not available."""
    if headline not in ("city", "name"):
        raise ValueError("headline must be city or name")
    if headline == "name" and not name.strip():
        raise ValueError("a name is required for the name headline")
    if couple is not None and graphs:
        raise ValueError("the graphs band is not available on a poster for two")
    S = STYLES[style]
    pieces = privacy.clip_to_extent(pieces, extent)  # only what is drawn is written (no hidden PDF/SVG lines)
    wmm, hmm = SIZES[size]
    k = wmm / 297.0
    fig = plt.figure(figsize=(wmm * MM, hmm * MM), dpi=dpi, facecolor=S["bg"])
    # A physical millimetre coordinate system keeps type and margins predictable.
    page = fig.add_axes([0, 0, 1, 1], zorder=10)
    page.set_xlim(0, wmm)
    page.set_ylim(hmm, 0)
    page.axis("off")

    def text(x, y, value, pt=9, font=TEXT, color=None, align="left", max_w=None, height=None, clear=None):
        if not value:
            return (0, 0)
        path, trans, w, h = place_text(value, x, y, k, pt, font, align, max_w, height, clear)
        # snap=False: a glyph made only of straight strokes (E, F, H, L, T) would be snapped to whole pixels and lose
        # its thin stems at small sizes
        page.add_patch(PathPatch(path, transform=trans + page.transData, color=color or S["ink"], lw=0, snap=False))
        return w, h

    def rule(y, left=18, right=279, width=0.45, color=None):
        page.plot([left * k, right * k], [y * k, y * k], color=color or S["rule"], lw=width * k)

    def runs(parts, x, y, pt=9, font=TEXT, align="left", max_w=None, height=None, clear=None):
        """A line in several colours, set like one string (place_runs)."""
        items, w = place_runs(parts, x, y, k, pt, font, align, max_w, height, clear)
        for path, trans, colour in items:
            page.add_patch(PathPatch(path, transform=trans + page.transData, color=colour, lw=0, snap=False))
        return w

    es = lang == "es"
    who = S["who"] if couple is not None else None
    # in a poster for two, colour only says who: each name in its own ink, the rest in the paper's ink and grey
    pair = None if couple is None else [(couple.names[0].strip().upper(), who[0]), (" & ", S["sub"]),
                                        (couple.names[1].strip().upper(), who[4])]
    text(18, 13, "P I N L E S S", pt=8.4)
    text(279, 13, ("ARCHIVO DE RUNNING A DOS" if es else "RUNNING ARCHIVE FOR TWO") if couple is not None else
         ("ARCHIVO DE RUNNING" if es else "RUNNING ARCHIVE"), pt=6.6, align="right", color=S["sub"])
    title = name if headline == "name" else city or ("MI CIUDAD" if es else "MY CITY")
    if couple is not None and headline == "name":  # both names: one line, or two when one would be low
        for parts, y, kw in name_lines(pair, k):
            runs(parts, 18, y, **kw)
    else:
        text(18, 24, title.upper(), pt=235, font=DISPLAY,
             max_w=215, height=57, clear=(15, 36, 17.0))  # an accent must not touch the PINLESS label
    text(279, 27, str(year), pt=43, font=REGULAR, color=S["accent"] if couple is None else S["ink"], align="right",
         max_w=39)
    text(279, 47, "TEMPORADA" if es else "SEASON", pt=6.0, align="right", color=S["sub"])
    if headline == "name":
        text(18, 83.5, city.upper(), pt=11.2, max_w=125)
        text(279, 84, "UN AÑO EN MOVIMIENTO" if es else "A YEAR IN MOTION", pt=7.2,
             color=S["sub"], align="right")
    else:
        text(18, 84, "UN AÑO EN MOVIMIENTO" if es else "A YEAR IN MOTION", pt=7.2, color=S["sub"])
        if couple is None:
            text(279, 83.5, name.upper(), pt=13.2, align="right", max_w=105)
        else:
            runs(pair, 279, 83.5, pt=13.2, align="right", max_w=105)

    m, map_top, map_w, map_h = 18 * k, 94 * k, wmm - 36 * k, (190 if graphs else 222) * k
    ax = fig.add_axes([m / wmm, 1 - (map_top + map_h) / hmm, map_w / wmm, map_h / hmm], zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    draw_basemap(ax, basemap, extent, S, k, map_w)  # map_w: millimetres of paper
    drawn = draw_relief(ax, relief, extent, S, basemap[2], k, map_w) if relief is not None else False

    # Fixed physical density resolution makes PDF and PNG frequencies independent
    # of export DPI. No blur is added around removed sensitive regions.
    W = int(round(map_w * MM * 220))
    H = int(round(map_h * MM * 220))
    if couple is not None:
        return _render_couple(fig, ax, page, text, runs, rule, pieces, extent, couple, S, k, W, H, lang, wmm, hmm, dpi,
                              out_path, pdf_path, svg_path, preview_path, relief.dataset if drawn else None)
    acc = heat_raster(pieces, extent, W, H, max(1, int(round(0.22 * 220 / 25.4 * k))))
    segments, counts, ids = _density_segments(pieces, extent, acc, with_ids=True)
    cap = max(3.0, float(np.percentile(counts, 98.5))) if len(counts) else 1
    if color_by in ("heat", "single"):
        raw = np.zeros(len(segments))
    elif activities is None:  # nothing to colour by: route_colours falls back to one ink and says so
        raw = np.full(len(segments), np.nan)
    else:
        raw = colouring.raw_values(color_by, ids, segments, activities, tracks_xy)
    (values,), legend, cmap = route_colours(color_by, [raw], S)
    draw_routes(ax, segments, counts, cap, cmap, k, glow=S.get("glow", 0.0), values=values)
    if graphs and activities is not None:  # km per month, time of day, direction: aggregates only
        draw_band(fig, text, S, k, wmm, hmm, activities, tracks_xy, lang, 289, 29, big_font=DISPLAY)

    rule(323)
    km_w, _ = text(18, 331, _num(stats["km"], lang), pt=109, font=DISPLAY, max_w=119, height=28)
    text(18 + km_w + 4, 348, "km", pt=24, font=REGULAR, color=S["accent"])
    text(18, 364, "KILÓMETROS RECORRIDOS" if es else "KILOMETRES COVERED", pt=6.8, color=S["sub"])
    for x, value, label in ((180, _num(stats["n"], lang), "SALIDAS" if es else "RUNS"),
                            (239, _num(stats["hours"], lang), "HORAS REGISTRADAS" if es else "RECORDED HOURS")):
        text(x, 334, value, pt=56, font=DISPLAY, max_w=39, height=17)
        text(x, 357, label, pt=6.4, color=S["sub"], max_w=40)
    rule(371, width=0.32)
    records = (
        (fmt_time(stats.get("best5k_s")), "MEJOR 5K" if es else "BEST 5K"),
        (fmt_time(stats.get("best10k_s")), "MEJOR 10K" if es else "BEST 10K"),
        (_num(stats["longest_km"], lang, 1) + " km", "TIRADA MÁS LARGA" if es else "LONGEST RUN"),
        (_num(stats["streets"], lang) if "streets" in stats else "—", "CALLES DISTINTAS" if es else "DISTINCT STREETS"),
    )
    for i, (value, label) in enumerate(records):
        x = 18 + i * 65.25
        text(x, 375, value, pt=25, font=DISPLAY, max_w=58, height=6.4)
        text(x, 385.5, label, pt=6.3, color=S["sub"], max_w=58)
    rule(392)
    text(18, 399, "CADA SALIDA DEJA HUELLA." if es else "EVERY RUN LEAVES A TRACE.", pt=6.8)
    draw_legend(page.plot, text, k, S, cmap, legend, lang, 399)
    footer_credits(text, S, lang, 409, 238, 238, relief.dataset if drawn else None)

    _save(fig, S, hmm, dpi, out_path, pdf_path, svg_path, preview_path)
    return {"W": W, "H": H, "cap": cap, "max": int(acc.max()) if acc.size else 0, "legend": legend, "relief": drawn}


def _save(fig, S, hmm, dpi, out_path, pdf_path, svg_path, preview_path):
    """The same files for every editorial poster, then the figure is closed."""
    # No timestamps, activity ids, source paths, sensor values or author metadata.
    fig.savefig(out_path, dpi=dpi, facecolor=S["bg"], metadata={"Software": "Pinless"})
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=S["bg"],
                    metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None,
                              "Title": None, "Author": None, "Subject": None, "Keywords": None})
    if svg_path:
        with matplotlib.rc_context(svg_rc("pinless-editorial")):
            fig.savefig(svg_path, facecolor=S["bg"], metadata={"Date": None, "Creator": "Pinless"})
    if preview_path:
        # Render directly from the same figure, avoiding a web-sized SVG or any
        # re-processing of tracks. 1600 px high, compressed, no personal EXIF.
        fig.savefig(preview_path, dpi=1600 * 25.4 / hmm, facecolor=S["bg"],
                    pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)


def _render_couple(fig, ax, page, text, runs, rule, pieces, extent, couple, S, k, W, H, lang, wmm, hmm, dpi, out_path,
                   pdf_path, svg_path, preview_path, relief_dataset=None):
    """The routes, numbers and key of a poster for two (render, couple=...): pieces are already cut to the map.
    relief_dataset: the elevation dataset when relief was drawn, for its credit line."""
    es = lang == "es"
    who = S["who"]
    segments, weight, values, classes = who_values(pieces, couple.owner, extent, W, H, k,
                                                   max(1, int(round(0.22 * 220 / 25.4 * k))))
    cap = max(3.0, float(np.percentile(weight, 98.5))) if len(weight) else 1
    cmap = ListedColormap(who)
    # thinnest first; within a weight the commonest colour on top (draw_routes)
    draw_routes(ax, segments, weight, cap, cmap, k, base=COUPLE_BASE, glow=S.get("glow", 0.0), values=values)
    length = np.hypot(*(segments[:, 1] - segments[:, 0]).T) if len(segments) else np.zeros(0)
    total = float(length.sum())
    who_frac = [float(length[classes == c].sum()) / total if total else 0.0 for c in range(5)]
    # how much of each person's drawn length is in their own two inks: none when they only ran the other's streets
    _, ids = _segments(pieces)
    own_ink = []
    for o, mine in ((0, (0, 1)), (1, (3, 4))):
        theirs = np.asarray(couple.owner)[ids] == o if len(ids) else np.zeros(0, dtype=bool)
        drawn = float(length[theirs].sum())
        own_ink.append(float(length[theirs & np.isin(classes, mine)].sum()) / drawn if drawn else 0.0)

    cc = couple_cells(couple, lang)
    rule(323)
    km_w, _ = text(18, 331, cc["total"], pt=109, font=DISPLAY, max_w=119, height=28)
    text(18 + km_w + 4, 348, "km", pt=24, font=REGULAR, color=S["ink"])
    wa = cc["wa"]  # the split of the km: the first person's part, a 1.2 mm gap, the second person's
    page.plot([18 * k, (18 + wa) * k], [362.2 * k, 362.2 * k], color=who[0], lw=2.2 * k, solid_capstyle="butt")
    page.plot([(18 + wa + 1.2) * k, 158 * k], [362.2 * k, 362.2 * k], color=who[4], lw=2.2 * k,
              solid_capstyle="butt")
    text(18, 365.5, cc["first"], pt=6.2, color=who[0], max_w=68)
    text(158, 365.5, cc["second"], pt=6.2, color=who[4], align="right", max_w=68)
    for x, (value, label) in zip((180, 239), cc["cells"]):
        text(x, 334, value, pt=56, font=DISPLAY, max_w=39, height=17)
        text(x, 357, label, pt=6.4, color=S["sub"], max_w=40)
    rule(371, width=0.32)
    for i, ((va, vb), label) in enumerate(cc["records"]):
        x = 18 + i * 65.25
        runs([(va, who[0]), (" · ", S["sub"]), (vb, who[4])], x, 375, pt=25, font=DISPLAY, max_w=58, height=6.4)
        text(x, 385.5, label, pt=6.3, color=S["sub"], max_w=58)
    value, label = cc["streets"]
    text(213.75, 375, value, pt=25, font=DISPLAY, max_w=58, height=6.4)
    text(213.75, 385.5, label, pt=6.3, color=S["sub"], max_w=58)
    rule(392)
    text(18, 399, "DOS HUELLAS, UN MAPA." if es else "TWO TRACES, ONE MAP.", pt=6.8)
    legend = {"mode": "who"}
    draw_legend(page.plot, text, k, S, cmap, legend, lang, 399, names=couple.names)
    footer_credits(text, S, lang, 409, 238, 238, relief_dataset)
    _save(fig, S, hmm, dpi, out_path, pdf_path, svg_path, preview_path)
    return {"W": W, "H": H, "cap": cap, "max": int(weight.max()) if len(weight) else 0, "legend": legend,
            "who_frac": who_frac, "own_ink": own_ink, "relief": relief_dataset is not None}
