"""Huella: a personal running archive, composed as an A-series art print.

All paths come from privacy-cut geometry. Density changes ink colour and stroke
weight only; no invented routes, snapping, endpoint markers or soft glow.
The PDF/SVG use vector streets, routes and outlined, bundled OFL typography.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from shapely.geometry.polygon import orient

from . import colouring, privacy
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
        "street": {"major": ("#d5d0c4", 0.32), "minor": ("#e1dbcf", 0.17), "path": ("#e7e1d5", 0.11)},
        "heat": ["#cb694e", "#bd5036", "#a63724", "#782b20", "#46231d"],
    },
    "carbon": {
        "bg": "#131e19", "ink": "#f2eddf", "sub": "#b0b5a5", "accent": "#dfa575",
        "green": "#1a261f", "water": "#1f3028", "water_line": "#2c4137", "rule": "#586055",
        "street": {"major": ("#314134", 0.32), "minor": ("#25352a", 0.17), "path": ("#203126", 0.11)},
        "heat": ["#bc875d", "#d19e6d", "#e0b582", "#eed0a8", "#fff2d7"],
    },
    "midnight": {
        "bg": "#0f1626", "ink": "#eef2f7", "sub": "#8a96ab", "accent": "#5fd4ff",
        "green": "#121c2f", "water": "#16223a", "water_line": "#22324d", "rule": "#2c3852",
        "street": {"major": ("#2a3650", 0.32), "minor": ("#1e283d", 0.17), "path": ("#1a2336", 0.11)},
        "heat": ["#1f6f9c", "#2aa3d6", "#5fd4ff", "#b7f0ff", "#ffffff"], "glow": 0.16,
    },
    "blueprint": {
        "bg": "#1d4a86", "ink": "#f3f7fc", "sub": "#b4c8e3", "accent": "#ffd45c",
        "green": "#204f8d", "water": "#1a4479", "water_line": "#3c6aa6", "rule": "#5d87bd",
        "street": {"major": ("#4a76ad", 0.32), "minor": ("#3a679f", 0.17), "path": ("#33609a", 0.11)},
        "heat": ["#a9c8ee", "#d3e4f8", "#ffffff", "#ffe9a8", "#ffd45c"],
    },
    "ink": {
        "bg": "#fbfaf7", "ink": "#111111", "sub": "#6b6b6b", "accent": "#111111",
        "green": "#f1f1ec", "water": "#e7eaec", "water_line": "#d6dadd", "rule": "#c9c9c4",
        "street": {"major": ("#d6d5d0", 0.32), "minor": ("#e2e1dc", 0.17), "path": ("#e9e8e3", 0.11)},
        "heat": ["#9c9c9c", "#6e6e6e", "#444444", "#222222", "#000000"],
    },
    "graphite": {
        "bg": "#222324", "ink": "#f1f1ef", "sub": "#9b9d9b", "accent": "#ff6b3d",
        "green": "#26282a", "water": "#2b2f33", "water_line": "#383d42", "rule": "#4a4c4e",
        "street": {"major": ("#3a3c3e", 0.32), "minor": ("#2e3032", 0.17), "path": ("#2a2c2e", 0.11)},
        "heat": ["#6a6c6e", "#9a9c9e", "#c9cbcc", "#ebecec", "#ffffff"], "glow": 0.10,
    },
    "neon": {
        "bg": "#0c0812", "ink": "#f6f0ff", "sub": "#9c8fb3", "accent": "#ff3d9a",
        "green": "#120d1a", "water": "#140f20", "water_line": "#221a33", "rule": "#33284a",
        "street": {"major": ("#2a2238", 0.32), "minor": ("#1d1729", 0.17), "path": ("#181322", 0.11)},
        "heat": ["#6a1fa0", "#b5179e", "#f72585", "#ff8c42", "#ffe066"], "glow": 0.20,
    },
    "lavender": {
        "bg": "#efebf5", "ink": "#2a2140", "sub": "#7d7395", "accent": "#6741d9",
        "green": "#e8e4ef", "water": "#e2e2f0", "water_line": "#d0d0e4", "rule": "#cfc8dc",
        "street": {"major": ("#dcd5e6", 0.32), "minor": ("#e5dfee", 0.17), "path": ("#eae5f1", 0.11)},
        "heat": ["#c0b3e8", "#9b87d8", "#7655c7", "#5233a3", "#2f1b66"],
    },
    "sunset": {
        "bg": "#fbf1e6", "ink": "#2a1d2e", "sub": "#8a7468", "accent": "#e8590c",
        "green": "#f4ecdc", "water": "#ece6e4", "water_line": "#ddd2cf", "rule": "#d8c8b8",
        "street": {"major": ("#e8dccd", 0.32), "minor": ("#efe5d8", 0.17), "path": ("#f2e9de", 0.11)},
        "heat": ["#ffb35c", "#ff8a2b", "#f25d1d", "#c2255c", "#5f3dc4"],
    },
    "forest": {
        "bg": "#eef1e8", "ink": "#1f2d1f", "sub": "#6d7a68", "accent": "#2f6b3a",
        "green": "#e1e8d6", "water": "#dde7e4", "water_line": "#c7d6d1", "rule": "#c3cbbb",
        "street": {"major": ("#d6dccd", 0.32), "minor": ("#e1e6d9", 0.17), "path": ("#e7ebe0", 0.11)},
        "heat": ["#a3c39b", "#6fa065", "#3f7a45", "#25522c", "#13301a"],
    },
    "blush": {
        "bg": "#f8ece9", "ink": "#3a1f24", "sub": "#9a7b7f", "accent": "#c2185b",
        "green": "#f2e6e1", "water": "#ece4e6", "water_line": "#dfd2d5", "rule": "#dccbc8",
        "street": {"major": ("#ead9d6", 0.32), "minor": ("#f0e2df", 0.17), "path": ("#f3e7e4", 0.11)},
        "heat": ["#f2a7b8", "#e06688", "#c2185b", "#880e4f", "#4a0629"],
    },
    "ocean": {
        "bg": "#edf4f7", "ink": "#0f2a3d", "sub": "#6b8595", "accent": "#0b6e99",
        "green": "#e4eeee", "water": "#d3e6ee", "water_line": "#bcd6e2", "rule": "#bfd0d9",
        "street": {"major": ("#d2dfe6", 0.32), "minor": ("#dde8ed", 0.17), "path": ("#e3ecf0", 0.11)},
        "heat": ["#8fd0e8", "#3fa9d1", "#0b6e99", "#0a4a6e", "#062a40"],
    },
}


LEVELS = 32  # route colour and weight steps: finer than the eye separates on paper, one path per step in a PDF/SVG
PRINT_TOL_MM = 0.04  # geometry is simplified to this, on paper: far below what a print shows


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


def draw_routes(ax, segments, counts, cap, cmap, k, base=0.55, gain=0.42, scale=1.0, glow=0.0, values=None):
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


def draw_legend(plot, text, k, S, cmap, legend, lang, y):
    """The key at the bottom right: what the line colour means. plot/text draw in A3-grid millimetres."""
    es = lang == "es"
    mode = legend.get("mode", "heat")
    names = {"heat": ("REPETICIÓN", "REPETITION"), "single": ("REPETICIÓN", "REPETITION"), "month": ("MES", "MONTH"),
             "pace": ("RITMO", "PACE"), "elevation": ("ALTITUD", "ELEVATION")}
    text(212, y, names[mode][0 if es else 1], pt=5.8, color=S["sub"], align="right")
    if mode in ("heat", "single"):
        for i, v in enumerate(np.linspace(0, 1, 6)):
            plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [(y + 1) * k, (y + 1) * k],
                 color=cmap(v if mode == "heat" else 0.85), lw=(0.55 + 0.95 * v) * k, solid_capstyle="round")
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


def map_aspect(size="A3", graphs=False):
    wmm, _ = SIZES[size]
    k = wmm / 297
    return (wmm - 36 * k) / ((190 if graphs else 222) * k)  # the graphs band takes 32 mm from the map


def _density_segments(pieces, extent, acc, with_ids=False):
    """Colour original segments from an activity-per-pixel density field.

    No interpolation, simplification or quantization changes the geometry.
    Sampling at both ends and the midpoint gives a stable frequency measure.
    """
    x0, x1, y0, y1 = extent
    h, w = acc.shape
    segments, ids = [], []
    for act, line in pieces:
        c = np.asarray(line.coords)
        if len(c) > 1:
            segments.append(np.stack((c[:-1], c[1:]), axis=1))
            ids.append(np.full(len(c) - 1, act))
    if not segments:
        return (np.empty((0, 2, 2)), np.empty(0)) + ((np.empty(0, dtype=int),) if with_ids else ())
    segments = np.concatenate(segments)
    probes = np.stack((segments[:, 0], segments.mean(axis=1), segments[:, 1]), axis=1)
    px = np.clip(((probes[:, :, 0] - x0) / (x1 - x0) * w).astype(int), 0, w - 1)
    py = np.clip(((y1 - probes[:, :, 1]) / (y1 - y0) * h).astype(int), 0, h - 1)
    density = np.median(acc[py, px], axis=1)
    if with_ids:  # the activity of each segment, for colourings that depend on the run
        return segments, np.maximum(density, 1), np.concatenate(ids)
    return segments, np.maximum(density, 1)


def render(pieces, basemap, extent, stats, out_path, style="huella", size="A3", dpi=300,
           name="", city="", year="", lang="es", pdf_path=None, svg_path=None,
           headline="city", preview_path=None, color_by="heat", activities=None, tracks_xy=None, graphs=False):
    """activities and tracks_xy (the runs the pieces come from, in the same order) are needed by colourings other
    than heat and single."""
    if headline not in ("city", "name"):
        raise ValueError("headline must be city or name")
    if headline == "name" and not name.strip():
        raise ValueError("a name is required for the name headline")
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

    es = lang == "es"
    text(18, 13, "P I N L E S S", pt=8.4)
    text(279, 13, "ARCHIVO DE RUNNING" if es else "RUNNING ARCHIVE", pt=6.6,
         align="right", color=S["sub"])
    title = name if headline == "name" else city or ("MI CIUDAD" if es else "MY CITY")
    text(18, 24, title.upper(), pt=235, font=DISPLAY,
         max_w=215, height=57, clear=(15, 36, 17.0))  # an accent must not touch the PINLESS label
    text(279, 27, str(year), pt=43, font=REGULAR, color=S["accent"], align="right", max_w=39)
    text(279, 47, "TEMPORADA" if es else "SEASON", pt=6.0, align="right", color=S["sub"])
    if headline == "name":
        text(18, 83.5, city.upper(), pt=11.2, max_w=125)
        text(279, 84, "UN AÑO EN MOVIMIENTO" if es else "A YEAR IN MOTION", pt=7.2,
             color=S["sub"], align="right")
    else:
        text(18, 84, "UN AÑO EN MOVIMIENTO" if es else "A YEAR IN MOTION", pt=7.2, color=S["sub"])
        text(279, 83.5, name.upper(), pt=13.2, align="right", max_w=105)

    m, map_top, map_w, map_h = 18 * k, 94 * k, wmm - 36 * k, (190 if graphs else 222) * k
    ax = fig.add_axes([m / wmm, 1 - (map_top + map_h) / hmm, map_w / wmm, map_h / hmm], zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    draw_basemap(ax, basemap, extent, S, k, map_w)  # map_w: millimetres of paper

    # Fixed physical density resolution makes PDF and PNG frequencies independent
    # of export DPI. No blur is added around removed sensitive regions.
    W = int(round(map_w * MM * 220))
    H = int(round(map_h * MM * 220))
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
    for x, value, label in ((180, str(stats["n"]), "SALIDAS" if es else "RUNS"),
                            (239, _num(stats["hours"], lang), "HORAS REGISTRADAS" if es else "RECORDED HOURS")):
        text(x, 334, value, pt=56, font=DISPLAY, max_w=39, height=17)
        text(x, 357, label, pt=6.4, color=S["sub"], max_w=40)
    rule(371, width=0.32)
    records = (
        (fmt_time(stats.get("best5k_s")), "MEJOR 5K" if es else "BEST 5K"),
        (fmt_time(stats.get("best10k_s")), "MEJOR 10K" if es else "BEST 10K"),
        (_num(stats["longest_km"], lang, 1) + " km", "TIRADA MÁS LARGA" if es else "LONGEST RUN"),
        (str(stats.get("streets", "—")), "CALLES DISTINTAS" if es else "DISTINCT STREETS"),
    )
    for i, (value, label) in enumerate(records):
        x = 18 + i * 65.25
        text(x, 375, value, pt=25, font=DISPLAY, max_w=58, height=6.4)
        text(x, 385.5, label, pt=6.3, color=S["sub"], max_w=58)
    rule(392)
    text(18, 399, "CADA SALIDA DEJA HUELLA." if es else "EVERY RUN LEAVES A TRACE.", pt=6.8)
    draw_legend(page.plot, text, k, S, cmap, legend, lang, 399)
    text(18, 409, "© OpenStreetMap contributors · ODbL · openstreetmap.org/copyright", pt=6.2,
         color=S["sub"], max_w=238)

    # No timestamps, activity ids, source paths, sensor values or author metadata.
    fig.savefig(out_path, dpi=dpi, facecolor=S["bg"], metadata={"Software": "Pinless"})
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=S["bg"],
                    metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None,
                              "Title": None, "Author": None, "Subject": None, "Keywords": None})
    if svg_path:
        with matplotlib.rc_context({"svg.hashsalt": "pinless-editorial", "svg.fonttype": "path"}):
            fig.savefig(svg_path, facecolor=S["bg"], metadata={"Date": None, "Creator": "Pinless"})
    if preview_path:
        # Render directly from the same figure, avoiding a web-sized SVG or any
        # re-processing of tracks. 1600 px high, compressed, no personal EXIF.
        fig.savefig(preview_path, dpi=1600 * 25.4 / hmm, facecolor=S["bg"],
                    pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return {"W": W, "H": H, "cap": cap, "max": int(acc.max()) if acc.size else 0, "legend": legend}
