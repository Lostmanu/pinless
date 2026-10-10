"""Huella: a personal running archive, composed as an A-series art print.

All paths come from privacy-cut geometry. Density changes ink colour and stroke
weight only; no invented routes, snapping, endpoint markers or soft glow.
The PDF/SVG use vector streets, routes and outlined, bundled OFL typography.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from . import privacy
from .render import MAJOR, MINOR, MM, SIZES, _hw, _lines, heat_raster
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
}


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


def map_aspect(size="A3"):
    wmm, _ = SIZES[size]
    k = wmm / 297
    return (wmm - 36 * k) / (222 * k)


def _density_segments(pieces, extent, acc):
    """Colour original segments from an activity-per-pixel density field.

    No interpolation, simplification or quantization changes the geometry.
    Sampling at both ends and the midpoint gives a stable frequency measure.
    """
    x0, x1, y0, y1 = extent
    h, w = acc.shape
    segments = []
    for _, line in pieces:
        c = np.asarray(line.coords)
        if len(c) > 1:
            segments.append(np.stack((c[:-1], c[1:]), axis=1))
    if not segments:
        return np.empty((0, 2, 2)), np.empty(0)
    segments = np.concatenate(segments)
    probes = np.stack((segments[:, 0], segments.mean(axis=1), segments[:, 1]), axis=1)
    px = np.clip(((probes[:, :, 0] - x0) / (x1 - x0) * w).astype(int), 0, w - 1)
    py = np.clip(((y1 - probes[:, :, 1]) / (y1 - y0) * h).astype(int), 0, h - 1)
    density = np.median(acc[py, px], axis=1)
    return segments, np.maximum(density, 1)


def render(pieces, basemap, extent, stats, out_path, style="huella", size="A3", dpi=300,
           name="", city="", year="", lang="es", pdf_path=None, svg_path=None,
           headline="city", preview_path=None):
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
        page.add_patch(PathPatch(path, transform=trans + page.transData, color=color or S["ink"], lw=0))
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

    m, map_top, map_w, map_h = 18 * k, 94 * k, wmm - 36 * k, 222 * k
    ax = fig.add_axes([m / wmm, 1 - (map_top + map_h) / hmm, map_w / wmm, map_h / hmm], zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    edges, green, water = basemap
    gp = green[green.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if len(gp):
        gp.plot(ax=ax, color=S["green"], linewidth=0, zorder=1)
    wp = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if len(wp):
        wp.plot(ax=ax, color=S["water"], linewidth=0, zorder=2)
    wl = water[water.geometry.type.isin(["LineString", "MultiLineString"])]
    if len(wl):
        ax.add_collection(LineCollection(_lines(wl.geometry), colors=S["water_line"], linewidths=0.7 * k, zorder=2))
    # OSM walk graphs contain reciprocal edges. Draw each geometry only once.
    edges = edges.loc[~edges.geometry.to_wkb().duplicated()].copy()
    hw = edges["highway"].map(_hw)
    for cls, mask in (("path", ~hw.isin(MAJOR | MINOR)), ("minor", hw.isin(MINOR)), ("major", hw.isin(MAJOR))):
        color, width = S["street"][cls]
        ax.add_collection(LineCollection(_lines(edges[mask].geometry), colors=color, linewidths=width * k,
                                         zorder=3, capstyle="round"))

    # Fixed physical density resolution makes PDF and PNG frequencies independent
    # of export DPI. No blur is added around removed sensitive regions.
    W = int(round(map_w * MM * 220))
    H = int(round(map_h * MM * 220))
    acc = heat_raster(pieces, extent, W, H, max(1, int(round(0.22 * 220 / 25.4 * k))))
    segments, counts = _density_segments(pieces, extent, acc)
    cap = max(3.0, float(np.percentile(counts, 98.5))) if len(counts) else 1
    cmap = LinearSegmentedColormap.from_list("huella", S["heat"])
    if len(segments):
        strength = np.clip(np.log1p(counts) / math.log1p(cap), 0, 1)
        order = np.argsort(strength, kind="stable")
        widths = (0.55 + 0.42 * strength) * k
        ax.add_collection(LineCollection(segments[order], colors=cmap(strength[order]), linewidths=widths[order],
                                         capstyle="round", joinstyle="round", zorder=5))

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
    text(212, 399, "REPETICIÓN" if es else "REPETITION", pt=5.8, color=S["sub"], align="right")
    for i, v in enumerate(np.linspace(0, 1, 6)):
        page.plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [400 * k, 400 * k],
                  color=cmap(v), lw=(0.55 + 0.95 * v) * k, solid_capstyle="round")
    text(279, 398, "+", pt=8, align="right", color=S["sub"])
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
    return {"W": W, "H": H, "cap": cap, "max": int(acc.max()) if acc.size else 0}
