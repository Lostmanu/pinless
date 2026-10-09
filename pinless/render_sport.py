"""Sport layout: map on top, a results sheet with bragging stats underneath."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image, ImageFilter

from .render import MAJOR, MINOR, MM, SIZES, _hw, _lines, heat_raster
from .stats import fmt_time

STYLES = {
    "volt": {
        "bg": "#0a0b0c", "water": "#0f1c22", "water_line": "#132631", "green": "#0f150f",
        "street": {"major": ("#2a2d30", 0.6), "minor": ("#1c1e21", 0.32), "path": ("#151719", 0.2)},
        "heat": ["#2f3d00", "#86b300", "#c8f000", "#e9ff7a", "#ffffff"], "glow": "#b8f000",
        "text": "#f4f4ef", "label": "#7d7f7a", "accent": "#d4ff2e", "rule": "#26292b",
    },
    "pista": {
        "bg": "#f6f5f2", "water": "#d3e2ea", "water_line": "#bcd2de", "green": "#e5eadf",
        "street": {"major": ("#c9c6bf", 0.6), "minor": ("#dedbd5", 0.32), "path": ("#e7e4df", 0.2)},
        "heat": ["#f2a594", "#ec5b3e", "#d9311c", "#9e180c", "#4f0904"], "glow": "#e0301e",
        "text": "#121212", "label": "#76746f", "accent": "#d9311c", "rule": "#d8d5cf",
    },
}
BIG = "Segoe UI Black"
TXT = "Bahnschrift"
LAB = {
    "es": {"km": "KILÓMETROS", "n": "CARRERAS", "h": "HORAS CORRIENDO", "b5": "MEJOR 5K", "b10": "MEJOR 10K",
           "long": "TIRADA MÁS LARGA", "streets": "calles distintas", "fav": "calle favorita", "streak": "días seguidos",
           "early": "salidas antes de las 8:00", "times": "veces"},
    "en": {"km": "KILOMETRES", "n": "RUNS", "h": "HOURS RUNNING", "b5": "BEST 5K", "b10": "BEST 10K",
           "long": "LONGEST RUN", "streets": "different streets", "fav": "favourite street", "streak": "day streak",
           "early": "runs before 8 am", "times": "times"},
}


def _num(x, lang, dec=0):
    s = f"{x:,.{dec}f}"
    if lang == "es":
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def render(pieces, basemap, extent, st_, out_path, style="volt", size="A3", dpi=300, name="", city="", year="",
           lang="es", pdf_path=None):
    S = STYLES[style]
    L = LAB[lang]
    wmm, hmm = SIZES[size]
    k = wmm / 297.0
    fig = plt.figure(figsize=(wmm * MM, hmm * MM), dpi=dpi, facecolor=S["bg"])
    m = 14 * k
    map_w = wmm - 2 * m
    map_h = 268 * k
    ax = fig.add_axes([m / wmm, 1 - (m + map_h) / hmm, map_w / wmm, map_h / hmm])
    ax.set_facecolor(S["bg"])
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
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
        ax.add_collection(LineCollection(_lines(wl.geometry), colors=S["water_line"], linewidths=1.6 * k, zorder=2))
    hw = edges["highway"].map(_hw)
    for cls, mask in (("path", ~hw.isin(MAJOR | MINOR)), ("minor", hw.isin(MINOR)), ("major", hw.isin(MAJOR))):
        col, lw = S["street"][cls]
        ax.add_collection(LineCollection(_lines(edges[mask].geometry), colors=col, linewidths=lw * k, zorder=3,
                                         capstyle="round"))
    W = int(round(map_w * MM * dpi))
    H = int(round(map_h * MM * dpi))
    acc = heat_raster(pieces, extent, W, H, max(1, int(round(0.24 * dpi / 25.4 * k))))
    nz = acc[acc > 0]
    cap = max(3.0, float(np.percentile(nz, 99.3))) if len(nz) else 1.0
    v = np.clip(np.log1p(acc) / math.log1p(cap), 0, 1)
    rgba = LinearSegmentedColormap.from_list("heat", S["heat"])(v)
    rgba[..., 3] = np.where(acc > 0, np.clip(0.4 + 0.6 * v, 0, 1), 0)
    glow = Image.fromarray((v * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(1, int(dpi / 55))))
    g = np.asarray(glow, dtype=float) / 255.0
    grgba = np.zeros((H, W, 4))
    grgba[..., :3] = to_rgb(S["glow"])
    grgba[..., 3] = np.clip(g * 0.6, 0, 0.55)
    ax.imshow(grgba, extent=(x0, x1, y0, y1), zorder=4, interpolation="bilinear")
    ax.imshow(rgba, extent=(x0, x1, y0, y1), zorder=5, interpolation="nearest")

    # ---- results sheet ----
    def T(xmm, ymm_from_top, text, **kw):
        fig.text(xmm / wmm, 1 - ymm_from_top / hmm, text, **kw)

    def rule(y, x_a=m, x_b=wmm - m, lw=0.9):
        fig.add_artist(plt.Line2D([x_a / wmm, x_b / wmm], [1 - y / hmm, 1 - y / hmm], color=S["rule"], linewidth=lw * k))

    top = m + map_h
    head = (name or city).upper()
    T(m, top + 21 * k, head, ha="left", va="center", color=S["text"], fontsize=44 * k, fontfamily=BIG)
    T(wmm - m, top + 15.5 * k, f"{city.upper()}  ·  {year}" if name else year, ha="right", va="center",
      color=S["accent"], fontsize=13 * k, fontfamily=TXT, fontweight="bold")
    T(wmm - m, top + 26 * k, f"{st_['active_days']} {'DÍAS' if lang == 'es' else 'DAYS'}  ·  {_num(st_['hours'], lang)} H",
      ha="right", va="center", color=S["label"], fontsize=9.5 * k, fontfamily=TXT)
    rule(top + 35 * k, lw=1.4)
    cells = [
        (_num(st_["km"], lang), L["km"]), (str(st_["n"]), L["n"]), (_num(st_["longest_km"], lang, 1) + " km", L["long"]),
        (fmt_time(st_["best5k_s"]), L["b5"]), (fmt_time(st_["best10k_s"]), L["b10"]),
        (str(st_.get("streets", "—")), L["streets"].upper()),
    ]
    colw = (wmm - 2 * m) / 3
    for i, (val, lab) in enumerate(cells):
        r, c = divmod(i, 3)
        cx = m + c * colw + 2 * k
        cy = top + 52 * k + r * 33 * k
        T(cx, cy, val, ha="left", va="center", color=S["text"] if i else S["accent"], fontsize=27 * k, fontfamily=BIG)
        T(cx, cy + 12.5 * k, lab, ha="left", va="center", color=S["label"], fontsize=7.6 * k, fontfamily=TXT)
        if c:
            x = m + c * colw - 2 * k
            fig.add_artist(plt.Line2D([x / wmm, x / wmm], [1 - (cy - 9 * k) / hmm, 1 - (cy + 16 * k) / hmm],
                                      color=S["rule"], linewidth=0.9 * k))
    rule(top + 102 * k)
    fav = st_.get("fav_street")
    foot = []
    if fav:
        foot.append(f"{L['fav']}: {fav} ({st_['fav_street_n']} {L['times']})")
    foot.append(f"{st_['streak']} {L['streak']}")
    foot.append(f"{st_['early_runs']} {L['early']}")
    T(m + 2 * k, top + 110 * k, "   ·   ".join(foot), ha="left", va="center", color=S["label"], fontsize=8 * k, fontfamily=TXT)
    T(wmm - m, hmm - 6 * k, "© OpenStreetMap contributors (ODbL) · openstreetmap.org/copyright", ha="right", va="bottom",
      color=S["label"], fontsize=5 * k, fontfamily=TXT, alpha=0.8)
    fig.savefig(out_path, dpi=dpi, facecolor=S["bg"])
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=S["bg"])
    plt.close(fig)
    return {"W": W, "H": H, "max": int(acc.max()) if acc.size else 0}
