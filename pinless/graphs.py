"""Small charts for the poster's graphs band, from aggregates only (no place can be read from them):

- kilometres per month (local months);
- time of day: when your runs start (local time), on a 24-hour rose;
- direction: which way your streets run, a 180-degree rose weighted by distance (a grid city shows its grid).
"""
from __future__ import annotations

import math

import numpy as np

from .ingest import distance_scale
from .localtime import local_start

MONTHS = {"en": "JFMAMJJASOND", "es": "EFMAMJJASOND"}


def km_per_month(activities, tracks_xy):
    km = np.zeros(12)
    for a, xy in zip(activities, tracks_xy):
        t = local_start(a)
        xy = np.asarray(xy)
        if t is not None and len(xy) > 1:
            own = float(np.hypot(*np.diff(xy, axis=0).T).sum())
            km[t.month - 1] += own * distance_scale(a, own) / 1000  # as Strava recorded it, where the export says
    return km


def start_hours(activities):
    """Fractional local hour of every start (0-24)."""
    return np.array([t.hour + t.minute / 60 for a in activities if (t := local_start(a)) is not None])


def bearings(tracks_xy, bins=18):
    """Distance run in each direction, folded to 0-180 degrees (north to south), in `bins` sectors."""
    hist = np.zeros(bins)
    for xy in tracks_xy:
        xy = np.asarray(xy)
        if len(xy) < 2:
            continue
        d = np.diff(xy, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        ang = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 180  # 0 = north, clockwise
        hist += np.bincount((ang / (180 / bins)).astype(int) % bins, weights=length, minlength=bins)
    return hist


def draw_band(fig, text, S, k, wmm, hmm, activities, tracks_xy, lang, top, height, big_font=None):
    """Three charts side by side in the band [top, top + height] (A3-grid millimetres), each labelled underneath on
    one shared line."""
    es = lang == "es"
    ink, sub, bar = S["ink"], S["sub"], S["heat"][1]
    label_y = top + height - 2.6
    chart_h = height - 8

    def axes(x, y, w, h, polar=False):
        ax = fig.add_axes([x * k / wmm, 1 - (y + h) * k / hmm, w * k / wmm, h * k / hmm], polar=polar, zorder=2)
        ax.set_facecolor("none")
        ax.axis("off")
        return ax

    # kilometres per month, the best month labelled on its bar
    km = km_per_month(activities, tracks_xy)
    x0, w = 18, 124
    ax = axes(x0, top + 2, w, chart_h - 2)
    ax.bar(np.arange(12), km, width=0.6, color=bar, linewidth=0)
    ax.set_xlim(-0.6, 11.6)
    peak = max(km.max(), 1)
    ax.set_ylim(0, peak * 1.22)
    for i, letter in enumerate(MONTHS["es" if es else "en"]):
        text(x0 + (i + 0.6) * w / 12.2, top + chart_h + 0.6, letter, pt=4.4, color=sub, align="center")
    best = int(km.argmax())
    text(x0 + (best + 0.6) * w / 12.2, top + 2 + (chart_h - 2) * (1 - km[best] / (peak * 1.22)) - 3.4,
         f"{km[best]:.0f}", pt=5.0, color=ink, align="center")
    text(x0, label_y, "KM POR MES" if es else "KM PER MONTH", pt=5.6, color=sub)

    # time of day: 24 hours around the clock face, midnight on top, and how many runs start before 8
    hours = start_hours(activities)
    counts = np.bincount(np.floor(hours).astype(int) % 24, minlength=24) if len(hours) else np.zeros(24)
    size, cx = chart_h, 176
    ax = axes(cx - size / 2, top, size, size, polar=True)
    theta = (np.arange(24) + 0.5) * 2 * math.pi / 24
    ax.bar(theta, counts / max(counts.max(), 1), width=2 * math.pi / 24 * 0.8, color=bar, linewidth=0, bottom=0.2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    for h in (0, 6, 12, 18):
        a = h * 2 * math.pi / 24
        text(cx + math.sin(a) * size * 0.6, top + size / 2 - math.cos(a) * size * 0.6 - 1.0, str(h), pt=4.2,
             color=sub, align="center")
    text(cx, label_y, "HORA DEL DÍA" if es else "TIME OF DAY", pt=5.6, color=sub, align="center")
    if len(hours):
        big = {"font": big_font} if big_font is not None else {}
        text(cx + size / 2 + 9, top + 2.5, f"{int((hours < 8).sum())}", pt=30, color=ink, height=7, **big)
        text(cx + size / 2 + 9, top + 10.5, "SALIDAS ANTES" if es else "RUNS BEFORE", pt=4.6, color=sub)
        text(cx + size / 2 + 9, top + 13.5, "DE LAS 8:00" if es else "8 AM", pt=4.6, color=sub)

    # direction: a symmetric 180-degree rose
    hist = bearings(tracks_xy)
    cx = 258
    ax = axes(cx - size / 2, top, size, size, polar=True)
    bins = len(hist)
    width = math.pi / bins
    r = hist / max(hist.max(), 1e-9)
    theta = (np.arange(bins) + 0.5) * width
    ax.bar(np.concatenate([theta, theta + math.pi]), np.concatenate([r, r]), width=width * 0.85, color=bar,
           linewidth=0, bottom=0.12)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    text(cx, top - 2.2, "N", pt=4.6, color=sub, align="center")
    text(cx, label_y, "DIRECCIÓN" if es else "DIRECTION", pt=5.6, color=sub, align="center")
