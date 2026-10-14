"""Race: one activity as a finisher print. The course, its elevation profile, distance, time, pace, climb and splits.

The caller passes the privacy-cut pieces and the along-track stretch the numbers may describe (race.cut): a public
course is described whole; a run allowed to start or end at a private place is described only where it is drawn,
so its profile and splits cannot be matched against the hidden ends. The date of the run is printed: a race poster
is about that day.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from shapely.geometry import LineString

from . import privacy
from .atlas import _Page as Page
from .colouring import fmt_pace
from .render import MM, SIZES
from .render_editorial import DISPLAY, REGULAR, STYLES, _line_path, draw_basemap, draw_routes, place_text
from .render_editorial import baseline_of, draw_relief, footer_credits, svg_rc
from .render_sport import _num
from .stats import fmt_time

MAP_TOP, MAP_H = 90.0, 222.0
UNIT_PT, UNIT_GAP_MM = 11, 1.2  # the units beside the numbers (pt x k, A3-grid mm), on the numbers' baseline
MONTHS = {"en": "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(),
          "es": "ENE FEB MAR ABR MAY JUN JUL AGO SEP OCT NOV DIC".split()}


def map_aspect(size="A3"):
    return (SIZES[size][0] - 36 * SIZES[size][0] / 297) / (MAP_H * SIZES[size][0] / 297)


def along(xy):
    xy = np.asarray(xy, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def profile(xy, ele):
    """Cumulative distance (m) and a lightly smoothed elevation (m) along the track."""
    dist = along(xy)
    e = np.asarray(ele, dtype=float)
    k = min(7, len(e) // 2 * 2 + 1)
    if k >= 3:  # running median: barometer and GPS altitude jitter
        pad = np.pad(e, k // 2, mode="edge")
        e = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    return dist, e


def climb(ele, step=3.0):
    """Total ascent (m), counting a rise only once it exceeds `step` metres (ignores jitter)."""
    total, ref = 0.0, ele[0] if len(ele) else 0.0
    for h in ele[1:]:
        if h - ref >= step:
            total += h - ref
            ref = h
        elif ref - h >= step:
            ref = h
    return total


def splits(dist, times):
    """(seconds for each full kilometre, (pace s/km, metres) of the last part or None when it is under 200 m)."""
    t = np.asarray(times, dtype=float)
    marks = np.arange(1000, dist[-1] + 1e-6, 1000)
    at = np.interp(marks, dist, t)
    full = np.diff(np.concatenate([[t[0]], at]))
    rest = dist[-1] - (marks[-1] if len(marks) else 0)
    last = None
    if rest > 200:
        last = ((t[-1] - (at[-1] if len(at) else t[0])) * 1000 / rest, rest)
    return full, last


def pace_along(dist, times, window=250.0):
    """Pace (s/km) at every track point, averaged over `window` metres: GPS noise would otherwise dominate."""
    t = np.asarray(times, dtype=float)
    lo = np.interp(np.clip(dist - window / 2, 0, dist[-1]), dist, t)
    hi = np.interp(np.clip(dist + window / 2, 0, dist[-1]), dist, t)
    span = np.clip(np.minimum(dist + window / 2, dist[-1]) - np.maximum(dist - window / 2, 0), 1.0, None)
    return (hi - lo) * 1000 / span


def restrict(xy, times, ele, stretch):
    """The track, times and elevations between two along-track distances, interpolated at both ends."""
    xy = np.asarray(xy, dtype=float)
    d = along(xy)
    if stretch is None or (stretch[0] <= 0 and stretch[1] >= d[-1]):
        return xy, times, ele
    lo, hi = stretch
    at = np.concatenate([[lo], d[(d > lo) & (d < hi)], [hi]])

    def take(v):
        return list(np.interp(at, d, np.asarray(v, dtype=float))) if v else v

    return np.column_stack([np.interp(at, d, xy[:, 0]), np.interp(at, d, xy[:, 1])]), take(times), take(ele)


def vertex_positions(coords, xy, dist, pairs):
    """Along-track distance of every vertex of a drawn piece. A piece is a stretch of the track: its vertices are the
    track's own, in order, except where it was cut. Find one of its segments among the track's (pairs: consecutive
    vertex pairs -> index) and count along from there, which stays exact where the course comes back along the same
    street (a projection onto the line cannot tell the way out from the way back); a piece with none is projected."""
    c = np.asarray(coords, dtype=float)
    pos = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
    for j in range(len(c) - 1):
        i = pairs.get((tuple(c[j]), tuple(c[j + 1])))
        if i is not None:
            return pos - pos[j] + dist[i]
    return shapely.line_locate_point(LineString(xy), shapely.points(c))


def render(activity, track_xy, pieces, basemap, extent, out_path, style="huella", size="A3", dpi=300, title="",
           city="", name="", date=None, lang="en", color_by="pace", pdf_path=None, svg_path=None, preview_path=None,
           times=None, ele=None, stretch=None, relief=None, scale=1.0):
    """activity: the run; track_xy: its (end-cleaned) projected track, with times and ele aligned to it (default: the
    activity's); pieces: the privacy-cut pieces (index, LineString), drawn; stretch: (start_m, end_m) along the track
    that the numbers describe (default: all of it); date: the local datetime of the run; relief: a relief.Relief for
    the map (relief.prepare_many), drawn under the course; scale: the share of the track's own length the course
    really measured (race.distance_scale), for its distance, pace, splits and the profile's km, never its drawing."""
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
    real = dist * scale  # what the course measured, for its numbers; the drawing keeps the track's own metres
    secs = float(times[-1] - times[0]) if times else 0.0
    if not secs > 0:  # a file with no clock, or a broken one: no time, pace or splits rather than nonsense
        times, secs = None, None

    # header
    T(18, 13, "P I N L E S S", pt=8.4)
    T(279, 13, "CARRERA" if es else "RACE", pt=6.6, align="right", color=S["sub"])
    head = (title or city or ("CARRERA" if es else "RACE")).upper()
    *_, title_h = place_text(head, 18, 22, k, 235, DISPLAY, max_w=212, height=40, clear=(15, 36, 17.0))
    # a long title is set smaller to fit the width: centre it in its band instead of leaving a hole under it
    T(18, 22 + max(0.0, 40 - title_h) / 2, head, pt=235, font=DISPLAY, max_w=212, height=40, clear=(15, 36, 17.0))
    if date is not None:
        T(279, 24, str(date.day), pt=43, font=REGULAR, color=S["accent"], align="right", max_w=39)
        T(279, 44, f"{MONTHS['es' if es else 'en'][date.month - 1]} {date.year}", pt=6.6, align="right",
          color=S["sub"])
    T(18, 70, city.upper(), pt=7.2, color=S["sub"])
    T(279, 69.5, name.upper(), pt=13.2, align="right", max_w=105)
    rule(80)

    # the course
    m, map_w, map_h = 18 * k, wmm - 36 * k, MAP_H * k
    ax = fig.add_axes([m / wmm, 1 - (MAP_TOP * k + map_h) / hmm, map_w / wmm, map_h / hmm], zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    draw_basemap(ax, basemap, extent, S, k, map_w)
    drawn = draw_relief(ax, relief, extent, S, basemap[2], k, map_w) if relief is not None else False
    cut = privacy.clip_to_extent(pieces, extent)
    coords = [c for c in (np.asarray(p.coords) for _, p in cut) if len(c) > 1]
    segments = np.concatenate([np.stack((c[:-1], c[1:]), axis=1) for c in coords]) if coords else np.empty((0, 2, 2))
    cmap = LinearSegmentedColormap.from_list("race", S["heat"])
    values, legend = None, {"mode": "single"}
    if len(segments) and color_by in ("pace", "elevation"):
        pts = list(map(tuple, xy))
        pairs = {pq: i for i, pq in enumerate(zip(pts[:-1], pts[1:]))}
        at = np.concatenate([(lambda p: (p[:-1] + p[1:]) / 2)(vertex_positions(c, xy, dist, pairs)) for c in coords])
        if color_by == "pace" and times:
            v = np.interp(at, dist, pace_along(real, times))
            lo, hi = np.percentile(v, [5, 95])
            values = 1 - np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)  # faster = stronger
            legend = {"mode": "pace", "lo": lo, "hi": hi}
        elif color_by == "elevation" and ele:
            v = np.interp(at, dist, profile(xy, ele)[1])
            lo, hi = float(v.min()), float(v.max())
            values = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
            legend = {"mode": "elevation", "lo": lo, "hi": hi}
    if values is None:
        values = np.full(len(segments), 0.85)
    if len(segments):  # a casing in the paper colour lifts the course off the streets
        ax.add_patch(PathPatch(_line_path(list(segments)), fill=False, edgecolor=S["bg"], linewidth=3.4 * k,
                               capstyle="round", joinstyle="round", zorder=4.8))
    draw_routes(ax, segments, np.full(len(segments), 30.0), 3.0, cmap, k, base=1.7, gain=0.0,
                glow=S.get("glow", 0.0), values=values)

    # elevation profile under the map
    ptop = MAP_TOP + MAP_H + 6
    if ele:
        d, e = profile(xy, ele)
        d = d * scale
        pax = fig.add_axes([18 * k / wmm, 1 - (ptop + 24) * k / hmm, 261 * k / wmm, 24 * k / hmm], zorder=2)
        pax.set_facecolor("none")
        base = e.min() - 0.15 * max(e.max() - e.min(), 10)
        pax.fill_between(d / 1000, base, e, color=S["heat"][0], alpha=0.55, linewidth=0)
        pax.plot(d / 1000, e, color=S["heat"][3], linewidth=0.9 * k)
        pax.set_xlim(0, d[-1] / 1000)
        pax.set_ylim(base, e.max() + 0.1 * max(e.max() - e.min(), 10))
        pax.axis("off")
        step = 5 if d[-1] > 12000 else 1
        for km_mark in range(0, int(d[-1] // 1000) + 1, step):
            T(18 + km_mark * 1000 / d[-1] * 261, ptop + 26, str(km_mark), pt=4.4, color=S["sub"], align="center")
        T(18, ptop - 3.2, "ALTITUD" if es else "ELEVATION", pt=5.6, color=S["sub"])
        T(279, ptop - 3.2, f"{e.min():.0f} – {e.max():.0f} m", pt=5.6, color=S["sub"], align="right")

    # numbers
    ytop = ptop + 34
    rule(ytop - 3)
    km = real[-1] / 1000
    cells = [(_num(km, lang, 2 if km < 50 else 1), "km", "DISTANCIA" if es else "DISTANCE"),
             (fmt_time(secs) if secs else "—", "", "TIEMPO" if es else "TIME"),
             (fmt_pace(secs / km)[:-3] if secs and km else "—", "/km", "RITMO" if es else "PACE"),
             (f"+{climb(profile(xy, ele)[1]):.0f}" if ele else "—", "m", "DESNIVEL" if es else "CLIMB")]
    for i, (value, unit, label) in enumerate(cells):
        x = 18 + i * 65.25
        fit = dict(pt=40, font=DISPLAY, max_w=52, height=11)
        w_, _ = T(x, ytop + 1, value, **fit)
        if unit:  # on the number's baseline
            T(x + w_ + UNIT_GAP_MM, ytop + 1 + baseline_of(value, k, **fit) - baseline_of(unit, k, pt=UNIT_PT,
                                                                                          font=REGULAR),
              unit, pt=UNIT_PT, font=REGULAR, color=S["accent"])
        T(x, ytop + 15, label, pt=6.0, color=S["sub"])

    # kilometre splits: one bar per full km, the last part (if any) apart and paler; the fastest is a full km
    full, last = splits(real, times) if times else (np.empty(0), None)
    fastest = None
    if len(full) and (full > 0).all():
        sp = np.append(full, last[0]) if last and last[0] > 0 else full
        stop = ytop + 24
        rule(stop - 2, width=0.32)
        sax = fig.add_axes([18 * k / wmm, 1 - (stop + 14) * k / hmm, 261 * k / wmm, 14 * k / hmm], zorder=2)
        sax.set_facecolor("none")
        fastest = int(np.argmin(full))
        speed = 1000 / sp
        colours = [S["accent"] if j == fastest else S["heat"][0] for j in range(len(full))]
        bars = sax.bar(np.arange(len(sp)), speed - speed.min() * 0.85, width=0.68, linewidth=0,
                       color=colours + ([S["heat"][0]] if len(sp) > len(full) else []))
        if len(sp) > len(full):
            bars[-1].set_alpha(0.45)
        sax.set_xlim(-0.6, len(sp) - 0.4)
        sax.axis("off")
        T(18, stop + 16.5, "PARCIALES POR KM" if es else "KM SPLITS", pt=5.6, color=S["sub"])
        best = f"KM {fastest + 1} · {fmt_pace(full[fastest])}"
        tail = ""
        if len(sp) > len(full):
            tail = f"   ÚLTIMOS {last[1]:.0f} M · {fmt_pace(last[0])}" if es else \
                f"   LAST {last[1]:.0f} M · {fmt_pace(last[0])}"
        T(279, stop + 16.5, (f"MÁS RÁPIDO: {best}" if es else f"FASTEST: {best}") + tail, pt=5.6, color=S["ink"],
          align="right")

    # footer: what the line colour means, attribution
    rule(398)
    T(18, 404, "CADA SALIDA DEJA HUELLA." if es else "EVERY RUN LEAVES A TRACE.", pt=6.8)
    if legend["mode"] in ("pace", "elevation"):
        T(212, 404, ("RITMO" if es else "PACE") if legend["mode"] == "pace" else ("ALTITUD" if es else "ELEVATION"),
          pt=5.8, color=S["sub"], align="right")
        for i, v in enumerate(np.linspace(0, 1, 6)):
            page.ax.plot([(218 + i * 8.4) * k, (225 + i * 8.4) * k], [405 * k, 405 * k], color=cmap(v), lw=1.3 * k,
                         solid_capstyle="round")
        lo_t = fmt_pace(legend["hi"]) if legend["mode"] == "pace" else f"{legend['lo']:.0f} m"
        hi_t = fmt_pace(legend["lo"]) if legend["mode"] == "pace" else f"{legend['hi']:.0f} m"
        T(218, 407.6, lo_t, pt=4.6, color=S["sub"])
        T(268, 407.6, hi_t, pt=4.6, color=S["sub"], align="right")
    footer_credits(T, S, lang, 412, 200, 238, relief.dataset if drawn else None)

    fig.savefig(out_path, dpi=dpi, facecolor=S["bg"], metadata={"Software": "Pinless"})
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=S["bg"],
                    metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None,
                              "Title": None, "Author": None, "Subject": None, "Keywords": None})
    if svg_path:
        with matplotlib.rc_context(svg_rc("pinless-race")):
            fig.savefig(svg_path, facecolor=S["bg"], metadata={"Date": None, "Creator": "Pinless"})
    if preview_path:
        fig.savefig(preview_path, dpi=1600 * 25.4 / hmm, facecolor=S["bg"], pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return {"km": km, "secs": secs, "legend": legend, "splits": (full, last), "fastest": fastest, "relief": drawn}
