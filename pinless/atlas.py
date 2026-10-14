"""Atlas: one small map per city you ran in, on one print.

Same pipeline as the single-city poster, applied city by city:
- group_by_city: activities grouped by where they happened (places less than link_km apart are joined).
- prepare: privacy PER CITY (own projection, own start/end clusters; with few runs a single loop run already marks
  its start, so a hotel or a family home gets a zone too). Offline; fails closed like the poster.
- render: a grid of maps in the editorial style. Line colour means the same in every map: how often you ran there.
Only each city's map box, and its centre rounded to about 1 km (for the name), are sent to OpenStreetMap services.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from pyproj import Transformer

from . import area, colouring, privacy, render
from . import relief
from .render import MM, SIZES, heat_raster
from .render_editorial import (DISPLAY, REGULAR, STYLES, TEXT, _density_segments, draw_basemap, draw_legend,
                               draw_routes, place_text, route_colours)
from .render_editorial import draw_relief, footer_credits, svg_rc
from .localtime import local_start
from .render_sport import _num

EARTH_KM = 6371.0088
# The atlas keeps its own route weights (pt x k): its small maps are drawn thinner than a city poster's, and its key
# draws the strokes it always drew
ATLAS_GAIN = 0.42
ATLAS_KEY = (0.55, ATLAS_GAIN)
TOP, BOTTOM, LEFT, RIGHT = 87.0, 372.0, 18.0, 279.0  # grid area on the A3 layout grid (mm, scaled for other sizes)
CAPTION, GAP_X, GAP_Y = 15.0, 8.0, 7.0
ASPECT_MIN, ASPECT_MAX = 0.75, 2.2  # no map narrower or flatter than this (width/height)


def _hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.subtract(lon2, lon1)) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def track_km(lonlat):
    a = np.asarray(lonlat, dtype=float)
    return float(_hav_km(a[:-1, 1], a[:-1, 0], a[1:, 1], a[1:, 0]).sum()) if len(a) > 1 else 0.0


def activity_km(act):
    """An activity's km as Strava recorded it where the export says (ingest.recorded_distance), else its track's."""
    return act.recorded_m / 1000 if getattr(act, "recorded_m", None) else track_km(act.lonlat)


@dataclass(eq=False)
class City:
    acts: list = field(repr=False)
    lat: float
    lon: float
    km: float
    first: dt.date | None
    last: dt.date | None
    name: str = ""
    country: str = ""
    crs: object = field(default=None, repr=False)
    tracks: list = field(default_factory=list, repr=False)
    pieces: list = field(default_factory=list, repr=False)
    report: object = field(default=None, repr=False)

    @property
    def n(self):
        return len(self.acts)

    @property
    def hours(self):
        return sum(a.times[-1] - a.times[0] for a in self.acts if a.times) / 3600


def group_by_city(activities, link_km=30.0):
    """Groups of activities whose median points chain less than link_km apart. The city with the most km comes first
    (home), then the others in the order you first ran there."""
    acts = [a for a in activities if len(a.lonlat) > 1]
    if not acts:
        return []
    anchors = np.array([np.median(np.asarray(a.lonlat, dtype=float), axis=0) for a in acts])  # (lon, lat)
    cells, inv = np.unique(np.round(anchors / 0.05).astype(np.int64), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    centres = cells * 0.05
    parent = list(range(len(cells)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cells) - 1):
        d = _hav_km(centres[i, 1], centres[i, 0], centres[i + 1:, 1], centres[i + 1:, 0])
        for j in np.nonzero(d <= link_km)[0] + i + 1:
            parent[find(int(j))] = find(i)
    groups = {}
    for k, a in enumerate(acts):
        groups.setdefault(find(int(inv[k])), []).append(a)
    cities = []
    for members in groups.values():
        pts = np.concatenate([np.asarray(a.lonlat, dtype=float) for a in members])
        lon, lat = np.median(pts, axis=0)
        days = sorted(a.start.date() for a in members if a.start)
        cities.append(City(acts=members, lat=float(lat), lon=float(lon), km=sum(activity_km(a) for a in members),
                           first=days[0] if days else None, last=days[-1] if days else None))
    home = max(cities, key=lambda c: c.km)
    rest = sorted((c for c in cities if c is not home), key=lambda c: (c.first or dt.date.max, -c.km))
    return [home] + rest


def prepare(cities, trim_m=300.0, zone_r=500.0, user_zones_ll=(), log=print):
    """Privacy per city, in place, in two passes, and fail closed (privacy.PrivacyError: nothing is exported).

    Pass 1 finds each city's own private places (start/end clusters; with few runs one loop run is enough).
    Pass 2 cuts every city with its own places AND every other city's places and the user's zones: an activity is
    grouped by its median, so a long one can belong to another city and still start at home.
    Every city is kept; those with nothing left to draw have no pieces and are reported."""
    for c in cities:
        c.crs = render.utm_crs(c.lon, c.lat)
        tr = Transformer.from_crs(4326, c.crs, always_xy=True)
        c.tracks = []
        for a in c.acts:
            arr = np.asarray(a.lonlat, dtype=float)
            c.tracks.append(np.column_stack(tr.transform(arr[:, 0], arr[:, 1])))
    places = []  # (city index, lat, lon) of every city's own private places; in memory only
    for i, c in enumerate(cities):
        _, rep = privacy.apply(c.tracks, trim_m=trim_m, zone_r=zone_r, min_pts_floor=2)
        inv = Transformer.from_crs(c.crs, 4326, always_xy=True)
        for x, y in rep.centres:
            lon, lat = inv.transform(x, y)
            places.append((i, lat, lon))
    for i, c in enumerate(cities):
        tr = Transformer.from_crs(4326, c.crs, always_xy=True)
        shared = [(la, lo, r) for la, lo, r in user_zones_ll] + [(la, lo, zone_r) for j, la, lo in places if j != i]
        zones = []
        for la, lo, r in shared:  # in this city's projection; far ones simply never touch its tracks
            zx, zy = tr.transform(lo, la)
            if math.isfinite(zx) and math.isfinite(zy):
                zones.append((zx, zy, r))
        c.pieces, c.report = privacy.apply(c.tracks, trim_m=trim_m, zone_r=zone_r, user_zones_xy=zones,
                                           min_pts_floor=2)
        if not c.pieces:
            log(f"[atlas] a city with {c.n} run{'' if c.n == 1 else 's'} has nothing left after the privacy cuts: "
                f"not drawn")
    return cities


def _rows(m, cols, top, mh):
    """m cells of height mh in rows of cols, from top; an incomplete last row is centred."""
    mw = (RIGHT - LEFT - (cols - 1) * GAP_X) / cols
    cells = []
    for i in range(m):
        r, c = divmod(i, cols)
        in_row = min(cols, m - r * cols)
        x0 = LEFT + (RIGHT - LEFT - (in_row * mw + (in_row - 1) * GAP_X)) / 2 + c * (mw + GAP_X)
        cells.append((x0, top + r * (mh + CAPTION + GAP_Y), mw, mh))
    return cells


def layout(n, target=1.25, hero_target=1.5, empty_cost=0.35):
    """Map cells (x, y, w, h) in A3 layout millimetres for n cities. Two families are scored by how far each map is
    from a pleasant shape (target width/height), plus a cost per empty cell: a uniform grid, or the first city (home)
    across the full width with the others in a grid below. No empty row; an incomplete last row is centred."""
    H, W = BOTTOM - TOP, RIGHT - LEFT
    best = None
    for cols in range(1, 5):
        rows = math.ceil(n / cols)
        if rows * cols - n >= cols:
            continue
        mw = (W - (cols - 1) * GAP_X) / cols
        mh = (H - (rows - 1) * GAP_Y - rows * CAPTION) / rows
        if mh < 25 or not ASPECT_MIN <= mw / mh <= ASPECT_MAX:
            continue
        score = abs(math.log(mw / mh / target)) + empty_cost * (rows * cols - n)
        if best is None or score < best[0]:
            best = (score, _rows(n, cols, TOP, mh))
    m = n - 1
    for cols in range(1, 5) if n >= 3 else ():
        rows = math.ceil(m / cols)
        if rows * cols - m >= cols:
            continue
        mw = (W - (cols - 1) * GAP_X) / cols
        avail = H - (rows + 1) * CAPTION - rows * GAP_Y
        for hh in range(60, int(avail)):
            mh = (avail - hh) / rows
            if mh < 25:
                break
            if not (ASPECT_MIN <= W / hh <= ASPECT_MAX and ASPECT_MIN <= mw / mh <= ASPECT_MAX):
                continue
            score = (abs(math.log(W / hh / hero_target)) + abs(math.log(mw / mh / target))) / 2 \
                + empty_cost * (rows * cols - m)
            if best is None or score < best[0]:
                best = (score, [(LEFT, TOP, W, float(hh))] + _rows(m, cols, TOP + hh + CAPTION + GAP_Y, mh))
    return best[1]


def fit(city, aspect, min_span_m=2500.0):
    """Map extent for a city: the area where you ran most there (as on the poster), grown to the cell aspect."""
    x0, x1, y0, y1 = render.fit_extent(area.auto_bbox(city.acts), city.crs, aspect)
    if x1 - x0 < min_span_m:
        cx, cy, hw = (x0 + x1) / 2, (y0 + y1) / 2, min_span_m / 2
        x0, x1, y0, y1 = cx - hw, cx + hw, cy - hw / aspect, cy + hw / aspect
    return x0, x1, y0, y1


class _Page:
    """Text and rules in physical millimetres of the A3 layout grid, scaled by k for other paper sizes."""

    def __init__(self, fig, wmm, hmm, k, S):
        self.k, self.S = k, S
        self.ax = fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ax.set_xlim(0, wmm)
        self.ax.set_ylim(hmm, 0)
        self.ax.axis("off")

    def text(self, x, y, value, pt=9, font=TEXT, color=None, align="left", max_w=None, height=None, clear=None):
        if not value:
            return (0, 0)
        path, trans, w, h = place_text(value, x, y, self.k, pt, font, align, max_w, height, clear)
        # snap=False: a glyph made only of straight strokes (E, F, H, L, T) would be snapped to whole pixels and lose
        # its thin stems at small sizes
        self.ax.add_patch(PathPatch(path, transform=trans + self.ax.transData, color=color or self.S["ink"], lw=0,
                                    snap=False))
        return w, h

    def rule(self, y, left=LEFT, right=RIGHT, width=0.45, color=None):
        self.ax.plot([left * self.k, right * self.k], [y * self.k, y * self.k], color=color or self.S["rule"],
                     lw=width * self.k)


def totals(cities):
    """The year's numbers over EVERY city run in, drawn or not."""
    return {"cities": len(cities), "countries": len({c.country for c in cities if c.country}),
            "km": sum(c.km for c in cities), "runs": sum(c.n for c in cities), "hours": sum(c.hours for c in cities)}


def render_atlas(cities, cells, extents, basemaps, out_path, style="huella", size="A3", dpi=300, name="", title="",
                 year="", lang="en", more=0, pdf_path=None, svg_path=None, preview_path=None, summary=None,
                 color_by="heat", reliefs=None):
    """cities: the drawn ones. summary: totals() over every city (defaults to the drawn ones); more: cities not drawn.
    reliefs: one relief.Relief or None per drawn city (relief.prepare_many), drawn under its routes."""
    S = STYLES[style]
    es = lang == "es"
    wmm, hmm = SIZES[size]
    k = wmm / 297.0
    fig = plt.figure(figsize=(wmm * MM, hmm * MM), dpi=dpi, facecolor=S["bg"])
    page = _Page(fig, wmm, hmm, k, S)
    T, rule = page.text, page.rule

    tot = summary or totals(cities)
    n_cities = tot["cities"]
    T(18, 13, "P I N L E S S", pt=8.4)
    T(279, 13, "ATLAS DE RUNNING" if es else "RUNNING ATLAS", pt=6.6, align="right", color=S["sub"])
    if title:
        head = title
    elif n_cities == 1:
        head = cities[0].name or "ATLAS"
    else:
        head = f"{n_cities} CIUDADES" if es else f"{n_cities} CITIES"
    T(18, 22, head.upper(), pt=235, font=DISPLAY, max_w=215, height=44, clear=(15, 36, 17.0))  # accents clear PINLESS
    T(279, 25, str(year), pt=43, font=REGULAR, color=S["accent"], align="right", max_w=39)
    T(279, 45, "TEMPORADA" if es else "SEASON", pt=6.0, align="right", color=S["sub"])
    T(18, 73, "UN AÑO EN MOVIMIENTO" if es else "A YEAR IN MOTION", pt=7.2, color=S["sub"])
    T(279, 72.5, name.upper(), pt=13.2, align="right", max_w=105)
    rule(80)

    # densities and colours first: one scale for every map, so a colour means the same thing everywhere
    dens = []
    for c, (x, y, w, h), ext in zip(cities, cells, extents):
        pieces = privacy.clip_to_extent(c.pieces, ext)  # only what is drawn is written (no hidden PDF/SVG lines)
        W, H = int(round(w * k * MM * 220)), int(round(h * k * MM * 220))
        acc = heat_raster(pieces, ext, W, H, max(1, int(round(0.22 * 220 / 25.4 * k))))
        segments, counts, ids = _density_segments(pieces, ext, acc, with_ids=True)
        raw = (colouring.raw_values(color_by, ids, segments, c.acts, c.tracks)
               if color_by not in ("heat", "single") else np.zeros(len(segments)))
        dens.append((acc, segments, counts, raw))
    every = np.concatenate([d[2] for d in dens if len(d[2])]) if any(len(d[2]) for d in dens) else np.ones(1)
    cap = max(3.0, float(np.percentile(every, 98.5)))
    values, legend, cmap = route_colours(color_by, [d[3] for d in dens], S)

    any_relief, dataset = False, None
    for i, (c, (x, y, w, h), ext, bm, (acc, segments, counts, _), vals) in enumerate(
            zip(cities, cells, extents, basemaps, dens, values)):
        ax = fig.add_axes([x * k / wmm, 1 - (y + h) * k / hmm, w * k / wmm, h * k / hmm], zorder=1)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.axis("off")
        sw = min(1.0, max(0.55, math.sqrt(w / (RIGHT - LEFT))))  # thinner streets in small maps
        if bm is not None:
            draw_basemap(ax, bm, ext, S, k, w * k, sw)  # w * k: millimetres of paper
        if reliefs and reliefs[i]:
            any_relief |= draw_relief(ax, reliefs[i], ext, S, bm[2] if bm is not None else None, k, w * k)
            dataset = reliefs[i].dataset
        draw_routes(ax, segments, counts, cap, cmap, k, base=0.5, gain=ATLAS_GAIN, scale=max(sw, 0.8),
                    glow=S.get("glow", 0.0), values=vals)
        # caption
        cy = y + h + 3.0
        T(x, cy + 0.4, f"{i + 1:02d}", pt=14, font=REGULAR, color=S["accent"], height=4.4)
        T(x + 7.5, cy, c.name.upper(), pt=40, font=DISPLAY, height=5.4,  # no name found: no label, never coordinates
          max_w=w * 0.68 - 7.5)
        T(x + w, cy, _num(c.km, lang), pt=40, font=DISPLAY, height=5.4, align="right", max_w=w * 0.3)
        T(x + 7.5, cy + 8.2, c.country.upper(), pt=6.2, color=S["sub"], max_w=w * 0.68 - 7.5)  # no dates, as on the poster
        runs = f"KM · {c.n} {'SALIDA' if es else 'RUN'}{'' if c.n == 1 else 'S'}"
        T(x + w, cy + 8.2, runs, pt=6.2, color=S["sub"], align="right", max_w=w * 0.3)

    rule(377)
    records = (
        (_num(tot["km"], lang), "KILÓMETROS" if es else "KILOMETRES"),
        (str(tot["runs"]), "SALIDAS" if es else "RUNS"),
        (str(tot["countries"]) if tot["countries"] else "—", "PAÍSES" if es else "COUNTRIES"),
        (_num(tot["hours"], lang), "HORAS REGISTRADAS" if es else "RECORDED HOURS"),
    )
    for i, (value, label) in enumerate(records):
        xx = 18 + i * 65.25
        T(xx, 381, value, pt=25, font=DISPLAY, max_w=58, height=6.4)
        T(xx, 391.5, label, pt=6.3, color=S["sub"], max_w=58)
    rule(398)
    T(18, 404, "CADA SALIDA DEJA HUELLA." if es else "EVERY RUN LEAVES A TRACE.", pt=6.8)
    draw_legend(page.ax.plot, T, k, S, cmap, legend, lang, 404, weights=ATLAS_KEY)
    footer_credits(T, S, lang, 412, 200, 220, dataset if any_relief else None)
    if more:
        note = (f"+{more} CIUDAD{'' if more == 1 else 'ES'} SIN MAPA" if es
                else f"+{more} MORE {'CITY' if more == 1 else 'CITIES'} NOT DRAWN")
        T(279, 412, note, pt=6.2, color=S["sub"], align="right")

    # No timestamps, activity ids, source paths or author metadata, as on the poster.
    fig.savefig(out_path, dpi=dpi, facecolor=S["bg"], metadata={"Software": "Pinless"})
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=S["bg"],
                    metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None,
                              "Title": None, "Author": None, "Subject": None, "Keywords": None})
    if svg_path:
        with matplotlib.rc_context(svg_rc("pinless-atlas")):
            fig.savefig(svg_path, facecolor=S["bg"], metadata={"Date": None, "Creator": "Pinless"})
    if preview_path:
        fig.savefig(preview_path, dpi=1600 * 25.4 / hmm, facecolor=S["bg"], pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return {"cap": cap, "max": int(max((d[0].max() for d in dens if d[0].size), default=0)), "legend": legend}


def run(sel, a, log=print):
    """CLI entry for --layout atlas. Returns the exit code."""
    cities = group_by_city(sel)
    if not cities:
        log("no activities with GPS to draw")
        return 2
    zones = []
    for z in a.zone:
        la, lo, r = map(float, z.split(","))
        zones.append((la, lo, r))
    try:  # privacy for EVERY city first: their private places protect each other
        prepare(cities, trim_m=a.trim, zone_r=a.zone_radius, user_zones_ll=zones, log=log)
    except privacy.PrivacyError as ex:
        log(f"[privacy] STOP: {ex}. Nothing was exported.")
        return 3
    drawable = [c for c in cities if c.pieces]
    if not drawable:
        log("nothing left to draw after the privacy cuts")
        return 2
    drawn = drawable
    if len(drawable) > a.max_cities:  # keep the first (home) and the biggest others, in their order
        keep = {id(c) for c in [drawable[0]] + sorted(drawable[1:], key=lambda c: -c.km)[: a.max_cities - 1]}
        drawn = [c for c in drawable if id(c) in keep]
    more = len(cities) - len(drawn)
    if more:
        left = [c for c in cities if c not in drawn]
        log(f"[atlas] {more} {'city' if more == 1 else 'cities'} not drawn ({sum(c.n for c in left)} runs, "
            f"{sum(c.km for c in left):.0f} km); they still count in the totals. --max-cities draws up to 12")
    for c in cities:  # every city, for the country count
        c.name, c.country = area.place_names(round(c.lat, 2), round(c.lon, 2), lang=a.lang)
    for i, c in enumerate(drawn):
        r = c.report
        log(f"[atlas] {i + 1:02d} {c.name or '(no name found)'}{', ' + c.country if c.country else ''} · "
            f"{c.n} run{'' if c.n == 1 else 's'} · {c.km:.0f} km · zones {r.zones_suggested} + {len(zones)} · "
            f"leak check {r.leak_check}")
    if len(cities) == 1:
        log("[atlas] only one city: the single-city poster (--layout editorial) shows it bigger")
    cells = layout(len(drawn))
    extents = [fit(c, w / h) for c, (x, y, w, h) in zip(drawn, cells)]
    basemaps = []
    for c, ext in zip(drawn, extents):
        try:
            basemaps.append(_basemap(c, ext, log))
        except RuntimeError as ex:  # every OpenStreetMap server busy: stop cleanly, what was downloaded stays cached
            log(f"[osm] STOP: {ex}. Run the same command again later; the cities already downloaded are kept.")
            return 4
    k = SIZES[a.size][0] / 297
    reliefs = relief.prepare_many([(ext, c.crs, w * k) for c, (x, y, w, h), ext in zip(drawn, cells, extents)],
                                  log=log) if getattr(a, "relief", "off") == "shade" else None
    years = sorted({t.year for c in cities for x in c.acts if (t := local_start(x))})
    year = str(a.year) if a.year else (f"{years[0]}–{years[-1]}" if len(years) > 1 else (str(years[0]) if years else ""))
    info = render_atlas(drawn, cells, extents, basemaps, a.out, style=a.style or "huella", size=a.size, dpi=a.dpi,
                        name=a.name, title=a.title, year=year, lang=a.lang, more=more, pdf_path=a.pdf,
                        svg_path=a.svg, preview_path=a.preview, summary=totals(cities), color_by=a.color_by,
                        reliefs=reliefs)
    if info.get("legend", {}).get("fallback"):
        log(f"[colour] no {a.color_by} data in these activities: drawn in one colour")
    log(f"[render] {a.out} · {len(drawn)} of {len(cities)} cities drawn · heat cap {info['cap']:.0f} · "
        f"max overlap {info['max']}")
    return 0


def _basemap(city, ext, log):
    """OSM layers for a map: its box plus ~200 m so streets reach the edge (cached; RuntimeError if no server answers)."""
    inv = Transformer.from_crs(city.crs, 4326, always_xy=True)
    lons, lats = inv.transform([ext[0], ext[1], ext[0], ext[1]], [ext[2], ext[2], ext[3], ext[3]])
    big = (min(lats) - 0.002, min(lons) - 0.002, max(lats) + 0.002, max(lons) + 0.002)
    return render.load_basemap(area.fetch_basemap(big, log=log), big, city.crs)
