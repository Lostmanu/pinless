"""--layout trail: one outdoor route (a hike, a trail run, a mountain loop) as a print, on a topographic map.

The privacy rules are the race print's (race.cut), reused as they are:
- Private places are learned from every OTHER activity of the export, whatever its type (a hike from home is
  learned from years of runs from there), plus every --zone. A route that starts or ends at one is refused unless
  allow_private_ends; then one unbroken stretch is drawn, well clear of those places, and every number describes it.
- A single GPX/TCX/FIT file is an export of one activity: there is nothing to learn places from, so only --zone
  protects its ends, and the log says so (and, for a loop, that it is drawn as if it started at a public trailhead).
- The title is never the activity's own name unless the user gives it with --title: a name can say where a walk
  starts. No start or finish pin is drawn.
Only the map box (for the OpenStreetMap layers, and the Copernicus cells of the relief and the WorldCover tiles of the
ground when they are on) and the middle of the drawn stretch rounded to about 1 km (for the place name; where it
names no place, as in the sea of a bay a walk goes round, the drawn point nearest to it, rounded the same way) leave
the machine.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

from . import area, ingest, landcover, privacy, race, relief, render, render_trail
from .localtime import local_start
from .render_race import along

ONE_ACTIVITY = ("[privacy] this export holds one activity (a single file): no private places can be learned from "
                "other activities, so only --zone protects where it starts and ends. If it starts at home, add --zone "
                "there.")
PAD = 0.10  # the map reaches this share of the route's longer side beyond it: room for the hills around it
KINDS = {"Run", "TrailRun", "Walk", "Hike", "Ride", "MountainBikeRide", "GravelRide", "EBikeRide", "BackcountrySki",
         "NordicSki", "AlpineSki", "Snowshoe", "RockClimbing", "Swim", "Unknown"}


def load_peaks(folder, bbox_ll, crs):
    """[(x, y, name, ele in metres or None)] of the named OSM summits in the box, projected to crs; [] when the
    cache has no summits layer (it could not be fetched yet)."""
    import os

    import geopandas as gpd
    from shapely.geometry import box

    path = os.path.join(folder, "peaks.gpkg")
    if not os.path.exists(path):
        return []
    gdf = gpd.read_file(path)
    if not len(gdf) or "name" not in gdf.columns:
        return []
    s, w, n, e = bbox_ll
    gdf = gdf[gdf.geometry.type == "Point"]
    gdf = gdf[gdf.intersects(box(w, s, e, n)) & gdf["name"].notna()].to_crs(crs)
    ele = gdf["ele"] if "ele" in gdf.columns else [None] * len(gdf)
    return [(float(p.x), float(p.y), str(nm), render_trail.parse_ele(v))
            for p, nm, v in zip(gdf.geometry, gdf["name"], ele)]


def load_glaciers(folder, crs):
    """[shapely polygons] of the OSM glaciers in the cache, projected to crs; None when the cache has no glacier layer
    (it could not be fetched yet)."""
    import os

    import geopandas as gpd

    path = os.path.join(folder, "glacier.gpkg")
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path)
    if not len(gdf):
        return []
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(crs)
    return [g for g in gdf.geometry if g is not None and not g.is_empty]


def ice_or_rock(ground_, relief_, folder, extent, crs, log=print):
    """The land cover with its snow and ice below the snowline drawn as rock where OpenStreetMap maps no glacier
    (landcover.rock_for_snow): WorldCover takes pale scree for snow. It needs both the heights of the relief and the
    glacier layer; without either, the snow and ice is drawn as it is."""
    if ground_ is None or not (np.asarray(ground_.classes) == landcover.SNOW).any():
        return ground_
    z = getattr(relief_, "z", None)
    glaciers = load_glaciers(folder, crs)
    if z is None or glaciers is None or np.shape(z) != np.shape(ground_.classes):
        return ground_
    x0, x1, y0, y1 = extent
    _, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)
    mask = landcover.glacier_mask(glaciers, extent, np.shape(z))
    out, share = landcover.rock_for_snow(ground_, z, lat, mask)
    if share > 0:
        log(f"[ground] snow and ice below the snowline where OpenStreetMap maps no glacier: drawn as rock "
            f"({f'{share * 100:.0f} %' if share >= 0.005 else 'under 1 %'} of the map)")
    return out


def _names_a_place(answer):
    """True when a reverse lookup (city, region, country) names a place below the country."""
    city, _, country = answer
    return bool(city) and city.strip().casefold() != (country or "").strip().casefold()


def place_of(pieces, crs, lang="en"):
    """(city, region, country) of the drawn stretch (area.place_region): the answer for the middle of its first piece,
    rounded to about 1 km. Where that names no place below the country (the middle of a walk round a bay lies in the
    sea, and the answer there is the country alone), one more point is asked: the drawn point nearest to the middle
    that rounds to another point, rounded the same way. A point of the printed route, never one cut away from it."""
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    mid = np.median(np.asarray(pieces[0][1].coords), axis=0)
    lon, lat = inv.transform(*mid)
    key = (round(float(lat), 2), round(float(lon), 2))
    out = area.place_region(*key, lang=lang)
    if _names_a_place(out) or not out[2]:  # a place, or no answer at all (offline): nothing more is asked
        return out
    drawn = np.concatenate([np.asarray(p.coords) for _, p in pieces])
    lons, lats = inv.transform(drawn[:, 0], drawn[:, 1])
    keys = [(round(float(la), 2), round(float(lo), 2)) for la, lo in zip(lats, lons)]
    other = [i for i, k in enumerate(keys) if k != key]
    if not other:
        return out
    i = min(other, key=lambda j: float(np.hypot(*(drawn[j] - mid))))
    again = area.place_region(*keys[i], lang=lang)
    return again if _names_a_place(again) else out


def run(sel, pool, a, log=print):
    """sel: the activities after the type/year filters; pool: every activity of the export (for the places).
    Returns the exit code, as race.run."""
    chosen = race.pick(sel, a.activity)
    if chosen is None:
        log(f"no activity matches --activity {a.activity}")
        return 2
    route = ingest.clean(chosen)
    if route is None:
        log("this activity has only stray GPS fixes")
        return 2
    kind = route.type if route.type in KINDS else "other"  # a file's own type is free text: never logged
    log(f"[trail] the route: one {kind} activity" + (f" of {len(sel)} (--activity or --type chooses another)"
                                                     if len(sel) > 1 else ""))
    lon, lat = np.median(np.asarray(route.lonlat, dtype=float), axis=0)
    crs = render.utm_crs(lon, lat)
    tr = Transformer.from_crs(4326, crs, always_xy=True)

    def project(act):
        arr = np.asarray(act.lonlat, dtype=float)
        return np.column_stack(tr.transform(arr[:, 0], arr[:, 1]))

    zones = []
    for z in a.zone:
        zla, zlo, zr = map(float, z.split(","))
        zx, zy = tr.transform(zlo, zla)
        zones.append((zx, zy, zr))
    track = project(route)
    others = [project(c) for x in pool if x is not chosen for c in [ingest.clean(x)] if c and len(c.lonlat) > 1]
    if not others:
        log(ONE_ACTIVITY)
    try:
        pieces, rep, learned, stretch = race.cut(track, others, a.trim, a.zone_radius, zones,
                                                 allow_private_ends=a.allow_private_ends)
    except race.PrivateEnds:
        log("[privacy] STOP: this route starts or ends at a private place (where your other activities start or "
            "end: home, work, a regular meeting point) or at a --zone. One route drawn from there would point at "
            "it. Pick another one, or pass --allow-private-ends to draw it without that end (and a random stretch "
            "more), with its numbers limited to what is drawn.")
        return 2
    except privacy.PrivacyError as ex:
        log(f"[privacy] STOP: {ex}. Nothing was exported.")
        return 3
    if not pieces:
        log("this activity has too little left after the privacy cuts (it stays near a private place)")
        return 2
    total = float(along(track)[-1])
    whole = stretch == (0.0, total)
    log(f"[privacy] private places learned from {len(others)} other activities: {learned} · km "
        f"{total / 1000:.1f} -> drawn {rep.km_out:.1f}"
        f"{' (a public route: numbers for all of it)' if whole else ' (numbers for the drawn stretch only)'} · "
        f"leak check {rep.leak_check}")
    if whole and np.hypot(*(track[0] - track[-1])) < race.LOOP_M:
        log("[privacy] this route starts and ends at the same place, and no other activity of yours starts there: "
            "it is drawn as a loop from a public trailhead. If it is not one (a hotel, a friend's home, your own "
            "door), add --zone there.")

    extent = race.frame_course(pieces, render_trail.map_aspect(a.size, profile=bool(route.ele)), pad_frac=PAD)
    try:
        big = race.osm_box(extent, crs)
    except race.AcrossDateline as ex:
        log(f"[osm] STOP: {ex}. Nothing was sent or exported.")
        return 2
    try:
        folder = area.fetch_basemap(big, log=log, extra=("peaks", "glacier"))
        basemap = render.load_basemap(folder, big, crs)
    except RuntimeError as ex:
        log(f"[osm] STOP: {ex}. Run the same command again later.")
        return 4
    peaks = load_peaks(folder, big, crs)
    k = render.SIZES[a.size][0] / 297
    map_w = (render_trail.RIGHT - render_trail.LEFT) * k
    mode = getattr(a, "relief", None) or "shade"  # the point of a mountain print: on unless --relief off
    relief_ = relief.prepare_many([(extent, crs, map_w)], log=log)[0] if mode == "shade" else None
    ground = getattr(a, "ground", None) or "nature"  # the colours of the land: on unless --ground style
    ground_ = landcover.prepare_many([(extent, crs, map_w)], log=log)[0] if ground == "nature" else None
    ground_ = ice_or_rock(ground_, relief_, folder, extent, crs, log=log)
    city, region, country = place_of(pieces, crs, lang=a.lang)
    if route.name and not a.title:
        log(f"[trail] title: {city or 'none'}. The activity's own name is not printed unless you pass it with "
            f"--title (names often say where a walk starts)")
    title = a.title or city
    named = a.title.strip().casefold()  # the city in the subline, unless the title already says it
    place = ", ".join(dict.fromkeys(v for v in (city if a.title else "", region, country)
                                    if v and v.strip().casefold() != named))
    color_by = a.color_by if a.color_by in ("pace", "elevation", "single") else "single"
    info = render_trail.render(route, track, pieces, basemap, extent, a.out, style=a.style or "huella", size=a.size,
                               dpi=a.dpi, title=title, place=place, name=a.name, date=local_start(route),
                               lang=a.lang, color_by=color_by, pdf_path=a.pdf, svg_path=a.svg,
                               preview_path=a.preview, times=route.times, ele=route.ele, stretch=stretch,
                               relief=relief_, peaks=peaks, ground=ground_, log=log)
    if info["legend"].get("flat"):
        log(f"[colour] this route climbs less than {render_trail.FLAT_M:.0f} m: drawn in one colour, since colouring "
            f"it by elevation would show only the altimeter's noise")
    elif info["legend"]["mode"] != color_by:
        log(f"[colour] this activity has no {color_by} data: drawn in one colour")
    log(f"[render] {a.out} · {title or 'no title'} · {info['km']:.2f} km · {len(info['summits'])} summits named")
    return 0
