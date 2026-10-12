"""Choose the poster area and fetch its OpenStreetMap data, so one command works for any city.

- auto_bbox: the area where the runner runs most (densest cluster of points), so holidays elsewhere drop out, rounded
  outward to a grid of about 1 km (snap_box).
- geocode_bbox: the bounding box of a named place (one Nominatim request through osmnx).
- city_name: the city/town near a point (one Nominatim reverse request, the point rounded to about 1 km), for the
  headline.
- fetch_basemap: streets (walk network), green, water, coastline and named summits for a bbox, cached on disk by
  rounded bbox.
Only the bbox of the area (and a point rounded to about 1 km for the city name) is sent to OpenStreetMap services;
activity files never leave the machine.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import urllib.parse
import urllib.request

import numpy as np

USER_AGENT = "pinless/0.1 (+https://github.com/Lostmanu/pinless)"


class NoSharedArea(ValueError):
    """auto_bbox with groups: no area where every group runs."""


GRID_DEG = 0.01  # the edges of an area found from your runs lie on a grid this many degrees of latitude apart


def snap_box(bbox, step=GRID_DEG):
    """(south, west, north, east) grown outward to a grid of about 1.1 km: step degrees of latitude, and the same
    distance in longitude (step / cos of the latitude, to the nearest degree).

    An edge found from your runs is a percentile of their points, and a few runs from one door at the edge of the area
    (a partner's lodging, a hotel) set it to within tens of metres of that door. The map's frame, the box sent to
    Overpass and the cache folder named after it all carry that edge, so it is rounded outward first: on the grid it
    says nothing finer than about 1 km, coarser than a private zone."""
    s, w, n, e = (float(v) for v in bbox)
    lon_step = step / max(math.cos(math.radians(round((s + n) / 2))), 0.1)

    def down(v, d):  # a value already on the grid (to a millimetre) stays where it is
        return round(math.floor(v / d + 1e-6) * d, 9)

    def up(v, d):
        return round(math.ceil(v / d - 1e-6) * d, 9)

    return (max(-90.0, down(s, step)), max(-180.0, down(w, lon_step)), min(90.0, up(n, step)),
            min(180.0, up(e, lon_step)))


def auto_bbox(activities, radius_km=7.0, cell_km=1.0, q=(1.0, 99.0), pad_frac=0.06, max_candidates=400, groups=None):
    """Bounding box (south, west, north, east) of the area where you run most.

    Score of a candidate centre = sum, over the 1 km cells within radius_km, of how many DIFFERENT activities touch
    each cell. Counting activities (not GPS points) makes it independent of the recording rate, and summing over a
    neighbourhood (not a single cell) stops a few identical loops elsewhere (a holiday) from winning.

    groups (one label per activity, e.g. 0 and 1 for two people): the area where EVERY group runs. Each group's score
    is divided by its best score, and a centre scores the smallest of those shares, so a place only one person runs
    cannot win. The box holds the 1st to 99th percentile of each group's own points near the winner, so a person
    whose watch records fewer points is not cut. Raises NoSharedArea when no area has runs of every group.

    Either way the box is rounded outward to a grid of about 1 km (snap_box), so its edges never give away where a
    run at the edge of the area started."""
    if groups is not None:
        return snap_box(_auto_bbox_groups(activities, groups, radius_km, cell_km, q, pad_frac, max_candidates))
    return snap_box(_auto_bbox_one(activities, radius_km, cell_km, q, pad_frac, max_candidates))


def _auto_bbox_one(activities, radius_km, cell_km, q, pad_frac, max_candidates):
    """auto_bbox for one person, before the box is rounded to the grid."""
    tracks = [np.asarray(a.lonlat) for a in activities if len(a.lonlat) > 1]
    pts = np.concatenate(tracks)
    if len(pts) > 400_000:  # keep it fast on very long histories
        pts = pts[:: len(pts) // 400_000 + 1]
    lat0 = float(np.median(pts[:, 1]))
    kx = 111.32 * math.cos(math.radians(lat0))  # km per degree of longitude
    ky = 110.57
    presence = {}
    for t in tracks:
        cells = set(zip(np.floor(t[:, 0] * kx / cell_km).astype(np.int64).tolist(),
                        np.floor(t[:, 1] * ky / cell_km).astype(np.int64).tolist()))
        for c in cells:
            presence[c] = presence.get(c, 0) + 1
    r = int(math.ceil(radius_km / cell_km))
    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    candidates = sorted(presence, key=presence.get, reverse=True)[:max_candidates]
    best, best_score = candidates[0], -1
    for (cx, cy) in candidates:
        score = sum(presence.get((cx + dx, cy + dy), 0) for dx, dy in offsets)
        if score > best_score:
            best, best_score = (cx, cy), score
    cx, cy = best
    clon, clat = (cx + 0.5) * cell_km / kx, (cy + 0.5) * cell_km / ky
    d = np.hypot((pts[:, 0] - clon) * kx, (pts[:, 1] - clat) * ky)
    near = pts[d <= radius_km]
    lo = np.percentile(near, q[0], axis=0)
    hi = np.percentile(near, q[1], axis=0)
    pad_lon, pad_lat = (hi - lo) * pad_frac
    return (float(lo[1] - pad_lat), float(lo[0] - pad_lon), float(hi[1] + pad_lat), float(hi[0] + pad_lon))


def _auto_bbox_groups(activities, groups, radius_km, cell_km, q, pad_frac, max_candidates):
    """auto_bbox for several people: the same cells, presence and neighbourhood sums, counted per group."""
    labelled = [(np.asarray(a.lonlat), g) for a, g in zip(activities, groups) if len(a.lonlat) > 1]
    labels = sorted(set(groups))
    if not labelled or {g for _, g in labelled} != set(labels):
        raise NoSharedArea("no area where every group runs")
    pts = np.concatenate([t for t, _ in labelled])
    if len(pts) > 400_000:  # keep it fast on very long histories
        pts = pts[:: len(pts) // 400_000 + 1]
    lat0 = float(np.median(pts[:, 1]))
    kx = 111.32 * math.cos(math.radians(lat0))  # km per degree of longitude
    ky = 110.57
    presence = {g: {} for g in labels}
    for t, g in labelled:
        cells = set(zip(np.floor(t[:, 0] * kx / cell_km).astype(np.int64).tolist(),
                        np.floor(t[:, 1] * ky / cell_km).astype(np.int64).tolist()))
        for c in cells:
            presence[g][c] = presence[g].get(c, 0) + 1
    r = int(math.ceil(radius_km / cell_km))
    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    candidates = sorted({c for g in labels for c in sorted(presence[g], key=presence[g].get, reverse=True)
                         [:max_candidates]})
    scores = np.array([[sum(presence[g].get((cx + dx, cy + dy), 0) for dx, dy in offsets) for g in labels]
                       for cx, cy in candidates], dtype=float)
    share = scores / scores.max(axis=0)
    worst, total = share.min(axis=1), share.sum(axis=1)
    best = max(range(len(candidates)), key=lambda i: (worst[i], total[i], -i))  # ties: the first cell in order
    if worst[best] <= 0:
        raise NoSharedArea("no area where every group runs")
    cx, cy = candidates[best]
    clon, clat = (cx + 0.5) * cell_km / kx, (cy + 0.5) * cell_km / ky
    # the percentiles of each group's own points, joined: pooled, a partner whose watch records ten times fewer
    # points would be a small share of them, and the 99th percentile would cut through their runs
    lows, highs = [], []
    for g in labels:
        own = np.concatenate([t for t, h in labelled if h == g])
        if len(own) > 400_000:
            own = own[:: len(own) // 400_000 + 1]
        d = np.hypot((own[:, 0] - clon) * kx, (own[:, 1] - clat) * ky)
        # a group whose cells in the neighbourhood lie just beyond radius_km still has its points within one more cell
        near = own[d <= radius_km] if (d <= radius_km).any() else own[d <= radius_km + 2 * cell_km]
        if len(near):
            lows.append(np.percentile(near, q[0], axis=0))
            highs.append(np.percentile(near, q[1], axis=0))
    if not lows:
        raise NoSharedArea("no area where every group runs")
    lo, hi = np.min(lows, axis=0), np.max(highs, axis=0)
    pad_lon, pad_lat = (hi - lo) * pad_frac
    return (float(lo[1] - pad_lat), float(lo[0] - pad_lon), float(hi[1] + pad_lat), float(hi[0] + pad_lon))


def geocode_bbox(place: str):
    import osmnx as ox

    gdf = ox.geocode_to_gdf(place)
    w, s, e, n = gdf.total_bounds
    return (float(s), float(w), float(n), float(e))


def city_name(lat: float, lon: float, lang: str = "en", timeout=20) -> str:
    """City/town/village name at a point, or '' if the lookup fails (offline is fine).

    The point is rounded to 0.01 degrees (about 1 km) before it is sent, as in place_names: the centre of the box of
    two people who run loops from one door lies a hundred metres or so from it, and a city needs no more."""
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
        {"lat": f"{lat:.2f}", "lon": f"{lon:.2f}", "format": "jsonv2", "zoom": 10, "accept-language": lang})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return ""
    adr = d.get("address", {})
    for k in ("city", "town", "village", "municipality", "county"):
        if adr.get(k):
            return adr[k]
    return d.get("name", "") or ""


_last_lookup = [0.0]


def place_names(lat: float, lon: float, lang: str = "en", timeout=20):
    """(city, country) at a point, or ('', '') if the lookup fails (offline is fine).

    The point is rounded to 0.01 degrees (about 1 km) before it is sent, answers are cached on disk, and requests are
    spaced at least 1.1 s apart (Nominatim usage policy)."""
    return _lookup(lat, lon, lang, timeout, region=False)


def place_region(lat: float, lon: float, lang: str = "en", timeout=20):
    """(city, region, country) at a point, or ('', '', '') if the lookup fails: the same request as place_names
    (the point rounded to about 1 km), with the region (state or province) as well, for a print in the hills where
    the nearest municipality says little."""
    return _lookup(lat, lon, lang, timeout, region=True)


def _lookup(lat, lon, lang, timeout, region):
    import time

    key = f"{lat:.2f},{lon:.2f},{lang}" + (",region" if region else "")
    empty = ("", "", "") if region else ("", "")
    path = os.path.join(cache_root(), "places.json")
    try:
        with open(path, encoding="utf-8") as f:
            known = json.load(f)
    except (OSError, ValueError):
        known = {}
    if key in known:
        return tuple(known[key])
    wait = 1.1 - (time.monotonic() - _last_lookup[0])
    if wait > 0:
        time.sleep(wait)
    _last_lookup[0] = time.monotonic()
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
        {"lat": f"{lat:.2f}", "lon": f"{lon:.2f}", "format": "jsonv2", "zoom": 10, "accept-language": lang})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return empty
    adr = d.get("address", {})
    city = next((adr[k] for k in ("city", "town", "village", "municipality", "county") if adr.get(k)), "")
    city = city or d.get("name", "") or ""
    if region:
        out = (city, next((adr[k] for k in ("state", "region", "province") if adr.get(k)), ""), adr.get("country", ""))
    else:
        out = (city, adr.get("country", ""))
    known[key] = list(out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(known, f, ensure_ascii=False, indent=0)
    return out


def cache_root() -> str:
    base = os.environ.get("PINLESS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "pinless")
    os.makedirs(base, exist_ok=True)
    return base


def _box_of(name):
    """(south, west, north, east) of a cache folder named bbox_S_W_N_E (m = minus), or None."""
    if not name.startswith("bbox_"):
        return None
    try:
        return tuple(float(v[1:]) * -1 if v.startswith("m") else float(v) for v in name[5:].split("_"))
    except ValueError:
        return None


def _cached_cover(root, bbox):
    """A complete cached folder whose box already contains bbox (the smallest one), or None."""
    s, w, n, e = bbox
    best = None
    for name in os.listdir(root) if os.path.isdir(root) else []:
        box_ = _box_of(name)
        if box_ is None or len(box_) != 4:
            continue
        cs, cw, cn, ce = box_
        folder = os.path.join(root, name)
        done = all(os.path.exists(os.path.join(folder, f)) for f in ("walk.graphml", "green.gpkg", "water.gpkg"))
        tol = 0.001  # folder names are rounded to 0.001 degrees, so a box can fall short by 0.0005 of what it holds
        if done and cs <= s + tol and cw <= w + tol and cn >= n - tol and ce >= e - tol:
            size = (cn - cs) * (ce - cw)
            if best is None or size < best[0]:
                best = (size, folder)
    return best[1] if best else None


OVERPASS = ("https://overpass-api.de/api", "https://overpass.private.coffee/api")  # main server, then a public mirror


@contextlib.contextmanager
def _pinned(host, ip):
    """osmnx pins one address per server through socket.gethostbyname; make that lookup return ip for host."""
    original = socket.gethostbyname
    socket.gethostbyname = lambda h: ip if h == host else original(h)
    try:
        yield
    finally:
        socket.gethostbyname = original


_answered = {}  # host -> the address that answered last: tried first next time, so a dead machine costs one timeout


def _addresses(host):
    try:
        ips = sorted(set(socket.gethostbyname_ex(host)[2]))
    except OSError:
        return [None]
    good = _answered.get(host)
    return ([good] if good in ips else []) + [ip for ip in ips if ip != good] or [None]


def _cache_written_since(t0):
    """osmnx keeps an Overpass answer in its cache only when it is a real one (HTTP OK and no server "remark" such as
    a timeout or out of memory). A file written during the call is therefore proof that OSM really answered."""
    import osmnx as ox

    folder = str(ox.settings.cache_folder)
    try:
        return any(f.endswith(".json") and os.path.getmtime(os.path.join(folder, f)) >= t0 - 1
                   for f in os.listdir(folder))
    except OSError:
        return False


def _overpass(call, log, empty_ok=True):
    """Run an osmnx download against the Overpass servers in turn (PINLESS_OVERPASS overrides the list), and against
    every address of each server: overpass-api.de is several machines, and one of them can be down while the others
    answer. "No features of that kind here" is an answer, not a failure, but only when OSM really sent it: a server
    that times out also answers with no data, and that is retried elsewhere. With empty_ok=False (streets) a real
    empty answer stops with RuntimeError too, so callers see one kind of failure."""
    import time

    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    urls = [os.environ["PINLESS_OVERPASS"]] if os.environ.get("PINLESS_OVERPASS") else list(OVERPASS)
    last = None
    for url in urls:
        host = urllib.parse.urlparse(url).hostname
        for ip in _addresses(host):
            ox.settings.overpass_url = url
            t0 = time.time()
            try:
                with _pinned(host, ip) if ip else contextlib.nullcontext():
                    out = call()
                _answered[host] = ip
                return out
            except InsufficientResponseError as ex:
                if _cache_written_since(t0):  # OSM really answered: there is nothing of that kind here
                    _answered[host] = ip
                    if empty_ok:
                        raise
                    raise RuntimeError("OpenStreetMap has no streets in this area") from ex
                last = ex
                log(f"[osm] {host}{f' ({ip})' if ip else ''} answered without data (busy?); trying the next one")
            except Exception as ex:  # busy or unreachable machine: try the next one
                last = ex
                log(f"[osm] {host}{f' ({ip})' if ip else ''} did not answer ({type(ex).__name__}); trying the next one")
    raise RuntimeError("no OpenStreetMap (Overpass) server answered; try again later") from last


LAYERS = (
    ("green", {"leisure": ["park", "garden", "pitch", "golf_course"], "landuse": ["grass", "forest", "recreation_ground",
                                                                               "meadow"],
               "natural": ["wood", "scrub", "grassland"]}),
    ("water", {"natural": ["water"], "waterway": ["river", "riverbank", "canal"], "water": True}),
    # OSM has no sea polygons: the sea (and estuaries like the Tagus) is whatever lies on the right of the coastline
    ("coast", {"natural": "coastline"}),
    # named summits for the trail print (points with name and ele)
    ("peaks", {"natural": "peak"}),
)
OPTIONAL = {"coast": "[osm] the coastline could not be fetched now: this map has no sea; run again later to add it",
            "peaks": "[osm] the summits could not be fetched now: this map names none; run again later to add them"}
ON_REQUEST = {"peaks"}  # only the trail print names summits: no other print asks Overpass for them


def fetch_basemap(bbox, root=None, log=print, extra=()):
    """Download (once) and cache the OSM layers for a bbox. Returns the folder that render.load_basemap reads.
    A layer is cached as empty only when OSM answers that there is nothing of that kind in the box; a server error
    stops instead, so a busy server never leaves a map without its parks or water. The layers of ON_REQUEST (the
    summits) are fetched only when named in extra, so a city poster on a cache made before them stays offline."""
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    root = root or cache_root()
    folder = _cached_cover(root, bbox)
    if folder:  # complete a reused folder (a layer added later, like the coast) for ITS whole box
        s, w, n, e = _box_of(os.path.basename(folder))
    else:
        s, w, n, e = bbox
        folder = os.path.join(root, f"bbox_{s:.3f}_{w:.3f}_{n:.3f}_{e:.3f}".replace("-", "m"))
    os.makedirs(folder, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(folder, "overpass")
    ox.settings.log_console = False
    ox.settings.requests_timeout = 180
    gpath = os.path.join(folder, "walk.graphml")
    box_ = (w, s, e, n)  # osmnx 2.x: (left, bottom, right, top)
    if not os.path.exists(gpath):
        log("[osm] downloading streets for the area (one time, cached)...")
        G = _overpass(lambda: ox.graph_from_bbox(box_, network_type="walk", simplify=True, retain_all=True), log,
                      empty_ok=False)
        ox.save_graphml(G, gpath)
    for nombre, tags in LAYERS:
        path = os.path.join(folder, f"{nombre}.gpkg")
        if os.path.exists(path) or (nombre in ON_REQUEST and nombre not in extra):
            continue
        log(f"[osm] downloading {nombre} for the area (one time, cached)...")
        try:
            gdf = _overpass(lambda: ox.features_from_bbox(box_, tags=tags), log)
            kinds = {"coast": ["LineString", "MultiLineString"], "peaks": ["Point"]}.get(
                nombre, ["LineString", "MultiLineString", "Polygon", "MultiPolygon"])
            gdf = gdf[gdf.geometry.type.isin(kinds)]
            cols = ("name", "ele") if nombre == "peaks" else ("leisure", "landuse", "natural", "waterway", "water",
                                                               "name")
            keep = [c for c in cols if c in gdf.columns]
            gdf = gdf[keep + ["geometry"]].reset_index(drop=True)
        except InsufficientResponseError:  # OSM answered: nothing of that kind in the area
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326)
        except RuntimeError:
            if nombre not in OPTIONAL:
                raise
            # the coast only paints the sea and the summits only name the hills: without a server now, the map is
            # drawn without them and nothing is cached, so the next run tries again
            log(OPTIONAL[nombre])
            continue
        gdf.to_file(path, driver="GPKG")
    return folder
