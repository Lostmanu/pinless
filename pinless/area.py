"""Choose the poster area and fetch its OpenStreetMap data, so one command works for any city.

- auto_bbox: the area where the runner runs most (densest cluster of points), so holidays elsewhere drop out.
- geocode_bbox: the bounding box of a named place (one Nominatim request through osmnx).
- city_name: the city/town at a point (one Nominatim reverse request), for the headline.
- fetch_basemap: streets (walk network), green and water for a bbox, cached on disk by rounded bbox.
Only the bbox of the area is sent to OpenStreetMap services; activity files never leave the machine.
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


def auto_bbox(activities, radius_km=7.0, cell_km=1.0, q=(1.0, 99.0), pad_frac=0.06, max_candidates=400):
    """Bounding box (south, west, north, east) of the area where you run most.

    Score of a candidate centre = sum, over the 1 km cells within radius_km, of how many DIFFERENT activities touch
    each cell. Counting activities (not GPS points) makes it independent of the recording rate, and summing over a
    neighbourhood (not a single cell) stops a few identical loops elsewhere (a holiday) from winning."""
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


def geocode_bbox(place: str):
    import osmnx as ox

    gdf = ox.geocode_to_gdf(place)
    w, s, e, n = gdf.total_bounds
    return (float(s), float(w), float(n), float(e))


def city_name(lat: float, lon: float, lang: str = "en", timeout=20) -> str:
    """City/town/village name at a point, or '' if the lookup fails (offline is fine)."""
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
        {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "format": "jsonv2", "zoom": 10, "accept-language": lang})
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
    import time

    key = f"{lat:.2f},{lon:.2f},{lang}"
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
        return ("", "")
    adr = d.get("address", {})
    city = next((adr[k] for k in ("city", "town", "village", "municipality", "county") if adr.get(k)), "")
    out = (city or d.get("name", "") or "", adr.get("country", ""))
    known[key] = list(out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(known, f, ensure_ascii=False, indent=0)
    return out


def cache_root() -> str:
    base = os.environ.get("PINLESS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "pinless")
    os.makedirs(base, exist_ok=True)
    return base


def _cached_cover(root, bbox):
    """A complete cached folder whose box already contains bbox (the smallest one), or None."""
    s, w, n, e = bbox
    best = None
    for name in os.listdir(root) if os.path.isdir(root) else []:
        if not name.startswith("bbox_"):
            continue
        try:
            cs, cw, cn, ce = (float(v[1:]) * -1 if v.startswith("m") else float(v) for v in name[5:].split("_"))
        except ValueError:
            continue
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


def fetch_basemap(bbox, root=None, log=print):
    """Download (once) and cache the OSM layers for a bbox. Returns the folder that render.load_basemap reads.
    A layer is cached as empty only when OSM answers that there is nothing of that kind in the box; a server error
    stops instead, so a busy server never leaves a map without its parks or water."""
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    s, w, n, e = bbox
    root = root or cache_root()
    reuse = _cached_cover(root, bbox)
    if reuse:
        return reuse
    key = f"bbox_{s:.3f}_{w:.3f}_{n:.3f}_{e:.3f}".replace("-", "m")
    folder = os.path.join(root, key)
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
    for nombre, tags in (
        ("green", {"leisure": ["park", "garden", "pitch", "golf_course"], "landuse": ["grass", "forest", "recreation_ground", "meadow"],
                   "natural": ["wood", "scrub", "grassland"]}),
        ("water", {"natural": ["water"], "waterway": ["river", "riverbank", "canal"], "water": True}),
    ):
        path = os.path.join(folder, f"{nombre}.gpkg")
        if os.path.exists(path):
            continue
        log(f"[osm] downloading {nombre} for the area (one time, cached)...")
        try:
            gdf = _overpass(lambda: ox.features_from_bbox(box_, tags=tags), log)
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
            keep = [c for c in ("leisure", "landuse", "natural", "waterway", "water", "name") if c in gdf.columns]
            gdf = gdf[keep + ["geometry"]].reset_index(drop=True)
        except InsufficientResponseError:  # OSM answered: nothing of that kind in the area
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326)
        gdf.to_file(path, driver="GPKG")
    return folder
