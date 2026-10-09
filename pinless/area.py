"""Choose the poster area and fetch its OpenStreetMap data, so one command works for any city.

- auto_bbox: the area where the runner runs most (densest cluster of points), so holidays elsewhere drop out.
- geocode_bbox: the bounding box of a named place (one Nominatim request through osmnx).
- city_name: the city/town at a point (one Nominatim reverse request), for the headline.
- fetch_basemap: streets (walk network), green and water for a bbox, cached on disk by rounded bbox.
Only the bbox of the area is sent to OpenStreetMap services; activity files never leave the machine.
"""
from __future__ import annotations

import json
import math
import os
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


def cache_root() -> str:
    base = os.environ.get("PINLESS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "pinless")
    os.makedirs(base, exist_ok=True)
    return base


def fetch_basemap(bbox, root=None, log=print):
    """Download (once) and cache the OSM layers for a bbox. Returns the folder that render.load_basemap reads."""
    import osmnx as ox

    s, w, n, e = bbox
    key = f"bbox_{s:.3f}_{w:.3f}_{n:.3f}_{e:.3f}".replace("-", "m")
    folder = os.path.join(root or cache_root(), key)
    os.makedirs(folder, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(folder, "overpass")
    ox.settings.log_console = False
    ox.settings.requests_timeout = 600
    gpath = os.path.join(folder, "walk.graphml")
    box_ = (w, s, e, n)  # osmnx 2.x: (left, bottom, right, top)
    if not os.path.exists(gpath):
        log("[osm] downloading streets for the area (one time, cached)...")
        G = ox.graph_from_bbox(box_, network_type="walk", simplify=True, retain_all=True)
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
            gdf = ox.features_from_bbox(box_, tags=tags)
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
            keep = [c for c in ("leisure", "landuse", "natural", "waterway", "water", "name") if c in gdf.columns]
            gdf = gdf[keep + ["geometry"]].reset_index(drop=True)
        except Exception:  # no features of that kind in the area
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326)
        gdf.to_file(path, driver="GPKG")
    return folder
