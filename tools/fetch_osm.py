"""Download and cache the OpenStreetMap data a city poster needs (walk network, parks, water).

Usage: python tools/fetch_osm.py --name madrid --lat 40.4100 --lon -3.7150 --dist 7000
Writes cache/<name>/walk.graphml, cache/<name>/green.gpkg and cache/<name>/water.gpkg.
Uses the Overpass API through osmnx once and caches the result; never touches tile servers.
"""
import argparse
import os
import sys
import time

import osmnx as ox

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--lat", type=float, required=True)
p.add_argument("--lon", type=float, required=True)
p.add_argument("--dist", type=int, default=7000, help="radius in metres")
a = p.parse_args()

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", a.name)
os.makedirs(base, exist_ok=True)
ox.settings.use_cache = True
ox.settings.cache_folder = os.path.join(base, "overpass")
ox.settings.log_console = False
ox.settings.requests_timeout = 600

t0 = time.time()
gpath = os.path.join(base, "walk.graphml")
if not os.path.exists(gpath):
    G = ox.graph_from_point((a.lat, a.lon), dist=a.dist, network_type="walk", simplify=True, retain_all=False)
    ox.save_graphml(G, gpath)
    print(f"walk graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time() - t0:.0f}s)")
else:
    print("walk graph cached")

for nombre, tags in (
    ("green", {"leisure": ["park", "garden", "pitch", "golf_course"], "landuse": ["grass", "forest", "recreation_ground", "meadow"],
               "natural": ["wood", "scrub", "grassland"]}),
    ("water", {"natural": ["water"], "waterway": ["river", "riverbank", "canal"], "water": True}),
):
    path = os.path.join(base, f"{nombre}.gpkg")
    if os.path.exists(path):
        print(f"{nombre} cached")
        continue
    t1 = time.time()
    gdf = ox.features_from_point((a.lat, a.lon), tags=tags, dist=a.dist)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
    keep = [c for c in ("leisure", "landuse", "natural", "waterway", "water", "name") if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]].reset_index(drop=True)
    gdf.to_file(path, driver="GPKG")
    print(f"{nombre}: {len(gdf)} features ({time.time() - t1:.0f}s)")
print(f"done in {time.time() - t0:.0f}s -> {base}")
