"""Named OpenStreetMap streets around the demo runs, so the street tests run offline on any machine.

Data © OpenStreetMap contributors, ODbL. Built by tools/make_street_fixture.py from the local Madrid walk graph.
"""
import gzip
import json
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "madrid_streets.json.gz")
SCALE = 1e7  # lon/lat stored as integers in 1e-7 degrees (OSM's own precision, so lossless), delta-encoded per segment


def load(crs):
    """GeoDataFrame with `name` and `geometry`, projected to crs (same shape stats.compute gets from osmnx)."""
    with gzip.open(PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)
    names, geoms = [], []
    for name, segments in data["streets"].items():
        for flat in segments:
            lonlat = np.cumsum(np.asarray(flat, dtype=np.int64).reshape(-1, 2), axis=0) / SCALE
            names.append(name)
            geoms.append(LineString(lonlat))
    return gpd.GeoDataFrame({"name": names}, geometry=geoms, crs=4326).to_crs(crs)
