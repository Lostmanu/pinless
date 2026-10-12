"""Generate a SYNTHETIC hike on real OpenStreetMap paths and the Copernicus elevation model, written as one GPX file.

Nobody's real data: a fictional hiker walks a loop from the Puerto de Cotos car park in the Sierra de Guadarrama, up
the ridge over Dos Hermanas to Penalara and the Risco de los Claveles, down to the Laguna de los Pajaros and back
along the valley past the Laguna Grande to the car park, on the OSM walk network (paths and tracks preferred over
roads).

- Elevation: sampled from the Copernicus DEM (GLO-30) through pinless.relief's reader, at the true positions, with
  a little barometer noise. A development tool may read the DEM; the poster code never samples it.
- Clock: Tobler's hiking function on the slope over 100 m, at most 30 % (a path zigzags where the map draws it
  straight), for a relaxed hiker at 85 % of it; a 20 minute stop at the summit and 15 by the Laguna de los Pajaros,
  standing still with the fix wandering a few metres; a point every 10 s.
- Positions: correlated GPS noise of about 3 m.
Usage: python tools/simulate_hike.py --out demo/hike_demo.gpx
Then: pinless demo/hike_demo.gpx --layout trail --title "Peñalara" --name "Lucía"
"""
import argparse
import datetime as dt
import math
import os
import random
import sys

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pinless import area, relief  # noqa: E402

BOX = (40.790, -4.020, 40.880, -3.900)  # south, west, north, east: the hills around the loop
# Public landmarks (approximate), in walking order; the route snaps each to the nearest node of the walk network
WAYPOINTS = [("Puerto de Cotos", 40.8222, -3.9630),
             ("Hermana Mayor", 40.8399, -3.9629), ("Penalara", 40.8500, -3.9561),
             ("Risco de los Claveles", 40.8551, -3.9523), ("Laguna de los Pajaros", 40.8604, -3.9476),
             ("Laguna Grande", 40.8399, -3.9574), ("Puerto de Cotos", 40.8222, -3.9630)]
STOPS = {"Penalara": 20 * 60, "Laguna de los Pajaros": 15 * 60}
FACTOR = {"path": 0.7, "track": 0.75, "footway": 0.8, "bridleway": 0.8, "steps": 1.5, "unclassified": 1.2,
          "service": 1.2, "residential": 1.3, "tertiary": 1.6, "secondary": 2.5, "primary": 3.0}
CRS = 25830  # ETRS89 / UTM 30N
STEP_S = 10.0


def _highway(d):
    h = d.get("highway", "path")
    return h[0] if isinstance(h, list) else h


def route_on_paths(folder):
    """The loop through the waypoints on the walk network, as (lon, lat) points, and the index where each waypoint
    is reached."""
    G = ox.load_graphml(os.path.join(folder, "walk.graphml"))
    for _, _, d in G.edges(data=True):
        d["w"] = d["length"] * FACTOR.get(_highway(d), 1.0)
    ids = np.array(list(G.nodes))
    xs = np.array([G.nodes[n]["x"] for n in ids])
    ys = np.array([G.nodes[n]["y"] for n in ids])
    kx = math.cos(math.radians(40.84))
    nodes = [ids[int(np.argmin(((xs - lon) * kx) ** 2 + (ys - lat) ** 2))] for _, lat, lon in WAYPOINTS]
    coords, reached = [], [0]
    for a, b in zip(nodes, nodes[1:]):
        path = nx.bidirectional_dijkstra(G, a, b, weight="w")[1]
        for u, v in zip(path, path[1:]):
            d = min(G[u][v].values(), key=lambda e: e["w"])
            cs = list(d["geometry"].coords) if "geometry" in d else [(G.nodes[u]["x"], G.nodes[u]["y"]),
                                                                       (G.nodes[v]["x"], G.nodes[v]["y"])]
            if math.dist(cs[0], (G.nodes[u]["x"], G.nodes[u]["y"])) > math.dist(cs[-1], (G.nodes[u]["x"],
                                                                                           G.nodes[u]["y"])):
                cs = cs[::-1]
            coords.extend(cs if not coords else cs[1:])
        reached.append(len(coords) - 1)
    return coords, reached


def dem_sampler(lonlat):
    """A function (lon, lat) -> metres from the Copernicus DEM at full resolution, over the box of the points."""
    root = os.path.join(area.cache_root(), "dem")
    lon, lat = np.asarray(lonlat).T
    box = (lat.min() - 0.01, lat.max() + 0.01, lon.min() - 0.01, lon.max() + 0.01)
    cells = relief._cells(box)
    dataset, headers = relief._dataset(cells, root)
    blocks = [(cell, 0, br, bc) for cell, br, bc in relief._blocks(dataset, headers, cells, 0, box)]
    relief._download(dataset, headers, blocks, root)
    mos = relief._mosaic(dataset, headers, cells, 0, box, root)
    return lambda lo, la: relief._sample(mos, lo, la)


def walk(coords, reached, start, seed):
    """[(lat, lon, ele, time)] every STEP_S seconds along the route."""
    rng = random.Random(seed)
    to_xy = Transformer.from_crs(4326, CRS, always_xy=True)
    to_ll = Transformer.from_crs(CRS, 4326, always_xy=True)
    height = dem_sampler(coords)
    xy = np.column_stack(to_xy.transform(*np.asarray(coords).T))
    line = LineString(xy)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    at_vertex = np.concatenate([[0.0], np.cumsum(seg)])
    stops = {float(at_vertex[reached[i]]): STOPS[name] for i, (name, *_) in enumerate(WAYPOINTS) if name in STOPS}
    s_grid = np.arange(0.0, line.length, 5.0)
    pts = np.array([line.interpolate(s).coords[0] for s in s_grid])
    lon, lat = to_ll.transform(pts[:, 0], pts[:, 1])
    z = height(lon, lat)
    slope = np.clip(np.gradient(np.convolve(np.pad(z, 10, mode="edge"), np.ones(21) / 21, mode="valid"), 5.0),
                    -0.3, 0.3)  # over 100 m, and no steeper than the zigzags of a path the map draws straight
    kmh = 0.85 * 6 * np.exp(-3.5 * np.abs(slope + 0.05))  # Tobler's hiking function
    t_grid = np.concatenate([[0.0], np.cumsum(5.0 / (kmh[:-1] / 3.6))])
    for s, pause in sorted(stops.items()):  # standing: the clock runs on, the hiker does not
        t_grid[s_grid > s] += pause
    times = np.arange(0.0, t_grid[-1], STEP_S)
    where = np.interp(times, t_grid, s_grid)  # flat across a stop: the same place for its whole length
    true = np.array([line.interpolate(s).coords[0] for s in where])
    lon, lat = to_ll.transform(true[:, 0], true[:, 1])
    ground = height(lon, lat)
    out, ex, ey, baro = [], 0.0, 0.0, 0.0
    phi, sigma = 0.9, 3.0
    for (x, y), h, t in zip(true, ground, times):
        ex = phi * ex + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        ey = phi * ey + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        baro = 0.95 * baro + rng.gauss(0, 0.3)
        lo, la = to_ll.transform(x + ex, y + ey)
        out.append((la, lo, round(float(h) + baro, 1), start + dt.timedelta(seconds=float(t))))
    return out


def gpx_text(points, name):
    rows = [f'<trkpt lat="{la:.6f}" lon="{lo:.6f}"><ele>{e}</ele><time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</time>'
            f'</trkpt>' for la, lo, e, t in points]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx creator="pinless simulate_hike (synthetic)" version="1.1" '
            'xmlns="http://www.topografix.com/GPX/1/1">\n<metadata><desc>Synthetic hike for the Pinless demo: '
            'nobody\'s real data.</desc></metadata>\n'
            f'<trk><name>{name}</name><type>hiking</type><trkseg>\n' + "\n".join(rows) + "\n</trkseg></trk></gpx>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "hike_demo.gpx"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--start", default="2025-07-12T07:40:00", help="UTC")
    a = ap.parse_args()
    folder = area.fetch_basemap(BOX, log=print)
    coords, reached = route_on_paths(folder)
    start = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
    points = walk(coords, reached, start, a.seed)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(gpx_text(points, "Penalara desde Cotos"))
    secs = (points[-1][3] - points[0][3]).total_seconds()
    print(f"{a.out}: {len(points)} points, {secs / 3600:.1f} h, {min(p[2] for p in points):.0f} to "
          f"{max(p[2] for p in points):.0f} m")


if __name__ == "__main__":
    main()
