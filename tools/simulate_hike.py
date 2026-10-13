"""Generate a SYNTHETIC hike on real OpenStreetMap paths and the Copernicus elevation model, written as one GPX file.

Nobody's real data. --route chooses the walk (public landmarks; the route snaps each to the nearest node of the OSM
walk network, paths and tracks preferred over roads):
- penalara (the trail demo, demo/hike_demo.gpx): a loop from the Puerto de Cotos car park in the Sierra de
  Guadarrama, up the ridge over Dos Hermanas to Penalara and the Risco de los Claveles, down to the Laguna de los
  Pajaros and back along the valley past the Laguna Grande; a 20 minute stop at the summit and 15 by the lake.
- tabernas (the desert demo, demo/hike_desert_demo.gpx): in the Desierto de Tabernas (Almeria), from the Camino de
  la Rellana up the tracks of the badlands to El Puntal and back the same way; a 20 minute stop at the top.
- trecime (the Dolomites demo, demo/hike_dolomites_demo.gpx): the classic loop round the Tre Cime di Lavaredo,
  anticlockwise from the Rifugio Auronzo: east under the south faces to the Rifugio Lavaredo, up to the Forcella
  Lavaredo, north to the Rifugio Locatelli (Dreizinnenhuette), down to the Langalm and back over the Forcella Col di
  Mezzo, on the CAI trails 101 and 105; 10 minutes at the Forcella Lavaredo and 30 at the Locatelli.
- lacblanc (the Alps demo, demo/hike_alps_demo.gpx): in Chamonix (France), from the top of the La Flegere cable car
  past the Lac de la Flegere and up to the Refuge du Lac Blanc and the shore of the lake, facing the Mont Blanc
  massif, then east past the Lacs des Cheserys and down to the Col des Montets; a 30 minute stop by the lake.
- saolourenco (the coast demo, demo/hike_coast_demo.gpx): Madeira's PR 8, the Vereda da Ponta de Sao Lourenco,
  from the car park above the Baia d'Abra out along the dry peninsula between the two seas, over the Pedra Furada
  viewpoint, to the Casa do Sardinha and back the same way (the climb to the Furado beyond it is closed: access=no
  on OSM); a 25 minute stop at the Casa do Sardinha.
- zermatt and ebro: short loops to look at snow and ice, and at farmland and rice fields, before a change to the
  colours of the ground (not demos: write them under out/).

- Paths: a via ferrata costs 50 times its length, so a route keeps off them where any walk goes round (OSM's walk
  network includes them, and some belong to hiking relations). With marked=True a route keeps to the marked
  hiking trails (OSM route=hiking relations, fetched once from Overpass and cached next to the walk network as
  hiking_ways.json): an unmarked way costs UNMARKED times more.
- Elevation: sampled from the Copernicus DEM (GLO-30) through pinless.relief's reader, at the true positions, with
  a little barometer noise. A development tool may read the DEM; the poster code never samples it. On a route by
  the sea (sea=True) only the land cells count, and where the DEM's sea mask covers a neck of land narrower than
  its cells the heights of the path on either side are joined along it.
- Clock: Tobler's hiking function on the slope over 100 m, at most 30 % (a path zigzags where the map draws it
  straight), for a relaxed hiker at 85 % of it, standing still at the stops with the fix wandering a few metres; a
  point every 10 s.
- Positions: correlated GPS noise of about 3 m.
Usage: python tools/simulate_hike.py [--route penalara] (writes the route's demo file, or --out)
Then: pinless demo/hike_demo.gpx --layout trail --title "Peñalara" --name "Lucía"
"""
import argparse
import datetime as dt
import json
import math
import os
import random
import sys

import networkx as nx
import numpy as np
import osmnx as ox
import requests
from pyproj import Transformer
from shapely.geometry import LineString

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pinless import area, relief  # noqa: E402

# Per route: (south, west, north, east) of the hills around it; its public landmarks (approximate) in walking order;
# the stops (seconds) at some of them; the GPX track name; the metric CRS; the UTC start; the default output file
ROUTES = {
    "penalara": dict(box=(40.790, -4.020, 40.880, -3.900),
                     waypoints=[("Puerto de Cotos", 40.8222, -3.9630), ("Hermana Mayor", 40.8399, -3.9629),
                                ("Penalara", 40.8500, -3.9561), ("Risco de los Claveles", 40.8551, -3.9523),
                                ("Laguna de los Pajaros", 40.8604, -3.9476), ("Laguna Grande", 40.8399, -3.9574),
                                ("Puerto de Cotos", 40.8222, -3.9630)],
                     stops={"Penalara": 20 * 60, "Laguna de los Pajaros": 15 * 60}, name="Penalara desde Cotos",
                     crs=25830, start="2025-07-12T07:40:00", out="demo/hike_demo.gpx"),  # ETRS89 / UTM 30N
    "tabernas": dict(box=(36.980, -2.560, 37.100, -2.360),
                     waypoints=[("Camino de la Rellana", 37.0298, -2.4131), ("El Puntal", 37.0008, -2.4128),
                                ("Camino de la Rellana", 37.0298, -2.4131)],
                     stops={"El Puntal": 20 * 60}, name="Desierto de Tabernas", crs=25830,
                     start="2025-03-15T08:10:00", out="demo/hike_desert_demo.gpx"),
    # the huts and saddles where OSM places them (the Langalm where the trail passes it); marked: on the CAI trails
    # 101 and 105, not the unmarked paths that run beside them
    "trecime": dict(box=(46.595, 12.260, 46.650, 12.335),
                    waypoints=[("Rifugio Auronzo", 46.6122, 12.2964), ("Rifugio Lavaredo", 46.6186, 12.3120),
                               ("Forcella Lavaredo", 46.6225, 12.3115), ("Rifugio Locatelli", 46.6369, 12.3108),
                               ("Langalm", 46.6256, 12.2906), ("Forcella Col di Mezzo", 46.6175, 12.2843),
                               ("Rifugio Auronzo", 46.6122, 12.2964)],
                    stops={"Forcella Lavaredo": 10 * 60, "Rifugio Locatelli": 30 * 60}, name="Tre Cime di Lavaredo",
                    crs=32633, start="2025-07-20T06:40:00", out="demo/hike_dolomites_demo.gpx",  # WGS 84 / UTM 33N
                    marked=True),
    # from the top of the La Flegere cable car up the path past the Lac de la Flegere to the Refuge du Lac Blanc and
    # the dam between its two lakes, then east past the Lacs des Cheserys and down through the Aiguilles Rouges
    # nature reserve to the Col des Montets
    "lacblanc": dict(box=(45.950, 6.865, 46.005, 6.935),
                     waypoints=[("La Flegere", 45.9607, 6.8870), ("Lac de la Flegere", 45.9697, 6.8879),
                                ("Refuge du Lac Blanc", 45.9812, 6.8916), ("Lac Blanc", 45.9822, 6.8902),
                                ("Lacs des Cheserys", 45.9851, 6.9020), ("Col des Montets", 45.9990, 6.9230)],
                     stops={"Lac Blanc": 30 * 60}, name="Lac Blanc et Cheserys", crs=32632,
                     start="2025-08-31T06:50:00", out="demo/hike_alps_demo.gpx"),  # WGS 84 / UTM 32N
    # PR 8 on OSM: the ways tagged fee=yes with the IFCN ticket link, from the guidepost by the roundabout at the end
    # of the ER214 to the Casa do Sardinha (the building, way 169877344); the Pedra Furada viewpoint keeps the walk
    # on the trail over the north cliffs, which is the PR 8, rather than any shortcut the network may offer
    "saolourenco": dict(box=(32.725, -16.725, 32.765, -16.645),
                        waypoints=[("Baia d'Abra", 32.7434, -16.7009), ("Pedra Furada", 32.7504, -16.6940),
                                   ("Casa do Sardinha", 32.7423, -16.6834), ("Pedra Furada", 32.7504, -16.6940),
                                   ("Baia d'Abra", 32.7434, -16.7009)],
                        stops={"Casa do Sardinha": 25 * 60}, name="Ponta de Sao Lourenco", crs=32628,
                        start="2025-04-27T08:20:00", out="demo/hike_coast_demo.gpx",  # WGS 84 / UTM 28N
                        sea=True),
    # checks of the ground's colours, not demos
    "ebro": dict(box=(40.600, 0.620, 40.700, 0.760),
                 waypoints=[("Carretera del Poblenou", 40.6550, 0.7060), ("Cami del Traves", 40.6500, 0.6650),
                            ("Carretera del Mig", 40.6370, 0.6850), ("Carretera del Poblenou", 40.6550, 0.7060)],
                 stops={}, name="Delta de l'Ebre", crs=25831, start="2025-06-08T07:30:00", out="out/ebro.gpx"),
    "zermatt": dict(box=(45.950, 7.700, 46.030, 7.820),
                    waypoints=[("Gornergrat", 45.9833, 7.7850), ("Riffelsee", 45.9870, 7.7595),
                               ("Riffelberg", 45.9915, 7.7510), ("Gugle", 45.9980, 7.7680),
                               ("Gornergrat", 45.9833, 7.7850)],
                    stops={"Riffelsee": 15 * 60}, name="Gornergrat", crs=32632, start="2025-08-02T07:00:00",
                    out="out/zermatt.gpx"),
}
FACTOR = {"path": 0.7, "track": 0.75, "footway": 0.8, "bridleway": 0.8, "steps": 1.5, "unclassified": 1.2,
          "service": 1.2, "residential": 1.3, "tertiary": 1.6, "secondary": 2.5, "primary": 3.0,
          "via_ferrata": 50.0}  # a demo hike keeps off the climbing routes, which OSM's walk network includes
STEP_S = 10.0
UNMARKED = 2.0  # with marked=True, a way on no marked hiking trail costs this much more


def marked_ways(folder, box, log=print):
    """The ids of the OSM ways on marked hiking trails (route=hiking relations) in the box, fetched once from
    Overpass and cached next to the walk network."""
    path =os.path.join(folder, "hiking_ways.json")
    if not os.path.exists(path):
        s, w, n, e = box
        query = f"[out:json][timeout:120];rel[route=hiking]({s},{w},{n},{e});out body;"
        urls = [os.environ["PINLESS_OVERPASS"]] if os.environ.get("PINLESS_OVERPASS") else list(area.OVERPASS)
        for url in urls:
            try:
                r = requests.post(url.rstrip("/") + "/interpreter", data={"data": query}, timeout=180)
                data = r.json() if r.ok else {}
            except (requests.RequestException, ValueError) as ex:
                log(f"[osm] {url}: {type(ex).__name__}")
                continue
            if "elements" in data and "remark" not in data:  # a remark is a timeout or an error: not the answer
                break
            log(f"[osm] {url}: HTTP {r.status_code}{' ' + data['remark'] if data.get('remark') else ''}")
        else:
            raise RuntimeError("no Overpass server answered for the marked trails; try again later")
        trails = {str(el["id"]): {"ref": el.get("tags", {}).get("ref"), "name": el.get("tags", {}).get("name"),
                                  "ways": [m["ref"] for m in el.get("members", []) if m["type"] == "way"]}
                  for el in data["elements"] if el["type"] == "relation"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trails, f, ensure_ascii=False)
    with open(path, encoding="utf-8") as f:
        return {w for t in json.load(f).values() for w in t["ways"]}


def _highway(d):
    h = d.get("highway", "path")
    if isinstance(h, list):  # an edge merged from several ways: a stretch of via ferrata makes it one
        return "via_ferrata" if "via_ferrata" in h else h[0]
    return h


def route_on_paths(folder, route):
    """The loop through the route's waypoints on the walk network, as (lon, lat) points, and the index where each
    waypoint is reached."""
    G = ox.load_graphml(os.path.join(folder, "walk.graphml"))
    marked = marked_ways(folder, route["box"]) if route.get("marked") else None  # osmid: one id, or a list
    for _, _, d in G.edges(data=True):
        d["w"] = d["length"] * FACTOR.get(_highway(d), 1.0)
        if marked is not None and not {int(w) for w in np.atleast_1d(d["osmid"])} & marked:
            d["w"] *= UNMARKED
    ids = np.array(list(G.nodes))
    xs = np.array([G.nodes[n]["x"] for n in ids])
    ys = np.array([G.nodes[n]["y"] for n in ids])
    kx = math.cos(math.radians((route["box"][0] + route["box"][2]) / 2))
    nodes = [ids[int(np.argmin(((xs - lon) * kx) ** 2 + (ys - lat) ** 2))] for _, lat, lon in route["waypoints"]]
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


def dem_sampler(lonlat, sea=False):
    """A function (lon, lat) -> metres from the Copernicus DEM at full resolution, over the box of the points.

    sea=True for a path along the sea: the DEM's sea is 0 m, and its mask also covers necks of land narrower than its
    30 m cells, at or a few metres below 0 m (the neck halfway along the Ponta de Sao Lourenco, 75 to 120 m wide on
    the OSM coastline, is a gap of sea 130 m long in the DEM), which would drop the path to the sea and back. Then
    each height is the bilinear mean of the LAND cells around the point only, and NaN where they weigh less than half
    (the caller joins the heights on either side along the path)."""
    root = os.path.join(area.cache_root(), "dem")
    lon, lat = np.asarray(lonlat).T
    box = (lat.min() - 0.01, lat.max() + 0.01, lon.min() - 0.01, lon.max() + 0.01)
    cells = relief._cells(box)
    dataset, headers = relief._dataset(cells, root)
    blocks = [(cell, 0, br, bc) for cell, br, bc in relief._blocks(dataset, headers, cells, 0, box)]
    relief._download(dataset, headers, blocks, root)
    mos = relief._mosaic(dataset, headers, cells, 0, box, root)
    if not sea:
        return lambda lo, la: relief._sample(mos, lo, la)
    land = (mos.z > relief.LAND_EPS_M).astype(np.float64)  # the sea is 0 m, and a masked neck at or below it
    weight = relief.Mosaic(land, mos.lon0, mos.dlon, mos.lat0, mos.dlat)
    masked = relief.Mosaic(mos.z * land, mos.lon0, mos.dlon, mos.lat0, mos.dlat)

    def height(lo, la):
        w = relief._sample(weight, lo, la)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w >= 0.5, relief._sample(masked, lo, la) / w, np.nan)
    return height


def _fill(at, z):
    """z with its NaN filled by linear interpolation over at (the distance along the path)."""
    bad = np.isnan(z)
    if bad.all():
        raise SystemExit("the DEM has no land under this path")
    return np.where(bad, np.interp(at, at[~bad], z[~bad]), z)


def walk(coords, reached, start, seed, route):
    """[(lat, lon, ele, time)] every STEP_S seconds along the route."""
    rng = random.Random(seed)
    to_xy = Transformer.from_crs(4326, route["crs"], always_xy=True)
    to_ll = Transformer.from_crs(route["crs"], 4326, always_xy=True)
    sea = route.get("sea", False)
    height = dem_sampler(coords, sea=sea)
    xy = np.column_stack(to_xy.transform(*np.asarray(coords).T))
    line = LineString(xy)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    at_vertex = np.concatenate([[0.0], np.cumsum(seg)])
    stops = {float(at_vertex[reached[i]]): route["stops"][name] for i, (name, *_) in enumerate(route["waypoints"])
             if name in route["stops"]}
    s_grid = np.arange(0.0, line.length, 5.0)
    pts = np.array([line.interpolate(s).coords[0] for s in s_grid])
    lon, lat = to_ll.transform(pts[:, 0], pts[:, 1])
    z = height(lon, lat)
    if sea:
        z = _fill(s_grid, z)
    slope =np.clip(np.gradient(np.convolve(np.pad(z, 10, mode="edge"), np.ones(21) / 21, mode="valid"), 5.0),
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
    if sea:  # across a neck the DEM masks as sea: the heights of the path on either side, joined
        ground = np.where(np.isnan(ground), np.interp(where, s_grid, z), ground)
    out, ex, ey, baro = [], 0.0, 0.0, 0.0
    phi, sigma = 0.9, 3.0
    for (x, y), h, t in zip(true, ground, times):
        ex = phi * ex + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        ey = phi * ey + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        baro = 0.95 * baro + rng.gauss(0, 0.3)
        lo, la = to_ll.transform(x + ex, y + ey)
        out.append((la, lo, round(float(h) + baro, 1), start + dt.timedelta(seconds=float(t))))
    return out


# What the file is and where its line and heights come from, with the notices their licences ask for (ASCII only)
METADATA = ("<metadata><desc>Synthetic hike for the Pinless demo: nobody's real data. The line follows OpenStreetMap "
            "paths with simulated GPS noise, and the clock and the stops are simulated (Tobler's hiking function). "
            "Paths: &#169; OpenStreetMap contributors, ODbL. Heights: sampled from the Copernicus DEM GLO-30, "
            "produced using Copernicus WorldDEM-30 &#169; DLR e.V. 2010-2014 and &#169; Airbus Defence and Space "
            "GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.</desc>"
            '<copyright author="OpenStreetMap contributors"><license>https://opendatacommons.org/licenses/odbl/1-0/'
            "</license></copyright></metadata>")


def gpx_text(points, name):
    rows = [f'<trkpt lat="{la:.6f}" lon="{lo:.6f}"><ele>{e}</ele><time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</time>'
            f'</trkpt>' for la, lo, e, t in points]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx creator="pinless simulate_hike (synthetic)" version="1.1" '
            f'xmlns="http://www.topografix.com/GPX/1/1">\n{METADATA}\n'
            f'<trk><name>{name}</name><type>hiking</type><trkseg>\n' + "\n".join(rows) + "\n</trkseg></trk></gpx>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="penalara", choices=sorted(ROUTES))
    ap.add_argument("--out", default=None, help="default: the route's demo file")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--start", default=None, help="UTC; default: the route's")
    a = ap.parse_args()
    route = ROUTES[a.route]
    out = a.out or os.path.join(ROOT, *route["out"].split("/"))
    folder = area.fetch_basemap(route["box"], log=print)
    coords, reached = route_on_paths(folder, route)
    start = dt.datetime.fromisoformat(a.start or route["start"]).replace(tzinfo=dt.timezone.utc)
    points = walk(coords, reached, start, a.seed, route)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(gpx_text(points, route["name"]))
    secs = (points[-1][3] - points[0][3]).total_seconds()
    print(f"{out}: {len(points)} points, {secs / 3600:.1f} h, {min(p[2] for p in points):.0f} to "
          f"{max(p[2] for p in points):.0f} m")


if __name__ == "__main__":
    main()
