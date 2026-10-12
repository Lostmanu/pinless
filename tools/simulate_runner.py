"""Generate a realistic SYNTHETIC runner and write it as a Strava-like bulk export (zip).

Nobody's real data: a fictional runner with a home and an office in Madrid runs one year on real OSM streets.
Usage: python tools/simulate_runner.py --city madrid --year 2025 --seed 7 --out demo/strava_export_madrid_demo.zip

A partner for the poster of two (pinless --with), who lives and works elsewhere, runs on fewer days, a little slower,
and runs some weekend runs side by side with the first runner (copied from that export, a few metres apart, the same
clock within 5 s), on one command line:
  python tools/simulate_runner.py --seed 11 --home 40.4310,-3.7165 --work 40.4275,-3.6790 --rate 0.6 --pace 1.06
    --together-from demo/strava_export_madrid_demo.zip --together 24
    --out demo/strava_export_madrid_partner_demo.zip --truth demo/truth_partner.json

Realism, by design:
- favourite routes repeated with variations (Madrid Rio, Retiro, Casa de Campo long runs) so repeated streets glow;
- routing that prefers parks, footways and cycleways and avoids big avenues;
- weekday 6-12 km, weekend long runs 15-24 km, realistic paces and start times, rest weeks;
- correlated GPS noise (worse under trees), rare spikes, standing still before the start;
- a second start cluster (lunch runs from the office) to exercise multi-zone privacy;
- races, a holiday abroad (outside the map area), walks, rides and treadmill runs without GPS.
The true home/work points are written to a SEPARATE file (truth.json) only for privacy tests; never into the export.
"""
import argparse
import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import random
import sys
import zipfile
from zoneinfo import ZoneInfo

import networkx as nx
import numpy as np
import osmnx as ox
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fictional places (approximate public landmarks; HOME and WORK are invented points on residential/office streets)
PLACES = {
    "home": (40.4005, -3.6955),            # Arganzuela (fictional doorstep)
    "work": (40.4452, -3.6935),            # Nuevos Ministerios office area (fictional)
    "toledo": (40.4024, -3.7173), "segovia": (40.4136, -3.7228), "rey": (40.4194, -3.7236),
    "matadero": (40.3920, -3.6975), "princesa": (40.3870, -3.6930), "caja_magica": (40.3700, -3.6870),
    "atocha_retiro": (40.4093, -3.6880), "estanque": (40.4180, -3.6835), "coches_n": (40.4205, -3.6880),
    "retiro_e": (40.4170, -3.6755), "rosaleda": (40.4105, -3.6800),
    "cc_lago": (40.4200, -3.7350), "cc_batan": (40.4050, -3.7590), "cc_garabitas": (40.4220, -3.7570),
    "cc_norte": (40.4370, -3.7650), "oeste": (40.4280, -3.7200), "debod": (40.4240, -3.7178),
    "colon": (40.4250, -3.6905), "cibeles": (40.4193, -3.6931), "canal": (40.4440, -3.7070),
    "castellana_n": (40.4590, -3.6900), "plaza_espana": (40.4233, -3.7123), "gran_via": (40.4200, -3.7020),
    "vallecas": (40.3920, -3.6585), "bernabeu": (40.4530, -3.6883),
}

# The races as run on the day: their courses measure 21.1 and 10.0 km along the recorded points (3 s sampling)
RACE_COURSES = {
    "Spring Half Marathon": ["castellana_n", "colon", "cibeles", "gran_via", "plaza_espana", "debod", "oeste", "rey",
                             "segovia", "toledo", "matadero", "atocha_retiro", "retiro_e", "estanque"],
    "New Year's Eve 10K": ["castellana_n", "colon", "cibeles", "atocha_retiro", "vallecas"],
}

# Route templates: sequence of places (start/end added automatically), weights, and kind
TEMPLATES = [
    {"name": "Madrid Rio corto", "via": ["toledo", "segovia", "toledo"], "w": 26, "kind": "easy"},
    {"name": "Madrid Rio sur", "via": ["matadero", "princesa", "matadero"], "w": 16, "kind": "easy"},
    {"name": "Retiro", "via": ["atocha_retiro", "rosaleda", "estanque", "atocha_retiro"], "w": 22, "kind": "easy"},
    {"name": "Barrio", "via": [], "w": 12, "kind": "short"},
    {"name": "Rio completo", "via": ["toledo", "segovia", "rey", "segovia", "toledo"], "w": 10, "kind": "mid"},
    {"name": "Retiro + Rio", "via": ["atocha_retiro", "estanque", "coches_n", "cibeles", "gran_via", "plaza_espana", "rey", "segovia", "toledo"], "w": 6, "kind": "mid"},
    {"name": "Rio + Oeste", "via": ["toledo", "segovia", "rey", "debod", "oeste", "rey", "segovia", "toledo"], "w": 5, "kind": "long"},
    {"name": "Casa de Campo", "via": ["toledo", "rey", "cc_lago", "cc_garabitas", "cc_batan", "cc_lago", "rey", "toledo"], "w": 9, "kind": "long"},
    {"name": "Casa de Campo norte", "via": ["toledo", "rey", "cc_lago", "cc_norte", "cc_garabitas", "cc_batan", "toledo"], "w": 4, "kind": "long"},
    {"name": "Largo rio completo", "via": ["caja_magica", "princesa", "matadero", "toledo", "segovia", "rey", "oeste", "rey", "toledo"], "w": 5, "kind": "long"},
]
WORK_TEMPLATES = [
    {"name": "Lunch run", "via": ["canal", "castellana_n", "work"], "w": 1, "kind": "work"},
    {"name": "Lunch run Castellana", "via": ["colon", "castellana_n"], "w": 1, "kind": "work"},
]
HIGHWAY_FACTOR = {"footway": 0.75, "path": 0.7, "pedestrian": 0.8, "track": 0.75, "cycleway": 0.7, "steps": 3.0,
                  "living_street": 0.9, "residential": 1.0, "service": 1.1, "unclassified": 1.0, "tertiary": 1.15,
                  "tertiary_link": 1.15, "secondary": 1.35, "secondary_link": 1.35, "primary": 1.6, "primary_link": 1.6,
                  "trunk": 2.5, "trunk_link": 2.5}


def highway_of(d):
    h = d.get("highway", "residential")
    return h[0] if isinstance(h, list) else h


def build_graph(city, rng, base=None, crs=25830):
    base = base or os.path.join(ROOT, "cache", city)
    G = ox.load_graphml(os.path.join(base, "walk.graphml"))
    green = gpd.read_file(os.path.join(base, "green.gpkg"))
    parks = green[(green.geometry.type.isin(["Polygon", "MultiPolygon"]))]
    parks = parks.to_crs(crs)
    parks = parks[parks.geometry.area > 30000]  # >= 3 ha
    tree = STRtree(list(parks.geometry.values))
    to_utm = Transformer.from_crs(4326, crs, always_xy=True)
    xs = np.array([d["x"] for _, d in G.nodes(data=True)])
    ys = np.array([d["y"] for _, d in G.nodes(data=True)])
    ux, uy = to_utm.transform(xs, ys)
    node_ids = np.array(list(G.nodes))
    utm = {n: (ux[i], uy[i]) for i, n in enumerate(node_ids)}
    for u, v, k, d in G.edges(keys=True, data=True):
        f = HIGHWAY_FACTOR.get(highway_of(d), 1.0)
        mx, my = (utm[u][0] + utm[v][0]) / 2, (utm[u][1] + utm[v][1]) / 2
        if len(tree.query(Point(mx, my), predicate="within")):
            f *= 0.65
        d["w"] = d["length"] * f
    return G, node_ids, ux, uy, to_utm


def nearest_node(node_ids, ux, uy, x, y):
    i = int(np.argmin((ux - x) ** 2 + (uy - y) ** 2))
    return node_ids[i]


def edge_coords(G, u, v):
    d = min(G[u][v].values(), key=lambda e: e["w"])
    if "geometry" in d:
        cs = list(d["geometry"].coords)
        su, sv = (G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])
        if math.dist(cs[0], su) > math.dist(cs[-1], su):
            cs = cs[::-1]
        return cs
    return [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]


def route_line(G, nodes_seq):
    coords = []
    for a, b in zip(nodes_seq, nodes_seq[1:]):
        if a == b:
            continue
        path = nx.bidirectional_dijkstra(G, a, b, weight="w")[1]
        for u, v in zip(path, path[1:]):
            cs = edge_coords(G, u, v)
            coords.extend(cs if not coords else cs[1:])
    return coords


def madrid_terrain(lat, lon):
    """Synthetic but plausible Madrid ground (m): rising to the north, the Manzanares valley in the west, gentle hills."""
    if not (40.30 < lat < 40.56 and -3.86 < lon < -3.54):  # away from Madrid (the beach holiday): near sea level
        return 8.0 + 2.0 * math.sin(lon * 300)
    river = -3.718 - 0.25 * (lat - 40.40)
    return (640 + 900 * (lat - 40.40) - 55 * math.exp(-((lon - river) / 0.008) ** 2)
            + 12 * math.sin(lon * 140) * math.cos(lat * 95))


def sample_track(coords_ll, to_utm, rng, pace_s_per_km, dt_s, start_time, trees_factor, crs=25830, ele0=650.0,
                 terrain=madrid_terrain):
    """Resample a route at constant speed, add AR(1) GPS noise and rare spikes; return list of (lat, lon, ele, t)."""
    xs, ys = to_utm.transform([c[0] for c in coords_ll], [c[1] for c in coords_ll])
    line = LineString(list(zip(xs, ys)))
    L = line.length
    speed = 1000.0 / pace_s_per_km
    pts, t, s = [], 0.0, 0.0
    ex = ey = 0.0
    sigma = 3.2 * trees_factor
    phi = 0.92
    # standing still before pressing start: a few points around the start
    p0 = line.interpolate(0)
    for _ in range(rng.randint(0, 8)):
        ex = phi * ex + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        ey = phi * ey + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        pts.append((p0.x + ex, p0.y + ey, t))
        t += dt_s
    while s <= L:
        p = line.interpolate(s)
        ex = phi * ex + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        ey = phi * ey + rng.gauss(0, sigma * math.sqrt(1 - phi ** 2))
        spike = (rng.gauss(0, 18), rng.gauss(0, 18)) if rng.random() < 0.0008 else (0.0, 0.0)
        pts.append((p.x + ex + spike[0], p.y + ey + spike[1], t))
        v = speed * rng.uniform(0.9, 1.1)
        s += v * dt_s
        t += dt_s
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = to_ll.transform([q[0] for q in pts], [q[1] for q in pts])
    out = []
    ele = ele0
    for i, q in enumerate(pts):
        ele += rng.gauss(0, 0.4)  # one draw per point, as before: the routes and times stay the same
        h = terrain(lat[i], lon[i]) + (ele - ele0) * 0.5 if terrain else ele  # barometer noise on the ground
        out.append((lat[i], lon[i], round(h, 1), start_time + dt.timedelta(seconds=q[2])))
    return out, L


def gpx_text(points, name, kind="running"):
    b = io.StringIO()
    b.write('<?xml version="1.0" encoding="UTF-8"?>\n<gpx creator="StravaGPX" version="1.1" xmlns="http://www.topografix.com/GPX/1/1" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
    b.write(f' <metadata>\n  <time>{points[0][3].strftime("%Y-%m-%dT%H:%M:%SZ")}</time>\n </metadata>\n')
    b.write(f' <trk>\n  <name>{name}</name>\n  <type>{kind}</type>\n  <trkseg>\n')
    for lat, lon, ele, t in points:
        b.write(f'   <trkpt lat="{lat:.7f}" lon="{lon:.7f}">\n    <ele>{ele}</ele>\n    <time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</time>\n   </trkpt>\n')
    b.write("  </trkseg>\n </trk>\n</gpx>\n")
    return b.getvalue()


def tcx_text(points, sport="Running"):
    b = io.StringIO()
    b.write('<?xml version="1.0" encoding="UTF-8"?>\n<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">\n')
    b.write(f' <Activities>\n  <Activity Sport="{sport}">\n   <Id>{points[0][3].strftime("%Y-%m-%dT%H:%M:%SZ")}</Id>\n   <Lap StartTime="{points[0][3].strftime("%Y-%m-%dT%H:%M:%SZ")}">\n    <Track>\n')
    for lat, lon, ele, t in points:
        b.write(f'     <Trackpoint><Time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</Time><Position><LatitudeDegrees>{lat:.7f}</LatitudeDegrees>'
                f'<LongitudeDegrees>{lon:.7f}</LongitudeDegrees></Position><AltitudeMeters>{ele}</AltitudeMeters></Trackpoint>\n')
    b.write("    </Track>\n   </Lap>\n  </Activity>\n </Activities>\n</TrainingCenterDatabase>\n")
    return b.getvalue()


def jitter(rng, latlon, r_m):
    ang = rng.uniform(0, 2 * math.pi)
    d = abs(rng.gauss(0, r_m))
    return (latlon[0] + d * math.sin(ang) / 111_320, latlon[1] + d * math.cos(ang) / (111_320 * math.cos(math.radians(latlon[0]))))


def ring_point(rng, latlon, rmin, rmax):
    ang = rng.uniform(0, 2 * math.pi)
    d = rng.uniform(rmin, rmax)
    return (latlon[0] + d * math.sin(ang) / 111_320, latlon[1] + d * math.cos(ang) / (111_320 * math.cos(math.radians(latlon[0]))))


def straight_route(rng, a, b, n=200):
    """Beach promenade style route for runs outside the cached city (holiday)."""
    return [(a[1] + (b[1] - a[1]) * i / n, a[0] + (b[0] - a[0]) * i / n) for i in range(n + 1)]


def to_utc(acts, zone):
    """The runner lives in local time; files store UTC, as Strava writes them. Shift each activity by its offset."""
    tz = ZoneInfo(zone)
    for x in acts:
        off = x["date"].replace(tzinfo=tz).utcoffset()
        x["date"] = x["date"] - off
        if x["pts"]:
            x["pts"] = [(la, lo, el, t - off) for la, lo, el, t in x["pts"]]


def together_runs(export, n, rng, first_id):
    """n weekend runs of another (synthetic) runner's export in Madrid, run side by side: the same streets a few metres
    apart (AR(1) GPS noise of about 3 m) and the same clock within 5 s. In UTC, as the export stores them."""
    sys.path.insert(0, ROOT)
    from pinless import ingest
    from pinless.localtime import local_start

    acts, _ = ingest.read_export(export)
    pool = [x for x in acts if x.type == "Run" and x.name == "Morning Run" and x.times and x.ele
            and (t := local_start(x)) is not None and t.weekday() >= 5
            and all(40.34 < la < 40.48 and -3.80 < lo < -3.63 for lo, la in x.lonlat)]
    if len(pool) < n:
        raise SystemExit(f"--together {n}: that export has only {len(pool)} weekend runs in Madrid")
    out = []
    for k, i in enumerate(sorted(rng.sample(range(len(pool)), n))):
        x = pool[i]
        shift = rng.uniform(-5.0, 5.0)  # the other watch's clock, within 5 s
        start = (x.start if x.start.tzinfo else x.start.replace(tzinfo=dt.timezone.utc)).replace(tzinfo=None)
        ex = ey = 0.0
        pts = []
        for (lon, lat), sec, ele in zip(x.lonlat, x.times, x.ele):
            ex = 0.92 * ex + rng.gauss(0, 3.0 * math.sqrt(1 - 0.92 ** 2))
            ey = 0.92 * ey + rng.gauss(0, 3.0 * math.sqrt(1 - 0.92 ** 2))
            pts.append((lat + ey / 110_540, lon + ex / (111_320 * math.cos(math.radians(lat))),
                        round(ele + rng.gauss(0, 0.4), 1), start + dt.timedelta(seconds=sec + shift)))
        km = sum(ingest._step_m(*q) for q in zip(x.lonlat, x.lonlat[1:])) / 1000
        out.append({"id": first_id + 1000 * (k + 1), "date": pts[0][3], "name": "Morning Run", "type": "Run",
                    "dist_km": km, "elapsed": int(x.times[-1] - x.times[0]), "fmt": "gpx", "pts": pts,
                    "kind": "together"})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="madrid")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dt", type=float, default=3.0, help="seconds between GPS points")
    p.add_argument("--out", default=os.path.join(ROOT, "demo", "strava_export_madrid_demo.zip"))
    p.add_argument("--home", default=None, help="LAT,LON of the runner's (fictional) home; default: the demo's")
    p.add_argument("--work", default=None, help="LAT,LON of the (fictional) workplace; default: the demo's")
    p.add_argument("--rate", type=float, default=1.0, help="share of the usual running days (below 1: a quieter runner)")
    p.add_argument("--pace", type=float, default=1.0, help="pace factor (1.06: 6 %% slower)")
    p.add_argument("--together-from", default=None, metavar="EXPORT",
                   help="another synthetic runner's export: --together of its weekend runs are run side by side")
    p.add_argument("--together", type=int, default=0, help="how many runs to copy from --together-from")
    p.add_argument("--truth", default=None, help="where to write the true places (default: truth.json next to --out)")
    a = p.parse_args()
    if bool(a.together) != bool(a.together_from):
        p.error("--together and --together-from go together")
    for key in ("home", "work"):
        if getattr(a, key):
            PLACES[key] = tuple(map(float, getattr(a, key).split(",")))
    rng = random.Random(a.seed)
    G, node_ids, ux, uy, to_utm = build_graph(a.city, rng)
    print(f"graph ready: {G.number_of_nodes()} nodes")

    def node_near(latlon, r=0.0):
        ll = jitter(rng, latlon, r) if r else latlon
        x, y = to_utm.transform(ll[1], ll[0])
        return nearest_node(node_ids, ux, uy, x, y)

    acts = []
    day = dt.date(a.year, 1, 1)
    act_id = 11_000_000_000 + rng.randint(0, 10_000_000)
    holiday = (dt.date(a.year, 8, 4), dt.date(a.year, 8, 17))
    rest_weeks = {rng.randint(5, 50) for _ in range(3)}
    races = {dt.date(a.year, 4, 6): ("Spring Half Marathon", ["castellana_n", "colon", "cibeles", "gran_via", "plaza_espana", "debod", "oeste", "rey", "segovia", "toledo", "matadero", "atocha_retiro", "estanque"]),
             dt.date(a.year, 12, 31): ("New Year's Eve 10K", ["bernabeu", "castellana_n", "colon", "cibeles", "atocha_retiro", "vallecas"])}
    while day.year == a.year:
        wd = day.weekday()
        week = day.isocalendar()[1]
        planned = None
        if day in races:
            planned = ("race",) + races[day]
        elif holiday[0] <= day <= holiday[1]:
            if rng.random() < 0.45:
                planned = ("holiday",)
        elif week in rest_weeks:
            if rng.random() < 0.12:
                planned = ("normal",)
        else:
            prob = {0: 0.5, 1: 0.55, 2: 0.55, 3: 0.5, 4: 0.25, 5: 0.6, 6: 0.75}[wd] * a.rate
            if rng.random() < prob:
                planned = ("normal",)
        if planned:
            act_id += rng.randint(30_000, 900_000)
            weekend = wd >= 5
            if planned[0] == "race":
                start_time = dt.datetime(day.year, day.month, day.day, 8 if day.month == 4 else 17, 30)
                start, end = PLACES[planned[2][0]], PLACES[planned[2][-1]]
                via = planned[2][1:-1]
                name, kind, pace, start_key = planned[1], "race", rng.uniform(285, 300) * a.pace, None
            elif planned[0] == "holiday":
                start_time = dt.datetime(day.year, day.month, day.day, 8, rng.randint(0, 50))
                name, kind, pace = "Morning Run", "holiday", rng.uniform(330, 370) * a.pace
            else:
                lunch = (not weekend) and rng.random() < 0.18
                if lunch:
                    tpl = rng.choice(WORK_TEMPLATES)
                    start_key = "work"
                    start_time = dt.datetime(day.year, day.month, day.day, 13, rng.randint(5, 40))
                    name = "Lunch Run"
                else:
                    pool = [t for t in TEMPLATES if (t["kind"] == "long") == (weekend and rng.random() < 0.55)] or TEMPLATES
                    tpl = rng.choices(pool, weights=[t["w"] for t in pool])[0]
                    start_key = "home"
                    if weekend:
                        start_time = dt.datetime(day.year, day.month, day.day, 9, rng.randint(0, 59))
                        name = "Morning Run"
                    elif rng.random() < 0.6:
                        start_time = dt.datetime(day.year, day.month, day.day, 7, rng.randint(0, 59))
                        name = "Morning Run"
                    else:
                        start_time = dt.datetime(day.year, day.month, day.day, 19, rng.randint(0, 59))
                        name = "Evening Run"
                kind = tpl["kind"]
                via = list(tpl["via"])
                if rng.random() < 0.35 and kind not in ("work", "short"):
                    via = via[::-1]
                pace = {"short": rng.uniform(335, 370), "easy": rng.uniform(320, 355), "mid": rng.uniform(305, 335), "long": rng.uniform(330, 365), "work": rng.uniform(315, 345)}[kind]
                pace *= a.pace
                start = end = PLACES[start_key]
            if planned[0] == "holiday":
                a0 = (39.4640 + rng.uniform(-0.002, 0.002), -0.3237)
                a1 = (39.4890 + rng.uniform(-0.002, 0.002), -0.3225)
                coords = straight_route(rng, a0, a1) + straight_route(rng, a1, a0)[1:]
            else:
                if kind == "easy" and len(via) > 2 and rng.random() < 0.35:  # turn around early
                    via = via[: max(1, len(via) // 2)] + via[: max(1, len(via) // 2)][::-1][1:]
                mid = [node_near(PLACES[k], 110) for k in via]
                if planned[0] != "race":
                    if kind == "short":  # neighbourhood loop, 1-2 random points 0.8-1.6 km away
                        mid = [node_near(ring_point(rng, start, 800, 1600), 0) for _ in range(rng.randint(1, 2))]
                    elif rng.random() < 0.08:  # exploration run: random waypoints 1-2.5 km away
                        mid = [node_near(ring_point(rng, start, 1000, 2500), 0) for _ in range(rng.randint(2, 3))]
                    if kind != "short" and rng.random() < 0.6:  # leave and come back by different nearby streets
                        mid = [node_near(ring_point(rng, start, 300, 700), 0)] + mid + [node_near(ring_point(rng, start, 300, 700), 0)]
                    if len(mid) > 2 and rng.random() < 0.3:  # small detour somewhere in the middle
                        j = rng.randint(1, len(mid) - 1)
                        mid.insert(j, node_near(ring_point(rng, PLACES[via[min(j, len(via) - 1)]] if via else start, 200, 700), 0))
                seq = [node_near(start, 25)] + mid + [node_near(end, 25)]
                coords = route_line(G, seq)
            trees = 1.5 if kind in ("long",) else 1.15
            pts, L = sample_track(coords, to_utm, rng, pace, a.dt, start_time, trees)
            if kind == "race":  # the course of the day, from its own random stream: the draws above stay as they
                # were, so every other run of the year is the same as before the courses were measured
                course = route_line(G, [nearest_node(node_ids, ux, uy, *to_utm.transform(PLACES[k][1], PLACES[k][0]))
                                        for k in RACE_COURSES[name]])
                pts, L = sample_track(course, to_utm, random.Random(a.seed * 1_000_003 + day.toordinal()), pace, a.dt,
                                      start_time, trees)
            fmt = rng.choices(["gpx", "gpx.gz", "tcx.gz"], weights=[60, 25, 15])[0]
            acts.append({"id": act_id, "date": start_time, "name": name, "type": "Run", "dist_km": L / 1000,
                         "elapsed": int((pts[-1][3] - pts[0][3]).total_seconds()), "fmt": fmt, "pts": pts, "kind": kind})
        # extra non-run activities
        if rng.random() < 0.03:
            act_id += rng.randint(30_000, 900_000)
            t0 = dt.datetime(day.year, day.month, day.day, 18, rng.randint(0, 59))
            seq = [node_near(PLACES["home"], 25), node_near(PLACES["estanque"], 200), node_near(PLACES["home"], 25)]
            pts, L = sample_track(route_line(G, seq), to_utm, rng, 720, a.dt, t0, 1.2)
            acts.append({"id": act_id, "date": t0, "name": "Afternoon Walk", "type": "Walk", "dist_km": L / 1000,
                         "elapsed": int((pts[-1][3] - pts[0][3]).total_seconds()), "fmt": "gpx", "pts": pts, "kind": "walk"})
        if rng.random() < 0.012:
            act_id += rng.randint(30_000, 900_000)
            t0 = dt.datetime(day.year, day.month, day.day, 18, rng.randint(0, 59))
            acts.append({"id": act_id, "date": t0, "name": "Treadmill Run", "type": "Run", "dist_km": rng.uniform(5, 10),
                         "elapsed": rng.randint(1800, 3600), "fmt": None, "pts": None, "kind": "indoor"})
        day += dt.timedelta(days=1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    to_utc(acts, "Europe/Madrid")
    if a.together:  # the shared runs take the place of that day's own run
        shared = together_runs(a.together_from, a.together, rng, act_id)
        days = {x["date"].date() for x in shared}
        acts = [x for x in acts if not (x["type"] == "Run" and x["kind"] != "race" and x["date"].date() in days)]
        acts = sorted(acts + shared, key=lambda x: x["date"])
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        rows = []
        for x in acts:
            fname = ""
            if x["pts"]:
                if x["fmt"] == "gpx":
                    fname = f"activities/{x['id']}.gpx"
                    z.writestr(fname, gpx_text(x["pts"], x["name"], "running" if x["type"] == "Run" else "walking"))
                elif x["fmt"] == "gpx.gz":
                    fname = f"activities/{x['id']}.gpx.gz"
                    z.writestr(fname, gzip.compress(gpx_text(x["pts"], x["name"]).encode()))
                else:
                    fname = f"activities/{x['id']}.tcx.gz"
                    z.writestr(fname, gzip.compress(tcx_text(x["pts"]).encode()))
            rows.append({"Activity ID": x["id"], "Activity Date": x["date"].strftime("%b %d, %Y, %I:%M:%S %p").replace(" 0", " "),
                         "Activity Name": x["name"], "Activity Type": x["type"], "Activity Description": "",
                         "Elapsed Time": x["elapsed"], "Distance": f"{x['dist_km']:.2f}", "Max Heart Rate": "",
                         "Relative Effort": "", "Commute": "false", "Activity Private Note": "", "Activity Gear": "",
                         "Filename": fname})
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        z.writestr("activities.csv", buf.getvalue())
    truth = {"home": PLACES["home"], "work": PLACES["work"], "note": "Synthetic runner. Only for privacy tests. Never ship with the export."}
    if a.together:
        truth = {"home": PLACES["home"], "work": PLACES["work"], "together": a.together, "note": truth["note"]}
    with open(a.truth or os.path.join(os.path.dirname(a.out), "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    runs = [x for x in acts if x["type"] == "Run" and x["pts"]]
    from collections import Counter
    print(f"{len(acts)} activities ({len(runs)} GPS runs, {sum(1 for x in acts if x['kind']=='indoor')} treadmill, "
          f"{sum(1 for x in acts if x['type']=='Walk')} walks); run km = {sum(x['dist_km'] for x in runs):.0f}")
    print("kinds:", Counter(x["kind"] for x in acts))
    print("zip:", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
