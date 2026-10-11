"""Add trips to the synthetic Madrid runner, for the atlas demo: Paris, Rome, Valencia, Zurich and a race in Lisbon.

Takes the Madrid demo export, removes what the runner did at home on travel days (including the old straight-line
Valencia holiday) and adds runs routed on the real OpenStreetMap streets of each city (downloaded once and cached by
pinless.area.fetch_basemap). Runs leave from and come back to a fictional lodging, as most travel runs do; the race is
point to point. Nobody's real data.

    python tools/simulate_trips.py --base demo/strava_export_madrid_demo.zip --out demo/strava_export_atlas_demo.zip

The lodgings go to a SEPARATE file (truth_atlas.json, next to --out) only for privacy tests, never into the export.
"""
import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import random
import sys
import zipfile

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import simulate_runner as sim  # noqa: E402
from pinless import area, render  # noqa: E402

# Fictional lodgings on residential streets and public landmarks (approximate), with loop templates.
TRIPS = [
    {"city": "Lisboa", "days": (dt.date(2025, 3, 9), dt.date(2025, 3, 9)), "runs": [dt.date(2025, 3, 9)],
     "half": 0.03, "ele": 20.0, "race": ["belem", "alcantara", "santos", "cais_sodre", "comercio"],
     "places": {"belem": (38.6950, -9.2050), "alcantara": (38.7030, -9.1780), "santos": (38.7060, -9.1560),
                "cais_sodre": (38.7060, -9.1450), "comercio": (38.7075, -9.1365)}},
    {"city": "Paris", "days": (dt.date(2025, 4, 18), dt.date(2025, 4, 21)),
     "runs": [dt.date(2025, 4, 18), dt.date(2025, 4, 19), dt.date(2025, 4, 21)], "half": 0.032, "ele": 40.0,
     "lodging": (48.8641, 2.3760),
     "places": {"canal": (48.8716, 2.3655), "buttes": (48.8809, 2.3828), "bastille": (48.8532, 2.3691),
                "ile_st_louis": (48.8517, 2.3567), "plantes": (48.8437, 2.3594), "lachaise": (48.8614, 2.3934),
                "villette": (48.8890, 2.3790)},
     "templates": [["canal", "villette", "buttes"], ["bastille", "ile_st_louis", "plantes", "bastille"],
                   ["lachaise", "buttes", "canal"]]},
    {"city": "Roma", "days": (dt.date(2025, 6, 5), dt.date(2025, 6, 8)),
     "runs": [dt.date(2025, 6, 6), dt.date(2025, 6, 7), dt.date(2025, 6, 8)], "half": 0.032, "ele": 30.0,
     "lodging": (41.8975, 12.4950),
     "places": {"colosseo": (41.8902, 12.4922), "circo": (41.8861, 12.4851), "caracalla": (41.8790, 12.4925),
                "aventino": (41.8833, 12.4800), "tevere": (41.8880, 12.4760), "borghese": (41.9128, 12.4852),
                "pincio": (41.9110, 12.4790), "popolo": (41.9107, 12.4763)},
     "templates": [["colosseo", "circo", "caracalla", "colosseo"], ["borghese", "pincio", "popolo"],
                   ["circo", "tevere", "aventino", "colosseo"]]},
    {"city": "Valencia", "days": (dt.date(2025, 8, 4), dt.date(2025, 8, 17)), "runs": None, "half": 0.026, "ele": 10.0,
     "lodging": (39.4705, -0.3290),
     "places": {"malvarrosa": (39.4850, -0.3230), "patacona": (39.4930, -0.3210), "marina": (39.4595, -0.3260),
                "artes": (39.4545, -0.3505), "turia": (39.4790, -0.3640)},
     "templates": [["malvarrosa", "patacona", "malvarrosa"], ["marina"], ["malvarrosa"],
                   ["artes", "turia", "artes"]]},
    {"city": "Zurich", "days": (dt.date(2025, 10, 10), dt.date(2025, 10, 12)),
     "runs": [dt.date(2025, 10, 11), dt.date(2025, 10, 12)], "half": 0.022, "ele": 410.0,
     "lodging": (47.3770, 8.5480),
     "places": {"bellevue": (47.3667, 8.5450), "utoquai": (47.3620, 8.5480), "zurichhorn": (47.3530, 8.5530),
                "mythenquai": (47.3580, 8.5370), "platzspitz": (47.3790, 8.5400), "lindenhof": (47.3730, 8.5410)},
     "templates": [["bellevue", "utoquai", "zurichhorn", "utoquai"], ["lindenhof", "platzspitz"],
                   ["bellevue", "mythenquai", "bellevue"]]},
]
CSV_DATE = "%b %d, %Y, %I:%M:%S %p"


def centre_of(trip):
    pts = list(trip["places"].values()) + ([trip["lodging"]] if "lodging" in trip else [])
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def on_trip(day):
    return any(t["days"][0] <= day <= t["days"][1] for t in TRIPS)


def trip_runs(trip, rng, act_id, dt_s):
    lat, lon = centre_of(trip)
    h = trip["half"]
    folder = area.fetch_basemap((lat - h, lon - h * 1.4, lat + h, lon + h * 1.4))
    crs = render.utm_crs(lon, lat)
    G, node_ids, ux, uy, to_utm = sim.build_graph(trip["city"], rng, base=folder, crs=crs)
    main_part = max(nx.weakly_connected_components(G), key=len)  # the download keeps loose fragments: route on the main one
    G = G.subgraph(main_part).copy()
    keep = np.isin(node_ids, list(main_part))
    node_ids, ux, uy = node_ids[keep], ux[keep], uy[keep]

    def node_near(latlon, r=0.0):
        ll = sim.jitter(rng, latlon, r) if r else latlon
        x, y = to_utm.transform(ll[1], ll[0])
        return sim.nearest_node(node_ids, ux, uy, x, y)

    days = trip["runs"]
    if days is None:  # a longer stay: run on about half of the days
        d, days = trip["days"][0], []
        while d <= trip["days"][1]:
            if rng.random() < 0.5:
                days.append(d)
            d += dt.timedelta(days=1)
    out = []
    for day in days:
        act_id += rng.randint(30_000, 900_000)
        start_time = dt.datetime(day.year, day.month, day.day, rng.choice([7, 8, 9]), rng.randint(0, 59))
        if "race" in trip:
            seq = [node_near(trip["places"][k], 20) for k in trip["race"]]
            name, pace, kind = "Morning Run", rng.uniform(280, 295), "race"
        else:
            via = list(rng.choice(trip["templates"]))
            if rng.random() < 0.4:
                via = via[::-1]
            mid = [node_near(trip["places"][k], 110) for k in via]
            if rng.random() < 0.6:  # leave and come back by different nearby streets
                mid = [node_near(sim.ring_point(rng, trip["lodging"], 300, 700))] + mid + \
                      [node_near(sim.ring_point(rng, trip["lodging"], 300, 700))]
            seq = [node_near(trip["lodging"], 25)] + mid + [node_near(trip["lodging"], 25)]
            name, pace, kind = "Morning Run", rng.uniform(320, 360), "trip"
        coords = sim.route_line(G, seq)
        pts, L = sim.sample_track(coords, to_utm, rng, pace, dt_s, start_time, 1.2, crs=crs, ele0=trip["ele"],
                                    terrain=None)
        out.append({"id": act_id, "date": start_time, "name": name, "type": "Run", "dist_km": L / 1000,
                    "elapsed": int((pts[-1][3] - pts[0][3]).total_seconds()), "pts": pts, "kind": kind,
                    "city": trip["city"], "fmt": rng.choices(["gpx", "gpx.gz", "tcx.gz"], weights=[60, 25, 15])[0]})
    return out, act_id


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.path.join(ROOT, "demo", "strava_export_madrid_demo.zip"))
    p.add_argument("--out", default=os.path.join(ROOT, "demo", "strava_export_atlas_demo.zip"))
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--dt", type=float, default=3.0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    with zipfile.ZipFile(a.base) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("activities.csv").decode("utf-8"))))
        kept = [r for r in rows if not on_trip(dt.datetime.strptime(r["Activity Date"], CSV_DATE).date())]
        files = {r["Filename"]: z.read(r["Filename"]) for r in kept if r["Filename"]}
    print(f"base: {len(rows)} activities, {len(rows) - len(kept)} removed on travel days")
    act_id = 12_000_000_000
    new = []
    for trip in TRIPS:
        runs, act_id = trip_runs(trip, rng, act_id, a.dt)
        new += runs
        print(f"{trip['city']}: {len(runs)} runs, {sum(r['dist_km'] for r in runs):.1f} km")
    for x in new:
        if x["fmt"] == "gpx":
            fname = f"activities/{x['id']}.gpx"
            files[fname] = sim.gpx_text(x["pts"], x["name"]).encode()
        elif x["fmt"] == "gpx.gz":
            fname = f"activities/{x['id']}.gpx.gz"
            files[fname] = gzip.compress(sim.gpx_text(x["pts"], x["name"]).encode(), mtime=0)
        else:
            fname = f"activities/{x['id']}.tcx.gz"
            files[fname] = gzip.compress(sim.tcx_text(x["pts"]).encode(), mtime=0)
        kept.append({"Activity ID": x["id"], "Activity Date": x["date"].strftime(CSV_DATE).replace(" 0", " "),
                     "Activity Name": x["name"], "Activity Type": "Run", "Activity Description": "",
                     "Elapsed Time": x["elapsed"], "Distance": f"{x['dist_km']:.2f}", "Max Heart Rate": "",
                     "Relative Effort": "", "Commute": "false", "Activity Private Note": "", "Activity Gear": "",
                     "Filename": fname})
    kept.sort(key=lambda r: dt.datetime.strptime(r["Activity Date"], CSV_DATE))
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
        z.writestr("activities.csv", buf.getvalue())
        for name in sorted(files):
            z.writestr(name, files[name])
    base_truth = json.load(open(os.path.join(os.path.dirname(a.base), "truth.json")))
    truth = {"home": base_truth["home"], "work": base_truth["work"],
             "lodgings": {t["city"]: list(t["lodging"]) for t in TRIPS if "lodging" in t},
             "note": "Synthetic runner. Only for privacy tests. Never ship with the export."}
    with open(os.path.join(os.path.dirname(a.out), "truth_atlas.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=1)
    print(f"{len(kept)} activities -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
