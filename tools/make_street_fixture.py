"""Build the street fixture the tests use (tests/data/madrid_streets.json.gz), so pytest needs no OSM download.

It keeps every named OpenStreetMap street that comes within --buffer metres of a raw demo run, and it keeps the WHOLE
street (all its segments): the favourite-street rule measures the distance from the whole street to the private zones.
Needs the local Madrid walk graph (tools/fetch_osm.py); the tests themselves do not.

    python tools/make_street_fixture.py            # write the fixture
    python tools/make_street_fixture.py --check    # street stats on the fixture vs the full graph, several privacy draws
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.strtree import STRtree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from pinless import ingest, privacy, render, stats  # noqa: E402
import _street_fixture  # noqa: E402

GRAPH = os.path.join(ROOT, "cache", "madrid", "walk.graphml")
EXPORT = os.path.join(ROOT, "demo", "strava_export_madrid_demo.zip")


def _name(n):
    return n[0] if isinstance(n, list) else n


def demo_tracks():
    acts, _ = ingest.read_export(EXPORT)
    runs = [a for a in acts if a.type == "Run"]
    crs = render.utm_crs(-3.71, 40.41)
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    tracks = [np.column_stack(tr.transform(np.asarray(a.lonlat)[:, 0], np.asarray(a.lonlat)[:, 1])) for a in runs]
    return runs, tracks, crs


def full_edges():
    import osmnx as ox
    G = ox.load_graphml(GRAPH)
    return ox.graph_to_gdfs(G, nodes=False)[["name", "geometry"]]


def build(buffer_m):
    _, tracks, crs = demo_tracks()
    edges = full_edges()
    named = edges[edges["name"].notna()].copy()
    named["name"] = named["name"].map(_name)
    utm = named.to_crs(crs)
    tree = STRtree(list(utm.geometry.values))
    touched = set()
    for xy in tracks:
        touched.update(utm["name"].values[tree.query(LineString(xy), predicate="dwithin", distance=buffer_m)])
    keep = named[named["name"].isin(touched)]
    streets, seen = {}, set()
    for nm, geom in zip(keep["name"].values, keep.geometry.values):
        xy = np.asarray(geom.coords)
        q = np.round(xy * _street_fixture.SCALE).astype(np.int64)
        if not np.array_equal(q / _street_fixture.SCALE, xy):
            raise SystemExit(f"coordinates of {nm} are not on OSM's 1e-7 grid: the fixture would not be exact")
        key = (nm, min(q.tobytes(), q[::-1].tobytes()))
        if key in seen:  # the walk graph stores each segment once per direction
            continue
        seen.add(key)
        streets.setdefault(nm, []).append(np.vstack([q[:1], np.diff(q, axis=0)]).ravel().tolist())
    data = {
        "about": "Named OpenStreetMap streets within %g m of the demo runs, whole streets kept. Built by "
                 "tools/make_street_fixture.py for the tests." % buffer_m,
        "license": "Data (c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright",
        "osm_downloaded": dt.date.fromtimestamp(os.path.getmtime(GRAPH)).isoformat(),
        "encoding": "per segment: flat [lon, lat, ...] in integer 1e-7 degrees, first point absolute, then deltas",
        "streets": {nm: sorted(segs) for nm, segs in sorted(streets.items())},
    }
    os.makedirs(os.path.dirname(_street_fixture.PATH), exist_ok=True)
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(_street_fixture.PATH, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)
    n_seg = sum(len(s) for s in streets.values())
    print(f"{len(streets)} streets, {n_seg} segments, {os.path.getsize(_street_fixture.PATH) / 1e6:.2f} MB "
          f"-> {os.path.relpath(_street_fixture.PATH, ROOT)}")


def check(draws):
    """Same privacy draw, two street sources: per-street counts, distinct streets and favourite street must match."""
    runs, tracks, crs = demo_tracks()
    full = full_edges().to_crs(crs)
    fixture = _street_fixture.load(crs)
    failed = 0
    for i in range(draws):
        pieces, rep = privacy.apply(tracks)
        kw = dict(private_pieces_xy=[np.asarray(p.coords) for _, p in pieces], piece_groups=[g for g, _ in pieces],
                  sensitive_xy=rep.centres)
        ca = stats.street_counts(kw["private_pieces_xy"], full, groups=kw["piece_groups"])
        cb = stats.street_counts(kw["private_pieces_xy"], fixture, groups=kw["piece_groups"])
        sa, sb = stats.compute(runs, tracks, full, **kw), stats.compute(runs, tracks, fixture, **kw)
        fa = (sa["streets"], sa["fav_street"], sa["fav_street_n"])
        fb = (sb["streets"], sb["fav_street"], sb["fav_street_n"])
        ok = ca == cb and fa == fb
        failed += not ok
        diff = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
        print(f"draw {i + 1}: {'same' if ok else 'DIFFERENT'} | full {fa} | fixture {fb} | streets that differ: {len(diff)}"
              + (f" {sorted(diff.items())[:5]}" if diff else ""))
    print(f"{draws - failed}/{draws} draws identical")
    return failed == 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=float, default=25.0, help="metres around the raw runs (street matching uses 14)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--draws", type=int, default=5)
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.draws) else 1)
    build(a.buffer)
