"""Privacy check against the synthetic runner's ground truth (home and work are known only to the generator).

Run: python -m pytest tests -q   (uses the committed demo/ export and tests/data street extract; no network)
"""
import json
import os

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import MultiLineString, Point

from pinless import ingest, privacy, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "demo", "strava_export_madrid_demo.zip")
TRUTH = os.path.join(ROOT, "demo", "truth.json")


@pytest.mark.parametrize("repeat", range(5))  # zones are random per export: check several draws
def test_no_track_near_true_home_or_work(repeat):
    acts, _ = ingest.read_export(EXPORT)
    runs = [a for a in acts if a.type == "Run"]
    crs = render.utm_crs(-3.71, 40.41)
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    tracks = []
    for a in runs:
        arr = np.asarray(a.lonlat)
        x, y = tr.transform(arr[:, 0], arr[:, 1])
        tracks.append(np.column_stack([x, y]))
    pieces, rep = privacy.apply(tracks)
    assert rep.zones_suggested >= 2, "home and work clusters should both be suggested"
    union = MultiLineString([p for _, p in pieces])
    truth = json.load(open(TRUTH))
    for key in ("home", "work"):
        lat, lon = truth[key]
        px, py = tr.transform(lon, lat)
        d = union.distance(Point(px, py))
        assert d >= 200, f"{key}: a published track passes {d:.0f} m from the true point"


def test_trim_removes_both_ends():
    xy = np.column_stack([np.linspace(0, 2000, 201), np.zeros(201)])
    out = privacy.trim_ends(xy, 300)
    assert out[0, 0] >= 300 and out[-1, 0] <= 1700


def test_fail_closed_when_zone_is_touched(monkeypatch):
    # a track that passes straight through a user zone must be cut; if something still touches it, export stops
    xy = np.column_stack([np.linspace(-3000, 3000, 601), np.zeros(601)])
    pieces, rep = privacy.apply([xy, xy + 5, xy - 5], user_zones_xy=[(0.0, 0.0, 400.0)], min_share=1.0)
    union = MultiLineString([p for _, p in pieces])
    assert union.distance(Point(0, 0)) >= 200
    assert rep.leak_check == "passed"


def test_street_counts_once_per_activity():
    import geopandas as gpd
    from shapely.geometry import LineString
    from pinless import stats
    edges = gpd.GeoDataFrame({"name": ["Calle A"]}, geometry=[LineString([(0, 0), (1000, 0)])])
    piece1 = np.column_stack([np.linspace(0, 400, 41), np.zeros(41)])
    piece2 = np.column_stack([np.linspace(600, 1000, 41), np.zeros(41)])
    sc = stats.street_counts([piece1, piece2], edges, groups=[7, 7])  # two pieces of the SAME activity
    assert sc["Calle A"] == 1


def test_street_counts_skip_a_corner_shared_by_two_streets():
    # a sample off a shared corner is equally close to both streets: it counts for neither, whatever the edge order
    import geopandas as gpd
    from shapely.geometry import LineString
    from pinless import stats
    a, b = LineString([(0, 0), (1000, 0)]), LineString([(1000, 0), (1000, 1000)])
    corner = np.array([[1005.0, -5.0], [1010.0, -10.0]])  # shorter than one sampling step: only (1005, -5) is sampled
    along_a = np.column_stack([np.linspace(0, 400, 41), np.zeros(41)])
    for names, geoms in ((["Calle A", "Calle B"], [a, b]), (["Calle B", "Calle A"], [b, a])):
        edges = gpd.GeoDataFrame({"name": names}, geometry=geoms)
        assert stats.street_counts([corner], edges) == {}
        assert stats.street_counts([along_a], edges) == {"Calle A": 1}


@pytest.mark.parametrize("repeat", range(3))  # zones are random per export: check several draws
def test_favourite_street_is_far_from_true_home_and_work(repeat):
    from _street_fixture import load
    from pinless import stats
    acts, _ = ingest.read_export(EXPORT)
    runs = [a for a in acts if a.type == "Run"]
    crs = render.utm_crs(-3.71, 40.41)
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    tracks = [np.column_stack(tr.transform(np.asarray(a.lonlat)[:, 0], np.asarray(a.lonlat)[:, 1])) for a in runs]
    pieces, rep = privacy.apply(tracks)
    edges = load(crs)  # named OSM streets around the demo runs (tools/make_street_fixture.py --check: same stats as the full graph)
    st = stats.compute(runs, tracks, edges, private_pieces_xy=[np.asarray(p.coords) for _, p in pieces],
                       piece_groups=[i for i, _ in pieces], sensitive_xy=rep.centres)
    fav = st["fav_street"]
    assert fav
    truth = json.load(open(TRUTH))
    named = edges[edges["name"].apply(lambda n: (n[0] if isinstance(n, list) else n) == fav)]
    for key in ("home", "work"):
        lat, lon = truth[key]
        px, py = tr.transform(lon, lat)
        assert named.distance(Point(px, py)).min() >= 700, f"favourite street {fav} is close to {key}"


def test_a_track_that_crosses_itself_is_not_split_where_it_crosses():
    """An out and back along one street with GPS noise crosses itself every few metres: the cut must not node the
    line there and drop the bits between crossings as stubs (it lost 113 km of the Madrid demo that way)."""
    x = np.linspace(0, 3000, 601)
    wobble = 3 * np.sin(x / 7.0)
    zigzag = np.vstack([np.column_stack([x, wobble]), np.column_stack([x[::-1], -wobble[::-1]])])
    homes = [np.column_stack([np.full(50, -4000.0 + i), np.linspace(0, 2000, 50)]) for i in range(6)]
    pieces, rep = privacy.apply([zigzag] + homes, trim_m=300.0, zone_r=500.0)
    mine = [p for i, p in pieces if i == 0]
    assert rep.zones_suggested >= 1 and len(mine) == 1, "far from every zone: one piece, crossings or not"
    assert mine[0].length == pytest.approx(privacy.LineString(privacy.trim_ends(zigzag, 300.0)).length)
    through = np.column_stack([np.linspace(-6000, -2000, 801), np.full(801, 100.0)])  # crosses the zone at (-4000, 0)
    pieces, _ = privacy.apply([through] + homes, trim_m=300.0, zone_r=500.0)
    cut = [p for i, p in pieces if i == 0]
    assert len(cut) == 2 and min(p.distance(Point(-4000, 0)) for p in cut) >= 200
