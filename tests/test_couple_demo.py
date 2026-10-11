"""The poster for two against the synthetic pair's ground truth: two runners with their own homes and workplaces, who
run some weekend runs side by side (the partner from the first runner's door). The true places and the number of
runs together are known only to the generator (demo/truth.json, demo/truth_partner.json)."""
import json
import os

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, box

from pinless import area, ingest, privacy, render, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")


def _pair():
    """Both demo runners' runs where both run, projected, as the --with flow selects them."""
    sel = []
    for name in ("strava_export_madrid_demo.zip", "strava_export_madrid_partner_demo.zip"):
        acts, _ = ingest.read_export(os.path.join(DEMO, name))
        sel.append([a for a in acts if a.type == "Run"])
    s, w, n, e = area.auto_bbox(sel[0] + sel[1], groups=[0] * len(sel[0]) + [1] * len(sel[1]))
    inside = box(w, s, e, n)
    sel = [[a for a in x if LineString(a.lonlat).intersects(inside)] for x in sel]
    tr = Transformer.from_crs(4326, render.utm_crs((w + e) / 2, (s + n) / 2), always_xy=True)
    tracks = [[np.column_stack(tr.transform(*np.asarray(a.lonlat).T)) for a in x] for x in sel]
    return sel, tracks, tr


@pytest.fixture(scope="module")
def pair():
    return _pair()


def _truth():
    with open(os.path.join(DEMO, "truth.json"), encoding="utf-8") as f:
        a = json.load(f)
    with open(os.path.join(DEMO, "truth_partner.json"), encoding="utf-8") as f:
        b = json.load(f)
    return a, b


@pytest.mark.parametrize("repeat", range(3))  # zones are random per export: check several draws
def test_couple_demo_keeps_lines_away_from_every_private_place(pair, repeat):
    _, (ta, tb), tr = pair
    zones, witnesses, counts = privacy.pair_places([ta, tb])
    pieces, rep = privacy.apply(ta + tb, places_xy=zones, witnesses_xy=witnesses)
    assert rep.leak_check == "passed" and min(counts["per"]) >= 2
    union = MultiLineString([p for _, p in pieces])
    truth_a, truth_b = _truth()
    for who, truth in (("first", truth_a), ("partner", truth_b)):
        for key in ("home", "work"):
            lat, lon = truth[key]
            d = union.distance(Point(*tr.transform(lon, lat)))
            assert d >= 200, f"{who} {key}: a published line passes {d:.0f} m from it"
    owners = {i < len(ta) for i, _ in pieces}
    assert owners == {True, False}, "both people are drawn"


def test_partner_demo_runs_together_match_the_truth(pair):
    (sel_a, sel_b), (ta, tb), _ = pair
    got = stats.together(sel_a, ta, sel_b, tb)
    assert got["known"] and got["runs"] == _truth()[1]["together"]
    assert 5 < got["km"] / got["runs"] < 30, "km side by side: whole weekend runs"
