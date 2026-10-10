"""Atlas: cities found from the activities, privacy per city (hotels and family homes too), a grid that fits the page."""
import datetime as dt
import json
import os

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point

from pinless import atlas, ingest, privacy
from pinless.ingest import Activity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "demo", "strava_export_atlas_demo.zip")
TRUTH = os.path.join(ROOT, "demo", "truth_atlas.json")


def _loop(lat, lon, r_deg=0.01, n=120):
    t = np.linspace(0, 2 * np.pi, n)
    return list(zip(lon + r_deg * np.cos(t), lat + r_deg * np.sin(t)))


def _act(i, lat, lon, day, r=0.01):
    return Activity(id=str(i), type="Run", start=dt.datetime.combine(day, dt.time(8)), lonlat=_loop(lat, lon, r),
                    source="x", times=list(np.linspace(0, 1800, 120)))


def test_group_by_city_home_first_then_in_order_of_first_visit():
    acts = [_act(i, 40.41, -3.70, dt.date(2025, 1 + i % 12, 3)) for i in range(20)]  # Madrid all year
    acts += [_act(100 + i, 40.54, -3.64, dt.date(2025, 5, 2 + i)) for i in range(2)]  # Alcobendas: same city
    acts += [_act(200, 39.86, -4.02, dt.date(2025, 2, 9))]  # Toledo, 70 km away: its own city
    acts += [_act(300 + i, 48.86, 2.35, dt.date(2025, 4, 18 + i)) for i in range(3)]  # Paris
    acts += [_act(400 + i, 41.89, 12.49, dt.date(2025, 6, 6 + i)) for i in range(2)]  # Rome
    cities = atlas.group_by_city(acts)
    assert [c.n for c in cities] == [22, 1, 3, 2]
    assert [round(c.lat) for c in cities] == [40, 40, 49, 42]
    assert cities[1].first == dt.date(2025, 2, 9) and cities[2].first == dt.date(2025, 4, 18)


@pytest.mark.parametrize("n", range(1, 13))
def test_layout_cells_fit_the_page_and_do_not_overlap(n):
    cells = atlas.layout(n)
    assert len(cells) == n
    boxes = [(x, y, x + w, y + h + atlas.CAPTION) for x, y, w, h in cells]
    for x0, y0, x1, y1 in boxes:
        assert atlas.LEFT - 1e-6 <= x0 < x1 <= atlas.RIGHT + 1e-6
        assert atlas.TOP - 1e-6 <= y0 < y1 <= atlas.BOTTOM + 1e-6
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] + 1e-6 or b[2] <= a[0] + 1e-6 or a[3] <= b[1] + 1e-6 or b[3] <= a[1] + 1e-6
    assert all(atlas.ASPECT_MIN <= w / h <= atlas.ASPECT_MAX for _, _, w, h in cells), "no strips, no slivers"


def test_a_single_loop_run_marks_its_start_when_the_floor_is_two():
    t = np.linspace(0, 2 * np.pi, 400)
    loop = np.column_stack([2500 * np.sin(t), 1200 - 1200 * np.cos(t)])  # starts and ends at (0, 0)
    _, rep3 = privacy.apply([loop])
    assert rep3.zones_suggested == 0  # the poster's default floor needs 3 start/end points
    pieces, rep2 = privacy.apply([loop], min_pts_floor=2)
    assert rep2.zones_suggested == 1 and rep2.leak_check == "passed"
    assert MultiLineString([p for _, p in pieces]).distance(Point(0, 0)) >= 200


@pytest.mark.parametrize("repeat", range(3))  # zones are random per export: check several draws
def test_atlas_demo_keeps_lines_away_from_home_work_and_every_lodging(repeat):
    acts, _ = ingest.read_export(EXPORT)
    cities = atlas.prepare(atlas.group_by_city([a for a in acts if a.type == "Run"]), log=lambda *_: None)
    assert len(cities) == 6
    truth = json.load(open(TRUTH, encoding="utf-8"))
    points = {"home": truth["home"], "work": truth["work"], **truth["lodgings"]}
    for c in cities:
        union = MultiLineString([p for _, p in c.pieces])
        tr = Transformer.from_crs(4326, c.crs, always_xy=True)
        for key, (lat, lon) in points.items():
            d = union.distance(Point(*tr.transform(lon, lat)))
            assert d >= 200, f"{key}: a published line passes {d:.0f} m from it"
    for key, (lat, lon) in points.items():  # the city around each private place is drawn, with a zone, checked
        c = min(cities, key=lambda c: float(atlas._hav_km(c.lat, c.lon, lat, lon)))
        assert float(atlas._hav_km(c.lat, c.lon, lat, lon)) < 10, f"{key}: its city is not on the atlas"
        assert c.report.zones_suggested >= 1 and c.report.leak_check == "passed", key


def test_an_accent_does_not_shrink_a_name():
    import matplotlib.pyplot as plt
    from pinless.render_editorial import DISPLAY, STYLES
    fig = plt.figure(figsize=(3, 3))
    page = atlas._Page(fig, 297, 420, 1.0, STYLES["huella"])
    plain, _ = page.text(0, 0, "CADIZ", pt=40, font=DISPLAY, height=5.4)
    accented, _ = page.text(0, 0, "CÁDIZ", pt=40, font=DISPLAY, height=5.4)
    plt.close(fig)
    assert abs(accented / plain - 1) < 0.01, "same letters with or without the accent"


def test_render_atlas_without_basemaps(tmp_path):
    acts = [_act(i, 40.41, -3.70, dt.date(2025, 1 + i % 12, 3), r=0.01 + 0.001 * i) for i in range(8)]
    acts += [_act(50 + i, 48.86, 2.35, dt.date(2025, 4, 18 + i)) for i in range(2)]
    cities = atlas.prepare(atlas.group_by_city(acts), log=lambda *_: None)
    for c in cities:
        c.name, c.country = "Somewhere", "Nowhere"
    cells = atlas.layout(len(cities))
    extents = [atlas.fit(c, w / h) for c, (_, _, w, h) in zip(cities, cells)]
    out = tmp_path / "atlas.png"
    atlas.render_atlas(cities, cells, extents, [None] * len(cities), str(out), dpi=40, name="Test", year="2025",
                       pdf_path=str(tmp_path / "atlas.pdf"))
    with Image.open(out) as im:
        assert abs(im.size[0] - 297 / 25.4 * 40) <= 1 and abs(im.size[1] - 420 / 25.4 * 40) <= 1  # A3 at 40 dpi
    assert (tmp_path / "atlas.pdf").stat().st_size > 0
