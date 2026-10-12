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


def _from(lat, lon, dx_km, dy_km):
    return lon + dx_km / (111.32 * np.cos(np.radians(lat))), lat + dy_km / 110.57


def test_home_zones_protect_an_activity_grouped_into_another_city():
    # 20 loops from home, plus one long run that starts at home, passes the door again and ends 100 km away: its
    # median is ~50 km away, so it belongs to another city, which must still cut it around home
    H = (40.41, -3.70)
    t = np.linspace(0, 2 * np.pi, 160)
    acts = []
    for i in range(20):
        r = 1.2 + 0.05 * i
        acts.append(Activity(id=str(i), type="Run", start=dt.datetime(2025, 1, 1 + i, 8),
                             lonlat=[_from(*H, r * (np.cos(a) - 1), r * np.sin(a)) for a in t], source="x"))
    out = [(0.7 * s, 0.0) for s in np.linspace(0, 1, 30)] + [(0.7 - 1.4 * s, 0.05) for s in np.linspace(0, 1, 60)]
    out += [(-0.7 - 99.3 * s, 0.05) for s in np.linspace(0, 1, 2000)]
    acts.append(Activity(id="far", type="Run", start=dt.datetime(2025, 3, 1, 8),
                         lonlat=[_from(*H, dx, dy) for dx, dy in out], source="x"))
    cities = atlas.prepare(atlas.group_by_city(acts), log=lambda *_: None)
    assert len(cities) == 2
    for c in cities:
        tr = Transformer.from_crs(4326, c.crs, always_xy=True)
        home = Point(*tr.transform(H[1], H[0]))
        assert MultiLineString([p for _, p in c.pieces]).distance(home) >= 200


def test_numbers_count_every_city_and_the_maps_skip_the_ones_privacy_empties(monkeypatch, tmp_path):
    import types
    acts = [_act(i, 40.41, -3.70, dt.date(2025, 1 + i % 12, 3), r=0.01 + 0.001 * i) for i in range(10)]  # home
    for j, (lat, lon, n) in enumerate(((48.86, 2.35, 4), (41.89, 12.49, 3), (47.37, 8.54, 2))):
        acts += [_act(100 * (j + 1) + i, lat, lon, dt.date(2025, 3 + j, 1 + i), r=0.012) for i in range(n)]
    acts += [_act(900, 52.37, 4.90, dt.date(2025, 9, 1), r=0.0008)]  # one tiny loop: its zone swallows it all
    captured = {}
    monkeypatch.setattr(atlas.area, "place_names", lambda lat, lon, lang="en": (f"C{lat:.0f}", f"K{lat:.0f}"))
    monkeypatch.setattr(atlas, "_basemap", lambda c, ext, log: None)
    monkeypatch.setattr(atlas, "render_atlas", lambda drawn, *a, **k: captured.update(drawn=drawn, **k) or
                        {"cap": 3.0, "max": 1})
    args = types.SimpleNamespace(zone=[], trim=300.0, zone_radius=500.0, max_cities=2, lang="en", year=None,
                                 style=None, size="A3", dpi=40, name="", title="", out=str(tmp_path / "a.png"),
                                 pdf=None, svg=None, preview=None, color_by="heat")
    assert atlas.run(acts, args, log=lambda *_: None) == 0
    drawn, s = captured["drawn"], captured["summary"]
    assert [round(c.lat) for c in drawn] == [40, 49], "home and the biggest other city"
    assert s["cities"] == 5 and captured["more"] == 3, "the headline counts every city, the note the ones not drawn"
    assert s["runs"] == len(acts) and s["countries"] == 5
    assert all(c.pieces for c in drawn)


def test_clip_keeps_a_course_that_crosses_itself_in_one_piece_with_its_own_vertices():
    import numpy as np

    rng = np.random.default_rng(3)
    walk = np.column_stack([np.linspace(0, 3000, 600), np.zeros(600)])
    stop = walk[300] + rng.normal(0, 3, (600, 2))  # ten minutes standing, the fix wandering: thousands of crossings
    course = np.vstack([walk[:300], stop, walk[300:], [[3000, 900], [1500, 900], [1500, 100]]])  # out and in
    out = privacy.clip_to_extent([(7, LineString(course))], (-100, 3100, -500, 500))
    assert [i for i, _ in out] == [7, 7], "two pieces: the course leaves the map once and comes back"
    first = np.asarray(out[0][1].coords)
    assert len(first) == 1201 and np.array_equal(first[:1200], course[:1200]), "not broken at its own crossings"
    for _, part in out:
        x0, y0, x1, y1 = part.bounds
        assert -100 - 1e-6 <= x0 and x1 <= 3100 + 1e-6 and -500 - 1e-6 <= y0 and y1 <= 500 + 1e-6


def test_clip_keeps_only_what_is_inside_the_map():
    pieces = [(1, LineString([(-100, 0), (1100, 0)])), (2, LineString([(2000, 0), (3000, 0)]))]
    out = privacy.clip_to_extent(pieces, (0, 1000, -500, 500))
    assert [i for i, _ in out] == [1]
    x0, _, x1, _ = out[0][1].bounds
    assert -1e-6 <= x0 and x1 <= 1000 + 1e-6


def _outside_lines_city():
    # a dense home area plus runs 12 km north in the same city: the map shows the dense area, the rest is outside it
    acts = [_act(i, 40.41, -3.70, dt.date(2025, 1 + i % 12, 3), r=0.018 + 0.001 * i) for i in range(20)]
    acts += [_act(100 + i, 40.41 + 12 / 110.57, -3.70, dt.date(2025, 5, 1 + i), r=0.014) for i in range(3)]
    cities = atlas.prepare(atlas.group_by_city(acts), log=lambda *_: None)
    assert len(cities) == 1
    return cities


def _spy_tracks(monkeypatch):
    # the routes are the paths drawn at zorder 5 (render_editorial.draw_routes, used by the poster and the atlas);
    # record every vertex handed to the PDF/SVG/PNG writer
    from pinless import render_editorial
    seen, real = [], render_editorial.PathPatch

    def spy(path, *a, **k):
        if k.get("zorder") == 5:
            seen.append(np.asarray(path.vertices))
        return real(path, *a, **k)

    monkeypatch.setattr(render_editorial, "PathPatch", spy)
    return seen


def _inside(pts, ext):
    x0, x1, y0, y1 = ext
    return bool(((pts[:, 0] >= x0 - 1e-6) & (pts[:, 0] <= x1 + 1e-6) & (pts[:, 1] >= y0 - 1e-6)
                 & (pts[:, 1] <= y1 + 1e-6)).all())


def test_atlas_writes_no_route_outside_its_map(monkeypatch, tmp_path):
    # a clip path would hide such lines in the PDF/SVG but keep them in the file: they must not be written at all
    cities = _outside_lines_city()
    cells = atlas.layout(1)
    ext = atlas.fit(cities[0], cells[0][2] / cells[0][3])
    assert not _inside(np.concatenate([np.asarray(p.coords) for _, p in cities[0].pieces]), ext), "setup: lines outside"
    seen = _spy_tracks(monkeypatch)
    atlas.render_atlas(cities, cells, [ext], [None], str(tmp_path / "a.png"), dpi=20, svg_path=str(tmp_path / "a.svg"))
    assert seen and _inside(np.concatenate(seen), ext)


def test_poster_writes_no_route_outside_its_map(monkeypatch, tmp_path):
    import geopandas as gpd
    from pinless import render_editorial
    cities = _outside_lines_city()
    ext = atlas.fit(cities[0], render_editorial.map_aspect("A3"))
    empty = gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])
    stats = {"km": 1.0, "n": 1, "hours": 1.0, "best5k_s": None, "best10k_s": None, "longest_km": 1.0, "streets": 1}
    seen = _spy_tracks(monkeypatch)
    render_editorial.render(cities[0].pieces, (empty, empty, empty), ext, stats, str(tmp_path / "p.png"), dpi=20,
                            city="X", year="2025", svg_path=str(tmp_path / "p.svg"))
    assert seen and _inside(np.concatenate(seen), ext)


def test_an_accent_under_the_label_does_not_cover_it():
    import matplotlib.pyplot as plt
    from pinless.render_editorial import DISPLAY, STYLES
    fig = plt.figure(figsize=(3, 3))
    page = atlas._Page(fig, 297, 420, 1.0, STYLES["huella"])
    from pinless.render_editorial import _outline
    tops = {}
    for word in ("ÁVILA", "CÁDIZ"):  # the headline call of render_atlas
        page.text(18, 22, word, pt=235, font=DISPLAY, max_w=215, height=44, clear=(15, 36, 17.0))
        patch = page.ax.patches[-1]
        v = _outline(patch.get_path(), patch.get_transform() + page.ax.transData.inverted(), 0.3)
        under = (v[:, 0] >= 15) & (v[:, 0] <= 36)  # the PINLESS label is x 18-32.7, y 13-15.2 mm
        tops[word] = (v[under, 1].min(), v[:, 1].max())
    plt.close(fig)
    assert tops["ÁVILA"][0] >= 17.0 - 1e-6, "no part of the accent of a first letter reaches the label"
    assert abs(tops["CÁDIZ"][1] - (22 + 44)) < 0.5, "CÁDIZ keeps the full capital height (baseline at 66 mm)"


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
