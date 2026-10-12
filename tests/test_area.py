"""The automatic area is where you run most, and the OSM download survives a busy server without caching a bad map."""
import datetime as dt
import os
import socket

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from osmnx._errors import InsufficientResponseError

from pinless import area
from pinless.ingest import Activity


def _loop(lat, lon, r_deg, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return list(zip(lon + r_deg * np.cos(t), lat + r_deg * np.sin(t)))


def test_auto_bbox_ignores_a_far_away_holiday():
    acts = [Activity(id=str(i), type="Run", start=dt.datetime(2025, 1, 1), lonlat=_loop(40.41, -3.70, 0.02 + 0.001 * i),
                     source="x") for i in range(30)]
    acts += [Activity(id=f"h{i}", type="Run", start=dt.datetime(2025, 8, 1), lonlat=_loop(39.47, -0.33, 0.01), source="y")
             for i in range(4)]
    s, w, n, e = area.auto_bbox(acts)
    assert s < 40.41 < n and w < -3.70 < e
    assert e < -1.0, "the Valencia holiday must not stretch the area east"
    assert (n - s) < 0.3 and (e - w) < 0.3


def _tiny_graph():
    G = nx.MultiDiGraph(crs="epsg:4326")
    G.add_node(1, x=-3.70, y=40.41)
    G.add_node(2, x=-3.69, y=40.41)
    G.add_edge(1, 2, length=850.0, highway="footway")
    return G


def test_fetch_basemap_tries_the_next_server_and_never_caches_a_failed_layer(tmp_path, monkeypatch):
    def graph(*a, **k):
        if ox.settings.overpass_url == area.OVERPASS[0]:
            raise ConnectionError("busy")
        return _tiny_graph()

    def features(*a, **k):
        raise ConnectionError("busy everywhere")

    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))  # no real DNS in tests
    monkeypatch.setattr(ox, "graph_from_bbox", graph)
    monkeypatch.setattr(ox, "features_from_bbox", features)
    with pytest.raises(RuntimeError, match="no OpenStreetMap"):
        area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None)
    (folder,) = tmp_path.iterdir()
    assert (folder / "walk.graphml").exists(), "the streets came from the second server"
    assert not (folder / "green.gpkg").exists(), "a server error must not be cached as 'no parks here'"


def test_overpass_tries_every_address_of_a_server_and_restores_dns(monkeypatch):
    # overpass-api.de is several machines; osmnx pins the one gethostbyname returns, which can be the one that is down
    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1", "192.0.2.2"]))
    original = socket.gethostbyname
    seen, lines = [], []

    def call():
        ip = socket.gethostbyname("overpass-api.de")
        seen.append(ip)
        if ip == "192.0.2.1":
            raise ConnectionError("this machine is down")
        return "streets"

    monkeypatch.setattr(area, "_answered", {})
    assert area._overpass(call, lines.append) == "streets"
    assert seen == ["192.0.2.1", "192.0.2.2"] and "192.0.2.1" in lines[0]
    assert socket.gethostbyname is original
    seen.clear()
    assert area._overpass(call, lines.append) == "streets"
    assert seen == ["192.0.2.2"], "the machine that answered is tried first next time"


def test_osmnx_still_pins_overpass_through_gethostbyname(monkeypatch):
    # _overpass relies on this osmnx behaviour to try each machine; if osmnx changes it, this test says so
    asked = []
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)  # osmnx wraps it: restore after the test
    monkeypatch.setattr(socket, "gethostbyname", lambda h: asked.append(h) or "192.0.2.9")
    ox._http._config_dns("https://overpass-api.de/api")
    assert asked == ["overpass-api.de"]


def test_a_busy_server_answer_without_data_is_retried_and_never_cached(tmp_path, monkeypatch):
    # Overpass answers a timeout with HTTP 200, a "remark" and no elements: osmnx then raises InsufficientResponseError
    # and does not keep the answer in its cache. That is a busy server, not "no parks here".
    def features(*a, **k):
        raise InsufficientResponseError("no matching features")

    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))
    monkeypatch.setattr(ox, "graph_from_bbox", lambda *a, **k: _tiny_graph())
    monkeypatch.setattr(ox, "features_from_bbox", features)
    with pytest.raises(RuntimeError, match="no OpenStreetMap"):
        area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None)
    (folder,) = tmp_path.iterdir()
    assert not (folder / "green.gpkg").exists()


def test_streets_that_never_come_stop_with_a_clean_error(tmp_path, monkeypatch):
    def graph(*a, **k):
        raise InsufficientResponseError("no data")

    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))
    monkeypatch.setattr(ox, "graph_from_bbox", graph)
    with pytest.raises(RuntimeError):  # the CLI turns RuntimeError into "[osm] STOP", never a traceback
        area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None)


def test_fetch_basemap_caches_an_empty_layer_only_when_osm_says_there_is_none(tmp_path, monkeypatch):
    def features(*a, **k):  # a real empty answer: osmnx keeps it in its cache, then finds no features
        folder = str(ox.settings.cache_folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"answer{len(os.listdir(folder))}.json"), "w") as f:
            f.write('{"elements": []}')
        raise InsufficientResponseError("no matching features")

    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))  # no real DNS in tests
    monkeypatch.setattr(ox, "graph_from_bbox", lambda *a, **k: _tiny_graph())
    monkeypatch.setattr(ox, "features_from_bbox", features)
    folder = area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None)
    assert len(gpd.read_file(os.path.join(folder, "green.gpkg"))) == 0


def test_a_cached_area_is_found_again_despite_the_rounded_folder_name(tmp_path, monkeypatch):
    monkeypatch.setattr(ox, "graph_from_bbox", lambda *a, **k: _tiny_graph())
    monkeypatch.setattr(ox, "features_from_bbox", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no network")))
    bbox = (40.40049, -3.71049, 40.42049, -3.68049)  # rounds to a folder that is 0.0005 short on every side
    folder = os.path.join(tmp_path, "bbox_40.400_m3.710_40.420_m3.680")
    os.makedirs(folder)
    ox.save_graphml(_tiny_graph(), os.path.join(folder, "walk.graphml"))
    for name in ("green", "water", "coast", "peaks"):
        gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326).to_file(os.path.join(folder, f"{name}.gpkg"))
    assert area._cached_cover(str(tmp_path), bbox) == folder
    assert area.fetch_basemap(bbox, root=str(tmp_path), log=lambda *_: None) == folder  # no download attempted


def test_an_older_cache_gets_its_coast_and_summits_for_its_own_whole_box(tmp_path, monkeypatch):
    # caches made before the coast and peaks layers are completed on first use, for the folder's box (not the smaller
    # request)
    asked = []

    def features(box_, tags):
        asked.append((box_, tags))
        folder = str(ox.settings.cache_folder)  # a real empty answer: osmnx keeps it in its cache
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "answer.json"), "w") as f:
            f.write('{"elements": []}')
        raise InsufficientResponseError("no coastline here")

    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))
    monkeypatch.setattr(ox, "features_from_bbox", features)
    folder = os.path.join(tmp_path, "bbox_40.300_m3.800_40.500_m3.600")
    os.makedirs(folder)
    ox.save_graphml(_tiny_graph(), os.path.join(folder, "walk.graphml"))
    for name in ("green", "water"):
        gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326).to_file(os.path.join(folder, f"{name}.gpkg"))
    assert area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None) == folder
    assert [t for _, t in asked] == [{"natural": "coastline"}], "a city poster never asks for the summits"
    assert not os.path.exists(os.path.join(folder, "peaks.gpkg"))
    assert area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None,
                              extra=("peaks",)) == folder
    assert [t for _, t in asked] == [{"natural": "coastline"}, {"natural": "peak"}], "the trail print does, once"
    assert {a[0] for a in asked} == {(-3.8, 40.3, -3.6, 40.5)}, "the folder's own box (west, south, east, north)"
    assert len(gpd.read_file(os.path.join(folder, "coast.gpkg"))) == 0
    assert len(gpd.read_file(os.path.join(folder, "peaks.gpkg"))) == 0


def test_a_city_poster_on_a_cache_without_summits_stays_offline_while_overpass_is_busy(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(area, "_overpass", lambda call, log, empty_ok=True: asked.append(1) or (_ for _ in ()).throw(
        RuntimeError("no OpenStreetMap (Overpass) server answered; try again later")))
    folder = os.path.join(tmp_path, "bbox_40.300_m3.800_40.500_m3.600")
    os.makedirs(folder)
    ox.save_graphml(_tiny_graph(), os.path.join(folder, "walk.graphml"))
    for name in ("green", "water", "coast"):
        gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326).to_file(os.path.join(folder, f"{name}.gpkg"))
    lines = []
    for _ in range(3):
        assert area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lines.append) == folder
    assert asked == [] and lines == [], "no request, no message: the summits are only the trail print's"
    area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lines.append, extra=("peaks",))
    assert asked == [1] and any("this map names none" in s_ for s_ in lines)


def _empty_answer():
    folder = str(ox.settings.cache_folder)  # a real empty answer: osmnx keeps it in its cache
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"a{len(os.listdir(folder))}.json"), "w") as f:
        f.write('{"elements": []}')
    raise InsufficientResponseError("none")


@pytest.mark.parametrize("layer, tags, said", [("coast", {"natural": "coastline"}, "this map has no sea"),
                                               ("peaks", {"natural": "peak"}, "this map names none")])
def test_a_missing_coast_or_summits_do_not_stop_the_map(tmp_path, monkeypatch, layer, tags, said):
    # the coast only paints the sea and the summits only name the hills: if no server answers, draw without them and
    # cache nothing (the next run retries)
    def features(box_, tags_):
        if tags_ == tags:
            raise ConnectionError("busy")
        _empty_answer()

    lines = []
    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))
    monkeypatch.setattr(ox, "graph_from_bbox", lambda *a, **k: _tiny_graph())
    monkeypatch.setattr(ox, "features_from_bbox", lambda box_, tags: features(box_, tags))
    folder = area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lines.append, extra=("peaks",))
    assert os.path.exists(os.path.join(folder, "green.gpkg"))
    assert not os.path.exists(os.path.join(folder, f"{layer}.gpkg"))
    assert any(said in s for s in lines)


def test_summits_are_cached_as_points_with_their_name_and_height(tmp_path, monkeypatch):
    from shapely.geometry import Point, Polygon

    def features(box_, tags):
        if tags != {"natural": "peak"}:
            _empty_answer()
        return gpd.GeoDataFrame({"natural": ["peak", "peak"], "name": ["Penalara", "Odd"], "ele": ["2428", "9"],
                                 "wikidata": ["Q1", "Q2"]},
                                geometry=[Point(-3.70, 40.41), Polygon([(-3.7, 40.4), (-3.69, 40.4), (-3.69, 40.41)])],
                                crs=4326)

    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))
    monkeypatch.setattr(ox, "graph_from_bbox", lambda *a, **k: _tiny_graph())
    monkeypatch.setattr(ox, "features_from_bbox", features)
    folder = area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lambda *_: None,
                                extra=("peaks",))
    peaks = gpd.read_file(os.path.join(folder, "peaks.gpkg"))
    assert list(peaks.columns) == ["name", "ele", "geometry"] and list(peaks.geometry.type) == ["Point"]
    assert peaks.loc[0, "name"] == "Penalara" and peaks.loc[0, "ele"] == "2428"


def test_the_city_lookup_sends_its_point_rounded_to_about_a_kilometre(monkeypatch):
    """The centre of the box of two people who run loops from one door lies about a hundred metres from it: the one
    Nominatim request for the city name gets that point rounded to 0.01 degrees, as the atlas does."""
    import io
    import urllib.request

    sent = []

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=None):
        sent.append(req.full_url)
        return Answer(b'{"address": {"city": "Madrid"}}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert area.city_name(40.41629, -3.69729) == "Madrid"
    assert len(sent) == 1 and "lat=40.42&lon=-3.70&" in sent[0], "rounded to 0.01 degrees"
    assert "4162" not in sent[0] and "6972" not in sent[0]


def test_the_area_for_two_keeps_the_partner_who_records_fewer_points():
    """The box is the 1st to 99th percentile of each person's own points near the winner, joined: pooled, a partner
    whose watch records ten times fewer points is about 2 % of them and a third of their runs fell outside."""
    lat, lon = 40.42, -3.70
    dlon = 4.5 / (111.32 * np.cos(np.radians(lat)))  # 4.5 km east

    def loops(n, r_km, pts, east=0.0):
        r = r_km / 111.32
        return [Activity(id=f"{east}{i}", type="Run", start=dt.datetime(2025, 1, 1),
                         lonlat=_loop(lat + r * np.sin(0.3 * i), lon + east + r * np.cos(0.3 * i), r, n=pts),
                         source="x") for i in range(n)]

    a, b = loops(200, 1.5, 1000), loops(40, 1.0, 100, east=dlon)
    s, w, n, e = area.auto_bbox(a + b, groups=[0] * len(a) + [1] * len(b))
    for acts in (a, b):
        pts = np.concatenate([np.asarray(x.lonlat) for x in acts])
        inside = (pts[:, 0] >= w) & (pts[:, 0] <= e) & (pts[:, 1] >= s) & (pts[:, 1] <= n)
        assert inside.mean() >= 0.99, inside.mean()


def _metres(xy, lat=40.42, lon=-3.70):
    """Metres east/north of a point in Madrid to (lon, lat)."""
    xy = np.asarray(xy, dtype=float)
    return list(zip(lon + xy[:, 0] / (111320 * np.cos(np.radians(lat))), lat + xy[:, 1] / 110540))


def _lodging_scene(north_m, n_loops=150):
    """First person: n_loops loops from a home 3 km north of the reference point. Second person: 10 out-and-back runs
    north from a lodging 2.5 km south of it (moved north_m further north), the southernmost points of either."""
    rng = np.random.default_rng(11)
    a = []
    for i in range(n_loops):
        r, ph = 1200 + 13 * (i % 31), 0.61 * i
        t = np.linspace(0, 2 * np.pi, 240) + ph
        xy = np.column_stack([r * (np.cos(t) - np.cos(ph)), 3000 + r * (np.sin(t) - np.sin(ph))])
        a.append(Activity(id=f"a{i}", type="Run", start=dt.datetime(2025, 1, 1), lonlat=_metres(xy), source="x"))
    b = []
    for i in range(10):
        go = np.column_stack([np.full(500, 300.0 + 40 * (i % 5)), np.linspace(-2500.0, 2500.0, 500)])
        xy = np.vstack([go, go[::-1]]) + rng.normal(0, 5, (1000, 2)) + (0.0, north_m)
        b.append(Activity(id=f"b{i}", type="Run", start=dt.datetime(2025, 1, 1), lonlat=_metres(xy), source="y"))
    return a, b


def test_the_area_s_edges_do_not_follow_a_door_at_the_edge_of_the_area():
    """The 1st percentile of a partner's out-and-back runs from a lodging is the lodging's latitude to within tens of
    metres, and it sets the south edge of the area: the map's frame and the box sent to Overpass would give it away.
    Moved north 15 m at a time, the lodging must not move the box (but once, where it crosses a grid line): its edges
    lie on a grid of about 1 km, for two people and for one."""
    lon_step = area.GRID_DEG / np.cos(np.radians(40))
    for two in (True, False):
        boxes = []
        for k in range(5):
            a, b = _lodging_scene(15.0 * k)
            box_ = area.auto_bbox(a + b, groups=[0] * len(a) + [1] * len(b)) if two else area.auto_bbox(b)
            s, w, n, e = box_
            assert box_ == area.snap_box(box_), "on the grid"
            for v, step in ((s, area.GRID_DEG), (n, area.GRID_DEG), (w, lon_step), (e, lon_step)):
                assert abs(v / step - round(v / step)) < 1e-6, (two, v)
            boxes.append(box_)
        assert len(set(boxes)) <= 2, (two, boxes)
        lodging = 40.42 - 2500 / 110540
        assert boxes[0][0] < lodging, "the box still holds every run"


def test_snap_box_rounds_outward_to_about_a_kilometre():
    s, w, n, e = area.snap_box((40.38267, -3.70123, 40.48911, -3.60002))
    assert (s, n) == (40.38, 40.49)
    assert w <= -3.70123 and e >= -3.60002 and 0 < -3.70123 - w < 0.0131 and 0 < e + 3.60002 < 0.0131
    assert area.snap_box((s, w, n, e)) == (s, w, n, e)
    assert area.snap_box((40.38, -3.7, 40.39, -3.69))[::2] == (40.38, 40.39), "a value on the grid stays"


def test_a_frame_fitted_to_the_drawn_lines_is_on_the_grid_too(monkeypatch):
    """--fit crops the map to the drawn lines: that crop is rounded outward to the same grid, so lines that end at a
    zone's edge near the map's border do not set the frame, or the box sent to Overpass, to the metre."""
    from pyproj import Transformer
    from shapely.geometry import LineString

    from pinless import frame, render
    crs = render.utm_crs(-3.70, 40.42)
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    fed, real = [], render.fit_extent
    monkeypatch.setattr(render, "fit_extent", lambda bbox, c, aspect: fed.append(tuple(bbox)) or real(bbox, c, aspect))
    extents = []
    for k in range(5):
        _, b = _lodging_scene(15.0 * k, n_loops=0)
        pieces = [(i, LineString(np.column_stack(tr.transform(*np.asarray(x.lonlat).T)))) for i, x in enumerate(b)]
        extents.append(frame.extent_for(pieces, (40.0, -4.2, 41.0, -3.2), crs, 0.8, True))
    assert all(box_ == area.snap_box(box_) for box_ in fed)
    assert len(set(extents)) <= 2, extents
