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
    for name in ("green", "water", "coast"):
        gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326).to_file(os.path.join(folder, f"{name}.gpkg"))
    assert area._cached_cover(str(tmp_path), bbox) == folder
    assert area.fetch_basemap(bbox, root=str(tmp_path), log=lambda *_: None) == folder  # no download attempted


def test_an_older_cache_gets_its_coast_for_its_own_whole_box(tmp_path, monkeypatch):
    # caches made before the coast layer are completed on first use, for the folder's box (not the smaller request)
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
    assert [t for _, t in asked] == [{"natural": "coastline"}]
    assert asked[0][0] == (-3.8, 40.3, -3.6, 40.5), "the folder's own box (west, south, east, north)"
    assert len(gpd.read_file(os.path.join(folder, "coast.gpkg"))) == 0


def test_a_missing_coast_does_not_stop_the_map(tmp_path, monkeypatch):
    # the coast only paints the sea: if no server answers, draw without it and cache nothing (the next run retries)
    def features(box_, tags):
        if tags == {"natural": "coastline"}:
            raise ConnectionError("busy")
        folder = str(ox.settings.cache_folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"a{len(os.listdir(folder))}.json"), "w") as f:
            f.write('{"elements": []}')
        raise InsufficientResponseError("none")

    lines = []
    monkeypatch.delenv("PINLESS_OVERPASS", raising=False)
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda h: (h, [], ["192.0.2.1"]))
    monkeypatch.setattr(ox, "graph_from_bbox", lambda *a, **k: _tiny_graph())
    monkeypatch.setattr(ox, "features_from_bbox", features)
    folder = area.fetch_basemap((40.40, -3.71, 40.42, -3.68), root=str(tmp_path), log=lines.append)
    assert os.path.exists(os.path.join(folder, "green.gpkg")) and not os.path.exists(os.path.join(folder, "coast.gpkg"))
    assert any("no sea" in s for s in lines)


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
