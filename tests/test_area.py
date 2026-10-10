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
