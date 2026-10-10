"""The map underneath: the sea comes from the coastline, and a PDF/SVG holds one path per kind, not one per street."""

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, Point

from pinless import render, render_editorial

BBOX = (39.45, -0.36, 39.49, -0.30)  # south, west, north, east (Valencia's beach)
CRS = render.utm_crs(-0.33, 39.47)
TR = Transformer.from_crs(4326, CRS, always_xy=True)


def _coast(*lines):
    return gpd.GeoDataFrame(geometry=[LineString(c) for c in lines], crs=4326)


def _is_sea(sea, lon, lat):
    return any(p.contains(Point(*TR.transform(lon, lat))) for p in sea)


def test_the_sea_is_on_the_right_of_the_coastline():
    north = render.sea_polygons(_coast([(-0.33, 39.40), (-0.33, 39.55)]), BBOX, CRS)  # land left (west), sea east
    assert _is_sea(north, -0.31, 39.47) and not _is_sea(north, -0.35, 39.47)
    south = render.sea_polygons(_coast([(-0.33, 39.55), (-0.33, 39.40)]), BBOX, CRS)
    assert _is_sea(south, -0.35, 39.47) and not _is_sea(south, -0.31, 39.47)


def test_an_island_stays_land_and_no_coast_means_no_sea():
    island = [(-0.32, 39.46), (-0.31, 39.46), (-0.31, 39.47), (-0.32, 39.47), (-0.32, 39.46)]  # counterclockwise
    sea = render.sea_polygons(_coast([(-0.33, 39.40), (-0.33, 39.55)], island), BBOX, CRS)
    assert not _is_sea(sea, -0.315, 39.465) and _is_sea(sea, -0.305, 39.48)
    assert render.sea_polygons(gpd.GeoDataFrame(geometry=[], crs=4326), BBOX, CRS) == []


def test_load_basemap_adds_the_sea_to_the_water(tmp_path):
    G = nx.MultiDiGraph(crs="epsg:4326")
    G.add_node(1, x=-0.35, y=39.47)
    G.add_node(2, x=-0.34, y=39.47)
    G.add_edge(1, 2, length=850.0, highway="footway")
    ox.save_graphml(G, str(tmp_path / "walk.graphml"))
    empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326)
    empty.to_file(tmp_path / "green.gpkg")
    river = LineString([(-0.35, 39.47), (-0.31, 39.47)])  # a river centre line that runs on into the sea
    gpd.GeoDataFrame({"waterway": ["river"]}, geometry=[river], crs=4326).to_file(tmp_path / "water.gpkg")
    _coast([(-0.33, 39.40), (-0.33, 39.55)]).to_file(tmp_path / "coast.gpkg")
    _, _, water = render.load_basemap(str(tmp_path), BBOX, CRS)
    assert sorted(water["natural"].fillna("river")) == ["river", "sea"] and water.crs == CRS
    line = water[water["waterway"] == "river"].geometry.iloc[0]
    assert line.bounds[2] <= TR.transform(-0.33, 39.47)[0] + 1, "the river line stops at the coast"


def _streets(n=400):
    rng = np.random.default_rng(0)
    lines, kinds = [], []
    for i in range(n):
        a = rng.uniform(150, 850, 2)  # well inside the map: nothing is cut by its edge
        line = LineString([a, a + rng.uniform(-60, 60, 2), a + rng.uniform(-60, 60, 2)])
        for g in (line, line.reverse()):  # OSM walk graphs hold every street once per direction
            lines.append(g)
            kinds.append(("footway", "residential", "primary")[i % 3])
    return gpd.GeoDataFrame({"highway": kinds}, geometry=lines)


def test_streets_are_drawn_once_each_as_one_path_per_kind(monkeypatch, tmp_path):
    seen, real = [], render_editorial.PathPatch

    def spy(path, *a, **k):
        if k.get("zorder") == 3:
            seen.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(render_editorial, "PathPatch", spy)
    empty = gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])
    stats = {"km": 1.0, "n": 1, "hours": 1.0, "best5k_s": None, "best10k_s": None, "longest_km": 1.0, "streets": 1}
    svg = tmp_path / "p.svg"
    render_editorial.render([], (_streets(), empty, empty), (0, 1000, 0, 1000), stats, str(tmp_path / "p.png"), dpi=20,
                            city="X", year="2025", svg_path=str(svg))
    assert len(seen) == 3, "one path per street kind"
    assert sum(int((p.codes == p.MOVETO).sum()) for p in seen) == 400, "each street once, whatever its direction"
    assert svg.read_text(encoding="utf-8").count("<path") < 300, "not one element per street"
