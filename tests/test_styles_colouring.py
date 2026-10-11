"""Twelve styles and five ways to colour the routes."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from pinless import colouring, render_editorial
from pinless.ingest import Activity

EMPTY = gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])
STATS = {"km": 12.0, "n": 2, "hours": 1.0, "best5k_s": 1500, "best10k_s": 3100, "longest_km": 10.0, "streets": 7}


def _run(i, start, xy_km, times=None, ele=None):
    # positions as metres in the map's projection; lonlat only needs to sit in Madrid for the time zone
    return Activity(id=str(i), type="Run", start=start, lonlat=[(-3.70, 40.41)] * len(xy_km), source="x",
                    times=times, ele=ele)


@pytest.mark.parametrize("style", sorted(render_editorial.STYLES))
def test_every_style_is_complete_and_draws(style, tmp_path):
    S = render_editorial.STYLES[style]
    assert {"bg", "ink", "sub", "accent", "green", "water", "water_line", "rule", "street", "heat", "who"} <= set(S)
    assert len(S["heat"]) == 5 and set(S["street"]) == {"major", "minor", "path"}
    assert len(S["who"]) == 5, "the poster for two: first, mostly first, both, mostly second, second"
    pieces = [(0, LineString([(100, 100), (900, 700)])), (1, LineString([(100, 700), (900, 100)]))]
    out = tmp_path / "p.png"
    render_editorial.render(pieces, (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850), STATS, str(out), style=style, dpi=20,
                            city="X", year="2025", lang="en")
    assert out.stat().st_size > 0


def test_month_uses_the_local_date():
    # 23:30 UTC on 31 December is already January in Madrid
    acts = [_run(0, dt.datetime(2025, 12, 31, 23, 30, tzinfo=dt.timezone.utc), [(0, 0), (1, 0)]),
            _run(1, dt.datetime(2025, 6, 15, 7, 0, tzinfo=dt.timezone.utc), [(0, 0), (1, 0)])]
    segs = np.zeros((2, 2, 2))
    raw = colouring.raw_values("month", np.array([0, 1]), segs, acts, [np.zeros((2, 2))] * 2)
    assert list(raw) == [0, 5]


def test_pace_is_per_run_and_faster_is_stronger():
    xy = np.column_stack([np.linspace(0, 5000, 51), np.zeros(51)])  # 5 km
    fast = _run(0, None, xy, times=list(np.linspace(0, 1250, 51)))  # 4:10/km
    slow = _run(1, None, xy, times=list(np.linspace(0, 1750, 51)))  # 5:50/km
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    ids = np.array([0] * 50 + [1] * 50)
    raw = colouring.raw_values("pace", ids, np.concatenate([segs, segs]), [fast, slow], [xy, xy])
    assert raw[0] == pytest.approx(250) and raw[-1] == pytest.approx(350)
    (vals,), legend = colouring.normalise("pace", [raw])
    assert vals[0] > vals[-1] and colouring.fmt_pace(legend["lo"]) <= colouring.fmt_pace(legend["hi"])


def test_elevation_is_read_where_each_segment_lies():
    xy = np.column_stack([np.linspace(0, 1000, 11), np.zeros(11)])
    run = _run(0, None, xy, ele=list(np.linspace(600, 700, 11)))  # climbs 100 m over 1 km
    segs = np.array([[[0, 0], [100, 0]], [[900, 0], [1000, 0]]], dtype=float)
    raw = colouring.raw_values("elevation", np.array([0, 0]), segs, [run], [xy])
    assert raw == pytest.approx([605, 695])


def test_single_colour_draws_one_ink(monkeypatch):
    import matplotlib.pyplot as plt
    seen, real = [], render_editorial.PathPatch

    def spy(path, *a, **k):
        if k.get("zorder") == 5:
            seen.append(k["edgecolor"])
        return real(path, *a, **k)

    monkeypatch.setattr(render_editorial, "PathPatch", spy)
    segs = np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0, 2], [1, 2]]], dtype=float)
    S = render_editorial.STYLES["huella"]
    (vals,), legend, cmap = render_editorial.route_colours("single", [np.zeros(3)], S)
    fig, ax = plt.subplots()
    render_editorial.draw_routes(ax, segs, np.array([1, 5, 20]), 20, cmap, 1.0, values=vals)
    plt.close(fig)
    assert len(set(map(tuple, seen))) == 1 and len(seen) == 3, "one colour, three weights"


def test_a_mode_without_data_falls_back_to_one_ink_and_says_so():
    S = render_editorial.STYLES["huella"]
    (vals,), legend, _ = render_editorial.route_colours("pace", [np.full(4, np.nan)], S)
    assert legend == {"mode": "single", "fallback": "pace"} and vals == pytest.approx([0.85] * 4)
    (vals,), legend, _ = render_editorial.route_colours("pace", [np.array([300.0, np.nan, 360.0])], S)
    assert legend["mode"] == "pace" and vals[1] == pytest.approx(0.5), "some data: the unknown run sits mid-scale"


def test_months_have_their_own_colours_and_the_commonest_is_drawn_on_top(monkeypatch):
    import matplotlib.pyplot as plt
    rgb = np.array([colouring.SEASONS(m / 11)[:3] for m in range(12)])
    gaps = np.abs(rgb[:, None] - rgb[None]).sum(axis=2) + np.eye(12)
    assert gaps.min() > 0.08, "no two months share a colour (January is not December)"
    seen, real = [], render_editorial.PathPatch

    def spy(path, *a, **k):
        if k.get("zorder") == 5:
            seen.append(tuple(k["edgecolor"]))
        return real(path, *a, **k)

    monkeypatch.setattr(render_editorial, "PathPatch", spy)
    # five streets at the same weight: one coloured late in the year, four early; the four go on top
    segs = np.array([[[0, i], [1, i]] for i in range(5)], dtype=float)
    values = np.array([0.9, 0.2, 0.2, 0.2, 0.2])
    fig, ax = plt.subplots()
    render_editorial.draw_routes(ax, segs, np.full(5, 3), 3, colouring.SEASONS, 1.0, values=values)
    plt.close(fig)
    top = render_editorial.LEVELS - 1
    assert len(seen) == 2 and seen[-1] == pytest.approx(colouring.SEASONS(round(0.2 * top) / top)), "commonest last"


def test_glyphs_are_not_snapped_to_pixels(monkeypatch, tmp_path):
    """A text of straight strokes only (the F of the month key, a T, a +) would be snapped to whole pixels and lose its
    thin stems at small sizes: every glyph path is drawn unsnapped."""
    snaps, real = [], render_editorial.PathPatch

    def spy(path, *a, **k):
        if "transform" in k:  # text: the routes and the basemap are drawn in data coordinates
            snaps.append(k.get("snap"))
        return real(path, *a, **k)

    monkeypatch.setattr(render_editorial, "PathPatch", spy)
    render_editorial.render([(0, LineString([(100, 100), (900, 700)]))], (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850),
                            STATS, str(tmp_path / "p.png"), dpi=20, city="X", year="2025", lang="es",
                            color_by="month", activities=[_run(0, dt.datetime(2025, 2, 1, tzinfo=dt.timezone.utc),
                                                               [(0, 0), (1, 0)])], tracks_xy=[np.zeros((2, 2))])
    assert len(snaps) > 20 and set(snaps) == {False}


def test_a_colour_mode_without_activities_falls_back_to_one_ink(tmp_path):
    info = render_editorial.render([(0, LineString([(100, 100), (900, 700)]))], (EMPTY, EMPTY, EMPTY),
                                   (0, 1000, 0, 850), STATS, str(tmp_path / "p.png"), dpi=20, city="X", year="2025",
                                   color_by="pace")
    assert info["legend"] == {"mode": "single", "fallback": "pace"}, "not a 0:00-0:01/km key"
