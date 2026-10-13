"""The trail print (one route on a topographic map): its numbers, its summits, its privacy and its command line."""
import datetime as dt
import glob
import itertools
import os
from xml.etree import ElementTree

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from pinless import atlas, cli, ingest, landcover, race, relief, render_editorial, render_trail, trail
from pinless.ingest import Activity
from test_race_graphs import EMPTY, HOME, POOL, _line, _lonlat, _loop

UTC = dt.timezone.utc
START = dt.datetime(2025, 7, 12, 7, 0, tzinfo=UTC)


def _hike(xy, i="hike", kind="Hike", speed=1.1, name="", start=START, ele=None):
    xy = np.asarray(xy, dtype=float)
    d = render_trail.along(xy)
    heights = list(1500 + 800 * np.sin(np.pi * d / d[-1])) if ele is None else ele
    return Activity(id=str(i), type=kind, start=start, lonlat=_lonlat(xy), source="x", times=list(d / speed),
                    ele=heights, name=name)


# --- numbers -----------------------------------------------------------------------------------------------------------

def test_route_type_tells_a_loop_an_out_and_back_and_a_point_to_point():
    rng = np.random.default_rng(1)
    assert render_trail.route_type(_loop(0, 0, 1500, n=400)) == "loop"
    out = _line((0, 0), (4000, 0), step=20.0)
    back = out[::-1] + rng.normal(0, 10, out.shape)  # the way back, 10 m of GPS noise off the way out
    noisy = np.vstack([out + rng.normal(0, 10, out.shape), back])
    assert render_trail.route_type(noisy) == "out_and_back", "its ends meet, but it retraces itself"
    assert render_trail.route_type(_line((0, 0), (6000, 0), (6000, 3000))) == "point_to_point"
    lollipop = np.vstack([_line((0, 0), (4000, 0)), _loop(4000, 0, 300, n=100)[1:], _line((4000, 0), (0, 0))[1:]])
    assert render_trail.route_type(lollipop) == "out_and_back", "a short loop at the far end of a long stick"
    wide = np.vstack([_line((0, 0), (4000, 0)), _line((4000, 0), (4000, 300), (0, 300), (0, 0))])
    assert render_trail.route_type(wide) == "loop", "back along a parallel trail 300 m away: a loop"
    assert render_trail.route_type(_line((0, 0), (3000, 0), (3000, 30), (0, 30))) == "out_and_back", "30 m apart"
    assert render_trail.route_type(_line((0, 0), (3000, 0), (3000, 260), (0, 260))) == "point_to_point"
    assert render_trail.route_type(_line((0, 0), (2000, 0), (2000, 2000), (240, 0))) == "loop", "ends 240 m apart"
    assert render_trail.route_type(_line((0, 0), (2000, 0), (2000, 2000), (0, 260))) == "point_to_point"


def test_moving_time_leaves_out_long_pauses_only():
    rng = np.random.default_rng(7)
    walk = np.column_stack([np.arange(1800) * 1.2, np.zeros(1800)])
    first = walk
    stop = first[-1] + rng.normal(0, 3, (600, 2))  # ten minutes at the summit, the fix wandering a few metres
    second = first[-1] + [1.2, 0] + walk
    short = second[-1] + rng.normal(0, 3, (100, 2))  # a stop under two minutes is part of the walk
    third = second[-1] + [1.2, 0] + walk
    xy = np.vstack([first, stop, second, short, third])
    t = np.arange(len(xy), dtype=float)
    secs, paused = render_trail.moving_time(xy, t)
    # the ten minutes go, less the 15 s at each edge where the 60 s window still sees walking; the short stop stays
    assert paused and secs == pytest.approx(len(xy) - 1 - 570, abs=20)
    secs, paused = render_trail.moving_time(walk, np.arange(1800.0))
    assert not paused and secs == 1799.0, "no pause: the elapsed time"
    gap = np.vstack([walk, walk[-1:] + [5, 0], walk + walk[-1] + [6, 0]])  # auto-pause: one 15-minute, 5 m step
    t = np.concatenate([np.arange(1800.0), [2700.0], 2701 + np.arange(1800.0)])
    secs, paused = render_trail.moving_time(gap, t)
    assert paused and secs == pytest.approx(t[-1] - 900, abs=40)
    assert render_trail.moving_time(walk, np.zeros(1800)) == (None, False), "a frozen clock: no time"
    # a ten-minute scramble: 0.25 m/s over the ground, 0.2 m/s up; slow on the map, but no pause
    scramble = np.column_stack([np.arange(600) * 0.25, np.zeros(600)])
    climb = list(2000 + np.arange(600) * 0.2)
    assert render_trail.moving_time(scramble, np.arange(600.0), climb) == (599.0, False)
    secs, paused = render_trail.moving_time(scramble, np.arange(600.0))
    assert paused and secs < 100, "without the heights it reads as a stop"
    secs, paused = render_trail.moving_time(xy[:4000], np.arange(4000.0), list(np.full(4000, 1500.0)))
    assert paused and secs == pytest.approx(3999 - 570, abs=20), "flat ground: the heights change nothing"
    assert render_trail.fmt_hours(4 * 3600 + 12 * 60 + 31) == "4:13" and render_trail.fmt_hours(52 * 60) == "0:52"


def _hairpins(leg_m, n_legs, speed, grade=0.18, hill=0.7):
    """A steady climb on hairpins across a hillside, recorded every second with heights: each leg runs mostly across
    the slope, climbs leg_m * grade and gains leg_m * grade / hill up the fall line."""
    rise = leg_m * grade
    up = rise / hill
    across = np.sqrt(leg_m ** 2 - up ** 2 - rise ** 2)
    corners = [(0.0, 0.0, 1000.0)]
    for i in range(n_legs):
        x, y, z = corners[-1]
        corners.append((x + (across if i % 2 == 0 else -across), y + up, z + rise))
    corners = np.array(corners)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(corners, axis=0), axis=1))])
    t = np.arange(0, s[-1] / speed, 1.0)
    pts = np.column_stack([np.interp(t * speed, s, corners[:, i]) for i in range(3)])
    return pts[:, :2], t, list(pts[:, 2])


def test_a_steady_climb_on_tight_hairpins_is_not_a_pause():
    for leg, speed in ((15, 0.55), (20, 0.6), (30, 0.6)):
        xy, t, ele = _hairpins(leg, 120, speed)
        assert render_trail.moving_time(xy, t, ele) == (t[-1], False), f"{leg} m legs at {speed} m/s, never stopping"
    # a whole day: an hour along the valley, 50 minutes up 15 m hairpins, half an hour's lunch at the top (the fix
    # wandering a few metres), 70 minutes down: 3:00 moving of 3:30
    rng = np.random.default_rng(0)
    valley = np.column_stack([np.arange(3600) * 1.2, np.zeros(3600)])
    climb_xy, _, climb_z = _hairpins(15, 400, 0.55)
    climb_xy, climb_z = climb_xy[:3000] - climb_xy[0] + valley[-1] + [1.2, 0], np.asarray(climb_z[:3000])
    lunch = climb_xy[-1] + rng.normal(0, 3, (1800, 2))
    down = lunch[-1] + np.column_stack([np.arange(1, 4201) * 1.0, np.zeros(4200)])
    xy = np.vstack([valley, climb_xy, lunch, down])
    ele = np.concatenate([np.full(3600, 1000.0), climb_z, climb_z[-1] + rng.normal(0, 2, 1800),
                          climb_z[-1] - np.arange(1, 4201) * 0.07])
    secs, paused = render_trail.moving_time(xy, np.arange(len(xy), dtype=float), list(ele))
    assert paused and secs == pytest.approx(3 * 3600, abs=60)
    mask = render_trail.paused_steps(xy, np.arange(len(xy), dtype=float), list(ele))
    assert mask.sum() == pytest.approx(30 * 60, abs=60) and not mask[:6600 - 30].any(), "only the lunch is a pause"
    assert not mask[6600 + 1800 + 30:].any()


def test_summit_heights_are_read_from_osm_text():
    cases = {"2428": 2428.0, "2428.5": 2428.5, "2428 m": 2428.0, " 2428m ": 2428.0, "2428,5": 2428.5,
             "7966 ft": 7966 * 0.3048, "about 2400": None, "": None, None: None, float("nan"): None, "12000": None}
    for text, want in cases.items():
        got = render_trail.parse_ele(text)
        assert got == (pytest.approx(want) if want is not None else None), text


# --- summits -----------------------------------------------------------------------------------------------------------

EXTENT = (0.0, 10000.0, 0.0, 10000.0 / render_trail.map_aspect())


def _frame():
    return box(render_trail.LEFT, render_trail.MAP_TOP, render_trail.RIGHT, render_trail.MAP_TOP + render_trail.MAP_H)


def test_summits_are_drawn_inside_the_map_highest_first_and_never_overlap():
    rng = np.random.default_rng(3)
    x0, x1, y0, y1 = EXTENT
    peaks = [(float(x), float(y), f"Peak {i}", float(h)) for i, (x, y, h) in
             enumerate(zip(rng.uniform(x0 - 3000, x1 + 3000, 80), rng.uniform(y0 - 3000, y1 + 3000, 80),
                           rng.uniform(1200, 2500, 80)))]
    peaks += [(5000.0, 4000.0, "", 3000.0), (5000.0, 4000.0, None, 3000.0)]  # unnamed: never drawn
    course = [_loop(5000, 4500, 3000, n=300)]
    for k in (1.0, 210 / 297, 500 / 297):
        placed = render_trail.place_peaks(peaks, EXTENT, k, course)
        assert 0 < len(placed) <= 6
        texts = [p[2] for p in placed]
        heights = [next(h for _, _, n, h in peaks if render_trail.peak_label(n, h) == t) for t in texts]
        assert heights == sorted(heights, reverse=True), "highest first"
        where = {render_trail.peak_label(n, h): (x, y) for x, y, n, h in peaks if n}
        frame = _frame()
        route = shapely.union_all([LineString(np.column_stack(render_trail._grid_xy(EXTENT, *c.T))) for c in course])
        boxes = [box(*p[3]) for p in placed]
        for (gx, gy, text, bounds, (lx, ly)), b in zip(placed, boxes):
            assert frame.contains(b) and frame.contains(Point(gx, gy)), "nothing outside the map"
            assert bounds[:2] == (lx, ly)
            assert b.distance(route) >= render_trail.ROUTE_CLEAR_MM - 1e-9, "clear of the casing, with a gap"
        for a, b in itertools.combinations(boxes, 2):
            assert not a.intersects(b), "no two labels overlap"
        assert all(x0 < where[t][0] < x1 and y0 < where[t][1] < y1 for t in texts), "only summits inside the map"


def test_a_label_that_cannot_be_placed_is_skipped():
    twins = [(5000.0, 4000.0, "Twin North", 2400.0), (5000.0, 4010.0, "Twin South", 2390.0)]
    placed = render_trail.place_peaks(twins, EXTENT, 1.0, [])
    assert [p[2] for p in placed] == ["TWIN NORTH 2400 m"], "the second triangle would sit on the first label"
    y1 = EXTENT[3]
    edge = [(9990.0, 4000.0, "At The Edge", 2000.0), (20.0, 20.0, "Corner", 2000.0), (5000.0, y1 - 20, "Top", 2000.0),
            (5000.0, 20.0, "Bottom", 2000.0)]
    assert render_trail.place_peaks(edge, EXTENT, 1.0, []) == [], "too close to the edge for its triangle"
    unknown = [(3000.0, 3000.0, "No Height", None), (7000.0, 3000.0, "Known", 1000.0)]
    assert [p[2] for p in render_trail.place_peaks(unknown, EXTENT, 1.0, [])] == ["KNOWN 1000 m", "NO HEIGHT"]
    many = [(1000.0 + 1100 * i, 2000.0 + 600 * (i % 3), f"P{i}", 2000.0 + i) for i in range(8)]
    assert len(render_trail.place_peaks(many, EXTENT, 1.0, [])) == 6


def test_a_label_moves_to_stay_clear_of_the_edge_the_course_and_other_labels():
    mm = (EXTENT[1] - EXTENT[0]) / (render_trail.RIGHT - render_trail.LEFT)  # metres per millimetre of the grid

    def side(placed):
        gx, _, _, (x0, _, x1, _), _ = placed
        return "right" if x0 > gx else "left" if x1 < gx else "above or below"

    (p,) = render_trail.place_peaks([(EXTENT[1] - 8 * mm, 4000.0, "Near The Right Edge", 2000.0)], EXTENT, 1.0, [])
    assert side(p) == "left", "its label would leave the map on the right"
    wall = [np.array([[5000 + 3 * mm, 0.0], [5000 + 3 * mm, EXTENT[3]]])]  # the course 3 mm right of the summit
    (p,) = render_trail.place_peaks([(5000.0, 4000.0, "By The Course", 2000.0)], EXTENT, 1.0, wall)
    assert side(p) == "left", "its label would cross the course"
    first = (5000.0, 4000.0, "First Summit", 2100.0)
    second = (5000.0 - 4 * mm, 4000.0 - 1.0 * mm, "Second Summit", 2000.0)  # down and left of the first
    a, b = render_trail.place_peaks([first, second], EXTENT, 1.0, [])
    assert side(a) == "right" and side(b) == "left", "its label would run into the first label"
    assert not box(*a[3]).intersects(box(*b[3]))


def test_a_label_goes_where_it_hides_no_lower_summit():
    """Three summits in a row, like the Tre Cime: the highest in the middle with a long name in two languages. Its
    label on the right would hide the third summit's triangle, on the left the second's, and above or below one of
    them, so it goes to a corner, and all three are named (the first place free of the map's edge, the course and
    the labels drawn before would leave the third summit unnamed)."""
    mm = (EXTENT[1] - EXTENT[0]) / (render_trail.RIGHT - render_trail.LEFT)
    row = [(5000.0, 4000.0, "Cima Grande - Grosse Zinne", 2999.0),
           (5000.0 - 12 * mm, 4000.0 - 1 * mm, "Cima Ovest - Westliche Zinne", 2973.0),
           (5000.0 + 12 * mm, 4000.0 + 1 * mm, "Cima Piccola - Kleine Zinne", 2857.0)]
    placed = render_trail.place_peaks(row, EXTENT, 1.0, [])
    assert [p[2] for p in placed] == ["CIMA GRANDE - GROSSE ZINNE 2999 m", "CIMA OVEST - WESTLICHE ZINNE 2973 m",
                                      "CIMA PICCOLA - KLEINE ZINNE 2857 m"]
    gx, gy, _, (x0, y0, x1, y1), _ = placed[0]
    assert x0 > gx and y0 > gy, "right and below: the first corner that hides neither"
    boxes = [box(*p[3]) for p in placed]
    for a, b in itertools.combinations(boxes, 2):
        assert not a.intersects(b)


def test_the_other_tops_of_a_mountain_are_not_named():
    """Two summits that share a base name within SAME_PEAK_M are one mountain: the higher is named by the base name
    alone. Farther apart, or with another base name, both are named."""
    m = render_trail.SAME_PEAK_M
    top = (5000.0, 4000.0, "Aiguilles Crochues - Sommet S", 2840.0)
    near = (5000.0 + 0.5 * m, 4000.0 + 0.2 * m, "Aiguilles Crochues", 2837.0)
    assert [p[2] for p in render_trail.place_peaks([near, top], EXTENT, 1.0, [])] == ["AIGUILLES CROCHUES 2840 m"]
    far = (5000.0 + 1.3 * m, 4000.0 + 2.5 * m, "Aiguilles Crochues", 2837.0)
    assert [p[2] for p in render_trail.place_peaks([far, top], EXTENT, 1.0, [])] == [
        "AIGUILLES CROCHUES - SOMMET S 2840 m", "AIGUILLES CROCHUES 2837 m"]
    other = (5000.0 + 0.5 * m, 4000.0 + 0.2 * m, "Aiguille de la Persévérance", 2837.0)
    assert len(render_trail.place_peaks([other, top], EXTENT, 1.0, [])) == 2
    assert render_trail.base_name("Cima Grande - Große Zinne") == "Cima Grande"
    assert render_trail.base_name("Yr Wyddfa / Snowdon") == "Yr Wyddfa"
    assert render_trail.base_name("Tête-à-Tête") == "Tête-à-Tête", "a hyphen inside a name is not a separator"


def test_a_label_keeps_its_gap_from_the_course_and_a_wandering_stop_does_not_stall_it():
    import time

    mm = (EXTENT[1] - EXTENT[0]) / (render_trail.RIGHT - render_trail.LEFT)
    clear = render_trail.ROUTE_CLEAR_MM
    assert clear == pytest.approx(render_trail.CASING_PT * render_trail.PT_MM / 2 + render_trail.ROUTE_GAP_MM)
    label_w = render_trail.place_text("GAP 2000 m", 0, 0, 1.0, render_trail.PEAK_PT, render_trail.TEXT)[2]
    x_right = 5000 + (render_trail.TRI_W / 2 + 0.7 + label_w) * mm  # where a label on the right ends
    for off, side in ((clear - 0.1, "left"), (clear + 0.1, "right")):  # the course this far beyond the label's end
        wall = [np.array([[x_right + off * mm, 0.0], [x_right + off * mm, EXTENT[3]]])]
        (p,) = render_trail.place_peaks([(5000.0, 4000.0, "Gap", 2000.0)], EXTENT, 1.0, wall)
        assert ("right" if p[3][0] > p[0] else "left") == side, off
    # twenty minutes standing by the course, the fix wandering a few metres every second: thousands of crossings
    rng = np.random.default_rng(4)
    walk = np.column_stack([np.linspace(1000, 9000, 4000), np.full(4000, 3000.0)])
    stop = walk[2000] + rng.normal(0, 3, (1200, 2))
    course = [np.vstack([walk[:2000], stop, walk[2000:]])]
    t0 = time.perf_counter()
    placed = render_trail.place_peaks([(5000.0, 3000.0 + 12 * mm, "Above", 2100.0)], EXTENT, 1.0, course)
    assert time.perf_counter() - t0 < 10, "a buffer of that course took minutes"
    route = LineString(np.column_stack(render_trail._grid_xy(EXTENT, *course[0].T)))
    assert [p[2] for p in placed] == ["ABOVE 2100 m"] and box(*placed[0][3]).distance(route) >= clear - 1e-9


def test_summit_labels_and_triangles_have_a_halo_in_the_paper_colour():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    placed = render_trail.place_peaks([(5000.0, 4000.0, "Peñalara", 2428.0)], EXTENT, 1.0, [])
    fig, ax = plt.subplots()
    S = render_editorial.STYLES["carbon"]
    render_trail.draw_summits(ax, placed, S, 1.0)
    assert len(ax.patches) == 2, "a triangle and a label"
    for patch in ax.patches:
        (halo,) = patch.get_path_effects()
        assert to_rgba(halo._gc["foreground"]) == to_rgba(S["bg"]) and halo._gc["linewidth"] == render_trail.HALO_PT
        assert halo._gc["joinstyle"] == "round", "a mitred halo sends spikes out of the corners of an M"
        assert to_rgba(patch.get_facecolor()) == to_rgba(S["ink"]), "in the ink: they read over the relief"
    plt.close(fig)


def test_a_round_halo_sends_no_spike_out_of_a_sharp_corner():
    """A label in a black ink with its white halo, on a black page: the white reaches no farther above the top of
    the letters (place_text sets their capital height at the label's top) than half the halo's width. A mitred halo
    sends spikes far above an M, out of the sharp corners where its strokes meet."""
    import matplotlib.pyplot as plt

    k, mm_px = 2.0, 600 / 25.4  # the scale of the label, and pixels per millimetre of the page at 600 dpi
    S = {"ink": "#000000", "bg": "#ffffff"}
    fig = plt.figure(figsize=(2, 1), dpi=600, facecolor="#000000")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 50.8)
    ax.set_ylim(25.4, 0)  # page millimetres, y down
    ax.axis("off")
    label_x, label_y = 7.0, 8.0  # A3-grid mm: the label's top at 16 mm of this page
    render_trail.draw_summits(ax, [(-20.0, 0.0, "MWMWM", None, (label_x, label_y))], S, k)  # its triangle off the page
    fig.canvas.draw()
    white = np.asarray(fig.canvas.buffer_rgba())[..., 0] > 128
    plt.close(fig)
    assert white.any(), "the halo is drawn"
    top_mm = np.flatnonzero(white.any(axis=1))[0] / mm_px
    half = render_trail.HALO_PT * k * render_trail.PT_MM / 2
    assert top_mm >= label_y * k - half - 2 / mm_px, (top_mm, label_y * k - half)


def test_summits_load_from_the_cache_with_their_heights(tmp_path):
    gpd.GeoDataFrame({"name": ["Peñalara", None, "Far"], "ele": ["2428 m", "2000", "1000"]},
                     geometry=[Point(-3.956, 40.851), Point(-3.95, 40.85), Point(0.0, 50.0)],
                     crs=4326).to_file(tmp_path / "peaks.gpkg")
    crs = trail.render.utm_crs(-3.95, 40.85)
    got = trail.load_peaks(str(tmp_path), (40.80, -4.00, 40.90, -3.90), crs)
    assert [(n, e) for _, _, n, e in got] == [("Peñalara", 2428.0)], "named, inside the box, height in metres"
    assert trail.load_peaks(str(tmp_path / "none"), (40.80, -4.00, 40.90, -3.90), crs) == []
    gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326).to_file(tmp_path / "empty.gpkg")
    (tmp_path / "empty.gpkg").replace(tmp_path / "peaks.gpkg")
    assert trail.load_peaks(str(tmp_path), (40.80, -4.00, 40.90, -3.90), crs) == []


# --- the page ----------------------------------------------------------------------------------------------------------

def _texts(monkeypatch):
    """Every text the page draws, as (value, x0, y, x1, y + height) in the layout grid."""
    boxes = []
    real = atlas.place_text

    def spy(value, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
        out = real(value, x, y, k, pt, font, align, max_w, height, clear)
        w, h = out[2], out[3]
        x0 = x - (w if align == "right" else w / 2 if align == "center" else 0)
        boxes.append((str(value), x0, y, x0 + w, y + h))
        return out

    def drawn(value, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
        out = real(value, x, y, k, pt, font, align, max_w, height, clear)
        if (x, y) != (0, 0):  # the summit labels, drawn with their halo (at 0, 0 a label is only measured)
            boxes.append((str(value), x, y, x + out[2], y + out[3]))
        return out

    monkeypatch.setattr(atlas, "place_text", spy)
    monkeypatch.setattr(render_trail, "place_text", drawn)
    return boxes


def _page(tmp_path, name="t.png", ele=None, **kw):
    xy = _loop(0, 0, 2000, n=600)
    act = _hike(xy, ele=ele)
    extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
    args = dict(dpi=20, title="Peñalara", place="Comunidad de Madrid, Spain", name="Lucia", date=START)
    args.update(kw)
    info = render_trail.render(act, xy, [(0, LineString(xy[30:-30]))], (EMPTY, EMPTY, EMPTY), extent,
                               str(tmp_path / name), **args)
    return info, act, xy


def test_the_high_point_is_marked_on_the_profile(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    info, act, xy = _page(tmp_path)
    d, e = render_trail.profile(xy, act.ele)
    assert info["high"] == pytest.approx(e.max()) and info["high"] == pytest.approx(2300, abs=2)
    label = [b for b in boxes if b[0] == f"{e.max():.0f} m"]
    assert len(label) == 1, "one altitude by the dot"
    _, x0, y_, x1, y1 = label[0]
    top, bottom = render_trail.PROFILE_TOP, render_trail.PROFILE_TOP + render_trail.PROFILE_H
    assert top < y_ and y1 < bottom, "inside the profile band, under the elevation header"
    want = render_trail.LEFT + d[int(np.argmax(e))] / d[-1] * (render_trail.RIGHT - render_trail.LEFT)
    assert (x0 + x1) / 2 == pytest.approx(want, abs=0.5), "above the highest point"
    cells = {b[0] for b in boxes if b[2] > render_trail.NUMBERS_TOP}
    assert {"2300", "LOOP", "HIGH POINT", "ROUTE TYPE", "LENGTH", "ELEVATION GAIN", "TIME"} <= cells
    assert info["route_type"] == "loop" and not info["paused"] and info["secs"] == pytest.approx(act.times[-1], abs=1)
    assert any(b[0] == "TRAIL" for b in boxes) and any(b[0] == "PEÑALARA" for b in boxes)


def test_the_kilometre_marks_never_run_into_each_other(monkeypatch, tmp_path):
    assert [render_trail.km_step(km * 1000) for km in (5, 12, 13, 24, 25, 60, 61, 120, 240, 840)] == [
        1, 1, 2, 2, 5, 5, 10, 10, 20, 100]
    boxes = _texts(monkeypatch)
    y = render_trail.PROFILE_TOP + render_trail.PROFILE_H + 2
    for km in (13, 300, 840):
        xy = _line((0, 0), (km * 1000, 0), step=km * 2.0)
        act = _hike(xy, speed=1.0)
        extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
        del boxes[:]
        render_trail.render(act, xy, [(0, LineString(xy))], (EMPTY, EMPTY, EMPTY), extent, str(tmp_path / "k.png"),
                            dpi=20)
        marks = sorted((b for b in boxes if b[2] == y), key=lambda b: b[1])
        assert 5 <= len(marks) <= render_trail.KM_MARKS + 1, km
        assert all(a[3] + 1 < b[1] for a, b in zip(marks, marks[1:])), f"{km} km: marks overlap"


def test_spanish_labels(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    _page(tmp_path, lang="es")
    texts = {b[0] for b in boxes}
    assert {"RUTA", "CIRCULAR", "TIPO DE RUTA", "COTA MÁXIMA", "DESNIVEL POSITIVO", "DISTANCIA", "TIEMPO",
            "CADA RUTA DEJA HUELLA.", "PERFIL", "JUL 2025"} <= texts


def test_the_elevation_key_and_the_profile_give_one_range_and_a_flat_route_is_not_coloured(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    info, act, xy = _page(tmp_path, color_by="elevation")  # the drawn piece leaves out 30 points at each end
    e = render_trail.profile(xy, act.ele)[1]
    assert (info["legend"]["lo"], info["legend"]["hi"]) == (pytest.approx(e.min()), pytest.approx(e.max()))
    texts = [b[0] for b in boxes]
    assert f"{e.min():.0f} – {e.max():.0f} m" in texts and f"{e.min():.0f} m" in texts and f"{e.max():.0f} m" in texts
    rng = np.random.default_rng(5)
    flat = list(2.0 + rng.normal(0, 0.6, 600))  # a walk in a river delta: the altimeter's noise and nothing else
    info, _, _ = _page(tmp_path, color_by="elevation", ele=flat)
    assert info["legend"] == {"mode": "single", "flat": True}
    info, _, _ = _page(tmp_path, color_by="elevation", ele=list(2.0 + np.linspace(0, 25, 600)))
    assert info["legend"]["mode"] == "elevation", "a climb of 25 m is coloured"


def test_summits_on_the_page_stay_inside_the_map(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    # the loop runs round (-2000, 0) at 2000 m; the map reaches about 550 m beyond it
    peaks = [(-2000.0, 0.0, "Inside", 2100.0), (99999.0, 0.0, "Far Away", 3000.0), (-2000.0, 2000.0, "On The Loop",
                                                                                      2050.0)]
    info, _, _ = _page(tmp_path, peaks=peaks)
    assert [s[2] for s in info["summits"]] == ["INSIDE 2100 m", "ON THE LOOP 2050 m"]
    assert not any("FAR AWAY" in b[0] for b in boxes), "a summit off the map is never drawn"
    labels = [b for b in boxes if b[0] in ("INSIDE 2100 m", "ON THE LOOP 2050 m")]
    assert len(labels) == 2 and all(_frame().contains(box(*b[1:])) for b in labels)


def test_every_style_draws_and_colours_follow_the_choice(tmp_path):
    for style in render_editorial.STYLES:
        info, _, _ = _page(tmp_path, name=f"{style}.png", style=style, size="A4")
        assert (tmp_path / f"{style}.png").stat().st_size > 0 and info["legend"]["mode"] == "single"
    info, _, _ = _page(tmp_path, color_by="elevation", svg_path=str(tmp_path / "t.svg"), pdf_path=str(tmp_path / "t.pdf"))
    assert info["legend"]["mode"] == "elevation" and (tmp_path / "t.svg").stat().st_size > 0
    info, _, _ = _page(tmp_path, color_by="pace")
    assert info["legend"]["mode"] == "pace"


def test_a_stop_does_not_move_the_pace_legend(tmp_path):
    rng = np.random.default_rng(2)
    n = 5400  # 90 minutes at 1.0 to 1.5 m/s (11:07 to 16:40 per km) on a wide curve
    speed = 1.25 + 0.25 * np.sin(np.arange(n) / 600)
    walk = np.cumsum(np.column_stack([np.cos(np.arange(n) / 1500), np.sin(np.arange(n) / 1500)]) * speed[:, None], 0)
    ends = []
    for stop in (0, 1200):  # then twenty minutes standing half way, the fix wandering a few metres every second
        xy = walk if not stop else np.vstack([walk[:n // 2], walk[n // 2] + rng.normal(0, 3, (stop, 2)), walk[n // 2:]])
        act = Activity(id="w", type="Hike", start=START, lonlat=_lonlat(xy), source="x",
                       times=list(np.arange(len(xy), dtype=float)))
        extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
        info = render_trail.render(act, xy, [(0, LineString(xy[300:-300]))], (EMPTY, EMPTY, EMPTY), extent,
                                   str(tmp_path / "p.png"), dpi=20, color_by="pace")
        assert info["legend"]["mode"] == "pace" and info["paused"] == bool(stop)
        ends.append((info["legend"]["lo"], info["legend"]["hi"]))
    (lo0, hi0), (lo1, hi1) = ends
    assert lo0 == pytest.approx(1000 / 1.5, rel=0.03) and hi0 == pytest.approx(1000 / 1.0, rel=0.03), "the walk's"
    assert lo1 == pytest.approx(lo0, rel=0.03) and hi1 == pytest.approx(hi0, rel=0.03), "the stop changes nothing"
    assert render_trail.weighted_percentiles([1, 2, 3, 100], [1, 1, 1, 0], (0, 100)) == [1.0, 3.0]
    assert render_trail.weighted_percentiles([5, 1], [0, 0], (50,)) == [3.0], "no weight: plain percentiles"


def test_the_trail_map_draws_its_paths_over_a_stronger_relief():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    def lstar(colour):
        return render_editorial._lstar(to_rgb(colour))

    for name, S in render_editorial.STYLES.items():
        before = repr(S)
        T = render_trail.trail_style(S)
        assert repr(S) == before, "STYLES itself is not changed"
        assert min(lstar(S["bg"]), lstar(S["sub"])) < lstar(T["trail"]) < max(lstar(S["bg"]), lstar(S["sub"])), name
        (_, shadow), (light_rgb, light) = T["relief"]
        (_, shadow0), _ = render_editorial.relief_colours(S)
        assert shadow >= shadow0 - 1e-9 and shadow > 0, f"{name}: the course is one bold line, the relief can show more"
        assert light <= render_editorial._keep_alpha(S, light_rgb, 0.6) + 1e-9, "the map under it keeps its contrast"
    own = dict(render_editorial.STYLES["huella"], relief=(((0, 0, 0), 0.1), ((1, 1, 1), 0.0)))
    assert render_trail.trail_style(own)["relief"] == own["relief"], "a style that sets its relief keeps it"
    kinds = ["track", "path", "footway", "residential", "primary", ["path", "steps"], "bridleway", "pedestrian",
             "cycleway", "steps", "corridor", "service", "trunk_link"]
    edges = gpd.GeoDataFrame({"highway": kinds}, geometry=[LineString([(0, i), (5000, i)])
                                                           for i in range(0, 100 * len(kinds), 100)])
    base, roads, trails = render_trail.split_network((edges, EMPTY, EMPTY))
    assert list(base[0]["highway"]) == ["footway", "pedestrian", "cycleway", "steps", "corridor"], \
        "a town's sidewalks, steps and cycleways in the street colours of every print"
    assert list(roads["highway"]) == ["residential", "primary", "service", "trunk_link"]
    assert list(trails["highway"]) == ["track", "path", ["path", "steps"], "bridleway"]
    fig, ax = plt.subplots()
    T = render_trail.trail_style(own)
    render_trail.trail_network(ax, trails, (0, 5000, -100, 1400), T, 1.0, 261)
    solid, dashed = ax.patches
    assert solid.get_linestyle() == "solid" and dashed.get_linestyle() == render_trail.FOOTPATH_DASH, \
        "tracks solid, footpaths dashed"
    assert solid.zorder == dashed.zorder == 3.6, "over the relief (3.5), under the course"
    assert len(solid.get_path().vertices) == 2 and len(dashed.get_path().vertices) == 6
    render_trail.trail_roads(ax, roads, (0, 5000, -100, 1400), T, 1.0, 261)
    minor, casing, major = ax.patches[2:]
    assert (minor.zorder, casing.zorder, major.zorder) == (3.61, 3.62, 3.63), "over the paths, under the course"
    assert to_rgb(casing.get_edgecolor()[:3]) == to_rgb(own["bg"]) and casing.get_linewidth() > major.get_linewidth()
    assert len(minor.get_path().vertices) == 4 and len(major.get_path().vertices) == 4
    plt.close(fig)


def test_woods_are_only_a_step_stronger_than_on_a_city_poster():
    """The land OpenStreetMap leaves without a wood or a meadow (the treeless summits, the very subject of the print)
    must not read as a blank cut-out: the woods stay close to the paper, and the relief carries the terrain."""
    from matplotlib.colors import to_rgb

    for name, S in render_editorial.STYLES.items():
        bg = render_editorial._lstar(to_rgb(S["bg"]))
        poster = abs(render_editorial._lstar(to_rgb(S["green"])) - bg)
        trail_ = abs(render_editorial._lstar(to_rgb(render_trail.trail_style(S)["green"])) - bg)
        assert poster <= trail_ <= 1.35 * poster + 0.3, name


def _lstar_image(path):
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(y > 216 / 24389, 116 * np.cbrt(y) - 16, y * 24389 / 27)


def test_roads_read_at_least_as_strongly_as_footpaths(tmp_path):
    """The contrast of what disappears when a layer is left out (the median change of L* over the pixels it covers):
    a paved road, major or minor, is never fainter than a footpath or a track."""
    from matplotlib.colors import to_rgb

    kinds = ["primary", "path", "residential", "track", "secondary", "path", "unclassified", "track"]
    edges = gpd.GeoDataFrame({"highway": kinds}, geometry=[LineString([(-3000, y), (3000, y)])
                                                           for y in np.linspace(-2500, 2500, len(kinds))])
    xy = _line((-200, -2900), (200, -2900))
    act = _hike(xy)
    extent = (-3500.0, 3500.0, -3500.0 / render_trail.map_aspect(), 3500.0 / render_trail.map_aspect())
    for style in ("huella", "carbon", "blueprint", "lavender"):
        out = {}
        for name, drop in (("all", ()), ("no major", ("primary", "secondary")),
                           ("no minor", ("residential", "unclassified")), ("no paths", ("path", "track"))):
            path = tmp_path / f"{style}_{name}.png"
            render_trail.render(act, xy, [(0, LineString(xy))], (edges[~edges["highway"].isin(drop)], EMPTY, EMPTY),
                                extent, str(path), style=style, dpi=60)
            out[name] = _lstar_image(path)

        def contrast(name):
            d = np.abs(out["all"] - out[name])
            return float(np.median(d[d > 0.5]))

        paths = contrast("no paths")
        assert contrast("no major") >= paths and contrast("no minor") >= paths, style
    for style, S in render_editorial.STYLES.items():
        T = render_trail.trail_style(S)
        bg = render_editorial._lstar(to_rgb(S["bg"]))
        trail_l, course_l = (abs(render_editorial._lstar(to_rgb(c)) - bg) for c in (T["trail"], S["accent"]))
        minor_l, major_l = (abs(render_editorial._lstar(to_rgb(T[f"road_{c}"])) - bg) for c in ("minor", "major"))
        assert trail_l - 0.3 <= minor_l <= major_l, f"{style}: roads never fainter than paths, major the strongest"
        assert major_l <= max(trail_l, 0.75 * course_l) + 0.3, f"{style}: the course leads (to 8-bit colour)"


# --- privacy and the command line --------------------------------------------------------------------------------------

def _trail_world(monkeypatch, acts, captured, tmp_path, prepared=None):
    if acts is not None:
        monkeypatch.setattr(ingest, "read_export", lambda path: (acts, ingest.IngestReport()))
    monkeypatch.setattr(trail.area, "fetch_basemap", lambda bbox, log=print, extra=(): captured.update(extra=extra)
                        or str(tmp_path))
    monkeypatch.setattr(trail.render, "load_basemap", lambda folder, bbox, crs: (EMPTY, EMPTY, EMPTY))
    monkeypatch.setattr(trail.area, "place_region", lambda lat, lon, lang="en": ("Rascafría", "Comunidad de Madrid",
                                                                                 "España" if lang == "es" else "Spain"))
    calls = [] if prepared is None else prepared
    monkeypatch.setattr(relief, "prepare_many", lambda maps, log=print: calls.append(maps) or [None] * len(maps))
    monkeypatch.setattr(landcover, "prepare_many", lambda maps, log=print: [None] * len(maps))
    real = render_trail.render

    def spy(*args, **kw):
        info = real(*args, **kw)
        captured.update(kw, track=args[1], pieces=args[2], extent=args[4], info=info)
        return info

    monkeypatch.setattr(trail.render_trail, "render", spy)
    return calls


def test_a_trail_from_home_is_refused_or_cut_far_from_it_with_its_numbers(monkeypatch, tmp_path, capsys):
    pool = [_hike(t, i, kind="Run") for i, t in enumerate(POOL)]  # years of runs from home teach where it is
    from_home = _hike(_loop(0, 0, 2500, n=500), "from-home", name="Walk from home")
    captured = {}
    _trail_world(monkeypatch, pool + [from_home], captured, tmp_path)
    base = ["x.zip", "--layout", "trail", "--activity", "from-home", "--dpi", "20", "--out", str(tmp_path / "t.png")]
    assert cli.main(base) == 2 and not captured, "a hike from home is refused by default"
    assert "starts or ends at a private place" in capsys.readouterr().out
    assert cli.main(base + ["--allow-private-ends"]) == 0
    lo, hi = captured["stretch"]
    assert captured["pieces"][0][1].distance(HOME) >= 1.6 * 500, "nothing within reach of the zone"
    assert captured["info"]["km"] == pytest.approx((hi - lo) / 1000, rel=0.01), "numbers for the drawn stretch"
    assert captured["info"]["km"] < render_trail.along(captured["track"])[-1] / 1000 - 1.0
    assert captured["title"] == "Rascafría" and captured["place"] == "Comunidad de Madrid, Spain"
    out = capsys.readouterr().out
    assert "not printed unless you pass it with --title" in out and "Walk from home" not in out


def test_a_hike_from_home_is_refused_with_type_hike_because_of_the_runs_from_home(monkeypatch, tmp_path, capsys):
    pool = [_hike(t, i, kind="Run") for i, t in enumerate(POOL)]
    captured = {}
    _trail_world(monkeypatch, pool + [_hike(_loop(0, 0, 2500, n=500), "from-home")], captured, tmp_path)
    assert cli.main(["x.zip", "--layout", "trail", "--type", "Hike", "--dpi", "20", "--out",
                     str(tmp_path / "t.png")]) == 2 and not captured, "the places come from every activity, not the hikes"
    assert "[privacy] STOP" in capsys.readouterr().out


def test_with_private_ends_allowed_every_number_describes_the_drawn_stretch(monkeypatch, tmp_path, capsys):
    from pinless.render_race import climb, restrict

    boxes = _texts(monkeypatch)
    pool = [_hike(t, i, kind="Run") for i, t in enumerate(POOL)]
    xy = _loop(0, 0, 2500, n=800)
    d = render_trail.along(xy)
    ele = list(1500 + 900 * np.exp(-((d - 300) / 400) ** 2) + 100 * np.sin(2 * np.pi * d / 3000))  # summit near home
    t = d / 1.1
    t[len(t) // 2:] += 1800  # a half-hour stop half way: the moving time is the stretch's, not the whole walk's
    walk = Activity(id="from-home", type="Hike", start=START, lonlat=_lonlat(xy), source="x", times=list(t), ele=ele)
    captured = {}
    _trail_world(monkeypatch, pool + [walk], captured, tmp_path)
    assert cli.main(["x.zip", "--layout", "trail", "--activity", "from-home", "--allow-private-ends", "--color-by",
                     "elevation", "--dpi", "20", "--out", str(tmp_path / "t.png")]) == 0
    out = capsys.readouterr().out
    assert "(numbers for the drawn stretch only)" in out and "public trailhead" not in out
    track, info = captured["track"], captured["info"]
    sx, st, se = restrict(track, captured["times"], captured["ele"], captured["stretch"])
    whole = (track, captured["times"], captured["ele"])
    e, e_all = render_trail.profile(sx, se)[1], render_trail.profile(track, captured["ele"])[1]
    assert info["route_type"] == render_trail.route_type(sx) == "point_to_point" != render_trail.route_type(track)
    assert info["secs"] == pytest.approx(render_trail.moving_time(sx, st, se)[0])
    assert info["secs"] != pytest.approx(render_trail.moving_time(*whole)[0], abs=60)
    assert info["high"] == pytest.approx(render_trail.high_point(sx, se)[1]) and info["high"] < e_all.max() - 100
    gain, gain_all = climb(e), climb(e_all)
    assert abs(gain - gain_all) > 50 and f"+{gain:.0f}" in [b[0] for b in boxes]
    assert f"{e.min():.0f} – {e.max():.0f} m" in [b[0] for b in boxes], "the profile of the stretch"
    assert (info["legend"]["lo"], info["legend"]["hi"]) == (pytest.approx(e.min()), pytest.approx(e.max()))
    assert info["legend"]["hi"] < e_all.max() - 100, "the elevation key of the stretch"


def test_the_place_is_looked_up_from_the_middle_of_the_drawn_stretch(monkeypatch, tmp_path):
    xy = _line((0, 0), (0, 12000), step=20.0)  # from home, straight north; a big --zone at home
    walk = _hike(xy, "north")
    captured, asked = {}, []
    _trail_world(monkeypatch, [walk], captured, tmp_path)
    monkeypatch.setattr(trail.area, "place_region", lambda lat, lon, lang="en": asked.append((lat, lon)) or
                        ("Somewhere", "", ""))
    lon0, lat0 = _lonlat([[0, 0]])[0]
    assert cli.main(["x.gpx", "--layout", "trail", "--zone", f"{lat0},{lon0},3000", "--allow-private-ends", "--dpi",
                     "20", "--out", str(tmp_path / "t.png")]) == 0
    piece = np.asarray(captured["pieces"][0][1].coords)
    crs = trail.render.utm_crs(*np.median(np.asarray(walk.lonlat), axis=0))
    inv = trail.Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = inv.transform(*np.median(piece, axis=0))
    lon_w, lat_w = inv.transform(*np.median(captured["track"], axis=0))
    assert asked == [(round(lat, 2), round(lon, 2))], "the middle of what is drawn, rounded to about 1 km"
    assert round(lat_w, 2) != round(lat, 2), "not the middle of the whole walk, which reaches towards home"


def test_a_middle_in_the_sea_is_looked_up_again_at_the_nearest_drawn_point(monkeypatch, tmp_path):
    """A walk round a bay: the middle of the drawn stretch lies in the sea, where the answer is the country alone. The
    drawn point nearest to it that rounds to another point is asked next, rounded the same way, and its place names
    the print; when that names no place either, the first answer stays."""
    t = np.linspace(0.1 * np.pi, 0.9 * np.pi, 600)
    walk = _hike(np.column_stack([4000 * np.cos(t), 4000 * np.sin(t)]), "bay")  # an arc round a bay 8 km wide
    places = {}
    asked = []

    def lookup(lat, lon, lang="en"):
        asked.append((lat, lon))
        return ("Portugal", "", "Portugal") if len(asked) == 1 else places.get("second", ("Machico", "Madeira",
                                                                                           "Portugal"))

    for second, title, place in ((None, "Machico", "Madeira, Portugal"),
                                 (("", "", "Portugal"), "Portugal", "Portugal")):
        if second:
            places["second"] = second
        captured = {}
        asked.clear()
        _trail_world(monkeypatch, [walk], captured, tmp_path)
        monkeypatch.setattr(trail.area, "place_region", lookup)
        assert cli.main(["x.gpx", "--layout", "trail", "--dpi", "20", "--out", str(tmp_path / "t.png")]) == 0
        pieces = captured["pieces"]
        crs = trail.render.utm_crs(*np.median(np.asarray(walk.lonlat), axis=0))
        inv = trail.Transformer.from_crs(crs, 4326, always_xy=True)
        mid = np.median(np.asarray(pieces[0][1].coords), axis=0)
        drawn = np.concatenate([np.asarray(p.coords) for _, p in pieces])
        lon_m, lat_m = inv.transform(*mid)
        first = (round(lat_m, 2), round(lon_m, 2))
        keys = [(round(la, 2), round(lo, 2)) for lo, la in (inv.transform(*p) for p in drawn)]
        near = min((i for i, k in enumerate(keys) if k != first), key=lambda i: np.hypot(*(drawn[i] - mid)))
        assert np.hypot(*(drawn[near] - mid)) > 500, "the middle lies in the bay, away from the walk"
        assert asked == [first, keys[near]]
        assert (captured["title"], captured["place"]) == (title, place)
    # a cove: the drawn point nearest to the middle rounds to the middle's own point, so the next one is asked
    asked.clear()
    places.pop("second")
    cove = _hike(np.column_stack([900 * np.cos(t), 900 * np.sin(t)]), "cove")
    captured = {}
    _trail_world(monkeypatch, [cove], captured, tmp_path)
    monkeypatch.setattr(trail.area, "place_region", lookup)
    assert cli.main(["x.gpx", "--layout", "trail", "--dpi", "20", "--out", str(tmp_path / "t.png")]) == 0
    pieces = captured["pieces"]
    mid = np.median(np.asarray(pieces[0][1].coords), axis=0)
    drawn = np.concatenate([np.asarray(p.coords) for _, p in pieces])
    lon_n, lat_n = inv.transform(*drawn[np.argmin(np.hypot(*(drawn - mid).T))])
    assert (round(lat_n, 2), round(lon_n, 2)) == asked[0], "the nearest drawn point rounds to the middle's point"
    assert len(asked) == 2 and asked[0] != asked[1] and captured["title"] == "Machico"


def test_a_free_text_type_is_logged_as_other(monkeypatch, tmp_path, capsys):
    walk = _hike(_loop(0, 0, 2500, n=500), kind="Evening walk from Calle Mayor 5")
    _trail_world(monkeypatch, [walk], {}, tmp_path)
    assert cli.main(["x.gpx", "--layout", "trail", "--dpi", "20", "--out", str(tmp_path / "t.png")]) == 0
    out = capsys.readouterr().out
    assert "[trail] the route: one other activity" in out and "Calle Mayor" not in out


def test_a_single_file_is_trimmed_at_both_ends(monkeypatch, tmp_path):
    loop = _hike(_loop(0, 0, 2500, n=500), "only")
    (tmp_path / "hike.gpx").write_bytes(_gpx(loop))
    captured = {}
    _trail_world(monkeypatch, None, captured, tmp_path)
    assert cli.main([str(tmp_path / "hike.gpx"), "--layout", "trail", "--dpi", "20", "--out",
                     str(tmp_path / "t.png")]) == 0
    track, piece = LineString(captured["track"]), captured["pieces"][0][1]
    first, last = (track.project(Point(piece.coords[i])) for i in (0, -1))
    assert first == pytest.approx(300, abs=1) and last == pytest.approx(track.length - 300, abs=1)
    assert piece.distance(Point(captured["track"][0])) > 290, "nothing drawn within 300 m of where it starts"


def _gpx(act):
    pts = "".join(f'<trkpt lat="{la:.7f}" lon="{lo:.7f}"><ele>{e:.1f}</ele><time>'
                  f'{(act.start + dt.timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")}</time></trkpt>'
                  for (lo, la), e, t in zip(act.lonlat, act.ele, act.times))
    return (f'<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><name>{act.name}</name>'
            f'<type>hiking</type><trkseg>{pts}</trkseg></trk></gpx>').encode()


def test_a_single_file_learns_no_places_and_says_so(monkeypatch, tmp_path, capsys):
    loop = _hike(_loop(0, 0, 2500, n=500), "only", name="Sunday loop", speed=1.0)
    (tmp_path / "hike.gpx").write_bytes(_gpx(loop))  # read for real: a single file is an export of one activity
    captured = {}
    _trail_world(monkeypatch, None, captured, tmp_path)
    assert cli.main([str(tmp_path / "hike.gpx"), "--layout", "trail", "--dpi", "20", "--title", "Sunday loop",
                     "--name", "Ana", "--out", str(tmp_path / "t.png")]) == 0
    out = capsys.readouterr().out
    assert "[ingest] files 1 · used 1" in out
    assert "one activity (a single file): no private places can be learned" in out
    assert "drawn as a loop from a public trailhead" in out, "the loop warning"
    assert captured["stretch"] == (0.0, pytest.approx(render_trail.along(captured["track"])[-1]))
    assert captured["title"] == "Sunday loop" and captured["place"] == "Rascafría, Comunidad de Madrid, Spain"
    assert cli.main([str(tmp_path / "hike.gpx"), "--layout", "trail", "--dpi", "20", "--title", "RASCAFRÍA ",
                     "--out", str(tmp_path / "t.png")]) == 0
    assert captured["place"] == "Comunidad de Madrid, Spain", "the title already names the place"
    zone = ["--zone", f"{_lonlat([[0, 0]])[0][1]},{_lonlat([[0, 0]])[0][0]},400"]
    assert cli.main([str(tmp_path / "hike.gpx"), "--layout", "trail", "--dpi", "20", "--out",
                     str(tmp_path / "t.png")] + zone) == 2
    assert "STOP" in capsys.readouterr().out, "a --zone at its start protects it"


def test_type_defaults_to_run_except_for_the_trail(monkeypatch, tmp_path):
    hike = _hike(_line((0, 0), (5000, 0), (5000, 4000)), "h", kind="Hike")
    run = _hike(_line((0, 0), (2000, 0)), "r", kind="Run")
    got = {}
    monkeypatch.setattr(ingest, "read_export", lambda path: ([hike, run], ingest.IngestReport()))
    monkeypatch.setattr(race, "run", lambda sel, pool, a: got.update(race=[x.id for x in sel], pool=pool) or 0)
    monkeypatch.setattr(trail, "run", lambda sel, pool, a: got.update(trail=[x.id for x in sel], tpool=pool) or 0)
    monkeypatch.setattr(atlas, "run", lambda sel, a: got.update(atlas=[x.id for x in sel]) or 0)
    assert cli.main(["x.zip", "--layout", "race"]) == 0 and got["race"] == ["r"], "race: runs only, as before"
    assert cli.main(["x.zip", "--layout", "atlas"]) == 0 and got["atlas"] == ["r"]
    assert cli.main(["x.zip", "--layout", "trail"]) == 0 and got["trail"] == ["h", "r"], "trail: any type"
    assert [x.id for x in got["tpool"]] == ["h", "r"], "places from every activity of the export"
    assert cli.main(["x.zip", "--layout", "trail", "--type", "hike"]) == 0 and got["trail"] == ["h"]
    assert [x.id for x in got["tpool"]] == ["h", "r"], "the places still come from every activity"
    monkeypatch.setattr(ingest, "read_export", lambda path: ([hike], ingest.IngestReport()))
    assert cli.main(["x.zip", "--layout", "race"]) == 2, "no runs: nothing to draw"


def test_relief_is_on_for_the_trail_only(monkeypatch, tmp_path):
    captured = {}
    calls = _trail_world(monkeypatch, [_hike(_loop(0, 0, 2500, n=500))], captured, tmp_path)
    args = ["h.gpx", "--layout", "trail", "--dpi", "20", "--size", "A4", "--out", str(tmp_path / "t.png")]
    assert cli.main(args) == 0 and len(calls) == 1
    (extent, crs, map_w), = calls[0]
    assert extent == captured["extent"] and map_w == pytest.approx(261 * 210 / 297), "the map box and paper width"
    assert cli.main(args + ["--relief", "off"]) == 0 and len(calls) == 1, "--relief off: nothing prepared"
    assert cli.main(args + ["--relief", "shade"]) == 0 and len(calls) == 2
    # the race print and the city poster still need --relief shade
    monkeypatch.setattr(race.area, "fetch_basemap", lambda bbox, log=print: "unused")
    monkeypatch.setattr(race.render, "load_basemap", lambda folder, bbox, crs: (EMPTY, EMPTY, EMPTY))
    monkeypatch.setattr(race.area, "place_names", lambda lat, lon, lang="en": ("Madrid", "Spain"))
    run = _hike(_line((0, 0), (6000, 2000)), "r", kind="Run")
    monkeypatch.setattr(ingest, "read_export", lambda path: ([run], ingest.IngestReport()))
    assert cli.main(["x.zip", "--layout", "race", "--dpi", "20", "--size", "A4", "--out",
                     str(tmp_path / "r.png")]) == 0 and len(calls) == 2
    assert cli.main(["x.zip", "--layout", "race", "--dpi", "20", "--size", "A4", "--relief", "shade", "--out",
                     str(tmp_path / "r.png")]) == 0 and len(calls) == 3


def test_a_full_trail_run_writes_png_pdf_and_preview(monkeypatch, tmp_path, capsys):
    gpd.GeoDataFrame({"name": ["Peñalara"], "ele": ["2428"]}, geometry=[Point(*_lonlat([[-2500, 1200]])[0])],
                     crs=4326).to_file(tmp_path / "peaks.gpkg")
    captured = {}
    _trail_world(monkeypatch, [_hike(_loop(0, 0, 2500, n=500), name="Morning Hike")], captured, tmp_path)
    folder = tmp_path / "out"
    folder.mkdir()
    assert cli.main(["hike.gpx", "--layout", "trail", "--title", "Peñalara", "--name", "Lucía", "--lang", "es",
                     "--style", "forest", "--dpi", "40", "--out", str(folder / "t.png"), "--pdf", str(folder / "t.pdf"),
                     "--preview", str(folder / "t.jpg"), "--color-by", "elevation"]) == 0
    assert sorted(p.name for p in folder.iterdir()) == ["t.jpg", "t.pdf", "t.png"]
    assert all((folder / f).stat().st_size > 0 for f in ("t.jpg", "t.pdf", "t.png"))
    assert [s[2] for s in captured["info"]["summits"]] == ["PEÑALARA 2428 m"] and captured["extra"] == ("peaks", "glacier")
    assert captured["info"]["legend"]["mode"] == "elevation" and captured["date"] == START
    pdf = (folder / "t.pdf").read_bytes()
    assert b"CreationDate" not in pdf and b"hike.gpx" not in pdf
    out = capsys.readouterr().out
    assert "1 summits named" in out and "Morning Hike" not in out


def test_a_route_across_the_180th_meridian_is_refused_before_anything_is_sent(monkeypatch, tmp_path, capsys):
    lons = np.linspace(179.98, 180.02, 400)  # a walk out and back across the meridian, on Taveuni (Fiji)
    lonlat = [((lo + 180) % 360 - 180, -16.80) for lo in np.concatenate([lons, lons[::-1]])]
    walk = Activity(id="d", type="Hike", start=START, lonlat=lonlat, source="x", times=list(np.arange(800) * 5.0),
                    ele=[100.0] * 800)
    captured = {}
    _trail_world(monkeypatch, [walk], captured, tmp_path)
    monkeypatch.setattr(race.area, "place_names", lambda *a, **k: pytest.fail("nothing may be looked up"))
    for layout in ("trail", "race"):
        assert cli.main(["x.gpx", "--layout", layout, "--type", "Hike", "--dpi", "20", "--out",
                         str(tmp_path / "t.png")]) == 2, layout
        assert "crosses the 180th meridian" in capsys.readouterr().out
    assert "extra" not in captured and "info" not in captured, "no OpenStreetMap request, no print"
    s, w, n, e = race.osm_box((497000.0, 503000.0, 4.47e6, 4.474e6), trail.render.utm_crs(-3.7, 40.4))
    assert -3.1 < w < -3 < e < -2.9 and 40.3 < s < n < 40.5, "a map elsewhere: its own box"


def test_trail_command_line_refusals(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ingest, "read_export", lambda path: ([], ingest.IngestReport()))
    for bad in (["--bbox", "40,-4,41,-3"], ["--place", "Madrid"], ["--cache", "madrid"], ["--fit"],
                ["--headline", "name", "--name", "A"], ["--graphs"], ["--with", "b.zip", "--with-name", "B",
                                                                       "--name", "A"], ["--color-by", "month"]):
        with pytest.raises(SystemExit) as ex:
            cli.main(["x.zip", "--layout", "trail"] + bad)
        assert ex.value.code == 2, bad
    assert cli.main(["x.zip", "--layout", "trail", "--style", "neon", "--svg", str(tmp_path / "t.svg")]) == 2
    assert "no activities" in capsys.readouterr().out


def test_a_style_file_draws_the_trail(monkeypatch, tmp_path):
    import json

    style = tmp_path / "alpine.json"
    style.write_text(json.dumps({"base": "forest", "accent": "#d9480f"}), encoding="utf-8")
    captured = {}
    _trail_world(monkeypatch, [_hike(_loop(0, 0, 2500, n=500))], captured, tmp_path)
    assert cli.main(["h.gpx", "--layout", "trail", "--style", str(style), "--dpi", "20", "--relief", "off", "--out",
                     str(tmp_path / "t.png")]) == 0
    assert captured["style"] == "file:alpine" and render_editorial.STYLES["file:alpine"]["accent"] == "#d9480f"
    assert (tmp_path / "t.png").stat().st_size > 0


# Every demo hike of tools/simulate_hike.py: the box its middle lies in (south, west, north, east), its route type,
# its length (km) and high point (m), and the minutes it stands still
DEMO_HIKES = {
    "hike_demo.gpx": ((40.79, -4.02, 40.88, -3.90), "loop", (12.5, 13.5), (2400, 2440), 35,
                      "round Penalara (2428 m), with stops of 20 and 15 minutes"),
    "hike_desert_demo.gpx": ((36.98, -2.56, 37.10, -2.36), "out_and_back", (14.5, 15.5), (670, 690), 20,
                             "up the badlands of the Desierto de Tabernas below El Puntal, 20 minutes at the top"),
    "hike_dolomites_demo.gpx": ((46.60, 12.28, 46.64, 12.32), "loop", (9.5, 10.5), (2440, 2465), 40,
                                "round the Tre Cime di Lavaredo over the Forcella Lavaredo (2454 m), with stops of "
                                "10 and 30 minutes"),
    "hike_alps_demo.gpx": ((45.95, 6.86, 46.00, 6.93), "point_to_point", (8.6, 9.4), (2350, 2375), 30,
                           "above Chamonix up to Lac Blanc (2352 m), 30 minutes by the lake, and down past the Lacs "
                           "des Cheserys to the Col des Montets"),
    "hike_coast_demo.gpx": ((32.73, -16.71, 32.76, -16.68), "out_and_back", (6.0, 6.8), (100, 115), 25,
                            "on the Ponta de Sao Lourenco (Madeira) over the cliffs by the Pedra Furada, 25 minutes "
                            "at the Casa do Sardinha"),
}


def _demo(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", name)


@pytest.mark.parametrize("name", sorted(DEMO_HIKES))
def test_each_demo_hike_reads_as_one_synthetic_walk_of_its_route(name):
    (s, w, n, e), kind, (km_lo, km_hi), (top_lo, top_hi), stopped, what = DEMO_HIKES[name]
    (act,), rep = ingest.read_export(_demo(name))
    assert rep.used == 1 and act.type == "Hike" and act.ele and act.times
    lon, lat = np.median(np.asarray(act.lonlat), axis=0)
    assert s < lat < n and w < lon < e, what
    crs = trail.render.utm_crs(lon, lat)
    xy = np.column_stack(trail.Transformer.from_crs(4326, crs, always_xy=True).transform(*np.asarray(act.lonlat).T))
    assert render_trail.route_type(xy) == kind and km_lo < render_trail.along(xy)[-1] / 1000 < km_hi, what
    assert top_lo < render_trail.high_point(xy, act.ele)[1] < top_hi, what
    assert min(act.ele) > 0, "never below the sea: a neck the elevation model masks as sea is crossed above it"
    secs, paused = render_trail.moving_time(xy, act.times, act.ele)
    assert paused and act.times[-1] - secs == pytest.approx(stopped * 60, abs=120), what


def test_every_demo_hike_is_checked_and_says_it_is_synthetic_with_its_sources():
    names = {os.path.basename(p) for p in glob.glob(_demo("hike_*.gpx"))}
    assert names == set(DEMO_HIKES), "a demo hike without its check above, or a check without its file"
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    for name in sorted(names):
        root = ElementTree.parse(_demo(name)).getroot()
        assert "synthetic" in root.get("creator") and len(root.findall("g:trk", ns)) == 1, name
        desc = root.find("g:metadata/g:desc", ns).text
        assert "Synthetic hike" in desc and "nobody's real data" in desc, name
        assert "© OpenStreetMap contributors, ODbL" in desc and "Copernicus WorldDEM-30" in desc, name
        rights = root.find("g:metadata/g:copyright", ns)
        assert rights.get("author") == "OpenStreetMap contributors", name
        assert rights.find("g:license", ns).text == "https://opendatacommons.org/licenses/odbl/1-0/", name


def test_a_planned_route_without_clock_or_heights_still_draws(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    xy = _line((0, 0), (3000, 0), (3000, 2500))
    plan = Activity(id="p", type="Unknown", start=None, lonlat=_lonlat(xy), source="x", times=None, ele=None)
    aspect = render_trail.map_aspect(profile=False)
    assert aspect < render_trail.map_aspect(), "no profile: the map takes its band"
    extent = race.frame_course([(0, LineString(xy))], aspect, pad_frac=0.1)
    x0, x1, y0, y1 = extent
    low = y0 + (y1 - y0) * 12 / render_trail.BARE_MAP_H  # 12 mm above the bottom of the taller map
    info = render_trail.render(plan, xy, [(0, LineString(xy[20:-20]))], (EMPTY, EMPTY, EMPTY), extent,
                               str(tmp_path / "p.png"), dpi=20, color_by="elevation",
                               peaks=[(x0 + (x1 - x0) * 0.3, low, "Low Hill", 900.0)])
    assert info["secs"] is None and info["high"] is None and info["route_type"] == "point_to_point"
    assert info["legend"]["mode"] == "single", "no heights: one colour"
    assert [s_[2] for s_ in info["summits"]] == ["LOW HILL 900 m"], "a summit in the band the profile would take"
    _, gy, _, (bx0, by0, bx1, by1), _ = info["summits"][0]
    assert render_trail.MAP_TOP + render_trail.MAP_H < gy and by1 < render_trail.MAP_TOP + render_trail.BARE_MAP_H
    cells = [b[0] for b in boxes if b[2] > render_trail.NUMBERS_TOP]
    assert cells.count("—") == 3 and "POINT TO POINT" in cells and "TIME" in cells
    assert "km" in cells and not {"m", "h"} & set(cells), "no unit next to a dash"
    assert not any(b[0] in ("ELEVATION", "PERFIL") for b in boxes), "no profile without heights"


def test_a_shrunk_value_sits_on_the_baseline_of_the_numbers(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    xy = _line((0, 0), (6000, 0), (6000, 3000))  # point to point: the longest value of the row
    for size, lang, value in (("A3", "en", "POINT TO POINT"), ("A4", "es", "IDA Y VUELTA"), ("50x70", "en", "LOOP")):
        del boxes[:]
        route = xy if value == "POINT TO POINT" else (np.vstack([xy, xy[::-1]]) if value == "IDA Y VUELTA"
                                                        else _loop(0, 0, 1500, n=400))
        a = _hike(route)
        extent = race.frame_course([(0, LineString(route))], render_trail.map_aspect(size), pad_frac=0.1)
        render_trail.render(a, route, [(0, LineString(route))], (EMPTY, EMPTY, EMPTY), extent, str(tmp_path / "b.png"),
                            dpi=20, size=size, lang=lang)
        row = {b[0]: b for b in boxes if render_trail.NUMBERS_TOP < b[2] < render_trail.NUMBERS_TOP + 13}
        kind, high = row[value], row[next(v for v in row if v.isdigit())]  # the high point: digits, no descender
        assert kind[4] == pytest.approx(high[4], abs=0.25), f"{size} {value}: one baseline (to the overshoot of an O)"


def test_a_single_file_is_warned_about_and_drawn_whatever_its_type_on_every_layout(monkeypatch, tmp_path, capsys):
    run = _hike(_line((0, 0), (3000, 0), (3000, 2200)), "commute", kind="Run", speed=3.0)
    (tmp_path / "run.gpx").write_bytes(_gpx(run).replace(b"<type>hiking</type>", b""))  # a planner names no type
    captured = {}
    _trail_world(monkeypatch, None, captured, tmp_path)
    monkeypatch.setattr(trail.area, "city_name", lambda lat, lon, lang="en": "Madrid")
    monkeypatch.setattr(trail.area, "place_names", lambda lat, lon, lang="en": ("Madrid", "Spain"))
    got = {}
    monkeypatch.setattr(atlas, "run", lambda sel, a: got.update(atlas=[(x.id, x.type) for x in sel]) or 0)
    for layout in ("editorial", "race", "atlas", "trail"):
        assert cli.main([str(tmp_path / "run.gpx"), "--layout", layout, "--dpi", "20", "--out",
                         str(tmp_path / f"{layout}.png")]) == 0, layout
        out = capsys.readouterr().out
        assert out.count(trail.ONE_ACTIVITY) == 1, f"{layout}: only --zone protects a single file's ends"
    assert got["atlas"] == [("run", "Unknown")], "no --type: the file's one activity, whatever its type"
    assert cli.main([str(tmp_path / "run.gpx"), "--type", "Ride", "--out", str(tmp_path / "x.png")]) == 2
    assert "this export has: Unknown 1" in capsys.readouterr().out
    monkeypatch.setattr(ingest, "read_export", lambda path: ([run, _hike(_loop(0, 0, 900), "h")], ingest.IngestReport()))
    assert cli.main(["x.zip", "--layout", "atlas"]) == 0 and got["atlas"] == [("commute", "Run")], "an export: runs"
    assert trail.ONE_ACTIVITY not in capsys.readouterr().out
    odd = Activity(id="o", type="Walk to Calle Mayor 5", start=START, lonlat=run.lonlat, source="x")
    monkeypatch.setattr(ingest, "read_export", lambda path: ([odd, _hike(_loop(0, 0, 900), "h")], ingest.IngestReport()))
    assert cli.main(["x.zip"]) == 2
    out = capsys.readouterr().out
    assert "this export has: other 1, Hike 1" in out and "Calle Mayor" not in out, "a file's own type is never logged"


# --- the spectacular relief: summits, units, profile, contours ---------------------------------------------------------

def _rows(monkeypatch):
    """Every text the page sets through its page axes: (value, x, y, the place_text keywords)."""
    seen = []
    real = atlas.place_text

    def spy(value, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
        if (x, y) != (0, 0):
            seen.append((str(value), x, y, k, dict(pt=pt, font=font, max_w=max_w, height=height)))
        return real(value, x, y, k, pt, font, align, max_w, height, clear)

    monkeypatch.setattr(atlas, "place_text", spy)
    return seen


def _baselines(seen, top, bottom):
    """{value: [(x, baseline in A3-grid mm)]} of the texts set with their top between top and bottom."""
    out = {}
    for value, x, y, k, kw in seen:
        if top <= y <= bottom:
            out.setdefault(value, []).append((x, y + render_editorial.baseline_of(value, k, **kw)))
    return out


def test_the_units_sit_on_the_baseline_of_their_numbers(monkeypatch, tmp_path):
    from pinless import render_race
    seen = _rows(monkeypatch)
    for size in ("A3", "A4"):
        del seen[:]
        _page(tmp_path, size=size)
        row = _baselines(seen, render_trail.NUMBERS_TOP, render_trail.NUMBERS_TOP + 13)
        units = [(u, x, b) for u in ("km", "m", "h") for x, b in row.get(u, [])]
        assert len(units) == 4, "km, m, m and h"
        for unit, x, b in units:  # the number just left of the unit
            value = max((v for v in row if v not in ("km", "m", "h")), key=lambda v: max(
                (xx for xx, _ in row[v] if xx < x), default=-1))
            assert b == pytest.approx(row[value][0][1], abs=0.1), f"{size} {unit}: on {value}'s baseline"
    del seen[:]
    xy = np.column_stack([np.linspace(0, 5000, 501), 800 * np.sin(np.linspace(0, 3, 501))])
    act = _hike(xy, kind="Run", speed=3.3)
    render_race.render(act, xy, [(0, LineString(xy[30:-30]))], (EMPTY, EMPTY, EMPTY), (-500, 5500, -2500, 2500),
                       str(tmp_path / "race.png"), dpi=20, title="Spring 5K", name="Lucia", date=act.start)
    ytop = min(y for v, x, y, k, kw in seen if v in ("km", "/km")) - 6
    row = _baselines(seen, ytop - 2, ytop + 12)
    for unit in ("km", "/km", "m"):
        (x, b), = row[unit]
        value = max((v for v in row if v not in ("km", "/km", "m")),
                    key=lambda v: max((xx for xx, _ in row[v] if xx < x), default=-1))
        assert b == pytest.approx(row[value][0][1], abs=0.1), f"race {unit}: on {value}'s baseline"


def test_the_profile_is_one_opaque_gradient_under_a_bolder_line(monkeypatch, tmp_path):
    from matplotlib.colors import to_rgb
    images = []
    real_fill = render_trail.draw_profile_fill
    monkeypatch.setattr(render_trail, "draw_profile_fill",
                        lambda *a, **k: images.append(real_fill(*a, **k)) or images[-1])
    info, act, xy = _page(tmp_path)
    (im,) = images
    a = np.asarray(im.get_array())
    assert a.shape == (256, 1, 3) and a.dtype == np.uint8, "opaque: no alpha, so no soft mask in a PDF"
    S = render_editorial.STYLES["huella"]
    paper, line = np.asarray(to_rgb(S["bg"])) * 255, np.asarray(to_rgb(info["course_ink"])) * 255
    for row, t in ((0, render_trail.PROFILE_FILL[0]), (-1, render_trail.PROFILE_FILL[1])):
        assert np.abs(a[row, 0] - (paper + (line - paper) * t)).max() <= 1, "pre-composited on the paper"
    assert im.get_clip_path() is not None, "clipped to the area under the line"
    assert im.get_extent()[3] == pytest.approx(max(render_trail.profile(xy, act.ele)[1])), "row 0 at the top"


def test_summits_are_set_in_the_ink_and_keep_clear_of_the_bolder_course():
    assert (render_trail.PEAK_PT, render_trail.HALO_PT, render_trail.COURSE_PT, render_trail.CASING_PT) == \
        (6.2, 1.9, 2.3, 4.3)
    assert render_trail.ROUTE_CLEAR_MM == pytest.approx(4.3 * render_trail.PT_MM / 2 + render_trail.ROUTE_GAP_MM)
    assert render_trail.ROUTE_CLEAR_MM > render_trail.CASING_PT * render_trail.PT_MM / 2, "clear of the casing"


def _hill_z(shape, extent, height=900.0, base=1400.0):
    H, W = shape
    x0, x1, y0, y1 = extent
    X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) * (x1 - x0) / W, y1 - (np.arange(H) + 0.5) * (y1 - y0) / H)
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4
    return (base + height * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r ** 2))).astype(np.float32)


@pytest.mark.parametrize("scale,step", [(21500, 10), (60000, 20), (150000, 50)])
def test_the_contour_interval_follows_the_scale(scale, step):
    w = 261 * scale / 1000  # metres of a 261 mm map at this scale
    extent = (0.0, w, 0.0, w * 0.94)
    z = _hill_z((120, 128), extent, height=0.25 * w * 0.3)  # gentle: the steep ground does not squeeze the lines
    assert render_trail.contour_step(z, extent, 261.0) == step


def test_steep_ground_takes_a_coarser_interval():
    extent = (0.0, 5611.5, 0.0, 5000.0)  # 1:21,500 on 261 mm
    yy, xx = np.mgrid[0:100, 0:110]
    z = 1000 + 2.0 * xx * 5611.5 / 110  # a slope of 2 m/m: 10 m lines would be 0.23 mm apart on paper
    assert render_trail.contour_step(z, extent, 261.0) == 20
    land = np.zeros(z.shape, bool)
    land[:, :5] = True
    assert render_trail.contour_step(np.where(land, z, 1000.0), extent, 261.0, land=~land) == 10, "only land counts"


def test_contours_stay_inside_the_map_and_off_the_water():
    extent = (1000.0, 6611.5, 2000.0, 7000.0)
    z = _hill_z((140, 150), extent)
    water = np.zeros(z.shape, bool)
    water[60:80, 20:60] = True  # a lake on the flank
    lines = render_trail.contour_lines(z, extent, 50, 261.0, render_trail._grow(water))
    assert lines and all(level > 0 and level % 50 == 0 for level, _ in lines)
    pts = np.concatenate([c for _, c in lines])
    x0, x1, y0, y1 = extent
    assert ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).all()
    H, W = z.shape
    col = np.clip(((pts[:, 0] - x0) / (x1 - x0) * W).astype(int), 0, W - 1)
    row = np.clip(((y1 - pts[:, 1]) / (y1 - y0) * H).astype(int), 0, H - 1)
    assert not water[row, col].any(), "no vertex over a pixel of water"
    assert len(render_trail.contour_lines(z, extent, 50, 261.0)) >= len(lines), "setup: the lake cuts some"
    bumps = z + np.where(np.hypot(*np.mgrid[0:140, 0:150] - np.array([20, 20])[:, None, None]) < 1.5, 60, 0)
    short = [c for _, c in render_trail.contour_lines(bumps, extent, 50, 261.0) if np.allclose(c[0], c[-1])]
    mm = (x1 - x0) / 261.0
    assert all(LineString(c).length >= render_trail.CONTOUR_MIN_LOOP_MM * mm for c in short), "no tiny rings"


def test_contours_are_two_paths_between_the_ground_and_the_paths(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    extent = (0.0, 5611.5, 0.0, 5000.0)
    z = _hill_z((140, 150), extent)
    fig, ax = plt.subplots()
    S = render_editorial.STYLES["huella"]
    info = render_trail.draw_contours(ax, z, extent, S, 1.0, 261.0, EMPTY, classes=np.full(z.shape, 10, np.uint8))
    minor, index = ax.patches
    assert minor.zorder == index.zorder == render_trail.CONTOUR_Z == 3.55, "over the ground (0.5), under paths (3.6)"
    assert (minor.get_linewidth(), index.get_linewidth()) == render_trail.CONTOUR_PT
    assert (minor.get_alpha(), index.get_alpha()) == render_trail.CONTOUR_ALPHA[False]
    assert to_rgb(minor.get_edgecolor()[:3]) == to_rgb(render_trail.CONTOUR_INK[False])
    assert minor.get_clip_path() is not None and index.get_clip_path() is not None, "clipped to the land"
    assert info["step"] == 10 and info["lines"] > 0 and info["vertices"] >= 2 * info["lines"]
    fig2, ax2 = plt.subplots()
    render_trail.draw_contours(ax2, z, extent, S, 1.0, 261.0, EMPTY)
    assert to_rgb(ax2.patches[0].get_edgecolor()[:3]) == to_rgb(S["sub"]), "without the land cover: sub colour"
    plt.close(fig)
    plt.close(fig2)


def test_contours_only_on_a_trail_map_with_heights(monkeypatch, tmp_path):
    xy = _loop(0, 0, 2000, n=600)
    extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
    shape = relief.raster_shape(render_trail.RIGHT - render_trail.LEFT, extent)
    z = _hill_z(shape, extent)
    shade = np.zeros(shape, np.float32)
    with_z = relief.Relief(shade, "glo30", 0, 0.5, smooth=shade, z=z, z_smooth=z)
    step = render_trail.contour_step(z, extent, render_trail.RIGHT - render_trail.LEFT)
    for rel, want in ((None, None), (relief.Relief(shade, "glo30", 0, 0.5), None), (with_z, step)):
        info, _, _ = _page(tmp_path, relief=rel)
        assert info["contours"] == want
    calls = []
    monkeypatch.setattr(render_trail, "draw_contours", lambda *a, **k: calls.append(a) or None)
    from pinless import render_editorial as re_
    re_.render([(0, LineString(xy))], (EMPTY, EMPTY, EMPTY), extent, {"km": 1.0, "n": 1, "hours": 1.0,
               "longest_km": 1.0}, str(tmp_path / "e.png"), dpi=20, relief=with_z)
    assert not calls, "never on the city poster"


def _slope_page(extent, shape, z0=1000.0, grade=0.2):
    """Heights of a plane rising east at grade on the pixel centres of the map, and the plane itself."""
    x0, x1, y0, y1 = extent
    H, W = shape

    def plane(x):
        return z0 + grade * (np.asarray(x) - x0)

    xs = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
    return np.tile(plane(xs), (H, 1)).astype(np.float64), plane


def test_index_lines_are_every_fifth_level_and_both_kinds_keep_off_the_water():
    import matplotlib.pyplot as plt
    extent = (0.0, 5611.5, 0.0, 5000.0)  # 1:21,500 on 261 mm: 10 m lines
    z, plane = _slope_page(extent, (140, 150))
    classes = np.full(z.shape, landcover.GRASS, np.uint8)
    classes[50:90, 40:110] = landcover.WATER  # a lake across the slope
    fig, ax = plt.subplots()
    info = render_trail.draw_contours(ax, z, extent, render_editorial.STYLES["huella"], 1.0, 261.0, EMPTY,
                                      classes=classes)
    assert info["step"] == 10
    by_width = {p.get_linewidth(): p for p in ax.patches}
    minor, index = by_width[render_trail.CONTOUR_PT[0]], by_width[render_trail.CONTOUR_PT[1]]
    every = render_trail.CONTOUR_INDEX * info["step"]
    for patch, is_index in ((minor, False), (index, True)):
        v = patch.get_path().vertices
        z_v = plane(v[:, 0])
        assert np.allclose(z_v / info["step"], np.round(z_v / info["step"]), atol=1e-6), "on the levels"
        on_index = np.isclose(z_v / every, np.round(z_v / every), atol=1e-6)
        assert on_index.all() if is_index else not on_index.any(), "index lines every fifth level, and only them"
        x0, x1, y0, y1 = extent
        H, W = z.shape
        codes = patch.get_path().codes
        joined = codes[1:] != patch.get_path().MOVETO  # the segments of each line, not the jumps between lines
        t = np.linspace(0, 1, 50)[:, None, None]
        pts = (v[:-1][joined] + (v[1:][joined] - v[:-1][joined]) * t).reshape(-1, 2)  # along every segment
        col = np.clip(((pts[:, 0] - x0) / (x1 - x0) * W).astype(int), 0, W - 1)
        row = np.clip(((y1 - pts[:, 1]) / (y1 - y0) * H).astype(int), 0, H - 1)
        assert not (classes[row, col] == landcover.WATER).any(), "no line over the land cover's water"
    plt.close(fig)


def test_land_below_the_sea_keeps_its_contour_lines():
    """By the Dead Sea or in Death Valley the land lies below 0 m: its lines are drawn like any other, only the sea's
    own level (0) is not (the coast is the water's edge)."""
    extent = (0.0, 5611.5, 0.0, 5000.0)
    import matplotlib.pyplot as plt
    z, plane = _slope_page(extent, (100, 110), z0=-440.0, grade=540.0 / 5611.5)  # from -440 m to +100 m
    step = render_trail.contour_step(z, extent, 261.0)
    levels = sorted({level for level, _ in render_trail.contour_lines(z, extent, step, 261.0)})
    assert step == 10 and levels[0] == -430 and levels[-1] == 90
    assert 0 not in levels and -10 in levels and 10 in levels
    fig, ax = plt.subplots()
    render_trail.draw_contours(ax, z, extent, render_editorial.STYLES["huella"], 1.0, 261.0, EMPTY)
    index = {p.get_linewidth(): p for p in ax.patches}[render_trail.CONTOUR_PT[1]]
    z_v = plane(index.get_path().vertices[:, 0])
    assert np.isclose(z_v, -50).any() and np.isclose(z_v, -400).any(), "index lines below the sea too"
    plt.close(fig)


def test_on_the_plain_paper_the_contours_stay_fainter_than_the_paths():
    """Without the land cover the lines are in the colour the paths are inked towards: an index line over the paper
    stays well closer to the paper than a path, so the paths read as paths and the lines as terrain."""
    from matplotlib.colors import to_rgb

    def lstar(rgb):
        return render_editorial._lstar(np.asarray(rgb, dtype=float))

    for name, S in render_editorial.STYLES.items():
        bg, sub = np.asarray(to_rgb(S["bg"])), np.asarray(to_rgb(S["sub"]))
        dark = lstar(bg) < 50
        minor_a, index_a = render_trail.CONTOUR_ALPHA_PLAIN[dark]
        trail_gap = abs(lstar(to_rgb(render_trail.trail_style(S)["trail"])) - lstar(bg))
        for a in (minor_a, index_a):
            gap = abs(lstar((1 - a) * bg + a * sub) - lstar(bg))
            assert gap <= 0.6 * trail_gap, f"{name}: a contour line at {a} reads fainter than a path"


def test_without_the_land_cover_the_paths_are_inked_for_the_shadow_of_the_relief(monkeypatch, tmp_path):
    """--ground style: the relief is an overlay that darkens the paper (lightens it on dark paper) under the paths;
    they are re-inked so that, against its shadow at the REF_PCT percentile, they keep their contrast."""
    from matplotlib.colors import to_rgb

    def lstar(c):
        return render_editorial._lstar(np.asarray(to_rgb(c), dtype=float))

    seen = []
    real = render_trail.trail_network
    monkeypatch.setattr(render_trail, "trail_network", lambda ax, trails, extent, S, *a: seen.append(S) or real(
        ax, trails, extent, S, *a))
    xy = _loop(0, 0, 2000, n=600)
    extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
    shape = relief.raster_shape(render_trail.RIGHT - render_trail.LEFT, extent)
    slope = np.tile(np.linspace(-1, 1, shape[1], dtype=np.float32), (shape[0], 1))
    for style in ("huella", "ink", "carbon"):
        S = render_editorial.STYLES[style]
        TS = render_trail.trail_style(S)
        seen.clear()
        _page(tmp_path, style=style, relief=relief.Relief(slope, "glo30", 0, 0.5))
        _page(tmp_path, style=style)
        shaded, plain = seen
        assert plain["trail"] == TS["trail"], f"{style}: no relief, the paths as they were"
        ref = render_trail.overlay_ref_l(S, TS["relief"], slope)
        lb = lstar(S["bg"])
        assert (ref > lb) if lb < 50 else (ref < lb - 5), f"{style}: the overlay's deep side"
        for key in render_trail.INK_KEYS:
            want = abs(lstar(TS[key]) - lb)
            assert abs(lstar(shaded[key]) - ref) >= want - 0.6, f"{style} {key}: against the shadow as the paper"


def test_inks_for_a_deep_shadow_stop_at_the_page_ink_in_their_order():
    """Paths, minor and major roads against a ground so deep that they cannot all keep their contrast: none goes past
    the page's own ink (a deep shadow pushed the roads to pure black), the paths keep theirs first, and the roads
    keep their order between the paths and the ink."""
    from matplotlib.colors import to_rgb

    def gap(c, S):
        return abs(render_editorial._lstar(np.asarray(to_rgb(c), dtype=float))
                   - render_editorial._lstar(np.asarray(to_rgb(S["bg"]), dtype=float)))

    for name, S in render_editorial.STYLES.items():
        TS = render_trail.trail_style(S)
        lb = render_editorial._lstar(np.asarray(to_rgb(S["bg"]), dtype=float))
        dark = lb < 50
        cap = gap(S["ink"], S)
        before = [gap(TS[key], S) for key in render_trail.INK_KEYS]
        # the paths still fit under the ink, the major roads would not: halfway between the two
        shift = cap - (before[0] + before[2]) / 2
        for depth, ref in (("halfway", (lb + shift) if dark else (lb - shift)), ("full", 100.0 if dark else 0.0)):
            deep = render_trail.shadow_inks(TS, S, ref, streets=True)
            g = [gap(deep[key], S) for key in render_trail.INK_KEYS]
            assert g[0] <= g[1] + 0.3 and g[1] <= g[2] + 0.3, f"{name}: paths, minor roads, major roads in order"
            assert abs(g[2] - max(cap, before[2])) <= 0.6, f"{name}: the strongest as far as the page ink, not past"
            assert g[0] >= min(cap, before[0] + abs(ref - lb)) - 0.6, f"{name}: the paths first"
            for key, b in zip(render_trail.INK_KEYS, before):
                assert gap(deep[key], S) >= b - 0.3, f"{name} {key}: never fainter than before"
            for cls, (colour, width) in TS["street"].items():
                assert deep["street"][cls][1] == width and gap(deep["street"][cls][0], S) <= max(cap, before[2]) + 0.6
            if depth == "halfway" and shift > 1 and before[2] > before[0] + 1:
                assert g[0] < g[2] - 0.5, f"{name}: where the ink leaves room, the roads still lead the paths"
        same = render_trail.shadow_inks(TS, S, 0.0 if dark else 100.0)
        assert all(same[key] == render_trail._ink_at(S["bg"], (S["sub"], S["ink"]), gap(TS[key], S))
                   for key in render_trail.INK_KEYS), f"{name}: a ground beyond the paper changes nothing"


def test_built_up_land_takes_the_smooth_shade_and_heights_feathered_at_its_edge():
    H, W = 200, 300
    map_w = 261.0
    classes = np.full((H, W), landcover.TREE, np.uint8)
    assert render_trail.built_weight(classes, map_w) is None, "no town, no weight"
    classes[60:140, 100:200] = landcover.BUILT
    w = render_trail.built_weight(classes, map_w)
    assert w.shape == (H, W) and w.dtype == np.float32 and 0 <= w.min() and w.max() <= 1
    assert (w[70:130, 110:190] == 1.0).all(), "the town itself"
    px = W / map_w
    reach = int(np.ceil(1 + 3 * render_trail.BUILT_FEATHER_MM * px))  # the pixel it grows by, then 3 sigma
    assert w[100, 200] > 0.5, "the pixel just outside its edge"
    assert w[100, 200 + reach:].max() < 0.02 and w[:60 - reach].max() < 0.02, "faded within the feather"
    crisp, smooth = np.full((H, W), 0.5, np.float32), np.full((H, W), -0.5, np.float32)
    z, z_smooth = np.full((H, W), 1000.0, np.float32), np.full((H, W), 900.0, np.float32)
    rel = relief.Relief(crisp, "glo30", 0, 0.5, smooth=smooth, z=z, z_smooth=z_smooth)
    ground = landcover.LandCover(classes, 0, 0.0)
    shade, heights = render_trail.trail_surface(rel, ground, map_w)
    assert shade[100, 150] == -0.5 and heights[100, 150] == 900.0, "over the town: the city ground filter"
    assert shade[100, 280] == 0.5 and heights[100, 280] == 1000.0, "far from it: the crisp surface"
    edge = shade[100, 195:200 + reach]
    assert (np.diff(edge) >= 0).all() and -0.5 < edge[5] < 0.5, "feathered across its edge"
    shade, heights = render_trail.trail_surface(rel, None, map_w)
    assert shade is crisp and heights is z, "--ground style: the crisp surface"
    shade, heights = render_trail.trail_surface(relief.Relief(crisp, "glo30", 0, 0.5, z=z), ground, map_w)
    assert shade is crisp and heights is z, "no smooth surface: the crisp one"
    assert render_trail.trail_surface(None, ground, map_w) == (None, None)
