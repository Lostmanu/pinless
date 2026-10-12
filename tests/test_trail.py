"""The trail print (one route on a topographic map): its numbers, its summits, its privacy and its command line."""
import datetime as dt
import itertools

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from pinless import atlas, cli, ingest, race, relief, render_editorial, render_trail, trail
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
            assert b.distance(route) >= render_trail.CASING_PT * render_trail.PT_MM / 2, "clear of the casing"
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

    monkeypatch.setattr(atlas, "place_text", spy)
    return boxes


def _page(tmp_path, name="t.png", **kw):
    xy = _loop(0, 0, 2000, n=600)
    act = _hike(xy)
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


def test_spanish_labels(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    _page(tmp_path, lang="es")
    texts = {b[0] for b in boxes}
    assert {"RUTA", "CIRCULAR", "TIPO DE RUTA", "COTA MÁXIMA", "DESNIVEL POSITIVO", "DISTANCIA", "TIEMPO",
            "CADA RUTA DEJA HUELLA.", "PERFIL", "JUL 2025"} <= texts


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
        assert shadow >= shadow0 and shadow > 0, f"{name}: the course is one bold line, the relief can show more"
        assert light <= render_editorial._keep_alpha(S, light_rgb, 0.6) + 1e-9, "the map under it keeps its contrast"
    own = dict(render_editorial.STYLES["huella"], relief=(((0, 0, 0), 0.1), ((1, 1, 1), 0.0)))
    assert render_trail.trail_style(own)["relief"] == own["relief"], "a style that sets its relief keeps it"
    edges = gpd.GeoDataFrame({"highway": ["track", "path", "footway", "residential", "primary", ["path", "steps"]]},
                             geometry=[LineString([(0, i), (5000, i)]) for i in range(0, 600, 100)])
    streets, trails = render_trail.split_network((edges, EMPTY, EMPTY))
    assert list(streets[0]["highway"]) == ["residential", "primary"] and len(trails) == 4
    fig, ax = plt.subplots()
    render_trail.trail_network(ax, trails, (0, 5000, -100, 700), render_trail.trail_style(own), 1.0, 261)
    solid, dashed = ax.patches
    assert solid.get_linestyle() == "solid" and dashed.get_linestyle() == render_trail.FOOTPATH_DASH, \
        "tracks solid, footpaths dashed"
    assert solid.zorder == dashed.zorder == 3.6, "over the relief (3.5), under the course"
    assert len(solid.get_path().vertices) == 2 and len(dashed.get_path().vertices) == 6
    plt.close(fig)


# --- privacy and the command line --------------------------------------------------------------------------------------

def _trail_world(monkeypatch, acts, captured, tmp_path, prepared=None):
    if acts is not None:
        monkeypatch.setattr(ingest, "read_export", lambda path: (acts, ingest.IngestReport()))
    monkeypatch.setattr(trail.area, "fetch_basemap", lambda bbox, log=print: str(tmp_path))
    monkeypatch.setattr(trail.render, "load_basemap", lambda folder, bbox, crs: (EMPTY, EMPTY, EMPTY))
    monkeypatch.setattr(trail.area, "place_region", lambda lat, lon, lang="en": ("Rascafría", "Comunidad de Madrid",
                                                                                 "España" if lang == "es" else "Spain"))
    calls = [] if prepared is None else prepared
    monkeypatch.setattr(relief, "prepare_many", lambda maps, log=print: calls.append(maps) or [None] * len(maps))
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
    assert [s[2] for s in captured["info"]["summits"]] == ["PEÑALARA 2428 m"]
    assert captured["info"]["legend"]["mode"] == "elevation" and captured["date"] == START
    pdf = (folder / "t.pdf").read_bytes()
    assert b"CreationDate" not in pdf and b"hike.gpx" not in pdf
    out = capsys.readouterr().out
    assert "1 summits named" in out and "Morning Hike" not in out


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


def test_the_demo_hike_reads_as_one_synthetic_loop():
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "hike_demo.gpx")
    (act,), rep = ingest.read_export(path)
    assert rep.used == 1 and act.type == "Hike" and act.ele and act.times
    crs = trail.render.utm_crs(*np.median(np.asarray(act.lonlat), axis=0))
    tr = trail.Transformer.from_crs(4326, crs, always_xy=True)
    xy = np.column_stack(tr.transform(*np.asarray(act.lonlat).T))
    assert render_trail.route_type(xy) == "loop" and 12.5 < render_trail.along(xy)[-1] / 1000 < 13.5
    assert 2400 < render_trail.high_point(xy, act.ele)[1] < 2440, "over Penalara (2428 m)"
    secs, paused = render_trail.moving_time(xy, act.times, act.ele)
    assert paused and act.times[-1] - secs == pytest.approx(35 * 60, abs=120), "the two stops, 20 and 15 minutes"


def test_a_planned_route_without_clock_or_heights_still_draws(monkeypatch, tmp_path):
    boxes = _texts(monkeypatch)
    xy = _line((0, 0), (3000, 0), (3000, 2500))
    plan = Activity(id="p", type="Unknown", start=None, lonlat=_lonlat(xy), source="x", times=None, ele=None)
    extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
    info = render_trail.render(plan, xy, [(0, LineString(xy[20:-20]))], (EMPTY, EMPTY, EMPTY), extent,
                               str(tmp_path / "p.png"), dpi=20, color_by="elevation")
    assert info["secs"] is None and info["high"] is None and info["route_type"] == "point_to_point"
    assert info["legend"]["mode"] == "single", "no heights: one colour"
    cells = [b[0] for b in boxes if b[2] > render_trail.NUMBERS_TOP]
    assert cells.count("—") == 3 and "POINT TO POINT" in cells and "TIME" in cells
    assert not any(b[0] in ("ELEVATION", "PERFIL") for b in boxes), "no profile without heights"
