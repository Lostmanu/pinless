"""The race poster (one activity) and the graphs band: numbers, privacy for a single course, and drawing."""
import datetime as dt
import math
import random

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from pinless import atlas, cli, graphs, ingest, privacy, race, render_editorial, render_race
from pinless.ingest import Activity

UTC = dt.timezone.utc
EMPTY = gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])
LAT0, LON0 = 40.42, -3.70  # Madrid


def _act(i, start, lonlat, times=None, ele=None, name=""):
    return Activity(id=str(i), type="Run", start=start, lonlat=lonlat, source="x", times=times, ele=ele, name=name)


def _lonlat(xy):
    """Metres east/north of (LON0, LAT0) to degrees: close enough over a few kilometres."""
    xy = np.asarray(xy, dtype=float)
    return [(LON0 + x / (111320 * math.cos(math.radians(LAT0))), LAT0 + y / 110540) for x, y in xy]


def _loop(cx, cy, r, n=120, phase=0.0):
    t = np.linspace(0, 2 * np.pi, n) + phase
    return np.column_stack([cx + r * (np.cos(t) - np.cos(phase)), cy + r * (np.sin(t) - np.sin(phase))])


def _line(*corners, step=10.0):
    out = [np.asarray(corners[0], dtype=float)[None]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        out.append(np.linspace(a, b, n)[1:])
    return np.vstack(out)


class Fixed:
    """A stand-in for SystemRandom: every random extra cut takes `value`, every keep-away radius its smallest."""

    def __init__(self, value):
        self.value = value

    def uniform(self, lo, hi):
        return self.value if (lo, hi) == race.EXTRA else lo


POOL = [_loop(0, 0, 1500 + 30 * i, phase=0.3 * i) for i in range(20)]  # twenty loops from home (0, 0)
HOME = Point(0, 0)


def test_pick_longest_latest_by_date_and_by_id():
    short = _act("a", dt.datetime(2025, 5, 1, 6, tzinfo=UTC), [(-3.70, 40.40), (-3.70, 40.41)])
    long_ = _act("b", dt.datetime(2025, 4, 6, 7, tzinfo=UTC), [(-3.70, 40.40), (-3.70, 40.50)])
    late = _act("c", dt.datetime(2025, 12, 31, 16, tzinfo=UTC), [(-3.70, 40.40), (-3.70, 40.42)])
    acts = [short, long_, late]
    assert race.pick(acts) is long_ and race.pick(acts, "latest") is late
    assert race.pick(acts, "2025-05-01") is short and race.pick(acts, "c") is late
    assert race.pick(acts, "2025-01-01") is None


def test_splits_count_full_km_and_keep_the_last_part_apart():
    dist = np.linspace(0, 2500, 251)  # 2.5 km at a steady 5:00/km
    times = dist * 0.3
    full, last = render_race.splits(dist, times)
    assert full == pytest.approx([300, 300]) and last == pytest.approx((300, 500)), "2 full km and the last 500 m"
    times = np.where(dist > 2000, 600 + (dist - 2000) * 0.2, times)  # a fast last 500 m (4:00/km)
    full, last = render_race.splits(dist, times)
    assert full == pytest.approx([300, 300]) and last[0] == pytest.approx(200), "a short sprint is not the fastest km"
    full, last = render_race.splits(np.linspace(0, 2100, 211), np.linspace(0, 630, 211))
    assert len(full) == 2 and last is None, "under 200 m left: no last bar"
    assert render_race.climb(np.array([600, 601, 600, 610, 605, 620, 619])) == pytest.approx(25)  # +10, dip, +15
    assert render_race.climb(np.array([600, 602, 600, 602, 600, 602])) == 0  # 2 m jitter is not climbing
    assert render_race.pace_along(dist, dist * 0.3)[125] == pytest.approx(300)


def test_restrict_and_vertex_positions_follow_the_track():
    xy = np.column_stack([np.linspace(0, 5000, 501), np.zeros(501)])
    part, t, e = render_race.restrict(xy, list(np.linspace(0, 1500, 501)), list(np.full(501, 600.0)), (1000, 4000))
    assert part[0] == pytest.approx([1000, 0]) and part[-1] == pytest.approx([4000, 0])
    assert t[0] == pytest.approx(300) and t[-1] == pytest.approx(1200) and len(t) == len(part) == len(e)
    # out and back along the very same street (the same vertices): the way back must not take the way out's place
    out_back = np.vstack([xy[:201], xy[199::-1]])  # 0 -> 2000 m -> 0
    d = render_race.along(out_back)
    pts = list(map(tuple, out_back))
    pairs = {pq: i for i, pq in enumerate(zip(pts[:-1], pts[1:]))}
    pos = render_race.vertex_positions(out_back[300:351], out_back, d, pairs)
    assert pos[0] == pytest.approx(3000) and pos[-1] == pytest.approx(3500)
    made_by_a_cut = np.vstack([[[1005.0, 0.0]], out_back[300:320]])  # its first vertex is not the track's
    assert render_race.vertex_positions(made_by_a_cut, out_back, d, pairs)[0] == pytest.approx(2995)


@pytest.mark.parametrize("seed", range(6))
def test_a_wait_at_the_door_does_not_spend_the_race_and_trail_trim(seed):
    """An out and back after a minute at the door, a fix a second wandering 4 m round it: measured along the track
    the wait used up the 300 m trim and the drawn line came within 1 to 42 m of the door (an adversarial review on
    13-10-2026). The trim is metres of movement now: the line keeps 300 m of walking away from the door."""
    rng = np.random.default_rng(100 + seed)
    o = np.array([20000.0, 20000.0])  # a public course: no place anywhere near
    stick = np.column_stack([np.linspace(0, 3000, 900), np.zeros(900)])
    xy = o + np.vstack([rng.normal(0, 4, (60, 2)), stick + rng.normal(0, 1.0, (900, 2)),
                        stick[::-1][1:] + rng.normal(0, 1.0, (899, 2))])
    pieces, rep, _, stretch = race.cut(xy, [], 300.0, 500.0)
    door = Point(o)
    assert len(pieces) == 1 and pieces[0][1].distance(door) > 280, "300 m of walking away from the door"
    assert stretch == (0.0, pytest.approx(render_race.along(xy)[-1])), "numbered whole, as a public course"


def test_a_course_passing_home_is_drawn_whole_and_numbered_whole():
    near_start = np.column_stack([np.linspace(-1500, 6000, 751), np.full(751, 400.0)])  # passes home after 1.5 km
    pieces, rep, _, stretch = race.cut(near_start, POOL, 300.0, 500.0)
    assert rep.zones_user == 0 and stretch[0] == 0.0, "a race passing home in its first km is a passing course"
    passing = np.column_stack([np.linspace(-6000, 6000, 601), np.full(601, 60.0)])  # a race 60 m from home
    head, tail = race._private_ends(passing, render_race.along(passing), [(0.0, 0.0, 500.0)],
                                    privacy.trim_span(passing, 300.0))
    assert head == [] and tail == [], "it starts and ends 6 km away: home is not one of its ends"
    pieces, rep, learned, stretch = race.cut(passing, POOL, 300.0, 500.0)
    assert learned >= 1 and rep.zones_user == 0, "home is known but not cut out of this course"
    assert len(pieces) == 1 and pieces[0][1].distance(HOME) < 100, "drawn whole: no gap pointing home"
    assert stretch == (0.0, pytest.approx(12000.0)), "its numbers describe the whole course"


def test_an_end_is_private_when_its_undrawn_metres_come_near_home():
    towards = _line((900, 0), (-6000, 0))  # the first fix 900 m away, then straight past the door
    with pytest.raises(race.PrivateEnds):
        race.cut(towards, POOL, 300.0, 500.0)
    sparse = np.vstack([[[1040.0, 0.0], [1030.0, 0.0], [731.0, 0.0]], _line((721, 0), (-6000, 0))])
    with pytest.raises(race.PrivateEnds):  # 300 m in, between two of its own points, it is 740 m from home
        race.cut(sparse, POOL, 300.0, 500.0)


def test_a_sparse_track_is_checked_between_its_points():
    sparse = np.array([[0.0, 0.0], [600.0, 0.0], [1200.0, 0.0], [2600.0, 0.0], [2600.0, 2000.0], [-2600.0, 2000.0],
                       [-2600.0, 700.0], [2600.0, 700.0], [6000.0, 700.0]])  # from home; one leg passes 700 m away
    pieces, _, _, (lo, hi) = race.cut(sparse, POOL, 300.0, 500.0, allow_private_ends=True, rng=Fixed(200))
    assert pieces[0][1].distance(HOME) >= 800 and pieces[0][1].length == pytest.approx(hi - lo, abs=1.0)


def test_the_leak_check_fails_closed(monkeypatch):
    monkeypatch.setattr(race, "substring", lambda line, a, b: LineString([(-3000, 0), (3000, 0)]))
    with pytest.raises(privacy.PrivacyError):
        race.cut(_loop(0, 0, 2000, n=400), POOL, 300.0, 500.0, allow_private_ends=True)


def test_a_run_from_home_is_refused_or_cut_far_from_it_with_its_numbers():
    from_home = _loop(0, 0, 2000, n=400)
    with pytest.raises(race.PrivateEnds):
        race.cut(from_home, POOL, 300.0, 500.0)
    short, long_ = (race.cut(from_home, POOL, 300.0, 500.0, allow_private_ends=True, rng=Fixed(u)) for u in (200, 800))
    for pieces, _, _, (lo, hi) in (short, long_):
        assert len(pieces) == 1 and pieces[0][1].distance(HOME) >= 1.6 * 500, "nothing within reach of any zone"
        assert pieces[0][1].length == pytest.approx(hi - lo, abs=1.0), "what is drawn is exactly what is numbered"
    (lo1, hi1), (lo2, hi2) = short[3], long_[3]
    assert lo2 - lo1 == pytest.approx(600) and hi1 - hi2 == pytest.approx(600), "a random 200-800 m more at each end"


def test_a_small_zone_still_gets_its_random_extra():
    out = _line((0, 0), (8000, 0))  # its 1.6 r reach (240 m) ends inside the 300 m trim
    starts = {u: race.cut(out, [], 300.0, 500.0, user_zones=[(0.0, 0.0, 150.0)], allow_private_ends=True,
                          rng=Fixed(u))[3][0] for u in (200, 800)}
    assert starts == {200: pytest.approx(500), 800: pytest.approx(1100)}


def test_an_out_and_back_does_not_place_home_at_a_fixed_distance():
    out_back = _line((0, 0), (5000, 0), (5000, 15), (0, 15))
    nearest = []
    for seed in range(300):
        pieces, _, _, _ = race.cut(out_back, POOL, 300.0, 500.0, allow_private_ends=True, rng=random.Random(seed))
        ends = np.asarray(pieces[0][1].coords)[[0, -1]]
        nearest.append(np.hypot(*ends.T).min())
    assert min(nearest) >= 1000 and np.percentile(nearest, 90) - np.percentile(nearest, 10) > 400


def test_a_run_from_home_to_somewhere_else_is_trimmed_at_its_far_end_too():
    out = _line((0, 0), (8000, 0))  # from home, 8 km east
    pieces, _, _, (lo, hi) = race.cut(out, POOL, 300.0, 500.0, allow_private_ends=True)
    assert hi == pytest.approx(8000 - 300), "the far end loses its 300 m like any other end"
    assert pieces[0][1].length == pytest.approx(hi - lo, abs=1.0)


def test_a_run_from_home_that_comes_back_past_the_door_keeps_one_stretch():
    """From home 6 km east, back past the door, then 5 km north to a cafe: home is cut wherever the run comes near
    it, and the print keeps one stretch, so no gap in the middle tells how far up the street the door is."""
    run = _line((0, 0), (6000, 0), (6000, 30), (0, 30), (0, 5000))
    for seed in range(20):
        pieces, _, _, (lo, hi) = race.cut(run, POOL, 300.0, 500.0, allow_private_ends=True, rng=random.Random(seed))
        assert len(pieces) == 1 and pieces[0][1].distance(HOME) >= 800
        assert pieces[0][1].length == pytest.approx(hi - lo, abs=1.0)


def test_an_out_and_back_from_home_numbers_what_it_draws():
    out_back = _line((0, 0), (5000, 0), (5000, 15), (0, 15))
    for seed in range(30):
        pieces, _, _, (lo, hi) = race.cut(out_back, POOL, 300.0, 500.0, allow_private_ends=True,
                                          rng=random.Random(seed))
        assert len(pieces) == 1 and pieces[0][1].length == pytest.approx(hi - lo, abs=1.0)


def test_a_zone_the_course_passes_leaves_one_numbered_stretch():
    passing = _line((-6000, 60), (6000, 60))
    pieces, _, _, (lo, hi) = race.cut(passing, POOL, 300.0, 500.0, user_zones=[(-2000.0, 0.0, 400.0)])
    assert len(pieces) == 1 and pieces[0][1].distance(Point(-2000, 0)) >= 1.6 * 400
    assert pieces[0][1].length == pytest.approx(hi - lo, abs=1.0)
    assert pieces[0][1].bounds[0] > -2000, "the longer side (here the second), numbered"


def test_a_new_home_is_private_before_years_of_history_learn_it():
    old = [_loop(0, 0, 1500 + i, phase=0.05 * i) for i in range(200)]
    new_home = [_loop(9000, 0, 1800 + 50 * i, phase=0.7 * i) for i in range(2)]
    race_loop = _loop(9000, 0, 2500)
    with pytest.raises(race.PrivateEnds):
        race.cut(race_loop, old + new_home, 300.0, 500.0)
    for late_fix in (260, 400, 700):  # the watch finds the satellites down the street, or the run starts there
        with pytest.raises(race.PrivateEnds):
            race.cut(_line((9000 + late_fix, 0), (9000, 0), (4000, 0)), old + new_home, 300.0, 500.0)
    with pytest.raises(race.PrivateEnds):  # and the same at the end: stopped 300 m before the door
        race.cut(_line((2000, 3000), (9000, 3000), (9000, 300)), old + new_home, 300.0, 500.0)
    pieces, _, _, stretch = race.cut(race_loop, old, 300.0, 500.0)
    assert stretch[0] == 0.0, "a loop from a place no other run starts from reads as a race start (the log says so)"


def test_stray_fixes_at_the_ends_are_dropped_however_often_they_repeat():
    from_home = _loop(0, 0, 2000)
    for k in (1, 2, 11, 30):
        run_t = list(k + render_race.along(from_home) / 3.0)  # running at 3 m/s
        clock = list(range(k)) + run_t + [run_t[-1] + 1 + i for i in range(k)]  # the stale fixes a second apart
        for stale in ([[2600.0, -1800.0]] * k, [[2600.0 + 4 * i, -1800.0 + 3 * i] for i in range(k)]):
            glitch = np.vstack([stale, from_home, stale])  # and the same at the end
            for secs in (None, clock):
                assert ingest.stray_ends(_lonlat(glitch), secs) == (k, k + len(from_home)), (k, secs is None)
    two_stale = np.vstack([[[2600.0, -1800.0]] * 5, [[-2500.0, 1900.0]] * 8, from_home])
    act = _act("2", None, _lonlat(two_stale))
    assert len(ingest.clean(act).lonlat) == len(from_home), "a second stale place is dropped too, in one call"
    sparse = np.column_stack([np.arange(25) * 350.0, np.zeros(25)])  # a fix every 350 m: a sparse recording
    assert ingest.stray_ends(_lonlat(sparse)) == (0, 25)
    assert ingest.stray_ends(_lonlat(sparse), list(np.arange(25) * 70.0)) == (0, 25)
    jumps = _act("j", None, _lonlat([[0, 0], [3000, 0], [6000, 0]]), times=[0.0, 1.0, 2.0])  # 3 km a second
    assert ingest.clean(jumps) is None, "nothing real left"
    # a stale place with 400 m of wandering, a jump, another one, a jump home: the second jump lies beyond the first
    # 500 m of movement until the first stale place is gone, so cleaning goes round again
    wander = [_line((2600, -1800), (2600, -1400)), _line((-2500, 1900), (-2500, 2300))]
    twice = np.vstack([wander[0], wander[1], from_home])
    assert len(ingest.clean(_act("w", None, _lonlat(twice))).lonlat) == len(from_home)
    gappy = _act("g", None, _lonlat(np.vstack([[[2600.0, -1800.0]], from_home])), times=[None] * (len(from_home) + 1))
    assert len(ingest.clean(gappy).lonlat) == len(from_home), "a clock with holes is no clock, not a crash"
    head, _ = race._private_ends(glitch, render_race.along(glitch), [(0.0, 0.0, 500.0)],
                                 privacy.trim_span(glitch, 300.0))
    # without cleaning, the first 300 m along the track all lie on the jump 3 km from home; the trim's 200 m in a
    # straight line from the first fix (the stale one) now reaches past the jump, so even then home is caught
    assert head == [(0.0, 0.0, 500.0)], "the start at home, caught even without cleaning"
    # a fix that drifts in by 200 m steps is not one big jump, but the clock shows it moving at 200 m/s
    drift = np.vstack([[[800.0, 0.0], [600.0, 0.0], [400.0, 0.0], [200.0, 0.0]], from_home])
    clock = [0.0, 1.0, 2.0, 3.0] + list(4 + render_race.along(from_home) * 0.3)
    assert ingest.stray_ends(_lonlat(drift), clock) == (4, len(drift))
    act = _act("d", dt.datetime(2025, 3, 2, 8, tzinfo=UTC), _lonlat(drift), times=clock, ele=[600.0] * len(drift))
    clean = ingest.clean(act)
    assert len(clean.lonlat) == len(from_home) and clean.times[0] == 0.0 and len(clean.ele) == len(from_home)
    assert clean.start == act.start + dt.timedelta(seconds=4), "the run starts at its first real fix"
    assert ingest.clean(clean) is clean


def test_home_keeps_its_zone_when_a_chain_of_starts_joins_it_to_a_busy_park():
    """Runs from home, one start every 180 m along the street, many more from a park 1.1 km away: one chain of grid
    cells whose mean sits far from home. Home must keep a centre of its own wherever it sits in its grid cell."""
    starts = [0.0] * 5 + [180.0, 360.0, 540.0, 720.0, 900.0] + [1100.0] * 15
    tracks = [_loop(x, 0, 1500, phase=0.4 * i) for i, x in enumerate(starts)]
    pieces, rep = privacy.apply(tracks, trim_m=50.0, zone_r=500.0)
    assert min(np.hypot(*c) for c in rep.centres) < 150, "a centre by home, well inside its 500 m zone"
    assert MultiLineString([p for _, p in pieces]).distance(HOME) >= 200
    rng = np.random.default_rng(3)
    for _ in range(40):  # 15 runs from home among 500, their fixes jittered across a cell edge
        h = rng.uniform(0, 180, 2)
        pts = [h + rng.uniform(-20, 20, 2) for _ in range(30)]
        pts += [h + (180.0 * j, 0) + rng.uniform(-20, 20, 2) for j in range(1, 6) for _ in range(2)]
        pts += [h + (1100.0, 0) + rng.uniform(-20, 20, 2) for _ in range(2 * 480)]
        found = privacy._clusters(np.array(pts), 180.0, 20)
        assert min(np.hypot(*(c - h)) for c, _ in found) < 250


def _cli_world(monkeypatch, acts, captured):
    monkeypatch.setattr(ingest, "read_export", lambda path: (acts, ingest.IngestReport()))
    monkeypatch.setattr(race.area, "fetch_basemap", lambda bbox, log=print: "unused")
    monkeypatch.setattr(race.render, "load_basemap", lambda folder, bbox, crs: (EMPTY, EMPTY, EMPTY))
    monkeypatch.setattr(race.area, "place_names", lambda lat, lon, lang="en": ("Madrid", "Spain"))
    real = render_race.render

    def spy(*args, **kw):
        info = real(*args, **kw)
        captured.update(kw, track=args[1], pieces=args[2], info=info)
        return info

    monkeypatch.setattr(race.render_race, "render", spy)


def _run_act(xy, i, start, name="", pace=0.3):
    d = render_race.along(xy)
    return _act(i, start, _lonlat(xy), times=list(d * pace), ele=list(650 + 20 * np.sin(d / 900)), name=name)


def test_cli_race_titles_with_the_city_and_refuses_a_run_from_home(monkeypatch, tmp_path):
    start = dt.datetime(2025, 3, 2, 8, tzinfo=UTC)
    # every other run opens and closes with two stale fixes, each somewhere else 3 km out: home is learned only once
    # they are cleaned away
    stale = [np.array([[3000 * np.cos(a), 3000 * np.sin(a)]] * 2) for a in np.linspace(0, 6, 2 * len(POOL))]
    pool = [_run_act(np.vstack([stale[2 * i], t, stale[2 * i + 1]]), i, start - dt.timedelta(days=i + 1))
            for i, t in enumerate(POOL)]
    away = _line((0, 0), (7000, 2500))  # from home to 7 km east
    glitch = np.vstack([[[2600.0, -1800.0]] * 2, away])  # and the watch's stale first fix 3 km away, twice
    long_run = _run_act(glitch, "home", start, name="Long run from Mum's house, Calle Mayor 12")
    captured = {}
    _cli_world(monkeypatch, pool + [long_run], captured)
    out = str(tmp_path / "race.png")
    base = ["x.zip", "--layout", "race", "--activity", "home", "--dpi", "20", "--out", out]
    assert cli.main(base) == 2 and not captured, "a run from home is refused by default"
    assert cli.main(base + ["--allow-private-ends"]) == 0
    assert captured["title"] == "Madrid", "the activity name is not printed unless asked"
    lo, hi = captured["stretch"]
    assert lo >= 1000 and captured["info"]["km"] == pytest.approx((hi - lo) / 1000, rel=0.01)
    assert captured["info"]["km"] < render_race.along(captured["track"])[-1] / 1000 - 1.0, "numbers of the drawn part"
    assert render_race.along(captured["track"])[-1] < 7500, "the stale fixes are not part of the run"
    assert cli.main(base + ["--allow-private-ends", "--title", "Sunday long run"]) == 0
    assert captured["title"] == "Sunday long run"


def test_cli_race_options(monkeypatch, tmp_path, capsys):
    nye = _act("nye", dt.datetime(2024, 12, 31, 23, 30, tzinfo=UTC), _lonlat([[0, 0], [0, 3000]]))  # 00:30 in Madrid
    got = {}
    monkeypatch.setattr(ingest, "read_export", lambda path: ([nye], ingest.IngestReport()))
    monkeypatch.setattr(race, "run", lambda sel, pool, a: got.update(sel=sel) or 0)
    assert cli.main(["x.zip", "--layout", "race", "--style", "midnight", "--year", "2025"]) == 0
    assert [x.id for x in got["sel"]] == ["nye"], "the year is the local one"
    assert cli.main(["x.zip", "--layout", "race", "--year", "2024"]) == 2
    monkeypatch.setattr(ingest, "read_export", lambda path: ([], ingest.IngestReport()))
    assert cli.main(["x.zip", "--layout", "race", "--style", "neon"]) == 2, "an empty export, not a KeyError"
    for bad in (["--activity", "2025-02-30"], ["--color-by", "month"], ["--allow-private-ends", "--layout", "atlas"]):
        with pytest.raises(SystemExit) as ex:
            cli.main(["x.zip", "--layout", "race"] + bad if "--layout" not in bad else ["x.zip"] + bad)
        assert ex.value.code == 2
    capsys.readouterr()


def test_race_poster_draws_without_a_basemap(tmp_path):
    xy = np.column_stack([np.linspace(0, 5000, 501), 800 * np.sin(np.linspace(0, 3, 501))])
    act = _act("r", dt.datetime(2025, 4, 6, 7, tzinfo=UTC), [(-3.70, 40.41)] * 501, times=list(np.linspace(0, 1500, 501)),
               ele=list(600 + 40 * np.sin(np.linspace(0, 6, 501))), name="Spring 5K")
    pieces = [(0, LineString(xy[30:-30]))]
    out = tmp_path / "race.png"
    info = render_race.render(act, xy, pieces, (EMPTY, EMPTY, EMPTY), (-500, 5500, -2500, 2500), str(out), dpi=20,
                              title="Spring 5K", city="Madrid, Spain", name="Lucia", date=act.start, color_by="pace",
                              svg_path=str(tmp_path / "race.svg"))
    assert out.stat().st_size > 0 and info["km"] == pytest.approx(5.0, rel=0.2) and info["legend"]["mode"] == "pace"
    no_clock = _act("n", act.start, act.lonlat, times=list(np.zeros(501)), ele=None)
    info = render_race.render(no_clock, xy, pieces, (EMPTY, EMPTY, EMPTY), (-500, 5500, -2500, 2500), str(out),
                              dpi=20, color_by="pace")
    assert info["secs"] is None and info["legend"]["mode"] == "single", "a frozen clock: no pace, no division by 0"


def test_the_fastest_split_is_a_full_kilometre(tmp_path):
    d = np.linspace(0, 3500, 351)  # 3 km at 5:00/km, then a 500 m sprint at 3:00/km
    t = np.where(d <= 3000, d * 0.3, 900 + (d - 3000) * 0.18)
    t[1000 < d] -= np.clip(d[1000 < d] - 1000, 0, 1000) * 0.02  # km 2 a little faster (4:40/km)
    xy = np.column_stack([d, np.zeros_like(d)])
    act = _act("s", dt.datetime(2025, 4, 6, 7, tzinfo=UTC), [(-3.70, 40.41)] * len(d), times=list(t))
    info = render_race.render(act, xy, [(0, LineString(xy[30:-30]))], (EMPTY, EMPTY, EMPTY), (-500, 4000, -2000, 2000),
                              str(tmp_path / "s.png"), dpi=20)
    assert info["fastest"] == 1 and info["splits"][1][0] == pytest.approx(180), "km 2, not the faster last 500 m"


def test_atlas_glyphs_are_not_snapped_to_pixels():
    fig = plt.figure()
    page = atlas._Page(fig, 297, 420, 1.0, render_editorial.STYLES["huella"])
    page.text(10, 10, "F", pt=4.2)
    plt.close(fig)
    assert page.ax.patches[-1].get_snap() is False


def test_graph_aggregates_use_local_time_and_fold_directions():
    acts = [_act(i, dt.datetime(2025, m, 10, 5, 30, tzinfo=UTC), [(-3.70, 40.41), (-3.70, 40.42)])
            for i, m in enumerate((1, 1, 7))]  # 6:30 in January, 7:30 in July (Madrid)
    tracks = [np.array([[0.0, 0.0], [0.0, 1000.0]])] * 3  # 1 km due north each
    km = graphs.km_per_month(acts, tracks)
    assert km[0] == pytest.approx(2.0) and km[6] == pytest.approx(1.0)
    assert sorted(graphs.start_hours(acts)) == pytest.approx([6.5, 6.5, 7.5])
    north_south = graphs.bearings([np.array([[0.0, 0.0], [0.0, 1000.0], [0.0, 0.0]])])
    assert north_south.argmax() == 0 and north_south.sum() == pytest.approx(2000)
