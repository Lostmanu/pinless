"""A poster for two: each person's places hide both lines, places split across the two exports are found, one zone per
place keeps its clearance, the area is where both run, and runs together need the same time and place."""
import datetime as dt
import math
import random

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from pinless import area, privacy, stats
from pinless.ingest import Activity

UTC = dt.timezone.utc
GEN = np.random.default_rng(20251011)


def loop(cx, cy, r, phase, n=160, noise=6.0):
    """A loop that starts and ends at (cx, cy), with GPS noise."""
    t = np.linspace(0, 2 * np.pi, n) + phase
    xy = np.column_stack([cx + r * (np.cos(t) - np.cos(phase)), cy + r * (np.sin(t) - np.sin(phase))])
    return xy + GEN.normal(0, noise, xy.shape)


def line(*corners, step=10.0, noise=0.0):
    out = [np.asarray(corners[0], dtype=float)[None]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        out.append(np.linspace(a, b, n)[1:])
    xy = np.vstack(out)
    return xy + GEN.normal(0, noise, xy.shape) if noise else xy


def through(p, angle=0.3, half=3000.0):
    """A run that passes straight over p, far from its own ends: only a zone can keep it away from p."""
    u = half * np.array([math.cos(angle), math.sin(angle)])
    return line(np.asarray(p) - u, np.asarray(p) + u, noise=4.0)


def loops_from(home, n, r0=1200.0):
    return [loop(home[0], home[1], r0 + 13 * (i % 31), phase=0.61 * i) for i in range(n)]


def couple_cut(ta, tb, rng=None, user=(), zone_r=500.0):
    """The couple path: places of both people, one forbidden area cutting both."""
    zones, witnesses, counts = privacy.pair_places([ta, tb], zone_r=zone_r)
    pieces, rep = privacy.apply(ta + tb, trim_m=300.0, zone_r=zone_r, user_zones_xy=user, places_xy=zones,
                                witnesses_xy=witnesses, rng=rng)
    return pieces, rep, zones, counts


def far(pieces, *points):
    union = MultiLineString([p for _, p in pieces])
    return min(union.distance(Point(p)) for p in points)


def near_any(found, p, d=250.0):
    return any(np.hypot(*(np.asarray(c) - p)) < d for c, _ in found)


def test_apply_with_given_places_matches_learned_places():
    tracks = loops_from((0.0, 0.0), 30) + loops_from((4000.0, 1000.0), 12)
    learned = privacy.find_places(tracks)
    assert len(learned) >= 2
    p1, r1 = privacy.apply(tracks, rng=random.Random(5))
    p2, r2 = privacy.apply(tracks, places_xy=[(c[0], c[1], 500.0) for c, _ in learned], rng=random.Random(5))
    assert [i for i, _ in p1] == [i for i, _ in p2]
    assert all(np.array_equal(np.asarray(a.coords), np.asarray(b.coords)) for (_, a), (_, b) in zip(p1, p2))
    assert len(r1.centres) == len(r2.centres) and all(np.array_equal(a, b) for a, b in zip(r1.centres, r2.centres))
    assert r1.zones_suggested == r2.zones_suggested == len(learned)


def test_merge_places_joins_close_centres_with_a_larger_radius(monkeypatch):
    assert privacy.merge_places([(0, 0), (170, 0)]) == [(85.0, 0.0, pytest.approx(712.5))]
    assert privacy.merge_places([(0, 0), (400, 0)]) == [(0.0, 0.0, 500.0), (400.0, 0.0, 500.0)]
    assert privacy.merge_places([(0, 0), (170, 0), (340, 0)]) == [(170.0, 0.0, pytest.approx(925.0))]
    assert privacy.merge_places([(400, 0), (0, 0), (170, 0)])[0][:2] == (400.0, 0.0), "in order of first member"
    # a --zone is never merged: it keeps its own centre and radius next to a place
    calls, real = [], privacy._irregular_zone
    monkeypatch.setattr(privacy, "_irregular_zone", lambda c, r, rng, n=14: calls.append((tuple(c), r)) or real(c, r, rng))
    privacy.apply([line((-4000, 2000), (4000, 2000))], places_xy=[(0.0, 0.0, 712.5)], user_zones_xy=[(100.0, 0.0, 300.0)],
                  rng=random.Random(1))
    assert calls == [((0.0, 0.0), 712.5), ((100.0, 0.0), 300.0)], "each place with its own radius, then the --zone"


def test_a_witness_is_kept_clear_or_nothing_is_exported():
    passing = [line((-5000, 100), (5000, 100))]  # 100 m from a witness that no zone covers
    with pytest.raises(privacy.PrivacyError):
        privacy.apply(passing, places_xy=[(-3000.0, 2000.0, 500.0)], witnesses_xy=[np.array([2000.0, 0.0])],
                      rng=random.Random(1))
    pieces, rep = privacy.apply(passing, places_xy=[(-3000.0, 2000.0, 500.0)], rng=random.Random(1))
    assert rep.leak_check == "passed" and len(rep.centres) == 1


class WorstDraw:
    """A scripted rng for _irregular_zone: the largest offset, straight away from a member that lies on the bisector
    of the widest possible gap between two vertices, and every vertex at its nearest."""

    def __init__(self, away):
        self.away, self.vertex = away, 0

    def uniform(self, lo, hi):
        if (lo, hi) == (-0.15, 0.15):  # vertex 13 turns back, vertex 0 forward: the gap between them is widest
            i, self.vertex = self.vertex, self.vertex + 1
            return {0: 0.15, 13: -0.15}.get(i, 0.0)
        if (lo, hi) == (0.85, 1.25):
            return 0.85
        if lo == 0 and abs(hi - 2 * math.pi) < 1e-12:
            return self.away
        return hi  # the offset: 0.35 r


def test_a_merged_zone_keeps_the_lone_zone_clearance_in_its_worst_draw():
    phi = -math.pi / 14  # the bisector of the gap between vertices 13 and 0
    u = np.array([math.cos(phi), math.sin(phi)])

    def clearance(d, grow):
        m = max(1, math.ceil(2 * d / 170) + 1) if d else 1
        members = [t * u for t in np.linspace(-d, d, m)]
        (x, y, r), = privacy.merge_places(members, grow=grow)
        zone = privacy._irregular_zone((x, y), r, WorstDraw((phi + math.pi) % (2 * math.pi)))
        member = Point(*(np.array([x, y]) + d * u))
        assert zone.contains(member)
        return zone.exterior.distance(member)

    for d in (0, 90, 180, 400):
        assert clearance(d, 2.5) >= 0.4411 * 500 - 1, d
    assert clearance(0, 2.5) == pytest.approx(0.4411 * 500, abs=0.5), "the scripted draw is the worst one"
    assert clearance(180, 2.0) < 200, "with 2 d the 200 m leak check would refuse at random"


def test_each_persons_places_cut_both_lines():
    ha, hb = np.array([0.0, 0.0]), np.array([6000.0, 0.0])
    ta = loops_from(ha, 300) + [line((5400, dy), (9000, dy), (9000, 3000 + 400 * k), noise=4.0)
                                for k, dy in enumerate((0.0, 20.0, -20.0))]
    tb = loops_from(hb, 80)
    solo, _ = privacy.apply(ta)
    assert far(solo, hb) < 20, "control: alone, the first person's runs past the other's door are drawn"
    pieces, rep, zones, counts = couple_cut(ta, tb, rng=random.Random(3))
    assert counts["per"][0] >= 1 and counts["per"][1] >= 1
    assert far(pieces, hb) >= 200 and far(pieces, ha) >= 200
    assert {i < len(ta) for i, _ in pieces} == {True, False}, "both people are still drawn"


def test_the_quieter_partner_keeps_their_own_places():
    ha, hb, gym = np.array([0.0, 0.0]), np.array([7000.0, 0.0]), np.array([7000.0, 4000.0])
    ta = loops_from(ha, 400)
    tb = loops_from(hb, 20) + loops_from(gym, 6, r0=900.0) + [through(gym)]
    assert not near_any(privacy.find_places(ta + tb), gym), "control: the union's threshold misses the gym"
    union_only, _ = privacy.apply(ta + tb, min_pts_floor=2)
    assert far(union_only, gym) < 20, "control: learning from the union of both exports leaves the gym drawn"
    pieces, _, _, _ = couple_cut(ta, tb, rng=random.Random(4))
    assert far(pieces, ha) >= 200 and far(pieces, hb) >= 200 and far(pieces, gym) >= 200


def _split_place(k, p=(6000.0, 4500.0)):
    ha, hb, p = np.array([0.0, 0.0]), np.array([0.0, 9000.0]), np.asarray(p, dtype=float)
    ends = [p + 4000 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(-1.2, 1.2, 2 * k)]
    ta = loops_from(ha, 300) + [line(p + GEN.normal(0, 10, 2), e, noise=4.0) for e in ends[:k]] + [through(p)]
    tb = loops_from(hb, 300) + [line(p + GEN.normal(0, 10, 2), e, noise=4.0) for e in ends[k:]]
    return ta, tb, p


def test_a_place_split_across_two_exports_is_hidden():
    ta, tb, p = _split_place(7)
    for tracks in (ta, tb, ta + tb):
        assert not near_any(privacy.find_places(tracks, min_pts_floor=2), p), "control: 7 + 7 ends at thresholds 12/12"
    pooled = privacy.pooled_places([ta, tb])
    assert near_any(pooled, p)
    per_person_only, _ = privacy.apply(ta + tb, places_xy=privacy.merge_places(
        [c for g in (ta, tb) for c, _ in privacy.find_places(g, min_pts_floor=2)]), rng=random.Random(6))
    assert far(per_person_only, p) < 20, "control: without the pooled rule a line passing there is drawn"
    pieces, _, zones, counts = couple_cut(ta, tb, rng=random.Random(6))
    assert counts["pooled_new"] == 1 and far(pieces, p) >= 200
    assert counts["per"] == [1, 1] and counts["zones"] == 3 and counts["merged"] == 0, "the log's numbers add up"
    ta, tb, p = _split_place(5)  # 5/12 + 5/12 = 0.83: not enough evidence, no zone there
    zones, _, _ = privacy.pair_places([ta, tb])
    assert not any(np.hypot(x - p[0], y - p[1]) < 250 for x, y, _ in zones)


def test_the_pooled_rule_at_its_edges():
    """The pooled rule takes every end (starts and finishes), labels each by its person and accepts exactly one full
    threshold of evidence: 6/12 + 6/12 is a place, and so is 20/40 + 1/2 when the two thresholds differ."""
    ta, tb, p = _split_place(6)  # 6/12 + 6/12 = 1 exactly
    assert near_any(privacy.pooled_places([ta, tb]), p)
    ta, tb, p = _split_place(7)
    ta = [t[::-1] for t in ta]
    tb = [t[::-1] for t in tb]  # the same runs ending at the place instead of starting there
    zones, _, counts = privacy.pair_places([ta, tb])
    assert counts["pooled_new"] == 1 and any(np.hypot(x - p[0], y - p[1]) < 250 for x, y, _ in zones)
    assert far(couple_cut(ta, tb, rng=random.Random(9))[0], p) >= 200
    ta, tb, p = _split_place(11)  # 22 one-way runs from p: 20 for the first person, 1 for the second
    first, second = ta[:311] + tb[300:309] + [ta[-1]], tb[:50] + [tb[309]]
    assert len(first) == 321 and len(second) == 51
    assert near_any(privacy.pooled_places([first, second], n_runs=(1000, 50)), p), "20/40 + 1/2 = 1"
    fewer = first[:300] + first[301:]
    assert not near_any(privacy.pooled_places([fewer, second], n_runs=(1000, 50)), p), "control: 19/40 + 1/2"


def test_pair_places_draws_its_zones_with_the_given_radius_and_keeps_every_centre_as_a_witness():
    ha, hb = np.array([0.0, 0.0]), np.array([6000.0, 0.0])
    ta, tb = loops_from(ha, 60), loops_from(hb, 60) + loops_from(ha + (100.0, 0.0), 10)  # B also starts by A's door
    zones, witnesses, counts = privacy.pair_places([ta, tb], zone_r=1000.0)
    per = [privacy.find_places(g, min_pts_floor=2) for g in (ta, tb)]
    pooled = privacy.pooled_places([ta, tb])
    assert len(witnesses) == sum(map(len, per)) + len(pooled) > len(zones), "every original centre, not the zones"
    for x, y, r in zones:
        members = [w for w in witnesses if np.hypot(w[0] - x, w[1] - y) < 180]
        d_max = max(np.hypot(w[0] - x, w[1] - y) for w in members)
        assert r == pytest.approx(1000.0 + 2.5 * d_max) and r >= 1000.0
    assert sorted(round(r) for _, _, r in privacy.pair_places([ta[:30], loops_from(hb, 30)], zone_r=1000.0)[0]) == \
        [1000, 1000], "lone places: exactly the radius asked for"


def test_a_place_next_to_a_bigger_one_gets_its_own_zone():
    """A gym 450 m from home shares a 3 x 3 block of cells with the home's many ends: that block's mean lies at home,
    so the gym needs a look of its own at the ends no centre covers yet. The same for a place where the two people's
    ends meet 450 m from one of their homes. 3 km away (the control) both were always found."""
    ha = np.array([0.0, 0.0])
    for g, label in ((np.array([3000.0, 0.0]), "control: 3 km from home"), (np.array([320.0, 320.0]), "450 m")):
        ta = loops_from(ha, 280) + loops_from(g, 20, r0=900.0) + [through(g)]  # 40 ends at the gym, threshold 12
        tb = loops_from(np.array([0.0, 9000.0]), 200)
        assert any(np.hypot(*(c - g)) < 60 for c, _ in privacy.find_places(ta)), label
        for seed in range(3):
            assert far(privacy.apply(ta, rng=random.Random(seed))[0], g) >= 200, (label, "solo")
            assert far(couple_cut(ta, tb, rng=random.Random(seed))[0], g) >= 200, (label, "couple")
    ta, tb, p = _split_place(7, p=(320.0, 320.0))  # 7 + 7 ends at thresholds 12/12, 450 m from the first home
    zones, witnesses, counts = privacy.pair_places([ta, tb])
    assert any(np.hypot(*(w - p)) < 60 for w in witnesses) and counts["pooled_new"] == 1
    for seed in range(3):
        assert far(couple_cut(ta, tb, rng=random.Random(seed))[0], p) >= 200, seed


def test_a_partner_with_few_runs_marks_a_single_loop_start():
    ha, lodging = np.array([0.0, 0.0]), np.array([8000.0, 2000.0])
    ta = loops_from(ha, 100)
    tb = [loop(lodging[0], lodging[1], 1400, phase=0.4), through(lodging, angle=1.2)]
    tb += [line((9000 + 700 * i, -3000 + 300 * i), (12000 - 400 * i, 4000 + 500 * i), noise=4.0) for i in range(6)]
    floor3, _ = privacy.apply(tb)
    assert far(floor3, lodging) < 20, "control: with the solo floor of 3 ends, one loop marks nothing"
    zones, witnesses, counts = privacy.pair_places([ta, tb])
    assert any(np.hypot(*(w - lodging)) < 180 for w in witnesses)
    pieces, _, _, _ = couple_cut(ta, tb, rng=random.Random(8))
    assert far(pieces, lodging) >= 200 and far(pieces, ha) >= 200


def test_a_shared_home_gets_one_zone(monkeypatch):
    door = np.array([500.0, -300.0])
    ta, tb = loops_from(door, 20) + [through(door)], loops_from(door, 20, r0=1500.0)
    calls, real = [], privacy._irregular_zone
    monkeypatch.setattr(privacy, "_irregular_zone", lambda c, r, rng, n=14: calls.append(np.asarray(c)) or real(c, r, rng))
    couple_cut(ta, tb, rng=random.Random(0))
    assert sum(np.hypot(*(c - door)) < 180 for c in calls) == 1, "one zone for the home both learn"
    monkeypatch.setattr(privacy, "_irregular_zone", real)
    for seed in range(20):
        pieces, _, _, _ = couple_cut(ta, tb, rng=random.Random(seed))
        assert far(pieces, door) >= 200


def _cut_ends(pieces, keep, centre=(0.0, 0.0), reach=1000.0):
    ends = [np.asarray(p.coords)[j] for i, p in pieces if keep(i) for j in (0, -1)]
    return [e for e in ends if np.hypot(*(e - centre)) < reach]


def test_both_inks_break_at_the_same_edge():
    ta = [line((-5000, 0), (5000, 0))]
    tb = [ta[0] + (0.0, 3.0)]
    zone = [(0.0, 0.0, 500.0)]
    for seed in range(20):
        pieces, _, _, _ = couple_cut(ta, tb, rng=random.Random(seed), user=zone)
        ea, eb = _cut_ends(pieces, lambda i: i == 0), _cut_ends(pieces, lambda i: i == 1)
        assert len(ea) == len(eb) == 2
        assert max(min(np.hypot(*(a - b)) for b in eb) for a in ea) <= 10, seed
    apart = 0
    for seed in range(20):  # control: each person cut alone draws their own zone
        pa, _ = privacy.apply(ta, user_zones_xy=zone, rng=random.Random(seed))
        pb, _ = privacy.apply(tb, user_zones_xy=zone, rng=random.Random(1000 + seed))
        ea, eb = _cut_ends(pa, lambda i: True), _cut_ends(pb, lambda i: True)
        apart += max(min(np.hypot(*(a - b)) for b in eb) for a in ea) > 50
    assert apart > 10


def _lonlat_loop(lat, lon, r_deg, n=200, phase=0.0):
    t = np.linspace(0, 2 * np.pi, n) + phase
    return list(zip(lon + r_deg * np.cos(t), lat + r_deg * np.sin(t)))


def _runs(where, n, first):
    return [Activity(id=str(first + i), type="Run", start=dt.datetime(2025, 1, 1),
                     lonlat=_lonlat_loop(*where, 0.012 + 0.0004 * i, phase=0.3 * i), source="x") for i in range(n)]


def test_auto_bbox_with_groups_picks_where_both_run():
    x, y = (40.42, -3.70), (40.48, -3.37)  # Madrid, and a town 28 km east
    a = _runs(x, 50, 0) + _runs(y, 10, 100)
    b = _runs(y, 30, 200)
    s, w, n, e = area.auto_bbox(a + b, groups=[0] * len(a) + [1] * len(b))
    assert s < y[0] < n and w < y[1] < e and not (s < x[0] < n and w < x[1] < e)
    s, w, n, e = area.auto_bbox(a + b)
    assert s < x[0] < n and w < x[1] < e, "alone, the busier place wins"
    # the result without groups is exactly the one before groups existed, rounded outward to the 1 km grid
    old = area.snap_box((40.38707439709491, -3.73294147338975, 40.452967765907836, -3.6670670730911157))
    assert area.auto_bbox(a + b) == old and area.auto_bbox(a + b, groups=None) == old
    with pytest.raises(ValueError):
        area.auto_bbox(_runs(x, 20, 0) + b, groups=[0] * 20 + [1] * len(b))


def _act(i, start, xy, times):
    return Activity(id=str(i), type="Run", start=start, lonlat=[(-3.7, 40.4)] * len(xy), source="x",
                    times=None if times is None else list(times))


def test_runs_together_need_the_same_time_and_place():
    route = line((0, 0), (4000, 0), (4000, 1000), (0, 1000), step=5.0)  # 10 km
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    t0 = dt.datetime(2025, 5, 4, 8, tzinfo=UTC)
    clock = d / 3.3  # about 50 minutes
    a = [_act("a", t0, route, clock)]
    side = route + (0.0, 5.0)
    got =stats.together(a, [route], [_act("b", t0, side, clock)], [side])
    assert got["runs"] == 1 and got["km"] == pytest.approx(d[-1] / 1000, abs=0.3) and got["known"]
    later = stats.together(a, [route], [_act("b", t0 + dt.timedelta(hours=1), side, clock)], [side])
    assert later["runs"] == 0
    away = route + (0.0, 2000.0)
    assert stats.together(a, [route], [_act("b", t0, away, clock)], [away])["runs"] == 0
    slower = stats.together(a, [route], [_act("b", t0, side, clock / 0.9)], [side])
    assert slower["runs"] == 0, "the same race, drifting apart"
    half = len(route) // 2  # the second watch restarted halfway: two files over one run
    split = [_act("b1", t0, side[:half], clock[:half]),
             _act("b2", t0 + dt.timedelta(seconds=float(clock[half])), side[half:], clock[half:] - clock[half])]
    got = stats.together(a, [route], split, [side[:half], side[half:]])
    assert got["runs"] == 1, "one to one"
    naive = _act("b", t0.replace(tzinfo=None), side, clock)  # a naive start is UTC
    assert stats.together(a, [route], [naive], [side])["runs"] == 1
    no_clock = stats.together(a, [route], [_act("b", None, side, None)], [side])
    assert no_clock == {"runs": 0, "km": 0.0, "known": False, "no_clock": (0, 1)}
    twice = [_act("a1", t0, route, clock), _act("a2", t0, route + (0.0, 2.0), clock)]  # two devices on one run
    got = stats.together(twice, [route, route + (0.0, 2.0)], [_act("b", t0, side, clock)], [side])
    assert got["runs"] == 1 <= min(2, 1)
    paused = clock.copy()
    paused[100:140] = paused[100]  # a paused watch repeats its stamp
    assert stats.together(a, [route], [_act("b", t0, side, paused)], [side])["runs"] == 1


def test_each_runs_together_rule_is_needed():
    """One case per rule that only that rule turns down, at the same clock: a parallel street 100 m away (median gap),
    60 % side by side and then 2 km apart (share of the shared time within 150 m), a partner who joins for the last
    15 of 45 minutes (half of the shorter run), an 8-minute jog alongside (10 minutes at least)."""
    route = line((0, 0), (9000, 0), step=5.0)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    t0 = dt.datetime(2025, 5, 4, 8, tzinfo=UTC)
    clock = d / 3.3  # 45 minutes
    a = [_act("a", t0, route, clock)]

    def runs(xy, times, start=t0):
        return stats.together(a, [route], [_act("b", start, xy, times)], [xy])["runs"]

    assert runs(route + (0.0, 5.0), clock) == 1, "control: side by side"
    assert runs(route + (0.0, 100.0), clock) == 0, "a parallel street"
    apart = route + np.where(d[:, None] > 0.6 * d[-1], (0.0, 2000.0), (0.0, 5.0))
    assert runs(apart, clock) == 0, "together for 60 %, then 2 km apart"
    late = clock >= clock[-1] - 900  # the last 15 minutes of the first run, then 30 more minutes elsewhere
    tail = np.vstack([route[late], line(route[-1] + (0.0, 50.0), (9000, 6000), step=5.0)])
    td = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(tail, axis=0).T))]) / 3.3
    assert runs(tail, td, t0 + dt.timedelta(seconds=float(clock[late][0]))) == 0, "joins for 15 of 45 minutes"
    short = clock <= 480
    eight = stats.together([_act("a8", t0, route[short], clock[short])], [route[short]],
                           [_act("b8", t0, route[short] + (0.0, 5.0), clock[short])], [route[short] + (0.0, 5.0)])
    assert eight["runs"] == 0, "an 8-minute jog alongside"


def test_runs_together_km_and_matching():
    """The km of a run together is the mean of what each covered over the shared time (a jittery watch covers more),
    and the runs are matched one to one, longest overlap first."""
    route = line((0, 0), (9000, 0), step=5.0)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    t0 = dt.datetime(2025, 5, 4, 8, tzinfo=UTC)
    clock = d / 3.3
    zig = route + np.column_stack([np.zeros(len(route)), np.where(np.arange(len(route)) % 2, 15.0, -15.0)])
    dz = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(zig, axis=0).T))])
    got = stats.together([_act("a", t0, route, clock)], [route], [_act("b", t0, zig, clock)], [zig])
    assert got["runs"] == 1 and got["km"] == pytest.approx((d[-1] + dz[-1]) / 2000, rel=0.01)
    # one long route run at 3.3 m/s: where everyone is at each moment; each activity is a window of it
    long = line((0, 0), (25000, 0), step=5.0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(long, axis=0).T))])
    t_abs = s / 3.3

    def window(name, m0, m1, dy):
        keep = (t_abs >= 60 * m0) & (t_abs <= 60 * m1)
        xy = long[keep] + (0.0, dy)
        return _act(name, t0 + dt.timedelta(seconds=float(t_abs[keep][0])), xy, t_abs[keep] - t_abs[keep][0]), xy

    (a0, xa0), (a1, xa1) = window("a0", 0, 60, 0.0), window("a1", 45, 105, 0.0)
    (b0, xb0), (b1, xb1) = window("b0", 35, 75, 5.0), window("b1", 0, 40, 5.0)
    # overlaps: a0-b1 40 min, a1-b0 30 min, a0-b0 25 min: longest first matches two pairs, index order only one
    got = stats.together([a0, a1], [xa0, xa1], [b0, b1], [xb0, xb1])
    assert got["runs"] == 2
    names = ["Calle A", "Calle B", "Calle C", "Calle D"]
    geoms = [LineString([(0, 0), (1000, 0)]), LineString([(0, 1000), (1000, 1000)]),
             LineString([(0, 2000), (1000, 2000)]), LineString([(3000, 3000), (3200, 3000)])]
    edges = gpd.GeoDataFrame({"name": names}, geometry=geoms)
    via_d = [(3000, 3000), (3200, 3000), (3600, 3600), (5000, 5000)]
    ta = [line((-600, 0), (1000, 0), (1000, 1000), (0, 1000), *via_d)]
    tb = [line((-600, 0), (1000, 0), (1300, 300), (1300, 2000), (0, 2000), *via_d)]  # clear of Calle B
    acts = [Activity(id="a", type="Run", start=None, lonlat=[(0, 0)] * len(ta[0]), source="x"),
            Activity(id="b", type="Run", start=None, lonlat=[(0, 0)] * len(tb[0]), source="x")]
    pieces, rep, _, _ = couple_cut(ta, tb, rng=random.Random(2), user=[(3100.0, 3000.0, 500.0)])

    def streets(k, tracks, cut=True):
        mine = [(i, p) for i, p in pieces if (i >= 1) == k]
        pp = [np.asarray(p.coords) for _, p in mine] if cut else tracks
        groups = [i for i, _ in mine] if cut else list(range(len(tracks)))
        return stats.compute([acts[k]], tracks, edges, private_pieces_xy=pp, piece_groups=groups,
                             keep_streets=True)["street_names"]

    sa, sb = streets(0, ta), streets(1, tb)
    assert sa & sb == {"Calle A"} and len(sa | sb) == 3
    assert streets(0, ta, cut=False) & streets(1, tb, cut=False) == {"Calle A", "Calle D"}, "control: full tracks"
    assert "street_names" not in stats.compute([acts[0]], ta, edges, private_pieces_xy=[ta[0]], piece_groups=[0])
