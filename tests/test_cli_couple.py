"""pinless EXPORT --with EXPORT2: flags checked before any file is read, each export read with its own type and named
by its person (never by its path), the same export twice refused before any request, and the whole flow offline."""
import dataclasses
import datetime as dt
import re

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point

from pinless import area, cli, couple, ingest, privacy, render, render_editorial
from pinless.ingest import Activity
from test_cli_editorial import EMPTY, _lonlat, madrid_runs, stub_world
from test_couple_privacy import line, loop, loops_from, through

BASE = ["a.zip", "--with", "b.zip", "--name", "Lucía", "--with-name", "Marcos"]
HA, HB = (0.0, 0.0), (1500.0, -1800.0)  # the two homes, metres from the map's reference point
BOX = "40.36,-3.80,40.50,-3.58"  # about 8 km west to 10 km east and 6 km south to 9 km north of that point
UTC = dt.timezone.utc


def _acts(tracks, first_id, day0=dt.date(2025, 1, 6), hour=7, pace=0.3):
    """Activities with a clock from tracks in metres east/north of the reference point, one a day."""
    out = []
    for i, xy in enumerate(tracks):
        d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        start = dt.datetime.combine(day0 + dt.timedelta(days=i), dt.time(hour), tzinfo=UTC)
        out.append(Activity(id=str(first_id + i), type="Run", start=start, lonlat=_lonlat(xy), source="x",
                            times=list(d * pace)))
    return out


def _to_map(*points):
    """Metres east/north of the reference point to the poster's projected metres."""
    tr = Transformer.from_crs(4326, render.utm_crs(-3.70, 40.42), always_xy=True)
    return [np.array(tr.transform(*ll)) for ll in _lonlat(points)]


def _spy_drawn(monkeypatch):
    """Every route vertex the poster draws (PathPatch at zorder 5), and what render received."""
    seen, drawn, real, real_patch = {}, [], render_editorial.render, render_editorial.PathPatch

    def spy(*args, **kw):
        info = real(*args, **kw)
        seen.update(kw, info=info, extent=args[2], pieces=args[0])
        return info

    def patch(path, *a, **k):
        if k.get("zorder") == 5:
            drawn.append(np.asarray(path.vertices))
        return real_patch(path, *a, **k)

    monkeypatch.setattr(render_editorial, "render", spy)
    monkeypatch.setattr(render_editorial, "PathPatch", patch)
    return seen, drawn


def _gap(drawn, p):
    pts = np.concatenate(drawn)
    return float(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]).min())


def _never(*a, **k):
    raise AssertionError("no file and no network before the flags are checked")


@pytest.mark.parametrize("argv,message", [
    (["a.zip", "--with-name", "Marcos"], "--with-name and --with-type need --with"),
    (["a.zip", "--with-type", "Carrera"], "--with-name and --with-type need --with"),
    (BASE + ["--layout", "atlas"], "available for the editorial layout"),
    (BASE + ["--layout", "race"], "available for the editorial layout"),
    (BASE + ["--layout", "classic"], "available for the editorial layout"),
    (BASE + ["--layout", "sport"], "available for the editorial layout"),
    (["a.zip", "--with", "b.zip", "--with-name", "Marcos"], "--with needs both names"),
    (["a.zip", "--with", "b.zip", "--name", "Lucía"], "--with needs both names"),
    (["a.zip", "--with", "b.zip", "--name", "Ana", "--with-name", " ana "], "they must differ"),
    (BASE + ["--color-by", "pace"], "--color-by does not apply"),
    (BASE + ["--color-by", "heat"], "--color-by does not apply"),
    (BASE + ["--graphs"], "--graphs is not available with --with"),
    (["SAME", "--with", "SAME", "--name", "Lucía", "--with-name", "Marcos"], "the same export as the first one"),
    (BASE + ["--bbox", "40.36,-3.78,40.48"], "--bbox needs four numbers"),
    (BASE + ["--bbox", "40.48,-3.78,40.36,-3.62"], "--bbox needs four numbers"),
    (["a.zip", "--bbox", "40.36,-3.78,40.48"], "--bbox needs four numbers"),
    (["a.zip", "--bbox", "40.36,-3.78,40.48,nan"], "--bbox needs four numbers"),
    (BASE + ["--zone", "40.41,-3.69,nan"], "--zone needs LAT,LON,RADIUS_M"),
    (BASE + ["--zone", "nan,nan,400"], "--zone needs LAT,LON,RADIUS_M"),
    (["a.zip", "--zone", "40.41,-3.69,400", "--zone", "40.41,-3.69"], "--zone needs LAT,LON,RADIUS_M"),
    (["a.zip", "--zone", "40.41,-183.69,400"], "--zone needs LAT,LON,RADIUS_M"),
    (["a.zip", "--zone", "-93.70,40.41,400"], "--zone needs LAT,LON,RADIUS_M"),
    (["a.zip", "--zone", "40.41,-3.69,0"], "--zone needs LAT,LON,RADIUS_M"),
    (["a.zip", "--layout", "atlas", "--zone", "40.41,-3.69,inf"], "--zone needs LAT,LON,RADIUS_M"),
    (["a.zip", "--layout", "race", "--zone", "40.41,-3.69,-50"], "--zone needs LAT,LON,RADIUS_M"),
])
def test_cli_couple_flags_are_checked(argv, message, monkeypatch, tmp_path, capsys):
    same = tmp_path / "export.zip"
    same.write_bytes(b"PK")
    monkeypatch.setattr(ingest, "read_export", _never)
    with pytest.raises(SystemExit) as ex:
        cli.main([str(same) if x == "SAME" else x for x in argv])
    assert ex.value.code == 2 and message in capsys.readouterr().err


def _by_path(monkeypatch, exports):
    stub_world(monkeypatch, lambda path: (exports[path], ingest.IngestReport(files_seen=len(exports[path]),
                                                                                used=len(exports[path]))))


def _partner(n=25, **kw):
    return madrid_runs(home=HB, work=(-2000.0, 1500.0), n_home=n, n_work=5, first_id=1000, hour=18, **kw)


def test_cli_couple_reads_each_export_with_its_own_type_and_names_the_person(monkeypatch, tmp_path, capsys):
    a_path, b_path = r"C:\exports\export_48213377.zip", r"C:\exports\export_99120455.zip"
    acts_b = [dataclasses.replace(x, type="Carrera") for x in _partner()]
    _by_path(monkeypatch, {a_path: madrid_runs(), b_path: acts_b})
    argv = [a_path, "--with", b_path, "--name", "Lucía", "--with-name", "Marcos", "--out", str(tmp_path / "p.png")]
    assert cli.main(argv) == 2
    out = capsys.readouterr().out
    assert "Marcos" in out and "Carrera" in out and "--with-type" in out
    assert "export_" not in out and "99120455" not in out and "48213377" not in out, "people are named, paths never"
    seen = {}
    monkeypatch.setattr(couple, "run", lambda acts_a, acts_b_, sel_a, sel_b, a: seen.update(sel_b=sel_b) or 0)
    assert cli.main(argv + ["--with-type", "Carrera"]) == 0
    assert seen["sel_b"] == acts_b
    capsys.readouterr()


def test_the_same_export_twice_is_refused_before_any_network(monkeypatch, tmp_path, capsys):
    acts = madrid_runs()
    _by_path(monkeypatch, {"a.zip": acts, "b.zip": list(acts)})
    monkeypatch.setattr(area, "fetch_basemap", _never)
    monkeypatch.setattr(area, "city_name", _never)
    monkeypatch.setattr(area, "geocode_bbox", _never)
    assert cli.main(BASE + ["--out", str(tmp_path / "p.png")]) == 2
    assert "same activities" in capsys.readouterr().out and not list(tmp_path.iterdir())
    acts_a = madrid_runs(n_home=30, n_work=0)
    acts_b = _partner(n=24) + [acts_a[0]]  # a partner who uploaded one of the other's runs when a watch died
    _by_path(monkeypatch, {"a.zip": acts_a, "b.zip": acts_b})
    assert cli.main(BASE + ["--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    assert "[couple] 1 activity is in both exports" in capsys.readouterr().out
    _by_path(monkeypatch, {"a.zip": acts, "b.zip": acts[:20]})  # a part of one export is still the same export
    assert cli.main(BASE + ["--out", str(tmp_path / "q.png")]) == 2
    assert "same activities (20 of 20)" in capsys.readouterr().out and not (tmp_path / "q.png").exists()


def test_a_mistyped_export_path_is_named_as_missing(monkeypatch, tmp_path, capsys):
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": []})  # neither file exists: the stub reads b.zip as empty
    assert cli.main(BASE + ["--out", str(tmp_path / "p.png")]) == 2
    out = capsys.readouterr().out
    assert "Marcos: there is no file or folder at the path given for --with" in out and "--with-type" not in out
    assert "b.zip" not in out and not list(tmp_path.iterdir())


def test_the_same_export_guard_at_its_edges():
    a = madrid_runs(n_home=60, n_work=40)
    own = _partner(n=15)  # 20 runs of the second person's own
    assert couple.same_export(a, a[:40]) == (True, 40), "the smaller export wholly inside the other"
    assert couple.same_export(a, own + a[:20]) == (True, 20), "half of the smaller export"
    assert couple.same_export(a, own + a[:19]) == (False, 19), "just under half"
    assert couple.same_export(a, own[:1] + a[5:6]) == (False, 1), "a new partner with one borrowed file"
    assert couple.same_export(a, own[:1] + a[:2]) == (False, 2), "two borrowed files are still a few"
    assert couple.same_export(a, a[5:6]) == (True, 1), "nothing of their own"
    assert couple.same_export(a, own) == (False, 0)


def _moved(acts, dlon, dlat):
    return [dataclasses.replace(x, lonlat=[(lo + dlon, la + dlat) for lo, la in x.lonlat]) for x in acts]


def test_a_partner_without_runs_in_the_area_is_named_and_refused(monkeypatch, tmp_path, capsys):
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _moved(_partner(), 3.3, -0.9)})  # 300 km away
    assert cli.main(BASE + ["--bbox", "40.36,-3.78,40.48,-3.62", "--out", str(tmp_path / "p.png")]) == 2
    out = capsys.readouterr().out
    assert "Marcos" in out and "--place" in out and not re.search(r"\d\.\d{4}", out)
    assert not list(tmp_path.iterdir())
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _moved(_partner(), 3.3, -0.9)})
    assert cli.main(BASE + ["--out", str(tmp_path / "p.png")]) == 2, "no area both run in"
    assert "never share an area" in capsys.readouterr().out and not list(tmp_path.iterdir())


def test_a_place_that_openstreetmap_cannot_find_is_not_a_missing_shared_area(monkeypatch, tmp_path, capsys):
    from osmnx._errors import InsufficientResponseError

    def nowhere(place):
        raise InsufficientResponseError("Nominatim geocoder returned 0 results")

    for argv in (BASE, ["a.zip"]):  # the poster for two, and the single poster
        _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _partner()})
        monkeypatch.setattr(area, "geocode_bbox", nowhere)
        monkeypatch.setattr(area, "fetch_basemap", _never)
        assert cli.main(argv + ["--place", "Atlantis, Nowhere", "--out", str(tmp_path / "p.png")]) == 2
        out = capsys.readouterr().out
        assert "did not find that --place" in out and "never share an area" not in out
        assert not list(tmp_path.iterdir())


def test_couple_leak_check_fails_closed(monkeypatch, tmp_path, capsys):
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _partner()})
    monkeypatch.setattr(privacy, "_outside", lambda xy, forbidden: [xy])  # a broken cut: every line uncut
    assert cli.main(BASE + ["--dpi", "20", "--out", str(tmp_path / "p.png"), "--pdf", str(tmp_path / "p.pdf")]) == 3
    assert "[privacy] STOP" in capsys.readouterr().out and not list(tmp_path.iterdir())


def _through(p, i, day):
    """A run straight over p (a visit to the other person's door), far from its own ends, with a clock."""
    xy = np.column_stack([np.linspace(p[0] - 3000, p[0] + 3000, 601), np.full(601, p[1] + 20.0)])
    return Activity(id=f"t{i}", type="Run", start=dt.datetime(2025, 3, day, 10, tzinfo=dt.timezone.utc),
                    lonlat=_lonlat(xy), source="x", times=list(np.arange(601) * 3.0))


def test_cli_couple_end_to_end_offline(monkeypatch, tmp_path, capsys):
    acts_a, acts_b = madrid_runs() + [_through(HB, 1, 2)], _partner() + [_through(HA, 2, 3), _through(HA, 3, 4)]
    _by_path(monkeypatch, {"a.zip": acts_a, "b.zip": acts_b})
    seen, drawn, real, real_patch = {}, [], render_editorial.render, render_editorial.PathPatch

    def spy(*args, **kw):
        info = real(*args, **kw)
        seen.update(kw, info=info, extent=args[2])
        return info

    def patch(path, *a, **k):
        if k.get("zorder") == 5:
            drawn.append(np.asarray(path.vertices))
        return real_patch(path, *a, **k)

    monkeypatch.setattr(render_editorial, "render", spy)
    monkeypatch.setattr(render_editorial, "PathPatch", patch)
    files = [tmp_path / f for f in ("p.png", "p.pdf", "p.jpg")]
    assert cli.main(BASE + ["--dpi", "20", "--style", "midnight", "--out", str(files[0]), "--pdf", str(files[1]),
                            "--preview", str(files[2])]) == 0
    out = capsys.readouterr().out
    c = seen["couple"]
    assert list(c.owner) == [0] * len(acts_a) + [1] * len(acts_b) and c.names == ("Lucía", "Marcos")
    assert seen["info"]["legend"] == {"mode": "who"} and seen["city"] == "Madrid" and seen["year"] == "2025"
    tr = Transformer.from_crs(4326, render.utm_crs(-3.70, 40.42), always_xy=True)
    homes = [Point(*tr.transform(*ll)) for ll in _lonlat([HA, HB])]
    pts = np.concatenate(drawn)
    for h in homes:
        assert np.hypot(pts[:, 0] - h.x, pts[:, 1] - h.y).min() >= 200, "no drawn vertex near either home"
    assert "Lucía" in out and "Marcos" in out and "[privacy] places: Lucía" in out
    assert not re.search(r"\d\.\d{4}", out) and "40.4" not in out and "-3.7" not in out, "no coordinates"
    assert all(f.exists() and f.stat().st_size > 0 for f in files)
    assert c.streets_both is None and "no street names in this map" in out, "an empty basemap names no street"
    assert "[render] p.png " in out and str(tmp_path) not in out, "the file name, never its folder"


def test_streets_you_both_ran_are_counted_on_the_drawn_lines(monkeypatch, tmp_path, capsys):
    """Three named streets, each run by both: 'Calle Dentro' on the map, 'Calle Fuera' 300 m beyond its top edge (in
    the basemap's margin) and 'Calle Casa' through the first person's door (inside its zone). Only the first one is on
    the drawn lines, so the poster says 1."""
    bbox = tuple(map(float, BOX.split(",")))
    crs = render.utm_crs((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    ext = render.fit_extent(bbox, crs, render_editorial.map_aspect("A3"))
    y_out = ext[3] - _to_map((0.0, 0.0))[0][1] + 300  # metres north of the reference point, beyond the map's top
    streets = {"Calle Dentro": [(-6000, 6500), (-3000, 6500)], "Calle Fuera": [(-1000, y_out), (1000, y_out)],
               "Calle Casa": [(-200, 0), (200, 0)]}
    edges = gpd.GeoDataFrame({"highway": ["residential"] * 3, "name": list(streets)},
                             geometry=[LineString(_to_map(*ends)) for ends in streets.values()])
    runs_a = [line((-6500, 5500), (-6500, 6500), (-2500, 6500), (-2500, 7500)),
              line((0, y_out - 2500), (0, y_out), (-1000, y_out), (1000, y_out), (1000, y_out - 2500)),
              line((-2500, 0), (2500, 0))]
    runs_b = [line((-6500, 7500), (-6500, 6500), (-2500, 6500), (-2500, 5500)),
              line((-3000, y_out - 2000), (-1000, y_out), (1000, y_out), (3000, y_out - 2000)),
              line((-3000, 0), (3000, 0))]
    _by_path(monkeypatch, {"a.zip": madrid_runs() + _acts(runs_a, 500, day0=dt.date(2025, 6, 1)),
                           "b.zip": _partner() + _acts(runs_b, 1500, day0=dt.date(2025, 6, 1), hour=19)})
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (edges, EMPTY, EMPTY))
    seen, _ = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    assert seen["extent"] == pytest.approx(ext) and edges.geometry[1].bounds[1] > ext[3] + 200
    assert seen["couple"].streets_both == 1 and "streets you both ran 1 of 1" in capsys.readouterr().out


def test_the_couple_flow_learns_each_person_s_places_and_cuts_both_at_once(monkeypatch, tmp_path, capsys):
    """A quiet partner (25 runs) has a gym with 8 ends: their own threshold (2) finds it, the union of both exports
    (threshold 9) would not. couple.run asks pair_places for both people's places with --zone-radius and cuts every
    track in ONE privacy.apply with those zones, every original centre as a witness and --trim."""
    gym = (-4000.0, 4000.0)
    ta = loops_from(HA, 200)
    tb = loops_from(HB, 20) + loops_from(gym, 4, r0=900.0) + [through(gym)]
    _by_path(monkeypatch, {"a.zip": _acts(ta, 0), "b.zip": _acts(tb, 5000, hour=18)})
    calls, real_pair, real_apply = {"pair": [], "apply": []}, privacy.pair_places, privacy.apply

    def pair(groups, **kw):
        out = real_pair(groups, **kw)
        calls["pair"].append(([len(g) for g in groups], kw, out))
        return out

    def apply(tracks, **kw):
        calls["apply"].append((len(tracks), kw))
        return real_apply(tracks, **kw)

    monkeypatch.setattr(privacy, "pair_places", pair)
    monkeypatch.setattr(privacy, "apply", apply)
    seen, drawn = _spy_drawn(monkeypatch)
    argv = BASE + ["--bbox", BOX, "--trim", "250", "--zone-radius", "800", "--dpi", "20", "--out", str(tmp_path / "p.png")]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    (sizes, kw, (zones, witnesses, counts)), = calls["pair"]
    assert sizes == [200, 25] and kw["zone_r"] == 800 and tuple(kw["n_runs"]) == (200, 25)
    assert counts["per"] == [1, 2] and "[privacy] places: Lucía 1 · Marcos 2" in out, "the gym is the partner's own"
    assert min(r for _, _, r in zones) >= 800
    (n, kw), = calls["apply"]
    assert n == 225 and kw["trim_m"] == 250 and kw["zone_r"] == 800
    assert kw["places_xy"] is zones and kw["witnesses_xy"] is witnesses and len(witnesses) > len(zones)
    g, ha, hb = _to_map(gym, HA, HB)
    assert _gap(drawn, g) >= 200, "the quiet partner's gym is hidden from both lines"
    assert min(_gap(drawn, ha), _gap(drawn, hb)) >= 0.4411 * 800 - 20, "800 m zones keep at least 353 m"


def test_a_zone_given_with_with_hides_both_people_s_lines(monkeypatch, tmp_path, capsys):
    """--zone with --with: both people run straight through P. Neither line comes within 200 m of it. Neither export
    has a clock, and the partner's runs are from the year before: the year label spans both."""
    p = (5000.0, -5000.0)
    acts_a = [dataclasses.replace(x, times=None) for x in madrid_runs() + _acts([through(p), through(p, 0.6)], 900)]
    acts_b = [dataclasses.replace(x, times=None, start=x.start.replace(year=2024))
              for x in _partner() + _acts([through(p, 1.0), through(p, 1.3)], 1900)]
    _by_path(monkeypatch, {"a.zip": acts_a, "b.zip": acts_b})
    seen, drawn = _spy_drawn(monkeypatch)
    (lon, lat), = _lonlat([p])
    assert cli.main(BASE + ["--bbox", BOX, "--zone", f"{lat},{lon},600", "--dpi", "20",
                            "--out", str(tmp_path / "p.png")]) == 0
    out = capsys.readouterr().out
    pz, = _to_map(p)
    assert _gap(drawn, pz) >= 200 and "user zones 1" in out
    assert "[together] no clock in Lucía's and Marcos's activities" in out
    assert seen["year"] == "2024–2025", "the year label spans both people's runs"
    assert f"{lat:.3f}" not in out and f"{lon:.3f}" not in out


def test_a_zone_that_misses_the_map_is_refused_before_any_request(monkeypatch, tmp_path, capsys):
    """A --zone given as LON,LAT (Madrid's -3.70,40.42 read as a point off Somalia), or anywhere off the map, hides
    nothing on it: the poster would show the place it was meant to hide while the log said 'user zones 1'. It is
    refused before anything is cut, fetched or written, on the poster for two and on the single poster, and the
    message names the zone by its number, never by its coordinates."""
    p = (800.0, 900.0)  # a friend's door both people run past
    acts_a = madrid_runs() + _acts([through(p), through(p, 0.6)], 900)
    acts_b = _partner() + _acts([through(p, 1.0)], 1900, hour=18)
    (lon, lat), = _lonlat([p])
    right, swapped = f"{lat:.5f},{lon:.5f},400", f"{lon:.5f},{lat:.5f},400"
    for argv in (BASE, ["a.zip"]):
        for zones, k in (([swapped], 1), ([right, swapped], 2), ([right, "40.42,-3.10,400"], 2)):
            _by_path(monkeypatch, {"a.zip": acts_a, "b.zip": acts_b})
            monkeypatch.setattr(area, "fetch_basemap", _never)
            monkeypatch.setattr(area, "city_name", _never)
            monkeypatch.setattr(privacy, "apply", _never)
            extra = [x for z in zones for x in ("--zone", z)]
            assert cli.main(argv + ["--bbox", BOX, *extra, "--pdf", str(tmp_path / "p.pdf"),
                                    "--out", str(tmp_path / "p.png")]) == 2, (argv, zones)
            out = capsys.readouterr().out
            assert f"--zone number {k} does not touch this map" in out and "latitude first" in out
            assert not re.search(r"\d\.\d{3}", out) and not list(tmp_path.iterdir())


def test_a_person_with_nothing_left_on_the_map_stops_the_export_before_any_request(monkeypatch, tmp_path, capsys):
    """The partner only runs 100 m loops from their door: after the end trims nothing of theirs is left, so nothing
    is exported and nothing is sent, rather than a poster of only one of the two."""
    tiny = [loop(*HB, 100, phase=0.5 * i) for i in range(12)]
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _acts(tiny, 5000, hour=18)})
    monkeypatch.setattr(area, "fetch_basemap", _never)
    monkeypatch.setattr(area, "city_name", _never)
    assert cli.main(BASE + ["--bbox", BOX, "--pdf", str(tmp_path / "p.pdf"), "--out", str(tmp_path / "p.png")]) == 2
    out = capsys.readouterr().out
    assert "[privacy] Marcos: nothing of yours is left on this map after the privacy cuts" in out
    assert "--trim" not in out and "--zone-radius" not in out and not list(tmp_path.iterdir())


def test_busy_map_servers_stop_the_poster_for_two(monkeypatch, tmp_path, capsys):
    def busy(bbox, log=print):
        raise RuntimeError("every OpenStreetMap server is busy")

    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _partner()})
    monkeypatch.setattr(area, "fetch_basemap", busy)
    monkeypatch.setattr(area, "city_name", _never)
    assert cli.main(BASE + ["--pdf", str(tmp_path / "p.pdf"), "--out", str(tmp_path / "p.png")]) == 4
    out = capsys.readouterr().out
    assert "[osm] STOP: every OpenStreetMap server is busy. Run the same command again later." in out
    assert not list(tmp_path.iterdir())


def test_the_first_person_without_runs_is_told_to_choose_the_type(monkeypatch, tmp_path, capsys):
    rides = [dataclasses.replace(x, type="Ride") for x in madrid_runs()]
    _by_path(monkeypatch, {"a.zip": rides, "b.zip": _partner()})
    assert cli.main(BASE + ["--out", str(tmp_path / "p.png")]) == 2
    out = capsys.readouterr().out
    assert "Lucía: no Run activities (this export has: Ride 48). Choose the type with --type." in out


def test_a_partner_who_only_ran_the_other_s_streets_is_told_their_ink_hardly_shows(monkeypatch, tmp_path, capsys):
    """Lucía ran two loops ten times each; Marcos ran each of them once with her. Every street is 'Lucía's' (under
    10 % his), so his blue is nowhere on the map while the names, the bar and the key show it: the log says so."""
    work = (3000.0, 2500.0)
    ta = [loop(*HA, 1400, phase=0.4) for _ in range(10)] + [loop(*work, 1300, phase=2.0) for _ in range(10)]
    tb = [ta[0] + (0.0, 3.0), ta[10] + (3.0, 0.0)]
    _by_path(monkeypatch, {"a.zip": _acts(ta, 0), "b.zip": _acts(tb, 5000, day0=dt.date(2025, 3, 1), hour=19)})
    seen, _ = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    out = capsys.readouterr().out
    assert seen["info"]["own_ink"][1] < 0.01 < seen["info"]["own_ink"][0]
    assert "[render] Marcos: under 1 % of your lines are in your own ink" in out and "Lucía's" in out
    assert "[render] Lucía: under" not in out


def test_files_in_both_exports_do_not_unhide_a_single_loop(monkeypatch, tmp_path, capsys):
    """39 of the first person's runs are also in the second export (allowed: under half of it). They are drawn for
    each, but they are not the second person's own runs: counted, they would raise that person's threshold from 2
    to 3 ends and a single loop from a lodging would no longer be a place."""
    lodging = (5000.0, 2000.0)
    ta = loops_from(HA, 120)
    own_b = [loop(*lodging, 1400, phase=0.4), through(lodging, angle=1.2)]
    own_b += [line((-7500 + 450 * i, -5800), (-7300 + 450 * i, 8200), noise=4.0) for i in range(38)]  # no two ends meet
    acts_a = _acts(ta, 0)
    _by_path(monkeypatch, {"a.zip": acts_a, "b.zip": _acts(own_b, 5000, hour=18) + acts_a[:39]})
    seen, drawn = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    assert "[couple] 39 activities are in both exports" in capsys.readouterr().out
    lodging_xy, = _to_map(lodging)
    assert _gap(drawn, lodging_xy) >= 200


def test_the_frame_and_the_overpass_box_do_not_follow_a_lodging_at_the_edge(monkeypatch, tmp_path, capsys):
    """The quieter partner's out-and-back runs from a lodging set the south edge of the area where both run: the 1st
    percentile of their points is the lodging's latitude to within tens of metres. The box behind the poster's frame
    lies on the 1 km grid, and moving the lodging 30 m north changes neither the frame nor the box sent to Overpass."""
    from test_area import _lodging_scene
    fed, sent, extents, real_fit = [], [], [], render.fit_extent
    for north in (0.0, 30.0):
        a, b = _lodging_scene(north, n_loops=60)
        _by_path(monkeypatch, {"a.zip": a, "b.zip": b})
        monkeypatch.setattr(area, "fetch_basemap", lambda bbox, log=print: sent.append(tuple(bbox)) or "unused")
        monkeypatch.setattr(render, "fit_extent",
                            lambda bbox, crs, aspect: fed.append(tuple(bbox)) or real_fit(bbox, crs, aspect))
        seen, _ = _spy_drawn(monkeypatch)
        assert cli.main(BASE + ["--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
        extents.append(seen["extent"])
    assert "[area] found where you both run most" in capsys.readouterr().out
    assert fed and all(box_ == area.snap_box(box_) for box_ in fed), "the frame's box is on the grid"
    assert extents[0] == extents[1] and len(sent) == 2 and sent[0] == sent[1]


def _km(acts, crs):
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    return sum(float(np.hypot(*np.diff(np.column_stack(tr.transform(*np.asarray(x.lonlat).T)), axis=0).T).sum())
               for x in acts) / 1000


def test_each_person_s_numbers_and_runs_together_reach_the_poster(monkeypatch, tmp_path, capsys):
    """Lucia 48 runs; Marcos 30 of his own, one run side by side with her (her run 5, 3 m aside, on the same clock)
    and 5 runs 300 km away. The poster gets each person's own runs and km, counted inside the map only, the one run
    together, and the log names who has activities outside the map."""
    acts_a = madrid_runs()
    with_her = dataclasses.replace(acts_a[5], id="m5", lonlat=[(lo, la + 3 / 110540) for lo, la in acts_a[5].lonlat])
    far_b = [dataclasses.replace(x, id=f"far{i}") for i, x in enumerate(_moved(_partner(n=5)[:5], 3.3, -0.9))]
    acts_b = _partner() + [with_her] + far_b
    _by_path(monkeypatch, {"a.zip": acts_a, "b.zip": acts_b})
    seen, _ = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    out = capsys.readouterr().out
    c = seen["couple"]
    assert [x["n"] for x in c.stats] == [48, 31]
    crs = render.utm_crs(-3.69, 40.43)  # the box's centre
    assert c.stats[0]["km"] == pytest.approx(_km(acts_a, crs), rel=1e-9)
    assert c.stats[1]["km"] == pytest.approx(_km(_partner() + [with_her], crs), rel=1e-9)
    assert c.together["runs"] == 1 and c.together["known"] and "[together] 1 runs side by side" in out
    assert "[area] Marcos: 31 of 36 activities are inside the map" in out and "[area] Lucía:" not in out


def test_streets_you_both_ran_are_the_streets_of_both(monkeypatch, tmp_path, capsys):
    """Lucia runs Calle Uno and Calle Dos, Marcos only Calle Uno: one street both ran, two in all."""
    streets = {"Calle Uno": [(-6000, 6500), (-3000, 6500)], "Calle Dos": [(-6000, -4500), (-3000, -4500)]}
    edges = gpd.GeoDataFrame({"highway": ["residential"] * 2, "name": list(streets)},
                             geometry=[LineString(_to_map(*ends)) for ends in streets.values()])
    runs_a = [line((-6500, 5500), (-6500, 6500), (-2500, 6500), (-2500, 7500)),
              line((-6500, -5500), (-6500, -4500), (-2500, -4500), (-2500, -3500))]
    runs_b = [line((-6500, 7500), (-6500, 6500), (-2500, 6500), (-2500, 5500))]
    _by_path(monkeypatch, {"a.zip": madrid_runs() + _acts(runs_a, 500, day0=dt.date(2025, 6, 1)),
                           "b.zip": _partner() + _acts(runs_b, 1500, day0=dt.date(2025, 6, 1), hour=19)})
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (edges, EMPTY, EMPTY))
    seen, _ = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    assert seen["couple"].streets_both == 1 and "streets you both ran 1 of 2" in capsys.readouterr().out


def test_a_person_whose_lines_are_all_beyond_the_map_edge_stops_the_export(monkeypatch, tmp_path, capsys):
    """Marcos's runs all start 150 m inside the west edge of the map and run 5 km west: they are in the area, but
    after the 300 m end trim every piece of his lies beyond the map's edge. Nothing of his would be drawn, so nothing
    is exported: what counts is what is left on the map, not what is left after the cuts."""
    bbox = tuple(map(float, BOX.split(",")))
    crs = render.utm_crs((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    ext = render.fit_extent(bbox, crs, render_editorial.map_aspect("A3"))
    x_edge = ext[0] - _to_map((0.0, 0.0))[0][0]  # the map's west edge, metres east of the reference point
    runs_b = [line((x_edge + 150, -3000.0 + 400 * i), (x_edge - 5000, -3000.0 + 400 * i)) for i in range(12)]
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _acts(runs_b, 5000, hour=18)})
    monkeypatch.setattr(area, "fetch_basemap", _never)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 2
    assert "[privacy] Marcos: nothing of yours is left on this map" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_files_in_both_exports_do_not_unhide_the_first_person_s_single_loop(monkeypatch, tmp_path, capsys):
    """The mirror of test_files_in_both_exports_do_not_unhide_a_single_loop: 39 of the SECOND person's runs are also
    in the FIRST export, whose owner has 40 runs of her own and one loop from a lodging. Counted as hers, the copies
    would raise her threshold from 2 to 3 ends and the loop's start would be drawn."""
    lodging = (5000.0, 2000.0)
    tb = loops_from(HB, 120)
    own_a = [loop(*lodging, 1400, phase=0.4), through(lodging, angle=1.2)]
    own_a += [line((-7500 + 450 * i, -5800), (-7300 + 450 * i, 8200), noise=4.0) for i in range(38)]
    acts_b = _acts(tb, 5000, hour=18)
    _by_path(monkeypatch, {"a.zip": _acts(own_a, 0) + acts_b[:39], "b.zip": acts_b})
    seen, drawn = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    assert "[couple] 39 activities are in both exports" in capsys.readouterr().out
    lodging_xy, = _to_map(lodging)
    assert _gap(drawn, lodging_xy) >= 200


def test_fit_and_lang_reach_the_poster_for_two(monkeypatch, tmp_path, capsys):
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _partner()})
    seen, _ = _spy_drawn(monkeypatch)
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    whole, lang = seen["extent"], seen["lang"]
    assert cli.main(BASE + ["--bbox", BOX, "--fit", "--lang", "es", "--dpi", "20",
                            "--out", str(tmp_path / "q.png")]) == 0
    assert lang == "en" and seen["lang"] == "es"
    assert seen["extent"][1] - seen["extent"][0] < 0.9 * (whole[1] - whole[0]), "--fit crops the map"
    capsys.readouterr()


def test_relief_reaches_the_poster_for_two_from_the_map_box_only(monkeypatch, tmp_path, capsys):
    from pinless import couple
    _by_path(monkeypatch, {"a.zip": madrid_runs(), "b.zip": _partner()})
    calls, seen, real = [], {}, render_editorial.render
    monkeypatch.setattr(couple.relief_mod, "prepare_many", lambda maps, log=print: calls.append(maps) or [None])
    monkeypatch.setattr(render_editorial, "render", lambda *a, **kw: seen.update(kw) or real(*a, **kw))
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
    assert calls == [] and seen["relief"] is None, "off by default: nothing is asked for"
    assert cli.main(BASE + ["--bbox", BOX, "--dpi", "20", "--relief", "shade", "--out", str(tmp_path / "q.png")]) == 0
    (maps,) = calls
    (extent, crs, width), = maps
    assert len(extent) == 4 and width == pytest.approx(297 * (1 - 36 / 297)), "one map: its box, CRS and paper width"
    capsys.readouterr()
