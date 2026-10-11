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
