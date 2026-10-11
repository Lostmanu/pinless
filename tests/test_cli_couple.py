"""pinless EXPORT --with EXPORT2: flags checked before any file is read, each export read with its own type and named
by its person (never by its path), the same export twice refused before any request, and the whole flow offline."""
import dataclasses
import datetime as dt
import re

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Point

from pinless import area, cli, couple, ingest, privacy, render, render_editorial
from pinless.ingest import Activity
from test_cli_editorial import _lonlat, madrid_runs, stub_world

BASE = ["a.zip", "--with", "b.zip", "--name", "Lucía", "--with-name", "Marcos"]
HA, HB = (0.0, 0.0), (1500.0, -1800.0)  # the two homes, metres from the map's reference point


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
