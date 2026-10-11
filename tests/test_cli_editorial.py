"""The solo editorial poster through the command line, offline: the flow every layout shares (area, privacy, extent,
basemap, headline, numbers, render) is pinned here, so moving it into pinless/frame.py cannot change it."""
import datetime as dt
import math

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import MultiLineString, Point

from pinless import area, cli, ingest, render, render_editorial
from pinless.ingest import Activity

UTC = dt.timezone.utc
EMPTY = gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])
LAT0, LON0 = 40.42, -3.70  # Madrid


def _lonlat(xy):
    """Metres east/north of (LON0, LAT0) to degrees: close enough over a few kilometres."""
    xy = np.asarray(xy, dtype=float)
    return [(LON0 + x / (111320 * math.cos(math.radians(LAT0))), LAT0 + y / 110540) for x, y in xy]


def _loop(cx, cy, r, n=160, phase=0.0):
    t = np.linspace(0, 2 * np.pi, n) + phase
    return np.column_stack([cx + r * (np.cos(t) - np.cos(phase)), cy + r * (np.sin(t) - np.sin(phase))])


def madrid_runs(home=(0.0, 0.0), work=(3000.0, 2500.0), n_home=40, n_work=8, first_id=0, day0=dt.date(2025, 1, 6),
                hour=7, pace=0.3):
    """Loops from a home and a few from a workplace, a day apart, with a clock (pace in s/m)."""
    acts = []
    for i in range(n_home + n_work):
        cx, cy = home if i < n_home else work
        xy = _loop(cx, cy, 1200 + 25 * (i % 17), phase=0.37 * i)
        d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        start = dt.datetime.combine(day0 + dt.timedelta(days=i), dt.time(hour), tzinfo=UTC)
        acts.append(Activity(id=str(first_id + i), type="Run", start=start, lonlat=_lonlat(xy), source="x",
                             times=list(d * pace)))
    return acts


def stub_world(monkeypatch, read):
    """No network: the export reader, the OSM download, the basemap and the city name are stand-ins."""
    monkeypatch.setattr(ingest, "read_export", read)
    monkeypatch.setattr(area, "fetch_basemap", lambda bbox, log=print: "unused")
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (EMPTY, EMPTY, EMPTY))
    monkeypatch.setattr(area, "city_name", lambda lat, lon, lang="en": "Madrid")


def test_solo_editorial_cli_flow_is_unchanged(monkeypatch, tmp_path, capsys):
    acts = madrid_runs()
    stub_world(monkeypatch, lambda path: (acts, ingest.IngestReport(files_seen=48, used=48)))
    seen, real = {}, render_editorial.render

    def spy(pieces, basemap, extent, stats, out_path, **kw):
        seen.update(kw, pieces=pieces, extent=extent, stats=stats, out=out_path)
        return real(pieces, basemap, extent, stats, out_path, **kw)

    monkeypatch.setattr(render_editorial, "render", spy)
    out = tmp_path / "p.png"
    assert cli.main(["x.zip", "--dpi", "20", "--name", "Lucia", "--out", str(out), "--pdf", str(tmp_path / "p.pdf")]) == 0
    text = capsys.readouterr().out
    assert "[area] found where you run most" in text and "[privacy] zones suggested" in text
    assert "[area] headline: Madrid" in text and "[stats] 48 runs" in text and "[render]" in text
    assert seen["city"] == "Madrid" and seen["year"] == "2025" and seen["color_by"] == "heat"
    assert seen["name"] == "Lucia" and seen["headline"] == "city" and seen["graphs"] is False
    s, w, n, e = area.auto_bbox(acts)
    crs = render.utm_crs((w + e) / 2, (s + n) / 2)
    assert seen["extent"] == pytest.approx(render.fit_extent((s, w, n, e), crs, render_editorial.map_aspect("A3")))
    assert seen["stats"]["n"] == 48 and seen["activities"] == acts and len(seen["tracks_xy"]) == 48
    union = MultiLineString([p for _, p in seen["pieces"]])
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    for lon, lat in _lonlat([(0, 0), (3000, 2500)]):  # home and work
        assert union.distance(Point(*tr.transform(lon, lat))) >= 200
    assert out.exists() and (tmp_path / "p.pdf").exists()
    # one year asked for: the label is that year; --fit crops to the drawn lines inside the box
    whole = seen["extent"]
    assert cli.main(["x.zip", "--dpi", "20", "--year", "2025", "--fit", "--out", str(out)]) == 0
    assert seen["year"] == "2025" and seen["extent"][1] - seen["extent"][0] <= whole[1] - whole[0] + 1e-6
    assert cli.main(["x.zip", "--dpi", "20", "--year", "2024", "--out", str(out)]) == 2, "no runs that year"
    capsys.readouterr()
