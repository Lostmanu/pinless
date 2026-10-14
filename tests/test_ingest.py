"""Reading an export: positions, times, elevation and the activity's name, from GPX and TCX inside a Strava-like zip."""
import datetime as dt
import zipfile

import numpy as np
import pytest

from pinless import ingest

GPX = b"""<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><type>running</type><trkseg>
<trkpt lat="40.41" lon="-3.70"><ele>650.5</ele><time>2025-04-06T08:30:00Z</time></trkpt>
<trkpt lat="40.411" lon="-3.70"><ele>655.0</ele><time>2025-04-06T08:31:00Z</time></trkpt>
</trkseg></trk></gpx>"""
TCX = b"""<?xml version="1.0"?><TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
<Activities><Activity Sport="Running"><Lap><Track>
<Trackpoint><Time>2025-05-01T07:00:00Z</Time><Position><LatitudeDegrees>40.41</LatitudeDegrees><LongitudeDegrees>-3.70</LongitudeDegrees></Position><AltitudeMeters>600</AltitudeMeters></Trackpoint>
<Trackpoint><Time>2025-05-01T07:00:05Z</Time><Position><LatitudeDegrees>40.411</LatitudeDegrees><LongitudeDegrees>-3.70</LongitudeDegrees></Position></Trackpoint>
</Track></Lap></Activity></Activities></TrainingCenterDatabase>"""
CSV = ("Activity ID,Activity Date,Activity Name,Activity Type,Filename\n"
       "1,\"Apr 6, 2025, 8:30:00 AM\",Valencia Half,Run,activities/1.gpx\n"
       "2,\"May 1, 2025, 7:00:00 AM\",Morning Run,Run,activities/2.tcx\n")


def test_elevation_and_names_are_read(tmp_path):
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("activities.csv", CSV)
        z.writestr("activities/1.gpx", GPX)
        z.writestr("activities/2.tcx", TCX)
    acts, rep = ingest.read_export(str(path))
    by_id = {a.id: a for a in acts}
    assert rep.used == 2
    assert by_id["1"].name == "Valencia Half" and by_id["1"].ele == [650.5, 655.0] and by_id["1"].times == [0.0, 60.0]
    assert by_id["2"].ele is None, "a point without altitude: no elevation rather than a broken profile"


def test_stray_fixes_are_dropped_on_import(tmp_path):
    stale = '<trkpt lat="40.44" lon="-3.66"><time>2025-04-06T08:29:{:02d}Z</time></trkpt>'
    run = '<trkpt lat="{:.5f}" lon="-3.70"><time>2025-04-06T08:30:{:02d}Z</time></trkpt>'
    pts = [stale.format(s) for s in (58, 59)] + [run.format(40.41 + 0.00003 * i, i) for i in range(40)]
    gpx = ('<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>' + "".join(pts)
           + "</trkseg></trk></gpx>").encode()
    (tmp_path / "a.gpx").write_bytes(gpx)
    (act,), rep = ingest.read_export(str(tmp_path))
    assert act.lonlat[0] == (-3.70, 40.41) and len(act.lonlat) == 40, "the stale fix, recorded twice, is gone"
    assert act.times[0] == 0.0 and act.start.second == 0 and act.start.minute == 30


HIKE = ("""<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><metadata><name>App export</name>
</metadata><trk><name>Loop from the car park</name><type>hiking</type><trkseg>"""
        + "".join(f'<trkpt lat="{40.82 + 0.0003 * i:.4f}" lon="-3.96"><ele>{1830 + 3 * i}</ele>'
                  f'<time>2025-07-12T07:{i // 2:02d}:{30 * (i % 2):02d}Z</time></trkpt>' for i in range(40))
        + "</trkseg></trk></gpx>").encode()


def test_a_single_gpx_file_is_an_export_of_one_activity(tmp_path):
    import gzip

    for name, data in (("hike.gpx", HIKE), ("hike.GPX.gz", gzip.compress(HIKE))):
        path = tmp_path / name
        path.write_bytes(data)
        (act,), rep = ingest.read_export(str(path))
        assert rep.files_seen == rep.used == 1
        assert act.type == "Hike" and act.name == "Loop from the car park", "type and name from the file"
        assert len(act.lonlat) == 40 and act.ele[-1] == 1830.0 + 3 * 39 and act.times[-1] == 39 * 30.0
    assert ingest.is_activity_file(str(tmp_path / "hike.gpx")) and not ingest.is_activity_file(str(tmp_path))


def test_a_planned_route_without_track_points_is_read_from_its_route_points(tmp_path):
    route = (b'<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><rte><name>Planned</name>'
             b'<rtept lat="40.82" lon="-3.96"><ele>1830</ele></rtept><rtept lat="40.83" lon="-3.95"><ele>1900</ele>'
             b'</rtept></rte></gpx>')
    (tmp_path / "route.gpx").write_bytes(route)
    (act,), _ = ingest.read_export(str(tmp_path / "route.gpx"))
    assert act.lonlat == [(-3.96, 40.82), (-3.95, 40.83)] and act.name == "Planned" and act.times is None
    assert act.type == "Unknown" and act.ele == [1830.0, 1900.0]


def _planner(step_s, ele):
    """A route drawn in a planner and saved by a converter (DownloadGPX, met on 14-10-2026): a point every ~220 m, the
    clock step_s seconds a point, the elevation ele(i)."""
    pts = "".join(f'<trkpt lat="{40.410 + 0.002 * i:.5f}" lon="-3.6886"><ele>{ele(i)}</ele>'
                  f'<time>{(dt.datetime(2026, 10, 14, 17, 33) + dt.timedelta(seconds=step_s * i)).isoformat()}Z'
                  f'</time></trkpt>' for i in range(20))
    return ('<?xml version="1.0"?><gpx version="1.1" creator="DownloadGPX" xmlns="http://www.topografix.com/GPX/1/1">'
            f'<trk><name> Retiro Loop</name><trkseg>{pts}</trkseg></trk></gpx>').encode()


def test_a_route_from_a_planner_keeps_its_points_and_drops_its_made_up_clock_and_flat_zero_heights(tmp_path):
    """A planner's route saved as GPX ran 4 km in 21 s at a height of 0 m: read as a clock, every 200 m step of it was
    a stray fix, one file was refused and the other kept 2 of its 22 points. A clock no body can keep (a median of more
    than MAX_MS) is no clock: the route keeps every point and is read without time, pace or date; heights that never
    change are no heights. A real clock and real heights over the same steps are kept."""
    path = tmp_path / "route.gpx"
    path.write_bytes(_planner(1, lambda i: 0))
    (act,), rep = ingest.read_export(str(path))
    assert len(act.lonlat) == 20, "every point of the route, none taken for a stray fix"
    assert act.times is None and act.start is None and act.ele is None
    assert rep.no_clock == 1 and rep.used == 1
    path.write_bytes(_planner(75, lambda i: 650 + i))  # 220 m in 75 s: a runner's clock, and a slope
    (act,), rep = ingest.read_export(str(path))
    assert len(act.lonlat) == 20 and act.times[-1] == 19 * 75.0 and act.ele[-1] == 669.0 and rep.no_clock == 0
    assert act.start == dt.datetime(2026, 10, 14, 17, 33, tzinfo=dt.timezone.utc)
    assert not ingest.plausible_clock([(0, 0), (0.001, 0), (0.002, 0)], [0.0, 0.0, 0.0]), "no step takes time"


def test_the_distance_strava_recorded_is_read_in_its_units_and_used_for_the_totals(tmp_path):
    """Strava's activities.csv carries the distance it measured, in the account's units (km or miles) and, in newer
    exports, again in metres under the same name, in the account's language. The value nearest the track's own
    length, in km, miles or metres, within RECORDED of it, is the activity's; anything else is ignored. The totals use
    it: a sum of the steps between fixes adds their noise."""
    from pinless import stats

    def gpx(n):  # n steps of ~111 m north
        pts = "".join(f'<trkpt lat="{40.41 + 0.001 * i:.4f}" lon="-3.70"><time>2025-04-06T08:{i // 2:02d}:'
                      f'{30 * (i % 2):02d}Z</time></trkpt>' for i in range(n + 1))
        return ('<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>' + pts
                + "</trkseg></trk></gpx>").encode()

    own = ingest.track_m([(-3.70, 40.41 + 0.001 * i) for i in range(19)])  # 18 steps: about 2 km
    rows = ("Activity ID,Activity Date,Activity Name,Activity Type,Distance,Filename,Distance\n"
            f'1,"Apr 6, 2025",A,Run,{own * 0.97 / 1000:.2f},activities/1.gpx,{own * 0.97:.1f}\n'
            f'2,"Apr 6, 2025",B,Run,{own * 0.98 / 1609.344:.3f},activities/2.gpx,\n'
            '3,"Apr 6, 2025",C,Run,9.0,activities/3.gpx,\n')
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("activities.csv", rows)
        for i in (1, 2, 3):
            z.writestr(f"activities/{i}.gpx", gpx(18))
    acts = {a.id: a for a in ingest.read_export(str(path))[0]}
    assert acts["1"].recorded_m == pytest.approx(own * 0.97, abs=0.1), "the metres column, nearest of km and m"
    assert acts["2"].recorded_m == pytest.approx(own * 0.98, rel=1e-3), "a value in miles"
    assert acts["3"].recorded_m is None, "9 km for a 2 km track is not this track's"
    es = rows.replace("Distance", "Distancia").replace("Activity ID", "ID de actividad").replace(
        "Activity Type", "Tipo de actividad").replace("Filename", "Nombre del archivo").replace(
        f"{own * 0.97 / 1000:.2f}", f'"{own * 0.97 / 1000:.2f}"'.replace(".", ","))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("activities.csv", es)
        z.writestr("activities/1.gpx", gpx(18))
    (es_act,) = ingest.read_export(str(path))[0]
    assert es_act.recorded_m == pytest.approx(own * 0.97, rel=1e-3), "Spanish headers, a decimal comma"
    xy = [np.column_stack([np.zeros(19), np.arange(19) * own / 18]) for _ in (1, 2, 3)]
    st = stats.compute([acts["1"], acts["2"], acts["3"]], xy)
    assert st["km"] == pytest.approx(own * (0.97 + 0.98 + 1) / 1000, rel=1e-3)


def test_a_single_tcx_course_and_a_single_fit_file(tmp_path):
    import datetime as dt

    from _fit_fixture import write_fit

    course = TCX.replace(b"<Activities><Activity", b"<Courses><Course><Name>Ridge walk</Name></Course><Activities>"
                                                   b"<Activity")
    course = course.replace(b"</Activities>", b"</Activities></Courses>")
    (tmp_path / "walk.tcx").write_bytes(course)
    (act,), _ = ingest.read_export(str(tmp_path / "walk.tcx"))
    assert act.name == "Ridge walk" and act.type == "Run" and act.times == [0.0, 5.0]
    start = dt.datetime(2025, 7, 12, 7, 0, tzinfo=dt.timezone.utc)
    pts = [(-3.96, 40.82 + 0.001 * i, 30.0 * i, 1830.0 + 2 * i) for i in range(20)]
    (tmp_path / "hike.fit").write_bytes(write_fit(pts, start))
    (act,), rep = ingest.read_export(str(tmp_path / "hike.fit"))
    assert rep.used == 1 and act.type == "Hike" and act.name == "" and act.start == start
    assert abs(act.lonlat[0][0] + 3.96) < 1e-6 and abs(act.lonlat[0][1] - 40.82) < 1e-6, "semicircles to degrees"
    assert act.times[-1] == 570.0 and abs(act.ele[-1] - 1868.0) < 0.3


def _gpx_tracks(acts):
    """One GPX file holding every activity as its own <trk>, as an app's "export all tracks" writes it."""
    def trk(a):
        pts = "".join(f'<trkpt lat="{la:.7f}" lon="{lo:.7f}"><time>'
                      f'{(a.start + dt.timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")}</time></trkpt>'
                      for (lo, la), t in zip(a.lonlat, a.times))
        return f"<trk><type>running</type><trkseg>{pts}</trkseg></trk>"

    return ('<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1">' + "".join(map(trk, acts))
            + "</gpx>").encode()


def test_every_track_of_one_file_is_an_activity_of_its_own(tmp_path):
    from test_cli_editorial import madrid_runs

    acts = madrid_runs(n_home=12, n_work=0)
    (tmp_path / "all.gpx").write_bytes(_gpx_tracks(acts))
    got, rep = ingest.read_export(str(tmp_path / "all.gpx"))
    assert rep.files_seen == 1 and rep.used == 12 and [a.id for a in got] == [f"all#{i}" for i in range(1, 13)]
    assert all(a.type == "Run" and a.lonlat[0] == pytest.approx(b.lonlat[0], abs=1e-6) and a.start == b.start
               for a, b in zip(got, acts)), "each track keeps its own start, place and clock"
    two_segments = HIKE.replace(b"</trkpt><trkpt", b"</trkpt></trkseg><trkseg><trkpt", 1)
    (tmp_path / "paused.gpx").write_bytes(two_segments)
    (one,), _ = ingest.read_export(str(tmp_path / "paused.gpx"))
    assert len(one.lonlat) == 40 and one.name == "Loop from the car park", "the segments of one track stay together"
    acts3 = TCX.replace(b"<Activities>", b"<Activities>" + TCX.split(b"<Activities>")[1].split(b"</Activities>")[0] * 2)
    (tmp_path / "three.tcx").write_bytes(acts3)
    got, rep = ingest.read_export(str(tmp_path / "three.tcx"))
    assert rep.used == 3 and [len(a.lonlat) for a in got] == [2, 2, 2] and [a.id for a in got] == [
        "three#1", "three#2", "three#3"]


def test_a_file_of_many_walks_from_home_hides_home_as_a_folder_of_them_does(monkeypatch, tmp_path, capsys):
    from pyproj import Transformer
    from shapely.geometry import MultiLineString, Point

    from pinless import area, cli, render, render_editorial
    from test_cli_editorial import EMPTY, _lonlat, madrid_runs

    acts = madrid_runs(n_home=12, n_work=0)
    (tmp_path / "all.gpx").write_bytes(_gpx_tracks(acts))
    folder = tmp_path / "folder"
    folder.mkdir()
    for i, a in enumerate(acts):
        (folder / f"{i}.gpx").write_bytes(_gpx_tracks([a]))
    monkeypatch.setattr(area, "fetch_basemap", lambda bbox, log=print: "unused")
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (EMPTY, EMPTY, EMPTY))
    monkeypatch.setattr(area, "city_name", lambda lat, lon, lang="en": "Madrid")
    seen = {}
    real = render_editorial.render

    def spy(pieces, basemap, extent, stats, out_path, **kw):
        seen.update(pieces=pieces, extent=extent)
        return real(pieces, basemap, extent, stats, out_path, **kw)

    monkeypatch.setattr(render_editorial, "render", spy)
    home = _lonlat([(0, 0)])[0]
    for export in (folder, tmp_path / "all.gpx"):
        assert cli.main([str(export), "--dpi", "20", "--out", str(tmp_path / "p.png")]) == 0
        crs = render.utm_crs(*home)
        hx, hy = Transformer.from_crs(4326, crs, always_xy=True).transform(*home)
        assert MultiLineString([p for _, p in seen["pieces"]]).distance(Point(hx, hy)) >= 200, export.name
        assert "zones suggested 1" in capsys.readouterr().out


def test_a_point_that_is_not_a_finite_number_is_dropped_and_a_height_that_is_not_one_is_missing(tmp_path):
    import math

    bad = HIKE.replace(b"<ele>1863</ele>", b"<ele>NaN</ele>").replace(b'lat="40.8260"', b'lat="NaN"')
    bad = bad.replace(b'lon="-3.96"><ele>1866', b'lon="inf"><ele>1866').replace(b'lat="40.8290"', b'lat="91.5"')
    (tmp_path / "bad.gpx").write_bytes(bad)
    (act,), _ = ingest.read_export(str(tmp_path / "bad.gpx"))
    assert len(act.lonlat) == 37 and all(math.isfinite(v) for p in act.lonlat for v in p), "three points dropped"
    assert act.ele is None, "a height that is not a number: no profile rather than a broken one"
    assert len(act.times) == 37
