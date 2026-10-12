"""Reading an export: positions, times, elevation and the activity's name, from GPX and TCX inside a Strava-like zip."""
import zipfile

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
