"""Reading an export: positions, times, elevation and the activity's name, from GPX and TCX inside a Strava-like zip."""
import zipfile

from pinless import ingest

GPX = b"""<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><type>running</type><trkseg>
<trkpt lat="40.41" lon="-3.70"><ele>650.5</ele><time>2025-04-06T08:30:00Z</time></trkpt>
<trkpt lat="40.42" lon="-3.70"><ele>655.0</ele><time>2025-04-06T08:31:00Z</time></trkpt>
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
