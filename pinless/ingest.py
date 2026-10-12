"""Read a Strava bulk export (zip or folder), a folder of GPX/TCX/FIT files or a single such file, locally.

Nothing leaves the machine. Corrupt or unreadable files are skipped and counted, never fatal. A single file (a hike
downloaded from a trail app or a watch) is an export of one activity: its type and name come from the file.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import math
import os
import zipfile
from dataclasses import dataclass, field, replace
from xml.etree import ElementTree as ET

# activities.csv headers vary with the account language; map the ones we need.
HEADERS = {
    "id": ["Activity ID", "ID de actividad", "ID de l'activité", "Aktivitäts-ID", "ID attività"],
    "type": ["Activity Type", "Tipo de actividad", "Type d'activité", "Aktivitätsart", "Tipo di attività"],
    "file": ["Filename", "Nombre del archivo", "Nom du fichier", "Dateiname", "Nome file"],
    "date": ["Activity Date", "Fecha de la actividad", "Date de l'activité", "Aktivitätsdatum", "Data attività"],
    "name": ["Activity Name", "Nombre de la actividad", "Nom de l'activité", "Aktivitätsname", "Nome attività"],
}
GPX_TYPES = {"running": "Run", "run": "Run", "9": "Run", "cycling": "Ride", "biking": "Ride", "ride": "Ride", "1": "Ride",
             "walking": "Walk", "walk": "Walk", "hiking": "Hike", "hike": "Hike", "10": "Walk"}
SUFFIXES = (".gpx", ".gpx.gz", ".tcx", ".tcx.gz", ".fit", ".fit.gz")


@dataclass
class Activity:
    id: str
    type: str
    start: dt.datetime | None
    lonlat: list  # [(lon, lat), ...]
    source: str
    times: list = None  # seconds since the first point, aligned with lonlat (None if unknown)
    ele: list = None  # metres above sea level, aligned with lonlat (None if the file has none)
    name: str = ""  # the activity's title in the export (a race name, or Strava's "Morning Run")


def _step_m(p, q):
    lat = math.radians((p[1] + q[1]) / 2)
    return 6371000.0 * math.hypot(math.radians(q[0] - p[0]) * math.cos(lat), math.radians(q[1] - p[1]))


def stray_ends(lonlat, secs=None, jump_m=300.0, window_m=500.0):
    """(lo, hi): the points to keep once the stray fixes at either end are dropped, or (0, 0) if nothing real is left.

    A watch can open with a stale position (yesterday's, or the last one before it found the satellites), recorded
    once or many times, and then jump to where it really is; it can also close with one. A step is stray when it is
    impossible for the body: with a clock, faster than 30 m/s over more than 50 m, or longer than jump_m at more than
    12 m/s or after a pause of over 10 minutes; without one, longer than jump_m and than 8 typical steps (so a sparse
    recording keeps its long steps). Everything before the last stray step within the first window_m metres of real
    movement goes, and the same at the end: counted in metres, not points, however many times the stale fix repeats."""
    n = len(lonlat)
    if n < 2:
        return (0, n)
    steps = [_step_m(lonlat[i], lonlat[i + 1]) for i in range(n - 1)]
    if secs is not None and (len(secs) != n or any(t is None for t in secs)):
        secs = None  # an incomplete clock is no clock
    if secs is not None:
        dts = [abs(secs[i + 1] - secs[i]) for i in range(n - 1)]
        stray = [d > 50 and (d / max(t, 1e-3) > 30 or (d > jump_m and (d / max(t, 1e-3) > 12 or t > 600)))
                 for d, t in zip(steps, dts)]
    else:
        limit = max(jump_m, 8 * sorted(steps)[len(steps) // 2])
        stray = [d > limit for d in steps]
    lo, moved = 0, 0.0
    for i in range(n - 1):
        if stray[i]:
            lo = i + 1
        else:
            moved += steps[i]
            if moved > window_m:
                break
    hi, moved = n, 0.0
    for i in range(n - 2, lo - 1, -1):
        if stray[i]:
            hi = i + 1
        else:
            moved += steps[i]
            if moved > window_m:
                break
    return (lo, hi) if hi - lo >= 2 else (0, 0)


def clean(act):
    """The activity without stray fixes at its ends (see stray_ends): the same object when there are none, None when
    nothing real is left. Repeated until nothing changes: dropping a stale prefix brings the next window closer."""
    for _ in range(5):
        lo, hi = stray_ends(act.lonlat, act.times)
        if (lo, hi) == (0, len(act.lonlat)):
            return act
        if hi - lo < 2:
            return None
        clock = act.times and len(act.times) == len(act.lonlat) and all(t is not None for t in act.times)
        times = [t - act.times[lo] for t in act.times[lo:hi]] if clock else None
        start = act.start + dt.timedelta(seconds=act.times[lo]) if act.start is not None and clock else act.start
        act = replace(act, lonlat=list(act.lonlat[lo:hi]), times=times, start=start,
                      ele=list(act.ele[lo:hi]) if act.ele else act.ele)
    return act


@dataclass
class IngestReport:
    files_seen: int = 0
    used: int = 0
    no_gps: int = 0
    unreadable: int = 0
    unsupported: int = 0
    reasons: dict = field(default_factory=dict)

    def skip(self, kind, name, why):
        setattr(self, kind, getattr(self, kind) + 1)
        self.reasons.setdefault(why, 0)
        self.reasons[why] += 1


def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _parse_time(s):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def _float(x):
    """A finite number, or None: "NaN", "inf" or text is no measurement."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _lonlat(lon, lat):
    """(lon, lat) in degrees, or None when either is missing, not a finite number or off the globe."""
    lo, la = _float(lon), _float(lat)
    if lo is None or la is None or not (-180 <= lo <= 180 and -90 <= la <= 90):
        return None
    return lo, la


def _text(el):
    return el.text.strip() if el is not None and el.text else ""


def _units(root, tags, point):
    """The elements of `tags` (in file order) that hold at least one `point`, or [root] when none does (points
    outside any of them)."""
    units = [e for e in root.iter() if e.tag in tags and next(e.iter(point), None) is not None]
    return units or [root]


def parse_gpx(data: bytes):
    """[(points, first time, type, times, elevations, name)], one per track (<trk>) of the file.

    Each track is an activity of its own: an app's "export all tracks" writes many walks into one file, and the ends
    of every one of them are private places to learn, as they are in a folder of files. The segments of a track (a
    pause in the recording) stay together. A file with no track points (a route planned in a trail app) is read from
    its routes (<rte>) instead. The file's own name (metadata) names a file of one track only."""
    root = _strip_ns(ET.fromstring(data))
    tag, unit = ("trkpt", "trk") if next(root.iter("trkpt"), None) is not None else ("rtept", "rte")
    units = _units(root, (unit,), tag)
    out = []
    for u in units:
        pts, ts, ele = [], [], []
        for p in u.iter(tag):
            xy = _lonlat(p.get("lon"), p.get("lat"))
            if xy is None:
                continue
            pts.append(xy)
            tt = p.find("time")
            ts.append(_parse_time(tt.text if tt is not None else None))
            ee = p.find("ele")
            ele.append(_float(ee.text if ee is not None else None))
        k = _text(u.find("type")) if u is not root else ""
        kind = GPX_TYPES.get(k.lower(), k) if k else None
        name = _text(u.find("name")) if u is not root else ""
        out.append((pts, ts[0] if ts else None, kind, ts, ele, name))
    if len(out) == 1 and not out[0][5]:
        out[0] = out[0][:5] + (_text(root.find("metadata/name")),)
    return out


def parse_tcx(data: bytes):
    """[(points, first time, type, times, elevations, name)], one per activity (<Activity>) or course (<Course>) of
    the file, for the same reason as parse_gpx; a course carries its name. The laps of an activity stay together."""
    root = _strip_ns(ET.fromstring(data.strip()))
    out = []
    for u in _units(root, ("Activity", "Course"), "Trackpoint"):
        pts, ts, ele = [], [], []
        for tp in u.iter("Trackpoint"):
            pos = tp.find("Position")
            if pos is None:
                continue
            xy = _lonlat(_text(pos.find("LongitudeDegrees")), _text(pos.find("LatitudeDegrees")))
            if xy is None:
                continue
            pts.append(xy)
            tt = tp.find("Time")
            ts.append(_parse_time(tt.text if tt is not None else None))
            aa = tp.find("AltitudeMeters")
            ele.append(_float(aa.text if aa is not None else None))
        sport = u.get("Sport") if u.tag == "Activity" else None
        kind = {"Running": "Run", "Biking": "Ride"}.get(sport, sport) if sport else None
        out.append((pts, ts[0] if ts else None, kind, ts, ele, _text(u.find("Name")) if u.tag == "Course" else ""))
    if len(out) == 1 and not out[0][5]:
        out[0] = out[0][:5] + (_text(root.find(".//Course/Name")),)
    return out


FIT_SPORTS = {"running": "Run", "cycling": "Ride", "walking": "Walk", "hiking": "Hike"}


def parse_fit(data: bytes):
    """[(points, first time, type, times, elevations, name)]: a FIT file is one activity and carries no name."""
    import fitdecode  # optional dependency
    pts, t0, kind, ts, ele = [], None, None, [], []
    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name in ("sport", "session") and frame.has_field("sport"):
                kind = FIT_SPORTS.get(str(frame.get_value("sport")), kind)
            if frame.name == "record" and frame.has_field("position_lat") and frame.has_field("position_long"):
                la, lo = frame.get_value("position_lat"), frame.get_value("position_long")
                if la is None or lo is None:
                    continue
                xy = _lonlat(lo * 180.0 / 2**31, la * 180.0 / 2**31)
                if xy is None:
                    continue
                pts.append(xy)
                ts.append(frame.get_value("timestamp") if frame.has_field("timestamp") else None)
                alt = next((frame.get_value(f) for f in ("enhanced_altitude", "altitude") if frame.has_field(f)), None)
                ele.append(_float(alt))
                if t0 is None:
                    t0 = ts[-1]
    return [(pts, t0, kind, ts, ele, "")]


def _decode(name: str, raw: bytes):
    low = name.lower()
    if low.endswith(".gz"):
        raw = gzip.decompress(raw)
        low = low[:-3]
    if low.endswith(".gpx"):
        return parse_gpx(raw)
    if low.endswith(".tcx"):
        return parse_tcx(raw)
    if low.endswith(".fit"):
        return parse_fit(raw)
    raise ValueError("unsupported")


def _col(row, key):
    for h in HEADERS[key]:
        if h in row:
            return row[h]
    return None


def _read_file(path):
    with open(path, "rb") as f:
        return f.read()


def is_activity_file(path: str) -> bool:
    """A single GPX, TCX or FIT file (also .gz), rather than an export zip or folder."""
    return os.path.isfile(path) and path.lower().endswith(SUFFIXES)


def read_export(path: str):
    """Return (activities, report). `path` may be a Strava export zip, an extracted export folder, any GPX/TCX/FIT
    folder, or a single GPX/TCX/FIT file (also .gz): an export of the activities it holds. Every track of a GPX file
    and every activity or course of a TCX file is an activity of its own (id "file#n" when a file holds several), so
    the ends of each are learned as private places, wherever the file came from."""
    rep = IngestReport()
    files = {}
    if is_activity_file(path):
        files[os.path.basename(path)] = lambda: _read_file(path)
    elif zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            if not n.endswith("/"):
                files[n] = lambda n=n: z.read(n)
    else:
        for d, _, fs in os.walk(path):
            for f in fs:
                full = os.path.join(d, f)
                files[os.path.relpath(full, path).replace("\\", "/")] = lambda full=full: open(full, "rb").read()
    meta = {}
    csv_name = next((n for n in files if n.split("/")[-1].lower() == "activities.csv"), None)
    if csv_name:
        text = files[csv_name]().decode("utf-8-sig", errors="replace")
        for row in csv.DictReader(io.StringIO(text)):
            fn = (_col(row, "file") or "").strip()
            if fn:
                meta[fn] = {"id": _col(row, "id"), "type": _col(row, "type"), "name": (_col(row, "name") or "").strip()}
            else:
                rep.skip("no_gps", "", "no GPS file (indoor or manual)")
    acts = []
    for name, read in sorted(files.items()):
        low = name.lower()
        if not low.endswith(SUFFIXES):
            continue
        rep.files_seen += 1
        try:
            tracks = [t for t in _decode(name, read()) if len(t[0]) >= 2]
        except ImportError:
            rep.skip("unsupported", name, "FIT file but fitdecode is not installed")
            continue
        except Exception:
            rep.skip("unreadable", name, "corrupt or unreadable file")
            continue
        if not tracks:
            rep.skip("no_gps", name, "file without GPS points")
            continue
        m = meta.get(name, {})
        base = str(m.get("id") or os.path.basename(name).split(".")[0])
        for n, (pts, t0, kind, ts, ele, title) in enumerate(tracks, 1):  # a file of many tracks: one activity each
            secs = None
            if ts and all(t is not None for t in ts) and len(ts) == len(pts):
                secs = [(t - ts[0]).total_seconds() for t in ts]
            heights = ele if ele and len(ele) == len(pts) and all(e is not None for e in ele) else None
            act = clean(Activity(id=base if len(tracks) == 1 else f"{base}#{n}",
                                 type=(m.get("type") or kind or "Unknown"), start=t0, lonlat=pts, source=name,
                                 times=secs, ele=heights, name=m.get("name") or title))
            if act is None:
                rep.skip("no_gps", name, "only stray GPS fixes")
                continue
            acts.append(act)
            rep.used += 1
    return acts, rep
