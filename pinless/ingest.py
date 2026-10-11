"""Read a Strava bulk export (zip or folder) or a folder of GPX/TCX/FIT files, locally.

Nothing leaves the machine. Corrupt or unreadable files are skipped and counted, never fatal.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import os
import zipfile
from dataclasses import dataclass, field
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
             "walking": "Walk", "walk": "Walk", "hiking": "Hike", "10": "Walk"}


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
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_gpx(data: bytes):
    root = _strip_ns(ET.fromstring(data))
    pts, t0, kind = [], None, None
    for trk in root.iter("trk"):
        k = trk.find("type")
        if k is not None and k.text:
            kind = GPX_TYPES.get(k.text.strip().lower(), k.text.strip())
    ts, ele = [], []
    for p in root.iter("trkpt"):
        try:
            pts.append((float(p.get("lon")), float(p.get("lat"))))
        except (TypeError, ValueError):
            continue
        tt = p.find("time")
        ts.append(_parse_time(tt.text if tt is not None else None))
        ee = p.find("ele")
        ele.append(_float(ee.text if ee is not None else None))
        if t0 is None:
            t0 = ts[-1]
    return pts, t0, kind, ts, ele


def parse_tcx(data: bytes):
    root = _strip_ns(ET.fromstring(data.strip()))
    pts, t0, kind, ts, ele = [], None, None, [], []
    act = root.find(".//Activity")
    if act is not None and act.get("Sport"):
        kind = {"Running": "Run", "Biking": "Ride"}.get(act.get("Sport"), act.get("Sport"))
    for tp in root.iter("Trackpoint"):
        pos = tp.find("Position")
        if pos is None:
            continue
        la, lo = pos.find("LatitudeDegrees"), pos.find("LongitudeDegrees")
        try:
            pts.append((float(lo.text), float(la.text)))
        except (AttributeError, TypeError, ValueError):
            continue
        tt = tp.find("Time")
        ts.append(_parse_time(tt.text if tt is not None else None))
        aa = tp.find("AltitudeMeters")
        ele.append(_float(aa.text if aa is not None else None))
        if t0 is None:
            t0 = ts[-1]
    return pts, t0, kind, ts, ele


def parse_fit(data: bytes):
    import fitdecode  # optional dependency
    pts, t0, kind, ts, ele = [], None, None, [], []
    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "sport" and frame.has_field("sport"):
                kind = {"running": "Run", "cycling": "Ride", "walking": "Walk"}.get(str(frame.get_value("sport")), kind)
            if frame.name == "record" and frame.has_field("position_lat") and frame.has_field("position_long"):
                la, lo = frame.get_value("position_lat"), frame.get_value("position_long")
                if la is None or lo is None:
                    continue
                pts.append((lo * 180.0 / 2**31, la * 180.0 / 2**31))
                ts.append(frame.get_value("timestamp") if frame.has_field("timestamp") else None)
                alt = next((frame.get_value(f) for f in ("enhanced_altitude", "altitude") if frame.has_field(f)), None)
                ele.append(_float(alt))
                if t0 is None:
                    t0 = ts[-1]
    return pts, t0, kind, ts, ele


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


def read_export(path: str):
    """Return (activities, report). `path` may be a Strava export zip, an extracted export folder or any GPX/TCX/FIT folder."""
    rep = IngestReport()
    files = {}
    if zipfile.is_zipfile(path):
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
        if not any(low.endswith(e) for e in (".gpx", ".gpx.gz", ".tcx", ".tcx.gz", ".fit", ".fit.gz")):
            continue
        rep.files_seen += 1
        try:
            pts, t0, kind, ts, ele = _decode(name, read())
        except ImportError:
            rep.skip("unsupported", name, "FIT file but fitdecode is not installed")
            continue
        except Exception:
            rep.skip("unreadable", name, "corrupt or unreadable file")
            continue
        if len(pts) < 2:
            rep.skip("no_gps", name, "file without GPS points")
            continue
        m = meta.get(name, {})
        secs = None
        if ts and all(t is not None for t in ts) and len(ts) == len(pts):
            secs = [(t - ts[0]).total_seconds() for t in ts]
        heights = ele if ele and len(ele) == len(pts) and all(e is not None for e in ele) else None
        acts.append(Activity(id=str(m.get("id") or os.path.basename(name).split(".")[0]),
                             type=(m.get("type") or kind or "Unknown"), start=t0, lonlat=pts, source=name, times=secs,
                             ele=heights, name=m.get("name", "")))
        rep.used += 1
    return acts, rep
