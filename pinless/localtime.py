"""Local wall-clock time of an activity. Files store UTC; a 7:00 run in Madrid is written as 5:00 or 6:00, so the
time zone is looked up, offline, from where the activity started."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo


@lru_cache(maxsize=4096)
def _zone(lon2: float, lat2: float):
    try:
        import tzfpy

        return ZoneInfo(tzfpy.get_tz(lon2, lat2) or "UTC")
    except Exception:  # no time zone data: stay in UTC rather than guess
        return dt.timezone.utc


def local_start(act):
    """The activity's start in local time where it began, or None if unknown. A naive time is taken as UTC."""
    if act.start is None or not act.lonlat:
        return None
    start = act.start if act.start.tzinfo else act.start.replace(tzinfo=dt.timezone.utc)
    lon, lat = act.lonlat[0]
    return start.astimezone(_zone(round(lon, 2), round(lat, 2)))
