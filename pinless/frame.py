"""The frame of a single-map poster, shared by the solo and the couple flows: the area, the map extent, the basemap,
the headline and the year label. Only the area's bounding box goes out (OpenStreetMap), and its centre rounded to
about 1 km for the city name, as in cli.main."""
from __future__ import annotations

import os

import numpy as np
from pyproj import Transformer

from . import area, render
from .localtime import local_start


class PlaceNotFound(ValueError):
    """--place: OpenStreetMap found no such place, or did not answer."""


PLACE_NOT_FOUND = ("[area] OpenStreetMap did not find that --place (or did not answer): check the name, for example "
                   "\"Valencia, Spain\", or pass --bbox")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def area_box(acts, a, log=print, groups=None):
    """(south, west, north, east): --bbox, --place, or where the activities are most (with groups: where every group
    is). Raises area.NoSharedArea when groups never share an area, and PlaceNotFound when OpenStreetMap does not
    find --place or does not answer."""
    if a.bbox:
        s, w, n, e = map(float, a.bbox.split(","))
    elif a.place:
        try:
            s, w, n, e = area.geocode_bbox(a.place)
        except ValueError as ex:  # osmnx: nothing found (InsufficientResponseError), or a server error status
            raise PlaceNotFound(str(a.place)) from ex
        log(f"[area] {a.place}")
    elif groups is not None:
        s, w, n, e = area.auto_bbox(acts, groups=groups)
        log("[area] found where you both run most")
    else:
        s, w, n, e = area.auto_bbox(acts)
        log("[area] found where you run most")
    return s, w, n, e


def user_zones(a, tr):
    """--zone lat,lon,radius_m as (x, y, r) in the map's projection (tr: lon/lat to the map)."""
    out = []
    for z in a.zone:
        la, lo, r = map(float, z.split(","))
        zx, zy = tr.transform(lo, la)
        out.append((zx, zy, r))
    return out


def extent_for(pieces, bbox, crs, aspect, fit):
    """The map extent: the box grown to the map aspect, or with fit, cropped to where the drawn pieces are."""
    s, w, n, e = bbox
    if fit and pieces:
        pts = np.concatenate([np.asarray(pc.coords) for _, pc in pieces])
        lo, hi = np.percentile(pts, 0.7, axis=0), np.percentile(pts, 99.3, axis=0)
        pad = 0.07 * max(hi - lo)
        inv = Transformer.from_crs(crs, 4326, always_xy=True)
        (w2, e2), (s2, n2) = inv.transform([lo[0] - pad, hi[0] + pad], [lo[1] - pad, hi[1] + pad])
        return render.fit_extent((max(s, s2), max(w, w2), min(n, n2), min(e, e2)), crs, aspect)
    return render.fit_extent((s, w, n, e), crs, aspect)


def basemap_for(extent, crs, a, log=print):
    """(edges, green, water) around the extent: --cache, or fetched once and cached. Raises RuntimeError when no
    OpenStreetMap server answers (the caller prints [osm] STOP)."""
    exp_ll = Transformer.from_crs(crs, 4326, always_xy=True).transform([extent[0], extent[1]], [extent[2], extent[3]])
    big = (min(exp_ll[1]) - 0.01, min(exp_ll[0]) - 0.01, max(exp_ll[1]) + 0.01, max(exp_ll[0]) + 0.01)
    if a.cache:
        folder = os.path.join(ROOT, "cache", a.cache)
    else:
        folder = area.fetch_basemap(big, log=log)
    return render.load_basemap(folder, big, crs)


def headline(a, bbox, log=print):
    """Sets a.title when it is empty: the --place name, or the city at the box centre (only that point, rounded to
    about 1 km, is sent)."""
    s, w, n, e = bbox
    if not a.title:
        a.title = (a.place.split(",")[0] if a.place else area.city_name((s + n) / 2, (w + e) / 2, lang=a.lang))
        if a.title:
            log(f"[area] headline: {a.title}")


def year_label(acts, a):
    """--year, or the span of local years of the activities ("2025", "2024–2025")."""
    years = sorted({t.year for x in acts if (t := local_start(x))})
    span = f"{years[0]}–{years[-1]}" if len(years) > 1 else (str(years[0]) if years else "")  # one year: "2025"
    return str(a.year) if a.year else span
