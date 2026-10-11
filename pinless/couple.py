"""--with: one editorial poster for two people, one ink each.

Same pipeline as the single poster, with the privacy a shared map needs:
- the area is where BOTH run (area.auto_bbox with groups), unless --place or --bbox says otherwise;
- every private place of either person (each found with that person's own threshold, so a partner with few runs
  keeps their home or gym), and every place where their ends meet, hides BOTH people's lines: one forbidden area is
  drawn once, so the two inks break at the same edge and no hole in one ink is crossed by the other;
- each person must still have something on the map after the cuts, or nothing is exported.
The line colour says whose street it is (the street's share, on paper, in five steps); the weight how often either
ran it. Logs carry names and counts only: never coordinates, paths, dates, activity names or ids.
"""
from __future__ import annotations

import datetime as dt
import os
from collections import Counter

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, box

from . import area, frame, privacy, render, render_editorial, stats


def _fingerprint(act):
    """(start in UTC to the second, number of points, first point to 1e-6 degrees): the same file in two exports."""
    start = None
    if act.start is not None:
        t = act.start if act.start.tzinfo else act.start.replace(tzinfo=dt.timezone.utc)
        start = t.astimezone(dt.timezone.utc).replace(microsecond=0)
    lon0, lat0 = act.lonlat[0] if act.lonlat else (None, None)
    return (start, len(act.lonlat), None if lon0 is None else round(lon0, 6), None if lat0 is None else round(lat0, 6))


def same_activities(acts_a, acts_b):
    """(how many activities are in both exports, that as a share of the smaller export)."""
    if not acts_a or not acts_b:
        return 0, 0.0
    shared = sum((Counter(map(_fingerprint, acts_a)) & Counter(map(_fingerprint, acts_b))).values())
    return shared, shared / min(len(acts_a), len(acts_b))


OWN_INK_MIN = 0.01  # below this share of a person's drawn length in their own inks, the log says so
SAME_MIN = 3  # fewest files in both exports that, with half of the smaller one, make them the same export


def same_export(acts_a, acts_b):
    """(refuse, shared): the two exports are one person's activities twice when the smaller one is wholly inside the
    other, or when at least SAME_MIN of its activities and half of it are. A few files in both (a partner who uploaded
    the other's run when a watch died) are allowed."""
    shared, frac = same_activities(acts_a, acts_b)
    whole = shared and shared == min(len(acts_a), len(acts_b))
    return bool(whole or (shared >= SAME_MIN and frac >= 0.5)), shared


def _km(tracks):
    return sum(float(np.hypot(*np.diff(t, axis=0).T).sum()) for t in tracks if len(t) > 1) / 1000


def run(acts_a, acts_b, sel_a, sel_b, a, log=print):
    """CLI entry for --with. acts_*: each export's activities; sel_*: after the type and year filters. Exit code."""
    na, nb = a.name.strip(), a.with_name.strip()
    types = (a.type, a.with_type or a.type)

    refuse, shared = same_export(acts_a, acts_b)
    if refuse:
        log(f"the two exports hold the same activities ({shared} of {min(len(acts_a), len(acts_b))}): pass each "
            f"person's own export")
        return 2
    if shared:
        log(f"[couple] {shared} activit{'y is' if shared == 1 else 'ies are'} in both exports (the same file); "
            f"{'it is' if shared == 1 else 'they are'} drawn for each of you")

    try:
        bbox = frame.area_box(sel_a + sel_b, a, log, groups=[0] * len(sel_a) + [1] * len(sel_b))
    except area.NoSharedArea:
        log("your runs never share an area: choose one with --place \"City, Country\" or --bbox")
        return 2
    except frame.PlaceNotFound:
        log(frame.PLACE_NOT_FOUND)
        return 2
    s, w, n, e = bbox
    area_ll = box(w, s, e, n)
    inside = []
    for name, sel, type_ in ((na, sel_a, types[0]), (nb, sel_b, types[1])):
        mine = [x for x in sel if LineString(x.lonlat).intersects(area_ll)]
        if not mine:
            log(f"{name} has no {type_} activities inside this map; choose a place you both ran in with --place "
                f"\"City, Country\" or --bbox")
            return 2
        if len(mine) < len(sel):
            log(f"[area] {name}: {len(mine)} of {len(sel)} activities are inside the map; the others are not counted")
        inside.append(mine)
    sel_a, sel_b = inside

    crs = render.utm_crs((w + e) / 2, (s + n) / 2)
    tr = Transformer.from_crs(4326, crs, always_xy=True)

    def project(x):
        arr = np.asarray(x.lonlat, dtype=float)
        return np.column_stack(tr.transform(arr[:, 0], arr[:, 1]))

    ta, tb = [project(x) for x in sel_a], [project(x) for x in sel_b]
    user = frame.user_zones(a, tr)
    aspect = render_editorial.map_aspect(a.size, False)
    off = frame.zone_off_map(user, render.fit_extent(bbox, crs, aspect))
    if off:  # a --zone that misses the map hides nothing on it: never a silent 'user zones 1'
        log(frame.ZONE_OFF_MAP.format(k=off))
        return 2
    # a file in both exports is drawn for each person but is nobody's own run when it comes to how often a place must
    # be a start or an end: copies of the other's runs would raise a person's threshold and unhide a single loop
    copies, _ = same_activities(sel_a, sel_b)
    try:
        zones, witnesses, counts = privacy.pair_places([ta, tb], zone_r=a.zone_radius,
                                                       n_runs=(len(ta) - copies, len(tb) - copies))
        pieces, rep = privacy.apply(ta + tb, trim_m=a.trim, zone_r=a.zone_radius, user_zones_xy=user,
                                    places_xy=zones, witnesses_xy=witnesses)
    except privacy.PrivacyError as ex:
        log(f"[privacy] STOP: {ex}. Nothing was exported.")
        return 3
    owner = np.array([0] * len(ta) + [1] * len(tb), dtype=np.int8)
    out_km = [sum(p.length for i, p in pieces if owner[i] == o) / 1000 for o in (0, 1)]
    log(f"[privacy] places: {na} {counts['per'][0]} · {nb} {counts['per'][1]} · shared ends {counts['pooled_new']} "
        f"-> zones {counts['zones']} ({counts['merged']} merged) · user zones {rep.zones_user} · km in "
        f"{_km(ta):.0f}+{_km(tb):.0f} -> out {out_km[0]:.0f}+{out_km[1]:.0f} · leak check {rep.leak_check}")

    tog = stats.together(sel_a, ta, sel_b, tb)
    if tog["known"]:
        log(f"[together] {tog['runs']} runs side by side · {tog['km']:.0f} km")
    else:
        missing = [nm for nm, k, sel in ((na, tog["no_clock"][0], sel_a), (nb, tog["no_clock"][1], sel_b))
                   if k == len(sel)]
        whose = " and ".join(f"{m}'s" for m in missing)
        log(f"[together] no clock in {whose} activities: runs together not counted")

    extent = frame.extent_for(pieces, bbox, crs, aspect, a.fit)
    clipped = privacy.clip_to_extent(pieces, extent)
    for o, name in enumerate((na, nb)):  # before any request: a poster never silently shows only one of the two
        if not any(owner[i] == o for i, _ in clipped):
            log(f"[privacy] {name}: nothing of yours is left on this map after the privacy cuts. Nothing was "
                f"exported.")
            return 2
    try:
        basemap = frame.basemap_for(extent, crs, a, log)
    except RuntimeError as ex:  # every OpenStreetMap server busy
        log(f"[osm] STOP: {ex}. Run the same command again later.")
        return 4
    frame.headline(a, bbox, log)

    edges = basemap[0]
    named = "name" in edges.columns and bool(edges["name"].notna().any())
    st = []
    for o, (sel, tracks) in enumerate(((sel_a, ta), (sel_b, tb))):
        mine = [(i, p) for i, p in clipped if owner[i] == o]  # streets on the drawn lines only, never beyond the map
        x = stats.compute(sel, tracks, edges, private_pieces_xy=[np.asarray(p.coords) for _, p in mine],
                          piece_groups=[i for i, _ in mine], sensitive_xy=rep.centres, keep_streets=True)
        for key in ("fav_street", "fav_street_n"):  # no favourite street on a poster for two
            x.pop(key, None)
        st.append(x)
    names_a, names_b = st[0].pop("street_names", set()), st[1].pop("street_names", set())
    streets_both = len(names_a & names_b) if named else None
    both = (f"streets you both ran {streets_both} of {len(names_a | names_b)}" if streets_both is not None
            else "no street names in this map")
    log(f"[stats] {na} {st[0]['n']} runs · {st[0]['km']:.0f} km · {st[0]['hours']:.0f} h | {nb} {st[1]['n']} runs · "
        f"{st[1]['km']:.0f} km · {st[1]['hours']:.0f} h · {both}")

    info = render_editorial.render(
        pieces, basemap, extent, None, a.out, style=a.style or "huella", size=a.size, dpi=a.dpi, name=na,
        city=a.title, year=frame.year_label(sel_a + sel_b, a), lang=a.lang, pdf_path=a.pdf, svg_path=a.svg,
        headline=a.headline, preview_path=a.preview, activities=sel_a + sel_b, tracks_xy=ta + tb,
        couple=render_editorial.Couple(names=(na, nb), owner=owner, stats=(st[0], st[1]), together=tog,
                                       streets_both=streets_both))
    log(f"[render] {os.path.basename(a.out)} {info['W']}x{info['H']} map px · heat cap {info['cap']:.0f} · drawn km by who: "
        f"{'/'.join(f'{100 * f:.0f}' for f in info['who_frac'])} %")
    for name, other, share in ((na, nb, info["own_ink"][0]), (nb, na, info["own_ink"][1])):
        if share < OWN_INK_MIN:  # the names, the bar and the key show this person's ink; the map hardly does
            log(f"[render] {name}: under {100 * OWN_INK_MIN:.0f} % of your lines are in your own ink: the streets you "
                f"ran on this map are mostly {other}'s, so they take {other}'s ink or the shared one")
    return 0
