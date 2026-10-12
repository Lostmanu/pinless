"""pinless EXPORT [--year 2025] [--name Lucía] [--place "Madrid"] [--layout editorial|atlas|race|trail] [--style huella]
pinless EXPORT --with EXPORT2 --name Lucía --with-name Marcos   (one poster for two, one ink each)
pinless hike.gpx --layout trail --title "Peñalara"   (one route on a topographic map; EXPORT may be one GPX/TCX/FIT)

With no area options it finds where you run most, fetches that area's OpenStreetMap data once (cached) and names the
city. Everything else runs locally: the export never leaves the machine; only the area's bounding box (and its centre
rounded to about 1 km, for the city name) is sent to OpenStreetMap services.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, box

from . import atlas, colouring, couple, frame, ingest, privacy, race, render, render_sport, render_editorial, stats
from . import trail
from . import relief, styles as own_styles
from .localtime import local_start

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _glue_negative_values(argv):
    """`--zone -33.87,151.21,500` and `--bbox -34.1,150.9,-33.6,151.4` (south of the equator, west of Greenwich) as
    `--zone=-33.87,...`: before Python 3.14 argparse takes a value that starts with '-' and is not a plain number for
    another option and stops with 'expected one argument'."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in ("--zone", "--bbox") and i + 1 < len(argv) and re.match(r"-[\d.]", argv[i + 1]):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="pinless")
    p.add_argument("export")
    p.add_argument("--bbox", default=None, help="south,west,north,east (degrees); default: where you run most")
    p.add_argument("--place", default=None, help="use a named place as the area, e.g. \"Valencia, Spain\"")
    p.add_argument("--cache", default=None, help="pre-fetched OSM folder name under ./cache (tools/fetch_osm.py)")
    p.add_argument("--title", default="", help="headline text; default: --place or the city found at the area centre")
    p.add_argument("--year", type=int, default=None, help="only activities that started that year (local time)")
    p.add_argument("--type", default=None,
                   help="activity type (Run, Ride, Walk, Hike...); default Run, and any type for the trail layout or a "
                        "single GPX, TCX or FIT file")
    p.add_argument("--layout", default="editorial", choices=["classic", "sport", "editorial", "atlas", "race", "trail"],
                   help="atlas: one small map per city you ran in; race: one activity as a finisher print; trail: one "
                        "route (a hike, a trail run) on a topographic map")
    p.add_argument("--max-cities", type=int, default=9, help="atlas: most cities on the print (1-12)")
    p.add_argument("--activity", default="longest",
                   help="race and trail: longest (default), latest, a date YYYY-MM-DD or an activity id")
    p.add_argument("--allow-private-ends", action="store_true",
                   help="race and trail: draw an activity that starts or ends at a private place, without that end "
                        "and a random stretch more, with its numbers limited to what is drawn")
    p.add_argument("--graphs", action="store_true",
                   help="editorial: add km per month, time of day and direction charts under the map")
    p.add_argument("--color-by", default=None, choices=colouring.MODES,
                   help="route colour: heat (how often), single, month, pace or elevation (editorial and atlas); "
                        "race: pace (default), elevation or single; trail: single (default), elevation or pace")
    p.add_argument("--name", default="", help="runner name for sport/editorial layouts (gifts)")
    p.add_argument("--headline", default="city", choices=["city", "name"],
                   help="editorial headline: city (default) or runner name")
    p.add_argument("--style", default=None,
                   help="classic: nightrun|ink · sport: volt|pista · editorial, atlas, race and trail: huella, carbon, "
                        "midnight, blueprint, ink, graphite, neon, lavender, sunset, forest, blush, ocean, or your "
                        "own style file (my-style.json; see python -m pinless.styles)")
    p.add_argument("--size", default="A3", choices=sorted(render.SIZES))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--lang", default="en", choices=["en", "es"])
    p.add_argument("--zone", action="append", default=[],
                   help="extra private zone lat,lon,radius_m (repeatable); with --with it hides both people's lines")
    p.add_argument("--trim", type=float, default=300.0)
    p.add_argument("--zone-radius", type=float, default=500.0)
    p.add_argument("--fit", action="store_true", help="crop the map to where the activities are (within the bbox)")
    p.add_argument("--out", default="poster.png")
    p.add_argument("--pdf", default=None)
    p.add_argument("--svg", default=None, help="vector export for the editorial layout")
    p.add_argument("--preview", default=None, help="lightweight editorial JPEG preview (1600 px high)")
    p.add_argument("--relief", default=None, choices=["off", "shade"],
                   help="shade: terrain relief under the map where the land is hilly (editorial, atlas, race and "
                        "trail), from the Copernicus elevation model on Amazon S3, which learns the 1-degree cells "
                        "around the map and, where the land is hilly, the blocks of about 30 km that cover it (never "
                        "your routes); downloaded once (cached); flat areas get none. Default: shade for the trail "
                        "layout, off for the others")
    p.add_argument("--ground", default=None, choices=["nature", "style"],
                   help="trail: nature (default) paints the map's ground in the colours of the land, green woods "
                        "and meadows, wheat fields, stone rock, white snow, straw and orange in a dry land, from ESA "
                        "WorldCover on Amazon S3, which learns the 3-degree tiles around the map and the blocks that "
                        "cover it (from about 9 by 7 km at 40 degrees of latitude; never your routes); downloaded "
                        "once, cached; style keeps the plain paper of the style")
    p.add_argument("--with", dest="with_export", default=None, metavar="EXPORT",
                   help="a second person's Strava export: one poster for two, one ink each (editorial)")
    p.add_argument("--with-name", default="", help="the second person's name (required with --with)")
    p.add_argument("--with-type", default=None,
                   help="activity type in the second export, when it names the sport differently (default: --type)")
    a = p.parse_args(_glue_negative_values(sys.argv[1:] if argv is None else list(argv)))
    # a trail can be any activity (a hike, a walk, a trail run), and so can a single file: a GPX from a trail app or
    # a route planner often names no type at all
    if a.type is None and (a.with_export or (a.layout != "trail" and not ingest.is_activity_file(a.export))):
        a.type = "Run"
    if a.relief is None:  # relief is the point of a mountain print
        a.relief = "shade" if a.layout == "trail" else "off"
    if a.ground is not None and a.layout != "trail":
        p.error("--ground is available for the trail layout only, for now")
    if a.ground is None:
        a.ground = "nature" if a.layout == "trail" else "style"
    if (a.with_name or a.with_type) and not a.with_export:
        p.error("--with-name and --with-type need --with")
    if a.with_export:
        if a.layout != "editorial":
            p.error("couple mode (--with) is available for the editorial layout")
        if not a.name.strip() or not a.with_name.strip():
            p.error("--with needs both names, for the headline and the key: --name and --with-name")
        if a.name.strip().casefold() == a.with_name.strip().casefold():
            p.error("the two names label the two inks: they must differ")
        if a.color_by is not None:
            p.error("in couple mode the colour says whose street it is: --color-by does not apply")
        if a.graphs:
            p.error("--graphs is not available with --with yet")
        if os.path.exists(a.export) and os.path.exists(a.with_export) and os.path.samefile(a.export, a.with_export):
            p.error("--with is the same export as the first one")
    if a.color_by is None:
        a.color_by = "heat"
    if a.bbox is not None and not _bbox_ok(a.bbox):
        p.error("--bbox needs four numbers south,west,north,east in degrees, with south below north and west below "
                "east")
    if not all(_zone_ok(z) for z in a.zone):
        p.error("--zone needs LAT,LON,RADIUS_M: the latitude first (-90 to 90), then the longitude (-180 to 180), then "
                "a radius in metres above 0")
    styles = {"classic": render.STYLES, "sport": render_sport.STYLES, "editorial": render_editorial.STYLES,
              "atlas": render_editorial.STYLES, "race": render_editorial.STYLES, "trail": render_editorial.STYLES}
    if own_styles.is_file(a.style):  # your own style: checked and loaded before any file is read
        if a.layout not in ("editorial", "atlas", "race", "trail"):
            p.error("a style file works with the editorial, atlas, race and trail layouts")
        try:
            a.style, notes = own_styles.register(a.style)
        except own_styles.StyleError as ex:
            p.error(f"--style: {ex}")
        if a.with_export and "who" not in render_editorial.STYLES[a.style]:
            p.error('--style: a poster for two needs "who", five inks (or a "base" style to take them from)')
        for note in notes:
            print(f"[style] {note}")
    elif a.style and a.style not in styles[a.layout]:
        p.error(f"style {a.style!r} is not available for {a.layout}; choose {', '.join(styles[a.layout])}, or "
                f"give a style file (my-style.json)")
    if a.graphs and a.layout != "editorial":
        p.error("--graphs is available for the editorial layout")
    if a.color_by != "heat" and a.layout not in ("editorial", "atlas", "race", "trail"):
        p.error("--color-by is available for the editorial, atlas, race and trail layouts")
    if a.svg and a.layout not in ("editorial", "atlas", "race", "trail"):
        p.error("--svg is available for the editorial, atlas, race and trail layouts")
    if a.relief != "off" and a.layout not in ("editorial", "atlas", "race", "trail"):
        p.error("--relief is available for the editorial, atlas, race and trail layouts")
    if (a.activity != "longest" or a.allow_private_ends) and a.layout not in ("race", "trail"):
        p.error("--activity and --allow-private-ends are for the race and trail layouts")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.activity):
        try:
            dt.date.fromisoformat(a.activity)
        except ValueError:
            p.error(f"--activity {a.activity}: not a real date (YYYY-MM-DD)")
    if a.layout == "race" and a.color_by not in ("heat", "pace", "elevation", "single"):
        p.error("the race layout colours its course by pace (default), elevation or single")
    if a.layout == "trail" and a.color_by not in ("heat", "pace", "elevation", "single"):
        p.error("the trail layout colours its route in one colour (default), or by elevation or pace")
    if a.layout in ("race", "trail") and (a.bbox or a.place or a.cache or a.fit or a.headline != "city" or a.graphs):
        p.error(f"the {a.layout} layout frames its course by itself: --bbox, --place, --cache, --fit, --headline and "
                f"--graphs do not apply")
    if a.layout == "atlas" and (a.bbox or a.place or a.cache or a.fit or a.headline != "city"):
        p.error("the atlas finds every city by itself: --bbox, --place, --cache, --fit and --headline do not apply")
    if not 1 <= a.max_cities <= 12:
        p.error("--max-cities must be between 1 and 12")
    if a.headline == "name" and a.layout != "editorial":
        p.error("--headline name is available for the editorial layout")
    if a.headline == "name" and not a.name.strip():
        p.error("--headline name requires --name")
    if a.preview and (a.layout not in ("editorial", "atlas", "race", "trail") or
                      os.path.splitext(a.preview)[1].lower() not in (".jpg", ".jpeg")):
        p.error("--preview requires the editorial, atlas, race or trail layout and a .jpg or .jpeg filename")
    t0 = time.time()

    if a.with_export:
        return _couple_main(a, t0)
    acts, rep = ingest.read_export(a.export)
    print(f"[ingest] files {rep.files_seen} · used {rep.used} · no GPS {rep.no_gps} · unreadable {rep.unreadable} · "
          f"unsupported {rep.unsupported}")
    if len(acts) == 1 and a.layout != "trail":  # the trail print says it where it learns its places
        print(trail.ONE_ACTIVITY)
    sel = _select(acts, a.type, a.year)
    if not sel:
        kinds = ", ".join(f"{t} {n}" for t, n in Counter(x.type if x.type in trail.KINDS else "other"  # free text
                                                         for x in acts).most_common(6))  # in a file: never logged
        print("no activities match the type/year filters" + (f" (this export has: {kinds}; choose the type with "
                                                             f"--type)" if kinds else ""))
        return 2
    if a.layout == "atlas":
        code = atlas.run(sel, a)
        print(f"[done] {time.time() - t0:.0f}s")
        return code
    if a.layout == "race":  # one activity; zones from every activity of that type, whatever the year
        code = race.run(sel, [x for x in acts if a.type is None or x.type.lower() == a.type.lower()], a)
        print(f"[done] {time.time() - t0:.0f}s")
        return code
    if a.layout == "trail":  # one activity; zones from every other activity of the export, whatever its type or year
        code = trail.run(sel, acts, a)
        print(f"[done] {time.time() - t0:.0f}s")
        return code
    try:
        s, w, n, e = frame.area_box(sel, a, print)
    except frame.PlaceNotFound:
        print(frame.PLACE_NOT_FOUND)
        return 2
    area_ll = box(w, s, e, n)
    sel = [x for x in sel if LineString(x.lonlat).intersects(area_ll)]
    if not sel:
        print("no activities match the filters inside the area")
        return 2
    crs = render.utm_crs((w + e) / 2, (s + n) / 2)
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    tracks = []
    total_m = 0.0
    for x in sel:
        arr = np.asarray(x.lonlat)
        xs, ys = tr.transform(arr[:, 0], arr[:, 1])
        xy = np.column_stack([xs, ys])
        total_m += float(np.hypot(*np.diff(xy, axis=0).T).sum())
        tracks.append((x.id, xy))
    user_zones = frame.user_zones(a, tr)
    if a.layout == "editorial":
        aspect = render_editorial.map_aspect(a.size, a.graphs)
    else:
        aspect = 1 / 1.22 if a.layout == "classic" else (render.SIZES[a.size][0] - 28 * render.SIZES[a.size][0] / 297) / (268 * render.SIZES[a.size][0] / 297)
    off = frame.zone_off_map(user_zones, render.fit_extent((s, w, n, e), crs, aspect))
    if off:
        print(frame.ZONE_OFF_MAP.format(k=off))
        return 2
    # privacy per activity, keeping activity ids for the heat (each activity counts once per pixel)
    pieces_all = []
    try:
        pieces, prep = privacy.apply([t for _, t in tracks], trim_m=a.trim, zone_r=a.zone_radius, user_zones_xy=user_zones)
    except privacy.PrivacyError as ex:
        print(f"[privacy] STOP: {ex}. Nothing was exported.")
        return 3
    pieces_all = pieces  # (activity_index, LineString): each activity counts once per pixel in the heat
    print(f"[privacy] zones suggested {prep.zones_suggested} · user zones {prep.zones_user} · "
          f"km in {prep.km_in:.0f} -> out {prep.km_out:.0f} · leak check {prep.leak_check}")
    extent = frame.extent_for(pieces, (s, w, n, e), crs, aspect, a.fit)
    try:
        basemap = frame.basemap_for(extent, crs, a, print)
    except RuntimeError as ex:  # every OpenStreetMap server busy
        print(f"[osm] STOP: {ex}. Run the same command again later.")
        return 4
    relief_ = relief.prepare_many([(extent, crs, render.SIZES[a.size][0] * (1 - 36 / 297))],
                                  log=print)[0] if a.relief == "shade" else None
    frame.headline(a, (s, w, n, e), print)
    stats_ = {"n": len(sel), "km": total_m / 1000, "year_label": frame.year_label(sel, a)}
    if a.layout in ("sport", "editorial"):
        drawn = privacy.clip_to_extent(pieces, extent)  # streets on the drawn lines only, never beyond the map
        st_ = stats.compute(sel, [t for _, t in tracks], basemap[0],
                            private_pieces_xy=[np.asarray(pc.coords) for _, pc in drawn],
                            piece_groups=[i for i, _ in drawn], sensitive_xy=prep.centres)
        print(f"[stats] {st_['n']} runs · {st_['km']:.0f} km · {st_['hours']:.0f} h · best 5k {stats.fmt_time(st_['best5k_s'])} · "
              f"best 10k {stats.fmt_time(st_['best10k_s'])} · {st_.get('streets')} streets · fav: {st_.get('fav_street')}")
        if a.layout == "editorial":
            info = render_editorial.render(pieces_all, basemap, extent, st_, a.out, style=a.style or "huella",
                                           size=a.size, dpi=a.dpi, name=a.name, city=a.title,
                                           year=stats_["year_label"], lang=a.lang, pdf_path=a.pdf, svg_path=a.svg,
                                           headline=a.headline, preview_path=a.preview, color_by=a.color_by,
                                           activities=sel, tracks_xy=[t for _, t in tracks], graphs=a.graphs,
                                           relief=relief_)
        else:
            info = render_sport.render(pieces_all, basemap, extent, st_, a.out, style=a.style or "volt", size=a.size, dpi=a.dpi,
                                       name=a.name, city=a.title, year=stats_["year_label"], lang=a.lang, pdf_path=a.pdf)
        info.setdefault("cap", 0)
        if info.get("legend", {}).get("fallback"):
            print(f"[colour] no {a.color_by} data in these activities: drawn in one colour")
    else:
        info = render.render(pieces_all, basemap, extent, stats_, a.out, style=a.style or "nightrun", size=a.size, dpi=a.dpi,
                             title=a.title, lang=a.lang, center_ll=((s + n) / 2, (w + e) / 2), pdf_path=a.pdf)
    print(f"[render] {a.out} {info['W']}x{info['H']} map px · heat cap {info['cap']:.0f} · max overlap {info['max']} · "
          f"{time.time() - t0:.0f}s")
    return 0


def _bbox_ok(text):
    """--bbox: four finite numbers south,west,north,east in degrees, south below north and west below east."""
    try:
        s, w, n, e = (float(v) for v in text.split(","))
    except ValueError:
        return False
    return -90 <= s < n <= 90 and -180 <= w < e <= 180


def _zone_ok(text):
    """--zone: three finite numbers LAT,LON,RADIUS_M, the latitude within -90 to 90, the longitude within -180 to 180
    and the radius above 0 metres."""
    try:
        la, lo, r = (float(v) for v in text.split(","))
    except ValueError:
        return False
    return -90 <= la <= 90 and -180 <= lo <= 180 and 0 < r < math.inf


def _select(acts, type_, year):
    """The activities of a type (case-insensitive; None: any type) and, when given, a year. The year of each run is
    the one where it was run: a New Year's Eve run in Madrid at 00:30 belongs to the new year."""
    return [x for x in acts if (type_ is None or x.type.lower() == type_.lower())
            and (year is None or ((t := local_start(x)) is not None and t.year == year))]


def _couple_main(a, t0):
    """--with: read both exports, name each person (never a path), select, then couple.run."""
    names = (a.name.strip(), a.with_name.strip())
    types = (a.type, a.with_type or a.type)
    read = []
    for name, path, flag in zip(names, (a.export, a.with_export), ("the first export", "--with")):
        acts, rep = ingest.read_export(path)
        if not rep.files_seen and not os.path.exists(path):  # a mistyped path reads as an empty export
            print(f"{name}: there is no file or folder at the path given for {flag}")
            return 2
        print(f"[ingest] {name}: files {rep.files_seen} · used {rep.used} · no GPS {rep.no_gps} · unreadable "
              f"{rep.unreadable} · unsupported {rep.unsupported}")
        read.append(acts)
    sels = []
    for k, (name, acts, type_) in enumerate(zip(names, read, types)):
        sel = _select(acts, type_, a.year)
        if not sel:
            kinds = ", ".join(f"{t} {n}" for t, n in Counter(x.type for x in acts).most_common(6))
            print(f"{name}: no {type_} activities{f' in {a.year}' if a.year else ''} "
                  f"({'this export has: ' + kinds if kinds else 'this export has no GPS activities'})."
                  + (" Choose the type with --type." if k == 0 else
                     " If this export names the sport differently, pass it with --with-type."))
            return 2
        sels.append(sel)
    code = couple.run(read[0], read[1], sels[0], sels[1], a)
    print(f"[done] {time.time() - t0:.0f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
