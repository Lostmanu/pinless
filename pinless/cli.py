"""pinless EXPORT [--year 2025] [--name Lucía] [--place "Madrid"] [--layout editorial] [--style huella]

With no area options it finds where you run most, fetches that area's OpenStreetMap data once (cached) and names the
city. Everything else runs locally: the export never leaves the machine; only the area's bounding box is sent to
OpenStreetMap services.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, box

from . import area, ingest, privacy, render, render_sport, render_editorial, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="pinless")
    p.add_argument("export")
    p.add_argument("--bbox", default=None, help="south,west,north,east (degrees); default: where you run most")
    p.add_argument("--place", default=None, help="use a named place as the area, e.g. \"Valencia, Spain\"")
    p.add_argument("--cache", default=None, help="pre-fetched OSM folder name under ./cache (tools/fetch_osm.py)")
    p.add_argument("--title", default="", help="headline city text; default: --place or the city found at the area centre")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--type", default="Run")
    p.add_argument("--layout", default="editorial", choices=["classic", "sport", "editorial"])
    p.add_argument("--name", default="", help="runner name for sport/editorial layouts (gifts)")
    p.add_argument("--headline", default="city", choices=["city", "name"],
                   help="editorial headline: city (default) or runner name")
    p.add_argument("--style", default=None, help="classic: nightrun|ink · sport: volt|pista · editorial: huella|carbon")
    p.add_argument("--size", default="A3", choices=sorted(render.SIZES))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--lang", default="en", choices=["en", "es"])
    p.add_argument("--zone", action="append", default=[], help="extra private zone lat,lon,radius_m (repeatable)")
    p.add_argument("--trim", type=float, default=300.0)
    p.add_argument("--zone-radius", type=float, default=500.0)
    p.add_argument("--fit", action="store_true", help="crop the map to where the activities are (within the bbox)")
    p.add_argument("--out", default="poster.png")
    p.add_argument("--pdf", default=None)
    p.add_argument("--svg", default=None, help="vector export for the editorial layout")
    p.add_argument("--preview", default=None, help="lightweight editorial JPEG preview (1600 px high)")
    a = p.parse_args(argv)
    styles = {"classic": render.STYLES, "sport": render_sport.STYLES, "editorial": render_editorial.STYLES}
    if a.style and a.style not in styles[a.layout]:
        p.error(f"style {a.style!r} is not available for {a.layout}; choose {', '.join(styles[a.layout])}")
    if a.svg and a.layout != "editorial":
        p.error("--svg is available for the editorial layout")
    if a.headline == "name" and a.layout != "editorial":
        p.error("--headline name is available for the editorial layout")
    if a.headline == "name" and not a.name.strip():
        p.error("--headline name requires --name")
    if a.preview and (a.layout != "editorial" or os.path.splitext(a.preview)[1].lower() not in (".jpg", ".jpeg")):
        p.error("--preview requires the editorial layout and a .jpg or .jpeg filename")
    t0 = time.time()

    acts, rep = ingest.read_export(a.export)
    print(f"[ingest] files {rep.files_seen} · used {rep.used} · no GPS {rep.no_gps} · unreadable {rep.unreadable} · "
          f"unsupported {rep.unsupported}")
    sel = [x for x in acts if x.type.lower() == a.type.lower() and (a.year is None or (x.start and x.start.year == a.year))]
    if not sel:
        print("no activities match the type/year filters")
        return 2
    if a.bbox:
        s, w, n, e = map(float, a.bbox.split(","))
    elif a.place:
        s, w, n, e = area.geocode_bbox(a.place)
        print(f"[area] {a.place}")
    else:
        s, w, n, e = area.auto_bbox(sel)
        print("[area] found where you run most")
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
    user_zones = []
    for z in a.zone:
        la, lo, r = map(float, z.split(","))
        zx, zy = tr.transform(lo, la)
        user_zones.append((zx, zy, r))
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
    if a.layout == "editorial":
        aspect = render_editorial.map_aspect(a.size)
    else:
        aspect = 1 / 1.22 if a.layout == "classic" else (render.SIZES[a.size][0] - 28 * render.SIZES[a.size][0] / 297) / (268 * render.SIZES[a.size][0] / 297)
    if a.fit and pieces:
        pts = np.concatenate([np.asarray(pc.coords) for _, pc in pieces])
        lo, hi = np.percentile(pts, 0.7, axis=0), np.percentile(pts, 99.3, axis=0)
        pad = 0.07 * max(hi - lo)
        inv = Transformer.from_crs(crs, 4326, always_xy=True)
        (w2, e2), (s2, n2) = inv.transform([lo[0] - pad, hi[0] + pad], [lo[1] - pad, hi[1] + pad])
        extent = render.fit_extent((max(s, s2), max(w, w2), min(n, n2), min(e, e2)), crs, aspect)
    else:
        extent = render.fit_extent((s, w, n, e), crs, aspect)
    exp_ll = Transformer.from_crs(crs, 4326, always_xy=True).transform([extent[0], extent[1]], [extent[2], extent[3]])
    big = (min(exp_ll[1]) - 0.01, min(exp_ll[0]) - 0.01, max(exp_ll[1]) + 0.01, max(exp_ll[0]) + 0.01)
    if a.cache:
        folder = os.path.join(ROOT, "cache", a.cache)
    else:
        folder = area.fetch_basemap(big)
    basemap = render.load_basemap(folder, big, crs)
    if not a.title:
        a.title = (a.place.split(",")[0] if a.place else area.city_name((s + n) / 2, (w + e) / 2, lang=a.lang))
        if a.title:
            print(f"[area] headline: {a.title}")
    years = sorted({x.start.year for x in sel if x.start})
    stats_ = {"n": len(sel), "km": total_m / 1000, "year_label": str(a.year) if a.year else (f"{years[0]}–{years[-1]}" if years else "")}
    if a.layout in ("sport", "editorial"):
        st_ = stats.compute(sel, [t for _, t in tracks], basemap[0],
                            private_pieces_xy=[np.asarray(pc.coords) for _, pc in pieces],
                            piece_groups=[i for i, _ in pieces], sensitive_xy=prep.centres)
        print(f"[stats] {st_['n']} runs · {st_['km']:.0f} km · {st_['hours']:.0f} h · best 5k {stats.fmt_time(st_['best5k_s'])} · "
              f"best 10k {stats.fmt_time(st_['best10k_s'])} · {st_.get('streets')} streets · fav: {st_.get('fav_street')}")
        if a.layout == "editorial":
            info = render_editorial.render(pieces_all, basemap, extent, st_, a.out, style=a.style or "huella",
                                           size=a.size, dpi=a.dpi, name=a.name, city=a.title,
                                           year=stats_["year_label"], lang=a.lang, pdf_path=a.pdf, svg_path=a.svg,
                                           headline=a.headline, preview_path=a.preview)
        else:
            info = render_sport.render(pieces_all, basemap, extent, st_, a.out, style=a.style or "volt", size=a.size, dpi=a.dpi,
                                       name=a.name, city=a.title, year=stats_["year_label"], lang=a.lang, pdf_path=a.pdf)
        info.setdefault("cap", 0)
    else:
        info = render.render(pieces_all, basemap, extent, stats_, a.out, style=a.style or "nightrun", size=a.size, dpi=a.dpi,
                             title=a.title, lang=a.lang, center_ll=((s + n) / 2, (w + e) / 2), pdf_path=a.pdf)
    print(f"[render] {a.out} {info['W']}x{info['H']} map px · heat cap {info['cap']:.0f} · max overlap {info['max']} · "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
