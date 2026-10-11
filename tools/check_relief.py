"""Check the terrain relief against the real Copernicus DEM, and draw what a person must look at before a merge.

Network; not run by the tests. Usage: python tools/check_relief.py --out out/relief [--city villalba] [--km 12]

1. Decodes one real GLO-30 block with the stdlib reader of pinless/relief.py and with Pillow (libtiff), and asserts
   that they are equal.
2. Prints the share of sloped land the gate measures in 13 places, and checks that the first eight land on the
   expected side of the 2% threshold.
3. Draws, for one hilly city: a sheet of the map in the twelve styles; the poster at A4, A3 and 50x70 with and
   without relief (printing the PNG, PDF and SVG size deltas); a race print (the casing over the relief). Then a
   Lisbon poster (the relief clipped off the sea and the Tagus), centred on the river front.
Uses the normal caches (PINLESS_CACHE is respected): the OSM data and the elevation of each place are fetched once.
When the Overpass servers are busy, a smaller --km lets an area already cached be drawn.
The routes are synthetic: shortest paths on the OSM walk network, nobody's real runs.
"""
import argparse
import datetime as dt
import io
import math
import os
import random
import struct
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from pinless import area, relief, render, render_editorial, render_race  # noqa: E402
from pinless.ingest import Activity  # noqa: E402

PLACES = [  # (name, lat, lon, expected side of the gate or None); public landmarks
    ("Madrid centre", 40.4168, -3.7038, "flat"), ("Casa de Campo", 40.42, -3.75, "flat"),
    ("El Pardo", 40.52, -3.77, "flat"), ("Alcala", 40.482, -3.364, "relief"),
    ("Collado Villalba", 40.635, -4.005, "relief"), ("Penalara", 40.85, -3.955, "relief"),
    ("Mexico City centre", 19.4326, -99.1332, "flat"), ("Ajusco", 19.2147, -99.2581, "relief"),
    ("Lisbon", 38.7223, -9.1393, None), ("Bilbao", 43.2630, -2.9350, None), ("Barcelona", 41.3874, 2.1686, None),
    ("Valencia", 39.4699, -0.3763, None), ("Zurich", 47.3769, 8.5417, None),
]
CITIES = {"villalba": ("Collado Villalba", 40.635, -4.005), "penalara": ("Penalara", 40.85, -3.955),
          "zurich": ("Zurich", 47.367, 8.5445)}
SEA = ("Lisboa", 38.7005, -9.171)  # the river front: the Tagus is sea in OSM (coastline)
STATS = {"km": 1234.0, "n": 187, "hours": 112.0, "best5k_s": 1260, "best10k_s": 2650, "longest_km": 21.1,
         "streets": 412}


def map_w(size):
    return render.SIZES[size][0] * (1 - 36 / 297)


def extent_at(lat, lon, km=12.0, aspect=None):
    crs = render.utm_crs(lon, lat)
    aspect = aspect or render_editorial.map_aspect("A3")
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    w = km * 1000
    return (x - w / 2, x + w / 2, y - w / aspect / 2, y + w / aspect / 2), crs


def osm_for(extent, crs):
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    lons, lats = inv.transform([extent[0], extent[1], extent[0], extent[1]], [extent[2], extent[2], extent[3], extent[3]])
    big = (min(lats) - 0.002, min(lons) - 0.002, max(lats) + 0.002, max(lons) + 0.002)  # as the atlas and race
    folder = area.fetch_basemap(big)
    return folder, render.load_basemap(folder, big, crs)


# 1 -------------------------------------------------------------------------------------------------------------------

def _tiff_around(tile, size):
    """A one-tile classic TIFF around a compressed Copernicus block, for Pillow to decode."""
    entries = [(256, 4, size), (257, 4, size), (258, 3, 32), (259, 3, 8), (262, 3, 1), (277, 3, 1), (284, 3, 1),
               (317, 3, 3), (322, 4, size), (323, 4, size), (324, 4, None), (325, 4, len(tile)), (339, 3, 3)]
    data_at = 8 + 2 + 12 * len(entries) + 4
    out = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(entries)))
    for tag, typ, value in entries:
        value = data_at if value is None else value
        out += struct.pack("<HHI", tag, typ, 1) + (struct.pack("<I", value) if typ == 4 else struct.pack("<HH", value, 0))
    out += struct.pack("<I", 0) + tile
    return bytes(out)


def check_decoder(root):
    cell, L, br, bc = (40, -4), 0, 1, 1
    hdr = relief._header("glo30", cell, root)
    lv = hdr.levels[L]
    i = br * math.ceil(lv.w / lv.tw) + bc
    tile = relief._fetch(relief.url("glo30", cell), lv.offsets[i], lv.offsets[i] + lv.counts[i] - 1)
    ours = relief._unpredict(zlib.decompress(tile), lv.th, lv.tw)
    with Image.open(io.BytesIO(_tiff_around(tile, lv.tw))) as im:
        pillow = np.asarray(im, dtype=np.float32)
    assert pillow.shape == ours.shape and np.array_equal(pillow, ours), "the stdlib reader and Pillow disagree"
    print(f"[decoder] one real GLO-30 block ({lv.tw}x{lv.th}, {len(tile):,} bytes): stdlib reader == Pillow, "
          f"{ours.min():.1f} to {ours.max():.1f} m")


# 2 -------------------------------------------------------------------------------------------------------------------

def gate_shares(root):
    wrong = []
    for name, lat, lon, expected in PLACES:
        extent, crs = extent_at(lat, lon)
        job = relief._Map(0, extent, crs, map_w("A3"))
        dataset, headers = relief._dataset(sorted(job.cells), root)
        present = [h for h in headers.values() if h is not None]
        share = relief._gate(job, dataset, headers, len(present[0].levels) - 1, root) if present else None
        side = "water" if share is None else "relief" if share >= relief.GATE_SHARE else "flat"
        note = "" if expected is None else ("  ok" if side == expected else f"  EXPECTED {expected}")
        if expected is not None and side != expected:
            wrong.append(name)
        print(f"[gate] {name:20s} {'-' if share is None else f'{100 * share:5.1f}%':>7s}  {side}{note}")
    return wrong


# 3 -------------------------------------------------------------------------------------------------------------------

def synthetic_runs(folder, crs, extent, n=60, seed=7):
    """Shortest paths between random street corners of the map, some repeated: drawn lines to judge the relief
    against, nobody's runs."""
    G = ox.load_graphml(os.path.join(folder, "walk.graphml"))
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    ids = list(G.nodes)
    xs, ys = tr.transform([G.nodes[v]["x"] for v in ids], [G.nodes[v]["y"] for v in ids])
    x0, x1, y0, y1 = extent
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    near = [v for v, x, y in zip(ids, xs, ys) if abs(x - cx) < (x1 - x0) * 0.35 and abs(y - cy) < (y1 - y0) * 0.35]
    rng = random.Random(seed)
    routes = []
    while len(routes) < n and near:
        a, b = rng.sample(near, 2)
        try:
            path = nx.shortest_path(G, a, b, weight="length")
        except nx.NetworkXNoPath:
            continue
        pts = []
        for u, v in zip(path[:-1], path[1:]):
            e = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0))
            if "geometry" in e:
                c = list(e["geometry"].coords)
                if math.hypot(c[0][0] - G.nodes[u]["x"], c[0][1] - G.nodes[u]["y"]) > 1e-9:
                    c = c[::-1]
            else:
                c = [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
            pts += c if not pts else c[1:]
        if len(pts) > 1:
            xy = np.column_stack(tr.transform(*np.asarray(pts).T))
            if LineString(xy).length > 1500:
                routes.append(xy)
    pieces = []
    for i, xy in enumerate(routes):
        for _ in range(1 + (i % 7 == 0) * 6):  # every seventh route run seven times
            pieces.append((len(pieces), LineString(xy)))
    return pieces, routes


def poster(path, pieces, basemap, extent, rel, style, size, dpi, city, pdf=False, svg=False):
    base = os.path.splitext(path)[0]
    render_editorial.render(pieces, basemap, extent, STATS, path, style=style, size=size, dpi=dpi, city=city,
                            year="2025", lang="en", pdf_path=base + ".pdf" if pdf else None,
                            svg_path=base + ".svg" if svg else None, relief=rel)


def style_sheet(out, pieces, basemap, extent, rel, city, dpi=110):
    tiles = []
    for style in render_editorial.STYLES:
        p = os.path.join(out, f"style_{style}.png")
        poster(p, pieces, basemap, extent, rel, style, "A3", dpi, city)
        with Image.open(p) as im:
            px = dpi / 25.4
            box = (round(18 * px), round(94 * px), round(279 * px), round(316 * px))  # the A3 map box
            tile = im.convert("RGB").crop(box)
        ImageDraw.Draw(tile).rectangle([0, 0, 150, 30], fill=(255, 255, 255))
        ImageDraw.Draw(tile).text((8, 8), style, fill=(0, 0, 0))
        tiles.append(tile)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (4 * w + 3 * 12, 3 * h + 2 * 12), (128, 128, 128))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * (w + 12), (i // 4) * (h + 12)))
    path = os.path.join(out, "sheet_12_styles.png")
    sheet.save(path)
    print(f"[sheet] {path}")


def size_deltas(out, pieces, basemap, extent, crs, city, style="huella"):
    for size in ("A4", "A3", "50x70"):
        (rel,) = relief.prepare_many([(extent, crs, map_w(size))], log=lambda *_: None)
        sizes = {}
        for tag, r in (("off", None), ("shade", rel)):
            p = os.path.join(out, f"poster_{size}_{tag}.png")
            t0 = time.time()
            poster(p, pieces, basemap, extent, r, style, size, 300, city, pdf=True, svg=True)
            sizes[tag] = {ext: os.path.getsize(os.path.splitext(p)[0] + ext) for ext in (".png", ".pdf", ".svg")}
            sizes[tag]["s"] = time.time() - t0
        line = " · ".join(f"{ext[1:].upper()} {sizes['off'][ext] / 1e6:.2f} -> {sizes['shade'][ext] / 1e6:.2f} MB "
                          f"({(sizes['shade'][ext] - sizes['off'][ext]) / 1e6:+.2f})" for ext in (".png", ".pdf", ".svg"))
        print(f"[size] {size}: relief {rel.shade.shape[1]}x{rel.shade.shape[0]} px · {line} · "
              f"{sizes['off']['s']:.0f}s -> {sizes['shade']['s']:.0f}s")


def hilliest(routes, rel, extent):
    """The route over the most relief (mean |shade| under it, longer ones preferred up to 3 km); the longest one
    when the city has no relief."""
    if rel is None:
        return max(routes, key=lambda r: LineString(r).length)
    H, W = rel.shade.shape
    x0, x1, y0, y1 = extent

    def score(xy):
        c = np.clip(((xy[:, 0] - x0) / (x1 - x0) * W).astype(int), 0, W - 1)
        r = np.clip(((y1 - xy[:, 1]) / (y1 - y0) * H).astype(int), 0, H - 1)
        return float(np.abs(rel.shade[r, c]).mean()) * min(1.0, LineString(xy).length / 3000)

    return max(routes, key=score)


def race_print(out, folder, basemap_of, crs, xy, city, style):
    """A race print of one synthetic course (the casing over the relief)."""
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    pad = 0.08 * max(x1 - x0, y1 - y0, 2000.0)
    aspect = render_race.map_aspect("A3")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0 + 2 * pad, 2000.0), max(y1 - y0 + 2 * pad, 2000.0 / aspect)
    w, h = (w, w / aspect) if w / h > aspect else (h * aspect, h)
    extent = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    d = render_race.along(xy)
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    lonlat = list(zip(*inv.transform(xy[:, 0], xy[:, 1])))
    act = Activity(id="synthetic", type="Run", start=dt.datetime(2025, 4, 6, 9, tzinfo=dt.timezone.utc),
                   lonlat=lonlat, source="synthetic", times=list(d * (0.29 + 0.02 * np.sin(d / 1300))),
                   ele=list(800 + 30 * np.sin(d / 1700)))
    (rel,) = relief.prepare_many([(extent, crs, map_w("A3"))], log=print)
    p = os.path.join(out, f"race_{style}.png")
    render_race.render(act, xy, [(0, LineString(xy[20:-20]))], basemap_of(extent), extent, p, style=style,
                       size="A3", dpi=200, title=city, city=city + ", Spain", date=act.start, color_by="pace",
                       relief=rel)
    print(f"[race] {p} · relief {'drawn' if rel is not None else 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "relief"))
    ap.add_argument("--city", default="villalba", choices=sorted(CITIES))
    ap.add_argument("--km", type=float, default=12.0, help="map width (km) of the city and Lisbon posters")
    ap.add_argument("--skip-sizes", action="store_true", help="skip the A4/A3/50x70 posters at 300 dpi")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = os.path.join(area.cache_root(), "dem")
    check_decoder(root)
    wrong = gate_shares(root)
    name, lat, lon = CITIES[a.city]
    extent, crs = extent_at(lat, lon, a.km)
    folder, basemap = osm_for(extent, crs)
    pieces, routes = synthetic_runs(folder, crs, extent)
    (rel,) = relief.prepare_many([(extent, crs, map_w("A3"))])
    print(f"[city] {name}: {'relief' if rel else 'no relief'}"
          f"{f' (level {rel.level}, {rel.dataset}, share {100 * rel.share:.1f}%)' if rel else ''}")
    style_sheet(a.out, pieces, basemap, extent, rel, name)
    if not a.skip_sizes:
        size_deltas(a.out, pieces, basemap, extent, crs, name)

    def basemap_of(ext):
        inv = Transformer.from_crs(crs, 4326, always_xy=True)
        lons, lats = inv.transform([ext[0], ext[1]], [ext[2], ext[3]])
        big = (min(lats) - 0.002, min(lons) - 0.002, max(lats) + 0.002, max(lons) + 0.002)
        return render.load_basemap(folder, big, crs)

    course = hilliest(routes, rel, extent)
    for style in ("huella", "midnight"):
        race_print(a.out, folder, basemap_of, crs, course, name, style)
    l_ext, l_crs = extent_at(SEA[1], SEA[2], a.km)
    l_folder, l_basemap = osm_for(l_ext, l_crs)
    l_pieces, _ = synthetic_runs(l_folder, l_crs, l_ext, n=40)
    (l_rel,) = relief.prepare_many([(l_ext, l_crs, map_w("A3"))])
    for style in ("huella", "carbon"):
        poster(os.path.join(a.out, f"lisbon_{style}.png"), l_pieces, l_basemap, l_ext, l_rel, style, "A3", 200,
               SEA[0], svg=style == "huella")
    print(f"[lisbon] relief {'drawn' if l_rel is not None else 'none'}; posters in {a.out}")
    if wrong:
        raise SystemExit(f"the gate put these places on the other side of the threshold: {', '.join(wrong)}")


if __name__ == "__main__":
    main()
