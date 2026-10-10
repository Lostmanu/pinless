"""Poster rendering: OSM vector base map + accumulated activity heat, print sizes."""
from __future__ import annotations

import math
import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

MM = 1 / 25.4
SIZES = {"A4": (210, 297), "A3": (297, 420), "50x70": (500, 700)}

STYLES = {
    "nightrun": {
        "bg": "#0c1016", "water": "#11263a", "water_line": "#173550", "green": "#121d17",
        "street": {"major": ("#323b49", 0.55), "minor": ("#222a35", 0.30), "path": ("#181e27", 0.18)},
        "heat": ["#7a2a12", "#ff5a1f", "#ff9e3d", "#ffd27a", "#fff6df"], "glow": "#ff6a2b",
        "text": "#ece8e1", "sub": "#8a94a3", "rule": "#2a3340",
    },
    "ink": {
        "bg": "#f4f0e8", "water": "#cddbe3", "water_line": "#b4c8d4", "green": "#e2e6d8",
        "street": {"major": ("#b9b2a6", 0.55), "minor": ("#d3cdc2", 0.32), "path": ("#e0dbd1", 0.22)},
        "heat": ["#5b6b8c", "#2b3a67", "#1b2447", "#0f1530", "#06091a"], "glow": "#2b3a67",
        "text": "#14161c", "sub": "#6c6a66", "rule": "#cfc8bc",
    },
}
MAJOR = {"primary", "secondary", "trunk", "primary_link", "secondary_link", "trunk_link"}
MINOR = {"tertiary", "residential", "unclassified", "living_street", "tertiary_link", "service", "road"}


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def load_basemap(city_dir: str, bbox_ll, crs):
    """(edges, green, water) for a box, projected to crs. The water includes the sea, built from the coastline."""
    s, w, n, e = bbox_ll
    area = box(w, s, e, n)
    G = ox.load_graphml(os.path.join(city_dir, "walk.graphml"))
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges = edges[[c for c in ("highway", "name", "geometry") if c in edges.columns]]
    edges = edges[edges.intersects(area)].to_crs(crs)
    green = gpd.read_file(os.path.join(city_dir, "green.gpkg"))
    green = green[green.intersects(area)].to_crs(crs)
    water = gpd.read_file(os.path.join(city_dir, "water.gpkg"))
    water = water[water.intersects(area)].to_crs(crs)
    coast_path = os.path.join(city_dir, "coast.gpkg")
    if os.path.exists(coast_path):  # caches made before the coast layer simply have no sea
        sea = sea_polygons(gpd.read_file(coast_path), bbox_ll, crs)
        if sea:
            water = gpd.GeoDataFrame(pd.concat([water, gpd.GeoDataFrame({"natural": ["sea"] * len(sea)}, geometry=sea,
                                                                         crs=crs)], ignore_index=True), crs=crs)
    return edges, green, water


def sea_polygons(coast, bbox_ll, crs, eps=2.0):
    """The sea inside the box, from OSM coastline ways (EPSG:4326 lines), as polygons in crs.

    OSM has no sea polygons: every coastline way is drawn with the land on its left and the water on its right. The
    coastlines split the box into faces; a point just right of each coastline segment votes "sea" for its face and a
    point just left votes "land", so a face is sea when it gets more sea votes."""
    lines = [p for g in coast.geometry if g is not None and not g.is_empty for p in getattr(g, "geoms", [g])
             if p.geom_type == "LineString"]
    if not lines:
        return []
    s, w, n, e = bbox_ll
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    t = np.linspace(0, 1, 60)[:-1]
    ring = np.concatenate([np.column_stack([w + (e - w) * t, np.full_like(t, s)]),
                           np.column_stack([np.full_like(t, e), s + (n - s) * t]),
                           np.column_stack([e - (e - w) * t, np.full_like(t, n)]),
                           np.column_stack([np.full_like(t, w), n - (n - s) * t])])
    frame = Polygon(np.column_stack(tr.transform(ring[:, 0], ring[:, 1]))).buffer(0)
    projected = [LineString(np.column_stack(tr.transform(*np.asarray(p.coords)[:, :2].T))) for p in lines]
    # node the WHOLE coastlines with the frame (a pre-clipped line can stop a hair short of the edge and split nothing);
    # what lies outside the frame only dangles and makes no face
    faces = [f for f in polygonize(unary_union([frame.boundary] + projected)) if f.representative_point().within(frame)]
    if not faces:
        return []
    votes = np.zeros(len(faces))
    tree = STRtree(faces)
    for line in projected:
        a = np.asarray(shapely.segmentize(line, 100.0).coords)  # a vote every 100 m at least
        d = np.diff(a, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        ok = length > 0
        if not ok.any():
            continue
        mid = ((a[:-1] + a[1:]) / 2)[ok]
        right = np.column_stack([d[ok, 1], -d[ok, 0]]) / length[ok, None]  # right-hand normal of the way
        for side, vote in ((1.0, 1.0), (-1.0, -1.0)):
            _, face = tree.query(shapely.points(mid + side * eps * right), predicate="within")
            np.add.at(votes, face, vote)
    return [f for f, v in zip(faces, votes) if v > 0]


def _hw(h):
    return h[0] if isinstance(h, list) else h


def _lines(geoms):
    segs = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "LineString":
                segs.append(np.asarray(part.coords))
            elif part.geom_type == "Polygon":
                segs.append(np.asarray(part.exterior.coords))
    return segs


def heat_raster(pieces, extent, W, H, width_px):
    """Count, per pixel, how many activities pass (each activity counts once per pixel)."""
    x0, x1, y0, y1 = extent
    acc = np.zeros((H, W), dtype=np.uint16)
    by_act = {}
    for act_id, line in pieces:
        by_act.setdefault(act_id, []).append(line)
    for lines in by_act.values():
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        for ln in lines:
            c = np.asarray(ln.coords)
            px = (c[:, 0] - x0) / (x1 - x0) * W
            py = (y1 - c[:, 1]) / (y1 - y0) * H
            d.line(list(zip(px, py)), fill=1, width=width_px, joint="curve")
        acc += np.asarray(img, dtype=np.uint16)
    return acc


def render(pieces, basemap, extent, stats, out_path, style="nightrun", size="A3", dpi=300, title="", lang="en",
           center_ll=(0, 0), pdf_path=None):
    st = STYLES[style]
    wmm, hmm = SIZES[size]
    k = wmm / 297.0  # scale typography and strokes with the paper
    fig = plt.figure(figsize=(wmm * MM, hmm * MM), dpi=dpi, facecolor=st["bg"])
    margin = 16 * k
    map_w = wmm - 2 * margin
    map_h = map_w * 1.22
    ax = fig.add_axes([margin / wmm, 1 - (margin + map_h) / hmm, map_w / wmm, map_h / hmm])
    ax.set_facecolor(st["bg"])
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    edges, green, water = basemap
    # base layers
    gp = green[green.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if len(gp):
        gp.plot(ax=ax, color=st["green"], linewidth=0, zorder=1)
    wp = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if len(wp):
        wp.plot(ax=ax, color=st["water"], linewidth=0, zorder=2)
    wl = water[water.geometry.type.isin(["LineString", "MultiLineString"])]
    if len(wl):
        ax.add_collection(LineCollection(_lines(wl.geometry), colors=st["water_line"], linewidths=1.6 * k, zorder=2))
    hw = edges["highway"].map(_hw)
    for cls, mask in (("path", ~hw.isin(MAJOR | MINOR)), ("minor", hw.isin(MINOR)), ("major", hw.isin(MAJOR))):
        col, lw = st["street"][cls]
        ax.add_collection(LineCollection(_lines(edges[mask].geometry), colors=col, linewidths=lw * k, zorder=3,
                                         capstyle="round"))
    # heat
    W = int(round(map_w * MM * dpi))
    H = int(round(map_h * MM * dpi))
    width_px = max(1, int(round(0.22 * dpi / 25.4 * k)))  # ~0.22 mm strokes
    acc = heat_raster(pieces, extent, W, H, width_px)
    nz = acc[acc > 0]
    cap = max(3.0, float(np.percentile(nz, 99.3))) if len(nz) else 1.0
    v = np.clip(np.log1p(acc) / math.log1p(cap), 0, 1)
    cmap = LinearSegmentedColormap.from_list("heat", st["heat"])
    rgba = cmap(v)
    rgba[..., 3] = np.where(acc > 0, np.clip(0.35 + 0.65 * v, 0, 1), 0)
    # soft glow under the strokes
    glow = Image.fromarray((v * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(1, int(dpi / 60))))
    g = np.asarray(glow, dtype=float) / 255.0
    grgba = np.zeros((H, W, 4))
    grgba[..., :3] = to_rgb(st["glow"])
    grgba[..., 3] = np.clip(g * 0.55, 0, 0.5)
    ax.imshow(grgba, extent=(x0, x1, y0, y1), zorder=4, interpolation="bilinear")
    ax.imshow(rgba, extent=(x0, x1, y0, y1), zorder=5, interpolation="nearest")
    # typography
    font = "Bahnschrift"
    tx = 0.5
    base = (margin + map_h) / hmm
    fig.text(tx, 1 - base - 0.045, " ".join(title.upper()), ha="center", va="center", color=st["text"],
             fontsize=46 * k, fontfamily=font, fontweight="bold")
    words = {"en": ("RUNS", "KM"), "es": ("CARRERAS", "KM")}[lang]
    km = f"{stats['km']:,.0f}".replace(",", "." if lang == "es" else ",")
    sub = f"{stats['year_label']}   ·   {stats['n']} {words[0]}   ·   {km} {words[1]}"
    fig.text(tx, 1 - base - 0.081, sub, ha="center", va="center", color=st["sub"], fontsize=13 * k, fontfamily=font)
    lat, lon = center_ll
    coords = f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}   {abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"
    fig.text(tx, 1 - base - 0.102, coords, ha="center", va="center", color=st["sub"], fontsize=8.5 * k, fontfamily=font)
    fig.add_artist(plt.Line2D([0.38, 0.62], [1 - base - 0.065, 1 - base - 0.065], color=st["rule"], linewidth=0.8 * k))
    fig.text(1 - margin / wmm, 0.012, "Map data © OpenStreetMap contributors (ODbL) · openstreetmap.org/copyright",
             ha="right", va="bottom", color=st["sub"], fontsize=5.2 * k, fontfamily=font, alpha=0.8)
    fig.savefig(out_path, dpi=dpi, facecolor=st["bg"])
    if pdf_path:
        fig.savefig(pdf_path, dpi=dpi, facecolor=st["bg"])
    plt.close(fig)
    return {"W": W, "H": H, "cap": cap, "max": int(acc.max()) if acc.size else 0}


def fit_extent(bbox_ll, crs, aspect):
    """Project the lat/lon box and grow it to the map aspect (width/height)."""
    s, w, n, e = bbox_ll
    t = Transformer.from_crs(4326, crs, always_xy=True)
    xs, ys = t.transform([w, e, w, e], [s, s, n, n])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cw, ch = x1 - x0, y1 - y0
    if cw / ch > aspect:
        nh = cw / aspect
        y0, y1 = (y0 + y1) / 2 - nh / 2, (y0 + y1) / 2 + nh / 2
    else:
        nw = ch * aspect
        x0, x1 = (x0 + x1) / 2 - nw / 2, (x0 + x1) / 2 + nw / 2
    return x0, x1, y0, y1
