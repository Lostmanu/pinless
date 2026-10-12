"""The ground of the trail map in the colours of the land, from ESA WorldCover 10 m (v200, 2021).

On by default for the trail print (--ground nature): woods green, meadows a lighter green, fields wheat, bare land and
deserts ochre, snow blue white, under the terrain relief, as on the terrain maps of hiking apps. What is requested
depends only on the map extents and the paper widths: for every 3-degree tile around a map, a 32 KB header and the
FULL rectangle of internal blocks that covers the padded map box, never the blocks a track touches. Tracks, zones and
activities never reach this module.

- The classes are read with HTTP Range requests through the relief's reader (no redirects, 206 only, the
  Content-Range checked, one retry) and decoded by zlib + numpy (8-bit samples, DEFLATE, no predictor), cached under
  the Pinless cache in landcover/. A tile the dataset does not have (open sea) is water and is not asked for again.
- The classes are resampled onto the map's own projected grid at 150 ppi, the relief's raster (nearest neighbour,
  every node projected), then generalised at the scale of the data: a lone 10 m pixel goes and the staircase of
  the data's pixels rounds into the outline of a painted map (generalise).
Any network or decoding failure means the style's plain ground (and one log line), never another data source.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgb
from pyproj import Transformer

from . import area, relief

HOST = "esa-worldcover.s3.eu-central-1.amazonaws.com"
HEADER_BYTES = 32768  # the seven IFDs of a real tile and their arrays end at byte 27556 (tests pin it)
TILE_DEG = 3  # a tile covers 3 x 3 degrees from its south-west corner
WIDTH0 = 36000  # its pixels on each side at full resolution: 1/12000 degree, about 9 m north to south
BLOCK = 1024  # the internal blocks of the real files, at every level: all that S3 learns inside a tile
NOMINAL_LEVELS = 7  # 36000, 18000, 9000, 4500, 2250, 1125 and 562 pixels
MAX_TILES = 16
BUDGET_BYTES = 64 * 2 ** 20

TREE, SHRUB, GRASS, CROP, BUILT, BARE, SNOW, WATER, WETLAND, MANGROVE, MOSS, NO_DATA = (
    10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0)
CLASSES = (NO_DATA, TREE, SHRUB, GRASS, CROP, BUILT, BARE, SNOW, WATER, WETLAND, MANGROVE, MOSS)
NAMES = {TREE: "tree cover", SHRUB: "shrubland", GRASS: "grassland", CROP: "cropland", BUILT: "built-up land",
         BARE: "bare land", SNOW: "snow and ice", WATER: "water", WETLAND: "wetland", MANGROVE: "mangroves",
         MOSS: "moss and lichen", NO_DATA: "land without data"}
_VALID = np.zeros(256, dtype=bool)
_VALID[list(CLASSES)] = True

# The attribution the ESA WorldCover licence (CC BY 4.0) asks for, in the wording of the 2021 map (v200)
NOTICE = ("© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA "
          "WorldCover consortium")


def notice(lang):
    """The credit line for a print with the land cover. The attribution itself stays in English, as the licence
    gives it."""
    return ("Cobertura del suelo " if lang == "es" else "Land cover ") + NOTICE


class LandCoverError(relief.ReliefError):
    """The land cover could not be fetched or decoded as expected: the print is drawn with the style's ground."""


class _Unsupported(LandCoverError):
    """A map the land cover cannot cover as it is: the same on every run, so no retry is suggested."""


@dataclass(frozen=True, eq=False)
class LandCover:
    classes: np.ndarray  # uint8 (H, W), row 0 north: the WorldCover class of every pixel of the map raster
    level: int
    dry: float = 0.0  # how dry the land is (dryness), 0 to 1


@dataclass(frozen=True)
class Header:
    S: int  # south latitude of the tile (degrees)
    W: int  # west longitude of the tile (degrees)
    levels: tuple  # relief.Level per level


# --- tiles and URLs -------------------------------------------------------------------------------------------------

def tile_of(lat, lon):
    """The 3-degree tile (south, west) holding a point: N39W006 covers 39 to 42 N and 6 to 3 W."""
    return (TILE_DEG * math.floor(lat / TILE_DEG), TILE_DEG * math.floor(lon / TILE_DEG))


def tile_name(tile):
    S, W = tile
    return f"{'N' if S >= 0 else 'S'}{abs(S):02d}{'E' if W >= 0 else 'W'}{abs(W):03d}"


def url(tile):
    return f"https://{HOST}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name(tile)}_Map.tif"


def _fetch(url_, first, last):
    """The relief's range reader; S3 answers 403 or 404 for a tile the dataset does not have (open sea)."""
    return relief._fetch(url_, first, last, absent=(403, 404))


# --- TIFF header ----------------------------------------------------------------------------------------------------

def read_header(buf, tile):
    """The levels of a WorldCover tile from the first bytes of its file. Anything that is not exactly the layout the
    block reader decodes raises LandCoverError."""
    try:
        return _read_header(buf, tile)
    except LandCoverError:
        raise
    except (relief.ReliefError, struct.error, IndexError, KeyError, ArithmeticError, TypeError) as ex:
        raise LandCoverError("unexpected TIFF layout") from ex


def _read_header(buf, tile):
    if bytes(buf[:4]) != b"II*\x00":
        raise LandCoverError("not a classic little-endian TIFF")
    off, = struct.unpack_from("<I", buf, 4)
    ifds, seen = [], set()
    while off:
        if off in seen or len(ifds) >= 16:
            raise LandCoverError("unexpected TIFF layout")
        seen.add(off)
        tags, off = relief._ifd(buf, off)
        ifds.append(tags)
    if len(ifds) != NOMINAL_LEVELS:  # the levels, and so the blocks a print can need, are fixed in advance
        raise LandCoverError("unexpected number of levels")
    S, W = tile
    t0 = ifds[0]
    if (relief._ints(t0, 256)[0], relief._ints(t0, 257)[0]) != (WIDTH0, WIDTH0):
        raise LandCoverError("unexpected raster size")
    geo = t0[34735]
    keys = {geo[i]: (geo[i + 1], geo[i + 3]) for i in range(4, 4 + 4 * geo[3], 4)}
    if keys.get(1025) != (0, 1):
        raise LandCoverError("not PixelIsArea")
    tie, scale = t0[33922], t0[33550]
    if len(tie) < 6 or any(abs(a - b) > 1e-9 for a, b in zip(tie[:6], (0, 0, 0, W, S + TILE_DEG, 0))):
        raise LandCoverError("unexpected georeference")
    if len(scale) < 2 or any(abs(s - TILE_DEG / WIDTH0) > 1e-12 for s in scale[:2]):
        raise LandCoverError("unexpected georeference")
    levels = []
    for L, t in enumerate(ifds):
        if (t[259][0] not in (8, 32946) or t.get(317, (1,)) != (1,) or t[258] != (8,) or t.get(339, (1,)) != (1,)
                or t.get(277, (1,)) != (1,) or t.get(284, (1,)) != (1,)):
            raise LandCoverError("an encoding the reader does not decode")
        w, h, tw, th = (relief._ints(t, tag)[0] for tag in (256, 257, 322, 323))
        if (tw, th) != (BLOCK, BLOCK):  # smaller blocks would tell S3 more about where the map is
            raise LandCoverError("blocks of another size")
        if t.get(254, (0,))[0] != (0 if L == 0 else 1) or (w, h) != (WIDTH0 >> L, WIDTH0 >> L):
            raise LandCoverError("an overview that is not a halving")
        offsets, counts = relief._ints(t, 324), relief._ints(t, 325)
        tiles = math.ceil(w / tw) * math.ceil(h / th)
        if len(offsets) != tiles or len(counts) != tiles:
            raise LandCoverError("unexpected TIFF layout")
        levels.append(relief.Level(w, h, tw, th, tuple(offsets), tuple(counts)))
    return Header(S, W, tuple(levels))


# --- cache and blocks -----------------------------------------------------------------------------------------------

def _tile_dir(root, tile):
    return os.path.join(root, "v200", tile_name(tile))


def _headers(tiles, root):
    """({tile: Header, or None for a tile the dataset does not have}, [the tiles found missing in this run]).

    A missing tile is the open sea only if the server, in this same run, sent the header of another tile of the print:
    a server or a proxy that refuses everything must not be taken for the sea. When every other header came from the
    cache, the first of them is asked for again to prove that the server answers; if it is refused too, the run
    fails (LandCoverError). The caller remembers the missing tiles (remember_absent) once the whole print is read."""
    out, fresh, answered = {}, [], False
    for tile in sorted(tiles):
        folder = _tile_dir(root, tile)
        if os.path.exists(os.path.join(folder, "absent")):
            out[tile] = None
            continue
        path = os.path.join(folder, "header.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                buf = f.read()
            try:
                out[tile] = read_header(buf, tile)
                continue
            except LandCoverError:
                os.remove(path)  # damaged: fetched again
        buf = _fetch(url(tile), 0, HEADER_BYTES - 1)
        if buf is None:
            out[tile] = None
            fresh.append(tile)
            continue
        out[tile] = read_header(buf, tile)
        answered = True
        relief._write_atomic(path, buf)  # only a header that parses is kept
    known = [t for t in sorted(out) if out[t] is not None]
    if fresh and known and not answered:
        witness = _fetch(url(known[0]), 0, HEADER_BYTES - 1)
        if witness is None:
            raise LandCoverError("the server refuses a tile it has")
        read_header(witness, known[0])
    return out, (fresh if known else [])


def remember_absent(tiles, root):
    """Mark the tiles as missing from the dataset (the open sea), so they are never asked for again."""
    for tile in tiles:
        relief._write_atomic(os.path.join(_tile_dir(root, tile), "absent"), b"")


def _decode(data, lv, br, bc):
    try:
        raw = zlib.decompress(data)
    except zlib.error as ex:
        raise LandCoverError("a block does not inflate") from ex
    if len(raw) != lv.th * lv.tw:
        raise LandCoverError("unexpected block size")
    a = np.frombuffer(raw, np.uint8).reshape(lv.th, lv.tw)[: min(lv.th, lv.h - br * lv.th),
                                                             : min(lv.tw, lv.w - bc * lv.tw)]
    if not _VALID[a].all():
        raise LandCoverError("a class the dataset does not have")
    return a


def _block_path(root, tile, L, br, bc):
    return os.path.join(_tile_dir(root, tile), f"L{L}_{br}_{bc}.zz")


def _block(tile, hdr, L, br, bc, root):
    """Block (br, bc) of level L, without its edge padding. The compressed bytes are cached only once they decode; a
    cached file that does not decode is deleted and fetched again, once."""
    lv = hdr.levels[L]
    path = _block_path(root, tile, L, br, bc)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        try:
            return _decode(data, lv, br, bc)
        except (LandCoverError, ValueError):
            os.remove(path)
    i = br * math.ceil(lv.w / lv.tw) + bc
    first, n = lv.offsets[i], lv.counts[i]
    if n <= 0:
        raise LandCoverError("an empty block")
    data = _fetch(url(tile), first, first + n - 1)
    try:
        if data is None:
            raise LandCoverError("a block is missing")
        a = _decode(data, lv, br, bc)
    except LandCoverError:
        # the bytes at the header's offsets are not that block: the file may have been replaced on the server since
        # its header was cached. Forget the tile, so the next run starts it afresh.
        shutil.rmtree(_tile_dir(root, tile), ignore_errors=True)
        raise
    relief._write_atomic(path, data)
    return a


# --- geometry -------------------------------------------------------------------------------------------------------

def _pixel_m(L):
    """North-south size (m) of a pixel of level L."""
    return TILE_DEG / (WIDTH0 >> L) * relief.M_PER_DEG_LAT


def _tiles(box):
    """Every tile the box (lat_south, lat_north, lon_west, lon_east) touches, as a full rectangle."""
    lat_a, lat_b, lon_a, lon_b = box
    return [(la, lo) for la in range(tile_of(lat_a, 0)[0], tile_of(lat_b, 0)[0] + 1, TILE_DEG)
            for lo in range(tile_of(0, lon_a)[1], tile_of(0, lon_b)[1] + 1, TILE_DEG)]


def _window(tile, L, box):
    """(i0, i1, j0, j1): the rows and columns of a tile at level L that cover the box, plus one pixel. Pixel (i, j)
    covers longitudes W + j * 3 / n to W + (j + 1) * 3 / n, and latitudes down from S + 3 - i * 3 / n."""
    S, W = tile
    n = WIDTH0 >> L
    lat_a, lat_b, lon_a, lon_b = box
    j0 = max(0, math.floor((lon_a - W) * n / TILE_DEG) - 1)
    j1 = min(n - 1, math.floor((lon_b - W) * n / TILE_DEG) + 1)
    i0 = max(0, math.floor((S + TILE_DEG - lat_b) * n / TILE_DEG) - 1)
    i1 = min(n - 1, math.floor((S + TILE_DEG - lat_a) * n / TILE_DEG) + 1)
    return i0, i1, j0, j1


class _Map:
    """One map of the print: its extent, projection and paper width, and what follows from them alone."""

    def __init__(self, i, extent, crs, map_w_mm):
        self.i, self.crs, self.map_w = i, crs, float(map_w_mm)
        self.extent = tuple(float(v) for v in extent)
        x0, x1, y0, y1 = self.extent
        if not (x1 > x0 and y1 > y0 and self.map_w > 0):
            raise ValueError("an empty map")
        self.shape = relief.raster_shape(self.map_w, self.extent)
        lon_c, lat_c = Transformer.from_crs(crs, 4326, always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)
        self.lat = float(lat_c)  # the latitude of the map's centre: how dry its land may be (dryness)
        self.px = (x1 - x0) / self.shape[1]  # the map raster's pixel (metres on the map's own projection)
        # every tile the map could need, whatever the level: fixed before any request is made
        lat_a, lat_b, lon_a, lon_b = relief._lonlat_box(self.extent, crs, max(map(self.pad, range(NOMINAL_LEVELS))))
        self.near_180 = lon_b >= 180 or lon_a < -180 or lon_b - lon_a > 180
        self.tiles = set() if self.near_180 else set(_tiles((lat_a, lat_b, lon_a, lon_b)))

    @staticmethod
    def pad(L):
        """Padding of the box at level L: two of its pixels (the nodes are pixel centres inside the frame)."""
        return 2 * _pixel_m(L)

    def box(self, L):
        box = relief._lonlat_box(self.extent, self.crs, self.pad(L))
        tiles = _tiles(box)
        if not set(tiles) <= self.tiles:
            raise _Unsupported("a map needs a tile outside its print")
        return box, tiles

    def level(self):
        """The coarsest level whose pixel is at most the map raster's pixel, or 0."""
        return max([L for L in range(NOMINAL_LEVELS) if _pixel_m(L) <= self.px] or [0])


def _level_blocks(job, headers, L):
    """[(tile, L, br, bc)]: the full rectangle of blocks of level L covering the padded map, tile by tile."""
    box, tiles = job.box(L)
    out = []
    for tile in sorted(tiles, key=lambda t: (-t[0], t[1])):
        hdr = headers.get(tile)
        if hdr is None:  # the open sea: no requests
            continue
        i0, i1, j0, j1 = _window(tile, L, box)
        out += [(tile, L, br, bc) for br in range(i0 // BLOCK, i1 // BLOCK + 1)
                for bc in range(j0 // BLOCK, j1 // BLOCK + 1)]
    return out


def _block_bytes(headers, tile, L, br, bc):
    lv = headers[tile].levels[L]
    return lv.counts[br * math.ceil(lv.w / lv.tw) + bc]


def _uncached(blocks, root):
    return [b for b in blocks if not os.path.exists(_block_path(root, *b))]


def _plan(job, headers, root):
    """(level, its blocks): the level of the map's pixel, one level coarser while the blocks still to download
    exceed BUDGET_BYTES; the coarsest level is always within budget."""
    L = job.level()
    while True:
        blocks = _level_blocks(job, headers, L)
        todo = _uncached(blocks, root)
        if L == NOMINAL_LEVELS - 1 or sum(_block_bytes(headers, *b) for b in todo) <= BUDGET_BYTES:
            return L, blocks
        L += 1


def _download(headers, blocks, root):
    """Every block of the set, in sorted order (the set's own order, never the order of the maps of a print)."""
    for tile, L, br, bc in sorted(set(blocks)):
        _block(tile, headers[tile], L, br, bc, root)


def _read_window(tile, hdr, L, win, root):
    i0, i1, j0, j1 = win
    out = np.empty((i1 - i0 + 1, j1 - j0 + 1), np.uint8)
    for br in range(i0 // BLOCK, i1 // BLOCK + 1):
        for bc in range(j0 // BLOCK, j1 // BLOCK + 1):
            a = _block(tile, hdr, L, br, bc, root)
            r0, c0 = br * BLOCK, bc * BLOCK
            ra, rb = max(r0, i0), min(r0 + a.shape[0] - 1, i1)
            ca, cb = max(c0, j0), min(c0 + a.shape[1] - 1, j1)
            out[ra - i0:rb - i0 + 1, ca - j0:cb - j0 + 1] = a[ra - r0:rb - r0 + 1, ca - c0:cb - c0 + 1]
    return out


def _nodes(job):
    """(lon, lat) of the centre of every pixel of the map raster, row 0 north: every node is projected."""
    x0, x1, y0, y1 = job.extent
    H, W = job.shape
    X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) * (x1 - x0) / W, y1 - (np.arange(H) + 0.5) * (y1 - y0) / H)
    lon, lat = Transformer.from_crs(job.crs, 4326, always_xy=True).transform(X.ravel(), Y.ravel())
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def _classes(job, headers, L, root):
    """The class of every pixel of the map raster: the WorldCover pixel under its centre (nearest neighbour); a tile
    the dataset does not have is water."""
    box, tiles = job.box(L)
    lon, lat = _nodes(job)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise _Unsupported("the map has no position")
    n = WIDTH0 >> L
    out = np.full(lon.shape, WATER, np.uint8)
    S_all = np.floor(lat / TILE_DEG).astype(np.int64) * TILE_DEG
    W_all = np.floor(lon / TILE_DEG).astype(np.int64) * TILE_DEG
    seen = 0
    for tile in tiles:
        sel = (S_all == tile[0]) & (W_all == tile[1])
        if not sel.any():
            continue
        seen += int(sel.sum())
        hdr = headers.get(tile)
        if hdr is None:
            continue
        i0, i1, j0, j1 = win = _window(tile, L, box)
        col = np.clip(np.floor((lon[sel] - tile[1]) * n / TILE_DEG).astype(np.intp), 0, n - 1) - j0
        row = np.clip(np.floor((tile[0] + TILE_DEG - lat[sel]) * n / TILE_DEG).astype(np.intp), 0, n - 1) - i0
        if row.min() < 0 or col.min() < 0 or row.max() > i1 - i0 or col.max() > j1 - j0:
            raise _Unsupported("a pixel of the map fell outside the land cover read")
        out[sel] = _read_window(tile, hdr, L, win, root)[row, col]
    if seen != lon.size:
        raise _Unsupported("a pixel of the map fell outside the tiles read")
    return out.reshape(job.shape)


def generalise(classes, sigma):
    """The classes generalised as a cartographer would: every class's mask (1 where it lies, 0 elsewhere) is blurred
    by a Gaussian of sigma raster pixels (edges repeated), and every pixel takes the class that weighs most around it,
    keeping its own on a tie with it (among other classes, a tie goes to the smaller code). A straight edge stays
    where it is, a patch narrower than about 1.4 sigma goes, corners and the staircase of the data's pixels round."""
    c = np.asarray(classes, dtype=np.uint8)
    present = np.unique(c)
    if len(present) < 2 or sigma <= 0:
        return c.copy()
    best, best_w = c.copy(), np.full(c.shape, -1.0, np.float32)
    own_w = np.zeros(c.shape, np.float32)
    for k in present:  # ascending
        mask = c == k
        w = _gauss(mask.astype(np.float32), sigma)
        better = w > best_w
        best[better], best_w[better] = k, w[better]
        own_w[mask] = w[mask]
    return np.where(own_w >= best_w, c, best).astype(np.uint8)


def general_sigma(level, raster_px_m, raster_px_mm, dry=0.0):
    """The sigma (raster pixels) of the generalisation of a map: GENERAL_DATA of the data's pixel (a lone 10 m pixel
    goes, a patch of three stays with rounded corners), never under GENERAL_MIN raster pixels, and in a dry land up to
    GENERAL_DRY_MM on paper, so the scattered scrub of a steppe does not print as a camouflage; never over
    GENERAL_MAX_MM on paper (a map of a kilometre or two, where a data pixel is millimetres wide)."""
    data = GENERAL_DATA * _pixel_m(level) / raster_px_m
    return min(max(GENERAL_MIN, data, float(dry) * GENERAL_DRY_MM / raster_px_mm), GENERAL_MAX_MM / raster_px_mm)


# --- colours --------------------------------------------------------------------------------------------------------

# The nature palette for light paper: the colours of the land, as a painted terrain map shows it. Bare land in a
# green land is rock and scree, a light stone.
NATURE = {TREE: "#7fa65a", SHRUB: "#a9b77a", GRASS: "#c8d89a", CROP: "#e6dcae", BUILT: "#d9d2c8", BARE: "#d4ccbb",
          SNOW: "#f4f8fb", WETLAND: "#9cc7b4", MANGROVE: "#5f9a72", MOSS: "#b7bfa6"}
# ... and in a dry land (dryness): grass and scrub in the straw and olive of a steppe, bare land the ochre of a desert
DRY = {GRASS: "#ddd09a", SHRUB: "#c2b47f", BARE: "#e2b47c"}
# WorldCover alone cannot tell a dry land from a mountain: the badlands of Tabernas (27 % bare, 49 % grassland) and the
# Picos de Europa (32 % bare, 51 % grassland) hold the same classes. Where the map lies can: a land is dry as far as
# it lies south of DRY_LAT (north of it in the south) and bare land covers DRY_BARE of the map; DESERT_BARE of bare
# land is a desert at any latitude; snow, ice, moss and lichen on COLD of the map make it a cold land, never a dry one.
DRY_LAT = (37.0, 42.0)  # degrees from the equator: dry at the first, not at all from the second
DRY_BARE = (0.10, 0.25)
DESERT_BARE = (0.50, 0.75)
COLD = 0.03
# Water on light paper: the style's water this share of the way to a lake blue, and its waterways to a stream blue
# (a style's pale water reads as snow on a green ground); dark paper keeps the style's own water
WATER_BLUE, STREAM_BLUE, WATER_MIX = "#a9cfe0", "#6ea6c4", 0.65
DARK_MIX = 0.35  # on dark paper each colour lies this share of the way from the paper to its light-paper colour
SMOOTH_PX = 1.0  # the Gaussian over the coloured raster (pixels at 150 ppi, 0.17 mm): class edges read painted
# The generalisation of the classes (general_sigma): half a data pixel, at least one raster pixel, in a dry land up to
# GENERAL_DRY_MM on paper, and never more than GENERAL_MAX_MM on paper
GENERAL_DATA, GENERAL_MIN, GENERAL_DRY_MM, GENERAL_MAX_MM = 0.5, 1.0, 0.0, 1.0


def _lstar(rgb):
    c = np.asarray(rgb, dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = float(lin @ [0.2126, 0.7152, 0.0722])
    return 116 * np.cbrt(y) - 16 if y > (6 / 29) ** 3 else y * (29 / 3) ** 3


def is_dark(S):
    return _lstar(to_rgb(S["bg"])) < 50


def _ramp(v, a, b):
    return min(1.0, max(0.0, (v - a) / (b - a)))


def dryness(shares, lat):
    """How dry the land of a map is, from 0 (a green land) to 1 (a dry one): see DRY_LAT. shares: {class: share of the
    map}; lat: the latitude of the map's centre."""
    bare = shares.get(BARE, 0.0)
    dry = max((1 - _ramp(abs(lat), *DRY_LAT)) * _ramp(bare, *DRY_BARE), _ramp(bare, *DESERT_BARE))
    return dry * (1 - _ramp(shares.get(SNOW, 0.0) + shares.get(MOSS, 0.0), 0.0, COLD))


def palette(S, dry=0.0):
    """{class: (r, g, b)} of the ground in a style: NATURE on light paper, its grass, scrub and bare land dry of the
    way to DRY; on dark paper the same colours pulled towards the paper (DARK_MIX of the way from it), darker and
    quieter, so a dark poster stays dark. Water as water() says, land without data the paper."""
    bg = np.asarray(to_rgb(S["bg"]))
    mix = DARK_MIX if is_dark(S) else 1.0
    land = {k: np.asarray(to_rgb(c)) for k, c in NATURE.items()}
    for k, c in DRY.items():
        land[k] = land[k] + (np.asarray(to_rgb(c)) - land[k]) * float(dry)
    out = {k: tuple(float(v) for v in bg + (c - bg) * mix) for k, c in land.items()}
    out[WATER] = water(S)
    out[NO_DATA] = tuple(float(v) for v in bg)
    return out


def water(S, line=False):
    """(r, g, b) of the water over the ground (line: of the waterways), as WATER_MIX says."""
    own = np.asarray(to_rgb(S["water_line" if line else "water"]))
    if is_dark(S):
        return tuple(float(v) for v in own)
    blue = np.asarray(to_rgb(STREAM_BLUE if line else WATER_BLUE))
    return tuple(float(v) for v in own + (blue - own) * WATER_MIX)


def _gauss(img, sigma):
    """Separable Gaussian of sigma pixels over the first two axes (edges repeated), in the image's own float type."""
    r = max(1, math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(img.dtype)
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(img, pad, mode="edge")
        n = img.shape[axis]
        img = sum(w * (p[t:t + n] if axis == 0 else p[:, t:t + n]) for t, w in enumerate(k))
    return img


def ground_rgb(classes, S, dry=0.0):
    """uint8 (H, W, 3): the classes in the style's ground colours, smoothed by SMOOTH_PX after colouring."""
    lut = np.zeros((256, 3))
    for k, c in palette(S, dry).items():
        lut[k] = c
    rgb = _gauss(lut[np.asarray(classes, dtype=np.uint8)], SMOOTH_PX)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def shares(classes):
    """{class: share of the map}."""
    values, counts = np.unique(np.asarray(classes), return_counts=True)
    return {int(v): float(c) / classes.size for v, c in zip(values, counts)}


# --- colour difference (CIEDE2000) ----------------------------------------------------------------------------------

def lab(rgb):
    """CIE L*a*b* (D65) of an sRGB colour (components in [0, 1])."""
    c = np.asarray(rgb, dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]]) @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(c1, c2):
    """CIEDE2000 difference of two sRGB colours."""
    return de2000(lab(to_rgb(c1)), lab(to_rgb(c2)))


def de2000(lab1, lab2):
    """CIEDE2000 difference of two L*a*b* colours (Sharma, Wu and Dalal 2005)."""
    (L1, a1, b1), (L2, a2, b2) = lab1, lab2
    cb = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cb ** 7 / (cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p, h2p = math.degrees(math.atan2(b1, a1p)) % 360, math.degrees(math.atan2(b2, a2p)) % 360
    dlp, dcp = L2 - L1, c2p - c1p
    dh = 0.0 if c1p * c2p == 0 else (h2p - h1p + 180) % 360 - 180
    dhp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    lbp, cbp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) > 180:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    else:
        hbp = (h1p + h2p) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    rc = 2 * math.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / math.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -math.sin(math.radians(2 * 30 * math.exp(-(((hbp - 275) / 25) ** 2)))) * rc
    return math.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2 + rt * (dcp / sc) * (dhp / sh))


# --- the maps of one print ------------------------------------------------------------------------------------------

def prepare_many(maps, log=print, root=None):
    """One LandCover (or None) per map of a print. maps: [(extent, crs, map_w_mm)]; nothing else about the print
    (tracks, zones, activities) reaches this function. Any network or decoding failure means the style's ground on
    the whole print and one log line naming only the kind of failure."""
    log("[ground] land cover from ESA WorldCover (Amazon S3; only the 3-degree tiles around the map are requested)")
    out = [None] * len(maps)

    def say(i, text):
        log(f"[ground] {f'map {i + 1}: ' if len(maps) > 1 else ''}{text}")

    try:
        root = root or os.path.join(area.cache_root(), "landcover")
        jobs = []
        for i, (extent, crs, map_w_mm) in enumerate(maps):
            job = _Map(i, extent, crs, map_w_mm)
            if job.near_180:
                say(i, "no land cover next to the 180th meridian: drawn with the style's ground")
            elif len(job.tiles) > MAX_TILES:
                say(i, "area too large for the land cover: drawn with the style's ground")
            else:
                jobs.append(job)
        if not jobs:
            return out
        headers, sea = _headers(set().union(*(j.tiles for j in jobs)), root)
        if all(h is None for h in headers.values()):
            log("[ground] no land cover here: drawn with the style's ground")
            return out
        plans = [(job,) + _plan(job, headers, root) for job in jobs]
        every = {b for *_, blocks in plans for b in blocks}  # maps that share a block download it once
        todo = _uncached(every, root)
        if todo:
            size = sum(_block_bytes(headers, *b) for b in todo)
            log(f"[ground] downloading {len(todo)} blocks, {size / 2 ** 20:.1f} MB (one time, cached)")
        _download(headers, every, root)
        for job, L, _ in plans:
            raw = _classes(job, headers, L, root)
            dry = dryness(shares(raw), job.lat)
            sigma = general_sigma(L, job.px, job.map_w / job.shape[1], dry)
            out[job.i] = LandCover(generalise(raw, sigma), L, dry)
            if dry > 0:
                say(job.i, "a dry land: grass and scrub in the colours of a steppe, bare land in the ochre of a "
                           "desert")
        remember_absent(sea, root)  # only once the whole print was read
    except _Unsupported:
        log("[ground] skipped: the land cover does not cover this map as it is; drawn with the style's ground")
        return [None] * len(maps)
    except (relief.ReliefError, OSError, zlib.error, ValueError) as ex:
        log(f"[ground] skipped ({type(ex).__name__}): drawn with the style's ground; run the same command again later "
            f"to add the land cover")
        return [None] * len(maps)
    return out
