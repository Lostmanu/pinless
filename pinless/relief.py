"""Shaded terrain relief under the map, from the Copernicus DEM (GLO-30; GLO-90 only where GLO-30 has a hole).

Opt-in (--relief shade). What is requested depends only on the map extents and the paper widths: for every 1-degree
cell around a map, a 16 KB header and the FULL rectangle of internal blocks that covers the padded map box, never the
blocks a track touches. Tracks, zones and activities never reach this module.

- The elevation is read with HTTP Range requests and decoded by a small stdlib zlib + numpy reader (DEFLATE with the
  floating point predictor), cached under the Pinless cache in dem/.
- A coarse probe decides whether the land is hilly enough to show: flat cities and open sea get nothing.
- The elevation is resampled onto the map's own projected grid, smoothed in proportion to the paper, lit from the
  north-west and returned as one shade raster per map, exactly the map box at 150 ppi.
Any network or decoding failure means a poster without relief (and one log line), never another data source.
"""
from __future__ import annotations

import http.client
import math
import os
import re
import shutil
import socket
import struct
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pyproj import Transformer

from . import area

HEADER_BYTES = 16384
TIMEOUT_S = 20
PPI = 150
SIGMA_MIN_M = 45.0  # smoothing of the detail relief: never less (GLO-30 is a surface model: roofs and trees)
SIGMA_PAPER_MM = 0.40  # ... and never less than this much paper
OPEN_M = 150.0  # ground filter: a grey opening this wide removes buildings and trees
OPEN_MAX_SPACING_M = 60.0  # ... only on grids at least this fine (in practice GLO-30 full resolution)
GATE_SIGMA_M = 250.0  # the probe: smoothing,
GATE_S = 0.15  # the shade that counts as a slope,
GATE_SHARE = 0.02  # and the share of the land that must be sloped for a map to get relief
LAND_EPS_M = 0.5  # a sample farther than this from 0 m is land
LAND_MIN_FRAC = 0.02  # less land than this in the map: mostly water, no relief
AZIMUTH_DEG = 315
ALTITUDE_DEG = 45
FULL_AT = 0.6  # a shade this strong gets the full alpha of its side
MAX_CELLS = 16
BUDGET_BYTES = 64 * 2 ** 20
VALID_M = (-500.0, 9000.0)
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON = 111320.0

DATASETS = {"glo30": ("copernicus-dem-30m", "COG_10"), "glo90": ("copernicus-dem-90m", "COG_30")}
HEIGHT0 = {"glo30": 3600, "glo90": 1200}  # rows of a cell at full resolution: 1 and 3 arc-seconds
NOMINAL_LEVELS = {"glo30": 4, "glo90": 3}  # 3600 to 450 rows, 1200 to 300: bounds the cells a print can need
# The blocks of the real files, at every level. They are all that S3 learns inside a cell, so their size is ours, not
# the server's: GLO-30 blocks of 1024 px are about 30 km at full resolution and the whole cell at the probe level;
# GLO-90 blocks of 2048 px are the whole cell at every level.
TILE = {"glo30": 1024, "glo90": 2048}

# Art. 6(b) of the licences, verbatim (the GLO-30 notice has no trademark sign; the GLO-90 one has it)
NOTICE = {"glo30": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
                   "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
          "glo90": "produced using Copernicus WorldDEM™-90 © DLR e.V. 2010-2014 and © Airbus Defence and "
                   "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"}


def notice(dataset, lang):
    """The credit line for a print with relief. The notice itself stays in English: the licence gives its wording."""
    return ("Relieve: " if lang == "es" else "Relief: ") + NOTICE[dataset]


class ReliefError(RuntimeError):
    """The elevation data could not be fetched or decoded as expected: the print is drawn without relief."""


class _Unsupported(ReliefError):
    """A map the elevation data cannot cover as it is: the same on every run, so no retry is suggested."""


@dataclass(frozen=True, eq=False)
class Relief:
    shade: np.ndarray  # float32 (H, W) in [-1, 1], row 0 north: < 0 shadow, > 0 light, 0 flat
    dataset: str
    level: int
    share: float  # the probe's share of sloped land


@dataclass(frozen=True)
class Level:
    w: int
    h: int
    tw: int
    th: int
    offsets: tuple
    counts: tuple


@dataclass(frozen=True)
class Header:
    W: int  # west longitude of the cell (degrees)
    N: int  # south latitude of the cell (degrees); its first row lies on N + 1
    width0: int
    height0: int
    levels: tuple


# --- cells and URLs -------------------------------------------------------------------------------------------------

def cell_of(lat, lon):
    """The 1-degree cell (south, west) holding a point."""
    return (math.floor(lat), math.floor(lon))


def cell_name(cell):
    lat, lon = cell
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"


def url(dataset, cell):
    bucket, cog = DATASETS[dataset]
    key = f"Copernicus_DSM_{cog}_{cell_name(cell)}_DEM"
    return f"https://{bucket}.s3.amazonaws.com/{key}/{key}.tif"


def _band(N):
    """Copernicus keeps pixels roughly square: above 50 degrees a cell has fewer columns than rows."""
    e = N if N >= 0 else -N - 1
    return 1 if e < 50 else 1.5 if e < 60 else 2 if e < 70 else 3 if e < 80 else 5 if e < 85 else 10


def _dims(dataset, cell, L):
    """(width0, height0, w, h) of a cell at level L, as its header must say (also for sea cells, which have none)."""
    h0 = HEIGHT0[dataset]
    w0 = h0 / _band(cell[0])
    s = 2 ** L
    if w0 != int(w0) or h0 % s or int(w0) % s:
        raise ReliefError("unexpected raster size")
    return int(w0), h0, int(w0) // s, h0 // s


# --- network ----------------------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # a 3xx is an HTTPError, never followed to another host


_OPENER = urllib.request.build_opener(_NoRedirect)
_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _fetch(url_, first, last):
    """Bytes first..last (inclusive) of url_, or None when the object does not exist (HTTP 404). A 5xx or a broken
    connection is tried once more; anything that is not exactly the requested range raises."""
    req = urllib.request.Request(url_, headers={"User-Agent": area.USER_AGENT, "Range": f"bytes={first}-{last}"})
    for attempt in (1, 2):
        try:
            with _OPENER.open(req, timeout=TIMEOUT_S) as r:
                if r.status != 206:  # checked before reading: a 200 would stream the whole 40 MB file
                    raise ReliefError("the server ignored the byte range")
                m = _RANGE.fullmatch((r.headers.get("Content-Range") or "").strip())
                end = min(last, int(m.group(3)) - 1) if m else -1
                if not m or int(m.group(1)) != first or int(m.group(2)) != end:
                    raise ReliefError("unexpected Content-Range")
                body = r.read(last - first + 2)
            if len(body) != end - first + 1:
                raise ReliefError("short body")
            return body
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return None
            if ex.code < 500 or attempt == 2:
                raise
        except (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError, http.client.HTTPException) as ex:
            if attempt == 2:
                if isinstance(ex, http.client.HTTPException):  # not an OSError: report it as a failed download
                    raise ReliefError("the connection broke") from ex
                raise


# --- TIFF header --------------------------------------------------------------------------------------------------

_FMT = {1: ("B", 1), 3: ("H", 2), 4: ("I", 4), 12: ("d", 8)}  # the TIFF types of every tag the reader needs


def _ifd(buf, off):
    """({tag: values}, next IFD offset) of the classic little-endian IFD at off."""
    n, = struct.unpack_from("<H", buf, off)
    end = off + 2 + 12 * n + 4
    if off + 2 > len(buf) or end > len(buf):
        raise ReliefError("unexpected TIFF layout")
    tags = {}
    for e in range(n):
        tag, typ, count, raw = struct.unpack_from("<HHI4s", buf, off + 2 + 12 * e)
        if typ not in _FMT:  # text and rationals: tags the reader never needs
            continue
        fmt, size = _FMT[typ]
        nbytes = size * count
        if nbytes <= 4:
            data = raw[:nbytes]
        else:
            at, = struct.unpack("<I", raw)
            if at + nbytes > len(buf):
                raise ReliefError("unexpected TIFF layout")
            data = bytes(buf[at:at + nbytes])
        tags[tag] = struct.unpack(f"<{count}{fmt}", data)
    nxt, = struct.unpack_from("<I", buf, end - 4)
    return tags, nxt


def read_header(buf, cell, dataset="glo30"):
    """The levels of a Copernicus DEM cell from the first bytes of its file. Anything that is not exactly the layout
    the block reader decodes raises ReliefError."""
    try:
        return _read_header(buf, cell, dataset)
    except (struct.error, IndexError, KeyError, ArithmeticError, TypeError) as ex:  # server data: never a crash
        raise ReliefError("unexpected TIFF layout") from ex


def _ints(t, tag):
    """The values of an integer tag; a tag of another type (a double ImageWidth, say) is not a layout we read."""
    v = t[tag]
    if not v or not all(type(x) is int for x in v):
        raise ReliefError("unexpected TIFF layout")
    return v


def _read_header(buf, cell, dataset):
    if bytes(buf[:4]) != b"II*\x00":
        raise ReliefError("not a classic little-endian TIFF")
    off, = struct.unpack_from("<I", buf, 4)
    ifds, seen = [], set()
    while off:
        if off in seen or len(ifds) >= 16:
            raise ReliefError("unexpected TIFF layout")
        seen.add(off)
        tags, off = _ifd(buf, off)
        ifds.append(tags)
    if not ifds:
        raise ReliefError("unexpected TIFF layout")
    N, W = cell
    t0 = ifds[0]
    width0, height0 = _ints(t0, 256)[0], _ints(t0, 257)[0]
    if height0 != HEIGHT0[dataset] or abs(width0 * _band(N) - height0) > 1e-9:  # before any division by them
        raise ReliefError("unexpected raster size")
    geo = t0[34735]
    keys = {geo[i]: (geo[i + 1], geo[i + 3]) for i in range(4, 4 + 4 * geo[3], 4)}
    if keys.get(1025) != (0, 2):
        raise ReliefError("not PixelIsPoint")
    tie, scale = t0[33922], t0[33550]
    if len(tie) < 6 or any(abs(a - b) > 1e-9 for a, b in zip(tie[:6], (0, 0, 0, W, N + 1, 0))):
        raise ReliefError("unexpected georeference")
    if abs(scale[0] - 1 / width0) > 1e-12 or abs(scale[1] - 1 / height0) > 1e-12:
        raise ReliefError("unexpected georeference")
    if len(ifds) != NOMINAL_LEVELS[dataset]:  # the probe level, and so the cells of a print, are fixed in advance
        raise ReliefError("unexpected number of levels")
    levels = []
    for L, t in enumerate(ifds):
        if (t[259][0] not in (8, 32946) or t[317][0] != 3 or t[339] != (3,) or t[258] != (32,)
                or t.get(277, (1,)) != (1,) or t.get(284, (1,)) != (1,)):
            raise ReliefError("an encoding the reader does not decode")
        w, h, tw, th = (_ints(t, tag)[0] for tag in (256, 257, 322, 323))
        if min(w, h, tw, th) <= 0:
            raise ReliefError("unexpected TIFF layout")
        if (tw, th) != (TILE[dataset], TILE[dataset]):  # smaller blocks would tell S3 more about where the map is
            raise ReliefError("blocks of another size")
        if L == 0:
            if t.get(254, (0,))[0] != 0:
                raise ReliefError("unexpected TIFF layout")
        elif t.get(254, (0,))[0] != 1 or w * 2 ** L != width0 or h * 2 ** L != height0:
            raise ReliefError("an overview that is not a halving")
        offsets, counts = _ints(t, 324), _ints(t, 325)
        tiles = math.ceil(w / tw) * math.ceil(h / th)
        if len(offsets) != tiles or len(counts) != tiles:
            raise ReliefError("unexpected TIFF layout")
        levels.append(Level(w, h, tw, th, tuple(offsets), tuple(counts)))
    if len(levels[-1].offsets) != 1:  # every map is probed: on a flat one, S3 must learn the cell and nothing more
        raise ReliefError("a probe level of several blocks")
    return Header(W, N, width0, height0, tuple(levels))


# --- cache and blocks -----------------------------------------------------------------------------------------------

def _cell_dir(root, dataset, cell):
    return os.path.join(root, dataset, cell_name(cell))


def _write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _header(dataset, cell, root):
    """The cell's header (cached), or None when the dataset has no such cell (HTTP 404, cached as 'absent')."""
    folder = _cell_dir(root, dataset, cell)
    if os.path.exists(os.path.join(folder, "absent")):
        return None
    path = os.path.join(folder, "header.bin")
    if os.path.exists(path):
        with open(path, "rb") as f:
            buf = f.read()
        try:
            return read_header(buf, cell, dataset)
        except ReliefError:
            os.remove(path)  # damaged: fetched again
    buf = _fetch(url(dataset, cell), 0, HEADER_BYTES - 1)
    if buf is None:
        _write_atomic(os.path.join(folder, "absent"), b"")
        return None
    hdr = read_header(buf, cell, dataset)
    _write_atomic(path, buf)  # only a header that parses is kept
    return hdr


def _unpredict(raw, rows, cols):
    """float32 rows from DEFLATE-inflated bytes written with the TIFF floating point predictor (3): every row holds
    its byte planes, most significant first, each byte stored as the difference from the one before it."""
    a = np.frombuffer(raw, np.uint8).reshape(rows, 4 * cols)
    u = np.cumsum(a, axis=1, dtype=np.uint8)
    return u.reshape(rows, 4, cols)[:, ::-1, :].transpose(0, 2, 1).copy().view("<f4").reshape(rows, cols)


def _decode(data, lv, br, bc):
    try:
        raw = zlib.decompress(data)
    except zlib.error as ex:
        raise ReliefError("a block does not inflate") from ex
    if len(raw) != lv.th * lv.tw * 4:
        raise ReliefError("unexpected block size")
    z = _unpredict(raw, lv.th, lv.tw)[: min(lv.th, lv.h - br * lv.th), : min(lv.tw, lv.w - bc * lv.tw)]
    if not (np.isfinite(z).all() and z.min() >= VALID_M[0] and z.max() <= VALID_M[1]):
        raise ReliefError("elevation out of range")
    return z


def _block_path(root, dataset, cell, L, br, bc):
    return os.path.join(_cell_dir(root, dataset, cell), f"L{L}_{br}_{bc}.zz")


def _block(dataset, cell, hdr, L, br, bc, root):
    """Block (br, bc) of level L, without its edge padding. The compressed bytes are cached only once they decode;
    a cached file that does not decode is deleted and fetched again, once."""
    lv = hdr.levels[L]
    path = _block_path(root, dataset, cell, L, br, bc)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        try:
            return _decode(data, lv, br, bc)
        except (ReliefError, ValueError):
            os.remove(path)
    i = br * math.ceil(lv.w / lv.tw) + bc
    first, n = lv.offsets[i], lv.counts[i]
    if n <= 0:
        raise ReliefError("an empty block")
    data = _fetch(url(dataset, cell), first, first + n - 1)
    try:
        if data is None:
            raise ReliefError("a block is missing")
        z = _decode(data, lv, br, bc)
    except ReliefError:
        # The bytes at the header's offsets are not that block: the file may have been replaced on the server since
        # its header was cached. Forget the cell (header and blocks), so the next run starts it afresh.
        shutil.rmtree(_cell_dir(root, dataset, cell), ignore_errors=True)
        raise
    _write_atomic(path, data)
    return z


# --- geometry -------------------------------------------------------------------------------------------------------

def raster_shape(map_w_mm, extent):
    """(H, W) of the relief raster of a map: its paper width at PPI, its height from the extent."""
    x0, x1, y0, y1 = extent
    W = round(round(map_w_mm, 6) / 25.4 * PPI)
    return (round(W * (y1 - y0) / (x1 - x0)), W)


def _lonlat_box(extent, crs, pad_m):
    """(lat_south, lat_north, lon_west, lon_east) of the map frame, padded by pad_m. The frame is densified (20
    points a side): projected edges are curved in degrees, so the four corners alone would miss a bulge."""
    x0, x1, y0, y1 = extent
    t = np.linspace(0, 1, 20)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full(20, x1), x1 - (x1 - x0) * t, np.full(20, x0)])
    ys = np.concatenate([np.full(20, y0), y0 + (y1 - y0) * t, np.full(20, y1), y1 - (y1 - y0) * t])
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(xs, ys)
    lat_a, lat_b, lon_a, lon_b = float(np.min(lat)), float(np.max(lat)), float(np.min(lon)), float(np.max(lon))
    if not all(map(math.isfinite, (lat_a, lat_b, lon_a, lon_b))):
        raise ReliefError("the map frame has no position")
    dlat = pad_m / M_PER_DEG_LAT
    dlon = pad_m / (M_PER_DEG_LON * max(0.05, math.cos(math.radians(max(abs(lat_a), abs(lat_b))))))
    return lat_a - dlat, lat_b + dlat, lon_a - dlon, lon_b + dlon


def _cells(box):
    """Every 1-degree cell the box touches, as a full rectangle."""
    lat_a, lat_b, lon_a, lon_b = box
    return [(la, lo) for la in range(math.floor(lat_a), math.floor(lat_b) + 1)
            for lo in range(math.floor(lon_a), math.floor(lon_b) + 1)]


def _window(cell, L, dims, box):
    """(i0, i1, j0, j1): the rows and columns of a cell at level L that cover the box, plus one pixel.

    The pixel centre of level L at (i, j) is lon = W + (2^L j + (2^L - 1) / 2) / width0, lat = N + 1 - (2^L i +
    (2^L - 1) / 2) / height0: an overview pixel is the mean of a 2^L block of full-resolution pixels."""
    w0, h0, w, h = dims
    lat_a, lat_b, lon_a, lon_b = box
    N, W = cell
    s = 2 ** L
    j0 = max(0, math.floor(((lon_a - W) * w0 - (s - 1) / 2) / s) - 1)
    j1 = min(w - 1, math.ceil(((lon_b - W) * w0 - (s - 1) / 2) / s) + 1)
    i0 = max(0, math.floor(((N + 1 - lat_b) * h0 - (s - 1) / 2) / s) - 1)
    i1 = min(h - 1, math.ceil(((N + 1 - lat_a) * h0 - (s - 1) / 2) / s) + 1)
    return i0, i1, j0, j1


def _blocks(dataset, headers, cells, L, box):
    """[(cell, br, bc)] of every land cell, row-major per cell: the full rectangle of blocks covering the box."""
    out = []
    for cell in sorted(cells, key=lambda c: (-c[0], c[1])):
        hdr = headers.get(cell)
        if hdr is None:  # sea: no requests
            continue
        lv = hdr.levels[L]
        i0, i1, j0, j1 = _window(cell, L, _dims(dataset, cell, L), box)
        out += [(cell, br, bc) for br in range(i0 // lv.th, i1 // lv.th + 1)
                for bc in range(j0 // lv.tw, j1 // lv.tw + 1)]
    return out


def _read_window(dataset, cell, hdr, L, win, root):
    i0, i1, j0, j1 = win
    lv = hdr.levels[L]
    out = np.empty((i1 - i0 + 1, j1 - j0 + 1), np.float32)
    for br in range(i0 // lv.th, i1 // lv.th + 1):
        for bc in range(j0 // lv.tw, j1 // lv.tw + 1):
            z = _block(dataset, cell, hdr, L, br, bc, root)
            r0, c0 = br * lv.th, bc * lv.tw
            ra, rb = max(r0, i0), min(r0 + z.shape[0] - 1, i1)
            ca, cb = max(c0, j0), min(c0 + z.shape[1] - 1, j1)
            out[ra - i0:rb - i0 + 1, ca - j0:cb - j0 + 1] = z[ra - r0:rb - r0 + 1, ca - c0:cb - c0 + 1]
    return out


@dataclass(frozen=True, eq=False)
class Mosaic:
    z: np.ndarray  # row 0 north
    lon0: float  # longitude of column 0
    dlon: float
    lat0: float  # latitude of row 0
    dlat: float


def _mosaic(dataset, headers, cells, L, box, root):
    """The cells' windows at level L as one grid. Within a latitude band the windows join west to east (Copernicus
    dropped each cell's last row and column, so they are contiguous); bands with fewer columns (above 50 degrees)
    are resampled along longitude to the columns of the widest one; sea cells are 0 m."""
    lats = sorted({c[0] for c in cells}, reverse=True)
    lons = sorted({c[1] for c in cells})
    bands = []
    for N in lats:
        parts = []
        for W in lons:
            cell = (N, W)
            dims = _dims(dataset, cell, L)
            win = _window(cell, L, dims, box)
            hdr = headers.get(cell)
            i0, i1, j0, j1 = win
            parts.append(np.zeros((i1 - i0 + 1, j1 - j0 + 1), np.float32) if hdr is None
                         else _read_window(dataset, cell, hdr, L, win, root))
            if W == lons[0]:
                w0, h0 = dims[:2]
                lon0 = W + (2 ** L * j0 + (2 ** L - 1) / 2) / w0
                lat0 = N + 1 - (2 ** L * i0 + (2 ** L - 1) / 2) / h0
                dlon, dlat = 2 ** L / w0, 2 ** L / h0
        bands.append((np.hstack(parts), lon0, dlon, lat0, dlat))
    wide = max(bands, key=lambda b: b[0].shape[1])
    _, lon0, dlon, lat0, dlat = wide
    n = wide[0].shape[1]
    rows = []
    for z, lo, dl, _, _ in bands:
        if z.shape[1] != n or lo != lon0 or dl != dlon:
            pos = np.clip((lon0 + np.arange(n) * dlon - lo) / dl, 0, z.shape[1] - 1)
            c0 = np.minimum(np.floor(pos).astype(np.intp), z.shape[1] - 2)
            t = (pos - c0).astype(np.float32)
            z = z[:, c0] + (z[:, c0 + 1] - z[:, c0]) * t
        rows.append(z)
    return Mosaic(np.vstack(rows), lon0, dlon, bands[0][3], dlat)


def _odd(v):
    """The nearest odd integer to v, at least 3."""
    return max(3, 2 * round((v - 1) / 2) + 1)


def _opening(z, ky, kx):
    """Grey opening (minimum, then maximum filter; separable, edges repeated): removes what stands out of the
    ground narrower than the window (buildings, trees) and leaves broader landforms."""
    def run(a, k, axis, fn):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k // 2, k // 2)
        return fn(np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, mode="edge"), k, axis=axis), axis=-1)

    z = run(run(z, ky, 0, np.min), kx, 1, np.min)
    return run(run(z, ky, 0, np.max), kx, 1, np.max)


def _ground(mos):
    """The ground filter on a grid fine enough to show buildings (N-S spacing up to OPEN_MAX_SPACING_M)."""
    sy = mos.dlat * M_PER_DEG_LAT
    if sy > OPEN_MAX_SPACING_M:
        return mos
    mid = mos.lat0 - mos.dlat * (mos.z.shape[0] - 1) / 2
    sx = M_PER_DEG_LON * math.cos(math.radians(mid)) * mos.dlon
    return Mosaic(_opening(mos.z, _odd(OPEN_M / sy), _odd(OPEN_M / sx)), mos.lon0, mos.dlon, mos.lat0, mos.dlat)


def _sample(mos, lon, lat):
    """Bilinear samples of the mosaic. A point outside it raises: clamping would paint streaks."""
    nr, nc = mos.z.shape
    col = (np.asarray(lon) - mos.lon0) / mos.dlon
    row = (mos.lat0 - np.asarray(lat)) / mos.dlat
    if not (np.isfinite(col).all() and np.isfinite(row).all() and col.min() >= 0 and row.min() >= 0
            and col.max() <= nc - 1 and row.max() <= nr - 1) or nr < 2 or nc < 2:
        raise _Unsupported("a sample fell outside the elevation data")
    c0 = np.minimum(np.floor(col).astype(np.intp), nc - 2)
    r0 = np.minimum(np.floor(row).astype(np.intp), nr - 2)
    tc, tr = col - c0, row - r0
    z = mos.z.astype(np.float64, copy=False)
    a, b, c, d = z[r0, c0], z[r0, c0 + 1], z[r0 + 1, c0], z[r0 + 1, c0 + 1]
    top = a + (b - a) * tc  # in this form a constant field stays exactly constant
    bottom = c + (d - c) * tc
    return top + (bottom - top) * tr


@dataclass(frozen=True)
class Grid:
    """The work grid of a map: nx x ny cells of cx x cy map units over the extent, plus a margin of m nodes."""
    cx: float
    cy: float
    nx: int
    ny: int
    m: int

    @property
    def reach(self):
        """How far the outermost node lies outside the extent."""
        return (self.m - 0.5) * max(self.cx, self.cy)


def _grid(extent, cell, sigma):
    x0, x1, y0, y1 = extent
    nx, ny = math.ceil((x1 - x0) / cell), math.ceil((y1 - y0) / cell)
    cx, cy = (x1 - x0) / nx, (y1 - y0) / ny
    return Grid(cx, cy, nx, ny, math.ceil(4 * sigma / min(cx, cy)) + 1)


def _nodes(extent, crs, g):
    """(lon, lat) of every node of the grid, row 0 north. Every node is projected: the grid is never larger than
    the output raster."""
    x0, x1, y0, y1 = extent
    xs = x0 + (np.arange(-g.m, g.nx + g.m) + 0.5) * g.cx
    ys = y1 - (np.arange(-g.m, g.ny + g.m) + 0.5) * g.cy
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(X.ravel(), Y.ravel())
    return np.asarray(lon).reshape(X.shape), np.asarray(lat).reshape(X.shape)


# --- shading ----------------------------------------------------------------------------------------------------------

def _gauss(z, sigma, g):
    """Separable Gaussian of sigma map units on the work grid (edges reflected; the margin absorbs them)."""
    for axis, step in ((1, g.cx), (0, g.cy)):
        s = sigma / step
        if s <= 0:
            continue
        r = max(1, math.ceil(4 * s))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        k /= k.sum()
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        zp = np.pad(z, pad, mode="reflect")
        n = z.shape[axis]
        out = np.zeros(z.shape)
        for t, wt in enumerate(k):  # every node gets the same sums: a constant field stays exactly constant
            out += wt * (zp[t:t + n] if axis == 0 else zp[:, t:t + n])
        z = out
    return z


def _shade(z, cx, cy):
    """Hillshade in [-1, 1] lit from AZIMUTH_DEG at ALTITUDE_DEG, row 0 north: 0 on flat ground, -1 in full
    shadow, 1 facing the light. No vertical exaggeration."""
    dzdrow, dzdx = np.gradient(z, cy, cx)
    dzdy = -dzdrow
    az, alt = math.radians(AZIMUTH_DEG), math.radians(ALTITUDE_DEG)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    hs = np.maximum(0.0, (-dzdx * lx - dzdy * ly + lz) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1))
    s = hs - lz
    return np.clip(np.where(s < 0, s / lz, s / (1 - lz)), -1.0, 1.0)


# --- the maps of one print ------------------------------------------------------------------------------------------

def _spacing(dataset, L):
    """N-S spacing (m) of level L."""
    return M_PER_DEG_LAT / HEIGHT0[dataset] * 2 ** L


class _Map:
    """One map of the print: its extent, projection and paper width, and what follows from them alone."""

    def __init__(self, i, extent, crs, map_w_mm):
        self.i, self.crs, self.map_w = i, crs, float(map_w_mm)
        self.extent = tuple(float(v) for v in extent)
        x0, x1, y0, y1 = self.extent
        if not (x1 > x0 and y1 > y0 and self.map_w > 0):
            raise ValueError("an empty map")
        self.shape = raster_shape(self.map_w, self.extent)
        self.sigma = max(SIGMA_MIN_M, SIGMA_PAPER_MM * (x1 - x0) / self.map_w)  # the detail smoothing
        self.probe_grid = _grid(self.extent, max(125.0, max(x1 - x0, y1 - y0) / 400), GATE_SIGMA_M)
        # every cell the map could need, whatever the dataset and level: fixed before any request is made
        spacings = [_spacing(ds, L) for ds in HEIGHT0 for L in range(NOMINAL_LEVELS[ds])]
        lat_a, lat_b, lon_a, lon_b = _lonlat_box(self.extent, crs, max(max(self.probe_pad(s), self.detail_pad(s))
                                                                       for s in spacings))
        # Cells and mosaics run west to east from -180 to 180: a map that reaches the 180th meridian gets no relief
        # (its frame wraps from 180 to -180, and there is no cell east of 179)
        self.near_180 = lon_b >= 180 or lon_a < -180 or lon_b - lon_a > 180
        self.cells = set() if self.near_180 else set(_cells((lat_a, lat_b, lon_a, lon_b)))

    def detail_grid(self, spacing):
        return _grid(self.extent, max((self.extent[1] - self.extent[0]) / self.shape[1], spacing / 2), self.sigma)

    def probe_pad(self, spacing):
        """Padding of the probe box: the outermost node, plus two pixels for the bilinear samples."""
        return self.probe_grid.reach + 2 * spacing

    def detail_pad(self, spacing):
        """Padding of the detail box: the outermost node, the ground filter, two pixels."""
        return self.detail_grid(spacing).reach + OPEN_M + 2 * spacing

    def box(self, pad):
        box = _lonlat_box(self.extent, self.crs, pad)
        cells = _cells(box)
        if not set(cells) <= self.cells:
            raise _Unsupported("a map needs a cell outside its print")
        return box, cells


def _dataset(cells, root):
    """('glo30', headers) unless a land cell is missing from GLO-30 and present in GLO-90: then ('glo90', headers),
    for every map of the print. A cell absent from both is sea (None). Network errors raise: they never switch."""
    h30 = {c: _header("glo30", c, root) for c in cells}
    h90 = {c: _header("glo90", c, root) for c in cells if h30[c] is None}
    if any(h is not None for h in h90.values()):
        return "glo90", {c: h90[c] if c in h90 else _header("glo90", c, root) for c in cells}
    return "glo30", h30


def _block_bytes(headers, cell, L, br, bc):
    lv = headers[cell].levels[L]
    return lv.counts[br * math.ceil(lv.w / lv.tw) + bc]


def _gate(job, dataset, headers, L, root):
    """The share of the map's land that slopes visibly, on level L (the coarsest); None when it is mostly water."""
    g = job.probe_grid
    box, cells = job.box(job.probe_pad(_spacing(dataset, L)))
    mos = _mosaic(dataset, headers, cells, L, box, root)
    raw = _sample(mos, *_nodes(job.extent, job.crs, g))
    s = _shade(_gauss(raw, GATE_SIGMA_M, g), g.cx, g.cy)
    inner = (slice(g.m, g.m + g.ny), slice(g.m, g.m + g.nx))
    land = np.abs(raw[inner]) > LAND_EPS_M  # unsmoothed, so the sea does not bleed into the land
    if land.mean() < LAND_MIN_FRAC:
        return None
    return float((np.abs(s[inner][land]) > GATE_S).mean())


def _detail_level(job, dataset, n_levels):
    """The coarsest level whose N-S spacing is within the map's smoothing (sigma), or 0."""
    return max([L for L in range(n_levels) if _spacing(dataset, L) <= job.sigma] or [0])


def _plan(job, dataset, headers, n_levels, root):
    """(level, blocks still to download): the detail level, one level coarser while those blocks exceed
    BUDGET_BYTES; the coarsest level is always within budget."""
    L = _detail_level(job, dataset, n_levels)
    while True:
        box, cells = job.box(job.detail_pad(_spacing(dataset, L)))
        todo = [(cell, L, br, bc) for cell, br, bc in _blocks(dataset, headers, cells, L, box)
                if not os.path.exists(_block_path(root, dataset, cell, L, br, bc))]
        if L == n_levels - 1 or sum(_block_bytes(headers, *b) for b in todo) <= BUDGET_BYTES:
            return L, todo
        L += 1


def _detail(job, dataset, headers, L, root):
    """The map's shade raster from level L: mosaic, ground filter, map grid, smoothing, shading, output size."""
    g = job.detail_grid(_spacing(dataset, L))
    box, cells = job.box(job.detail_pad(_spacing(dataset, L)))
    mos = _ground(_mosaic(dataset, headers, cells, L, box, root))
    z = _gauss(_sample(mos, *_nodes(job.extent, job.crs, g)), job.sigma, g)
    s = _shade(z, g.cx, g.cy)[g.m:g.m + g.ny, g.m:g.m + g.nx]
    H, W = job.shape
    out = np.asarray(Image.fromarray(s.astype(np.float32)).resize((W, H), Image.Resampling.BICUBIC), np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def prepare_many(maps, log=print, root=None):
    """One Relief (or None) per map of a print. maps: [(extent, crs, map_w_mm)]; nothing else about the print
    (tracks, zones, activities) reaches this function. Any network or decoding failure means no relief anywhere on
    the print and one log line; a map that is too large, flat or mostly water gets None and its own line."""
    log("[relief] terrain from the Copernicus elevation model (Amazon S3 learns the 1-degree cells around the map and, "
        "where the land is hilly, the blocks of about 30 km that cover it; never your routes)")
    out = [None] * len(maps)

    def say(i, text):
        log(f"[relief] {f'map {i + 1}: ' if len(maps) > 1 else ''}{text}")

    try:
        root = root or os.path.join(area.cache_root(), "dem")
        jobs = []
        for i, (extent, crs, map_w_mm) in enumerate(maps):
            job = _Map(i, extent, crs, map_w_mm)
            if job.near_180:
                say(i, "no relief next to the 180th meridian: drawn without it")
            elif len(job.cells) > MAX_CELLS:
                say(i, "area too large for relief: drawn without it")
            else:
                jobs.append(job)
        if not jobs:
            return out
        dataset, headers = _dataset(sorted(set().union(*(j.cells for j in jobs))), root)
        present = [h for h in headers.values() if h is not None]
        if not present:
            log("[relief] only sea here: no relief")
            return out
        n_levels = NOMINAL_LEVELS[dataset]  # read_header holds every cell to it
        passed = []
        for job in jobs:  # the probe: the coarsest level only; detail blocks are requested after it passes
            share = _gate(job, dataset, headers, n_levels - 1, root)
            if share is None:
                say(job.i, "mostly water here: no relief")
            elif share < GATE_SHARE:
                say(job.i, "flat here: no relief")
            else:
                passed.append((job, share))
        plans = [(job, share) + _plan(job, dataset, headers, n_levels, root) for job, share in passed]
        todo = list(dict.fromkeys(b for *_, blocks in plans for b in blocks))
        if todo:
            size = sum(_block_bytes(headers, *b) for b in todo)
            log(f"[relief] downloading {len(todo)} blocks, {size / 2 ** 20:.1f} MB (one time, cached)")
        for job, share, L, _ in plans:
            out[job.i] = Relief(_detail(job, dataset, headers, L, root), dataset, L, share)
    except _Unsupported:  # the same on every run: no point in running it again
        log("[relief] skipped: the elevation data does not cover this map as it is; the poster is drawn without relief")
        return [None] * len(maps)
    except (ReliefError, OSError, zlib.error, ValueError) as ex:
        log(f"[relief] skipped ({type(ex).__name__}): the poster is drawn without relief; run the same command again "
            f"later to add it")
        return [None] * len(maps)
    return out
