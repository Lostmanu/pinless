"""Shaded terrain relief under the map, from the Copernicus DEM (GLO-30; GLO-90 only where GLO-30 has a hole).

Opt-in (--relief shade). What is requested depends only on the map extents and the paper widths: for every 1-degree
cell around a map, a 16 KB header and the FULL rectangle of internal blocks that covers the padded map box, never the
blocks a track touches. Tracks, zones and activities never reach this module.

- The elevation is read with HTTP Range requests and decoded by a small stdlib zlib + numpy reader (DEFLATE with the
  floating point predictor), cached under the Pinless cache in dem/.
- A coarse probe decides whether the land is hilly enough to show: flat cities and open sea get nothing.
- The elevation is resampled onto the map's own projected grid (cubic convolution) and smoothed only as much as the
  paper shows. Three signals make the shade: a hillshade lit from the north-west for the form, Leland Brown's texture
  shading for the ridges and gullies at every scale, and the sky-view factor for the depth of cirques and gorges.
- Two shades come back per map, exactly the map box at 150 ppi: the crisp one of the surface itself, and a smooth one
  over a ground filter that takes the buildings of a town out (a normalised convolution of the streets, squares and
  parks between them), for city prints and the built-up land of the trail print. The heights of both surfaces come
  back too, smoothed for contour lines.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pyproj import Transformer

from . import area

HEADER_BYTES = 16384
TIMEOUT_S = 20
PPI = 150
# The level rule: the detail level is the coarsest whose spacing is within max(SIGMA_MIN_M, SIGMA_PAPER_MM of paper).
# It fixes which blocks a map downloads, so it stays as it was; the shade itself is smoothed far less (CRISP_*)
SIGMA_MIN_M = 45.0
SIGMA_PAPER_MM = 0.40
OPEN_M = 150.0  # city ground filter: a grey opening this wide finds the ground between buildings
OPEN_MAX_SPACING_M = 60.0  # ... only on grids at least this fine (in practice GLO-30 full resolution)
# The probe: smoothing of GATE_SIGMA_MM of paper, between GATE_SIGMA_MIN_M and GATE_SIGMA_M (a landform at the
# map's scale: 250 m on every map of 11 km or more on A3, a city's; 125 m on a hike of a few km, where 250 m into the
# sea flattened the cliffs of a peninsula 300 m wide),
GATE_SIGMA_M, GATE_SIGMA_MIN_M, GATE_SIGMA_MM = 250.0, 125.0, 6.0
GATE_S = 0.15  # the shade that counts as a slope,
GATE_SHARE = 0.02  # and the share of the land that must be sloped for a map to get relief
LAND_EPS_M = 0.5  # a sample farther than this from 0 m is land
LAND_MIN_FRAC = 0.02  # less land than this in the map: mostly water, no relief
AZIMUTH_DEG = 315
ALTITUDE_DEG = 45  # the probe's light (the gate)
CRISP_SIGMA_M, CRISP_SIGMA_MM = 8.0, 0.25  # crisp shade smoothing: max(8 m, 0.25 mm of paper)
SHADE_ALT_DEG = 55  # the detail hillshade's light
TEX_ALPHA, TEX_K, TEX_FLOOR = 0.8, 1.5, 1.0  # texture shading: detail exponent, contrast, normalisation floor (m)
SVF_DIRS, SVF_MM, SVF_MIN_M, SVF_MAX_M = 16, 14.0, 250.0, 600.0  # sky view: directions, radius (paper mm, bounds)
AO_FLOOR = 0.10  # the sky view's occlusion is normalised by max(p99, this): flat land gains no false depth
W_HS, W_TEX, W_AO, GAIN = 0.65, 0.6, 0.45, 1.25  # the blend of the three signals and its contrast
URBAN_DH_M, URBAN_SIGMA_M, URBAN_MIN_W, URBAN_SHADE_SIGMA_M = 2.5, 60.0, 0.02, 20.0  # the city ground filter
CONTOUR_SIGMA_M, CONTOUR_SIGMA_MM = 20.0, 0.20  # the heights for contour lines: max(20 m, 0.20 mm of paper)
RAMP_FULL, RAMP_GAMMA = 0.9, 0.85  # a drawn overlay's alpha = side alpha * min(1, |s| / 0.9) ** 0.85
SVF_ROWS = 24  # the sky view runs in bands of this many rows (threads)
_WORKERS = max(1, min(8, os.cpu_count() or 1))
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
    shade: np.ndarray  # float32 (H, W) in [-1, 1], row 0 north: < 0 shadow, > 0 light, 0 flat; the crisp surface
    dataset: str
    level: int
    share: float  # the probe's share of sloped land
    smooth: np.ndarray = None  # float32 (H, W): the shade over the city ground filter (buildings taken out)
    z: np.ndarray = None  # float32 (H, W): heights (m) of the surface, smoothed for contour lines
    z_smooth: np.ndarray = None  # float32 (H, W): heights (m) of the city ground, smoothed for contour lines


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


def _fetch(url_, first, last, absent=(404,)):
    """Bytes first..last (inclusive) of url_, or None when the object does not exist (an HTTP status in absent). A 5xx
    or a broken connection is tried once more; anything that is not exactly the requested range raises."""
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
            if ex.code in absent:
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


def _median3(z):
    """3 x 3 median (edges repeated): takes out single-pixel spikes and pits (a lone building or tree, a radar glint)
    and keeps the edges of ridges and gullies."""
    zp = np.pad(z, 1, mode="edge")
    n, m = z.shape
    return np.median(np.stack([zp[i:i + n, j:j + m] for i in range(3) for j in range(3)]), axis=0)


def _ground_urban(mos):
    """The ground under a town, on a grid fine enough to show buildings (N-S spacing up to OPEN_MAX_SPACING_M), without
    the terraces a grey opening leaves: the opening only finds the ground (the pixels it lowers by less than
    URBAN_DH_M: streets, squares, parks), and the ground between them is a normalised convolution of those pixels
    (Knutsson and Westin 1993), a Gaussian of URBAN_SIGMA_M through every one of them. Where no ground pixel lies near
    enough (URBAN_MIN_W), the opening itself."""
    sy = mos.dlat * M_PER_DEG_LAT
    if sy > OPEN_MAX_SPACING_M:
        return mos
    mid = mos.lat0 - mos.dlat * (mos.z.shape[0] - 1) / 2
    sx = M_PER_DEG_LON * math.cos(math.radians(mid)) * mos.dlon
    z = _median3(np.asarray(mos.z, dtype=np.float64))
    opened = _opening(z, _odd(OPEN_M / sy), _odd(OPEN_M / sx))
    ground = (z - opened < URBAN_DH_M).astype(np.float64)
    g = Grid(sx, sy, z.shape[1], z.shape[0], 0)
    w = _gauss(ground, URBAN_SIGMA_M, g)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = _gauss(z * ground, URBAN_SIGMA_M, g) / w
    return Mosaic(np.where(w > URBAN_MIN_W, est, opened), mos.lon0, mos.dlon, mos.lat0, mos.dlat)


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


def _keys(t):
    """The four weights of cubic convolution (Keys, a = -0.5) at fractional offset t: an interpolant through the
    samples whose slope is continuous, so a hillshade of it has no creases along the data's pixel edges."""
    t2, t3 = t * t, t * t * t
    return (-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2)


def _sample_cubic(mos, lon, lat, *more):
    """Cubic convolution samples of the mosaic, with _sample's checks (a point outside it raises); the outer taps
    repeat the edges. Written as differences from the tap at the floor, so a constant field stays exactly constant.
    more: other height grids of the same shape and position (the city ground), sampled at the same points; then a
    list of the samples of each."""
    nr, nc = mos.z.shape
    col = (np.asarray(lon) - mos.lon0) / mos.dlon
    row = (mos.lat0 - np.asarray(lat)) / mos.dlat
    if not (np.isfinite(col).all() and np.isfinite(row).all() and col.min() >= 0 and row.min() >= 0
            and col.max() <= nc - 1 and row.max() <= nr - 1) or nr < 4 or nc < 4:
        raise _Unsupported("a sample fell outside the elevation data")
    c0 = np.minimum(np.floor(col).astype(np.intp), nc - 2)
    r0 = np.minimum(np.floor(row).astype(np.intp), nr - 2)
    wc, wr = _keys(col - c0), _keys(row - r0)
    # the grid padded by one repeated edge pixel all round, so the 4 x 4 taps of every sample are inside it; flat
    # indices (np.take on the flat grid is several times faster than 2-D fancy indexing)
    at = (r0 + 1) * (nc + 2) + (c0 + 1)  # the tap at the floor, in the padded grid
    offsets = [[(i - 1) * (nc + 2) + (j - 1) for j in range(4)] for i in range(4)]

    def one(z):
        z = np.pad(np.asarray(z, dtype=np.float64), 1, mode="edge").ravel()
        lines = []
        for row_off in offsets:
            base = np.take(z, at + row_off[1])
            line = base.copy()
            for j in (0, 2, 3):  # the floor's own term is exactly 0
                line += wc[j] * (np.take(z, at + row_off[j]) - base)
            lines.append(line)
        out = lines[1].copy()
        for i in (0, 2, 3):
            out += wr[i] * (lines[i] - lines[1])
        return out

    if not more:
        return one(mos.z)
    if any(np.shape(m) != mos.z.shape for m in more):
        raise ValueError("grids of another shape")
    return [one(mos.z)] + [one(m) for m in more]


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


def _shade(z, cx, cy, alt_deg=ALTITUDE_DEG):
    """Hillshade in [-1, 1] lit from AZIMUTH_DEG at alt_deg, row 0 north: 0 on flat ground, -1 in full shadow, 1
    facing the light. No vertical exaggeration."""
    dzdrow, dzdx = np.gradient(z, cy, cx)
    dzdy = -dzdrow
    az, alt = math.radians(AZIMUTH_DEG), math.radians(alt_deg)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    hs = np.maximum(0.0, (-dzdx * lx - dzdy * ly + lz) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1))
    s = hs - lz
    return np.clip(np.where(s < 0, s / lz, s / (1 - lz)), -1.0, 1.0)


def _texture(z, cx, cy, alpha):
    """Leland Brown's terrain texture: the fractional Laplacian (-Lap)^(alpha/2) of z, by FFT on the even (mirrored)
    extension of the grid, so its edges add no seam, once the mean and the plane that fits best are taken out (both
    have no texture). Positive on ridges, negative in valleys, at every scale; exactly 0 on a constant."""
    ny, nx = z.shape
    # the mean and the plane that fits best taken out first: the texture of a plane is 0, but mirrored the plane would
    # fold into a ridge or a valley along the grid's edges and shade the whole map from one side to the other
    yy, xx = np.arange(ny) - (ny - 1) / 2, np.arange(nx) - (nx - 1) / 2
    z = z - z.flat[0]  # a constant becomes exactly 0 (its mean may not round back to it): no texture at all
    z = z - z.mean()
    z = z - (z @ xx).sum() / max(ny * (xx @ xx), 1e-12) * xx[None, :] \
        - (yy @ z).sum() / max(nx * (yy @ yy), 1e-12) * yy[:, None]
    # grown (edges repeated, south and east) to a size whose double the FFT factors fast: about twice as quick as a
    # size with a large prime factor; the map's nodes lie a margin away from that edge
    z = np.pad(z, ((0, _fft_size(ny) - ny), (0, _fft_size(nx) - nx)), mode="edge")
    zp = np.block([[z, z[:, ::-1]], [z[::-1], z[::-1, ::-1]]])
    ky = np.fft.fftfreq(zp.shape[0], d=cy)[:, None]
    kx = np.fft.rfftfreq(zp.shape[1], d=cx)[None, :]
    k = 2 * np.pi * np.sqrt(kx ** 2 + ky ** 2)
    return np.fft.irfft2(np.fft.rfft2(zp) * k ** alpha, s=zp.shape)[:ny, :nx]


def _fft_size(n):
    """The smallest m >= n whose double has no prime factor above 7."""
    m = n
    while True:
        r = 2 * m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def _sky_view(z, cx, cy, radius_m, n_dir=SVF_DIRS):
    """Sky-view factor (Zaksek et al. 2011) in [0, 1]: 1 on open ground, lower in gullies, cirques and under cliffs.
    The horizon in n_dir directions up to radius_m, at steps that grow with the distance (edges repeated; a horizon
    below the horizontal counts as the horizontal)."""
    ny, nx = z.shape
    c = min(cx, cy)
    steps = np.unique(np.round(np.geomspace(c, radius_m, max(4, int(round(radius_m / c)))) / c).astype(int))
    pad = int(steps.max()) + 1
    z = np.asarray(z, dtype=np.float32)
    zp = np.pad(z, pad, mode="edge")
    shifts = []  # (row offset, column offset, 1 / distance) of every step of every direction, in order
    for d in range(n_dir):
        a = 2 * math.pi * d / n_dir
        shifts.append([(oy, ox, np.float32(1 / math.hypot(ox * cx, oy * cy)))
                       for ox, oy in ((int(round(math.sin(a) * r)), int(round(-math.cos(a) * r))) for r in steps)
                       if ox or oy])

    def band(y0):  # SVF_ROWS rows, every direction, in cache: the work is memory bound, not arithmetic bound
        y1 = min(ny, y0 + SVF_ROWS)
        zb = z[y0:y1]
        acc = np.zeros(zb.shape, np.float32)
        best, tmp = np.empty(zb.shape, np.float32), np.empty(zb.shape, np.float32)
        for steps_d in shifts:
            best[:] = 0  # a horizon below the horizontal counts as the horizontal
            for oy, ox, inv in steps_d:
                np.subtract(zp[pad + oy + y0:pad + oy + y1, pad + ox:pad + ox + nx], zb, out=tmp)
                tmp *= inv
                np.maximum(best, tmp, out=best)
            np.multiply(best, best, out=tmp)
            tmp += 1
            np.sqrt(tmp, out=tmp)
            acc += best / tmp  # sin(atan(best)), the directions summed in the same order for every band
        return acc

    with ThreadPoolExecutor(_WORKERS) as ex:  # numpy lets go of the GIL; the bands do not depend on the threads
        acc = np.concatenate(list(ex.map(band, range(0, ny, SVF_ROWS)))).astype(np.float64)
    return 1.0 - acc / n_dir


def _blend(zs, g, radius):
    """The shade of the smoothed heights zs on the work grid: a hillshade at SHADE_ALT_DEG for the form, the texture
    for ridges and valleys at every scale, the sky view's occlusion for the depth of gullies and cirques, each
    normalised on the map's own nodes, blended and squeezed into [-1, 1]."""
    inner = (slice(g.m, g.m + g.ny), slice(g.m, g.m + g.nx))
    hs = _shade(zs, g.cx, g.cy, SHADE_ALT_DEG)
    tex = _texture(zs, g.cx, g.cy, TEX_ALPHA)
    tex = np.tanh(TEX_K * tex / max(float(np.percentile(np.abs(tex[inner]), 99)), TEX_FLOOR))
    ao = 1 - _sky_view(zs, g.cx, g.cy, radius)
    ao = np.clip(ao / max(float(np.percentile(ao[inner], 99)), AO_FLOOR), 0, 1)
    return np.tanh(GAIN * (W_HS * hs + W_TEX * tex - W_AO * ao)) / math.tanh(GAIN)


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
        self.sigma = max(SIGMA_MIN_M, SIGMA_PAPER_MM * (x1 - x0) / self.map_w)  # the level rule (_detail_level)
        self.tex_sigma = max(CRISP_SIGMA_M, CRISP_SIGMA_MM * (x1 - x0) / self.map_w)  # the shade's smoothing
        self.tex_radius = min(SVF_MAX_M, max(SVF_MIN_M, SVF_MM * (x1 - x0) / self.map_w))  # the sky view's reach
        self.contour_sigma = max(CONTOUR_SIGMA_M, CONTOUR_SIGMA_MM * (x1 - x0) / self.map_w)
        self.gate_sigma = min(GATE_SIGMA_M, max(GATE_SIGMA_MIN_M, GATE_SIGMA_MM * (x1 - x0) / self.map_w))
        self.probe_grid = _grid(self.extent, max(125.0, max(x1 - x0, y1 - y0) / 400), self.gate_sigma)
        # every cell the map could need, whatever the dataset and level: fixed before any request is made
        spacings = [_spacing(ds, L) for ds in HEIGHT0 for L in range(NOMINAL_LEVELS[ds])]
        lat_a, lat_b, lon_a, lon_b = _lonlat_box(self.extent, crs, max(max(self.probe_pad(s), self.detail_pad(s))
                                                                       for s in spacings))
        # Cells and mosaics run west to east from -180 to 180: a map that reaches the 180th meridian gets no relief
        # (its frame wraps from 180 to -180, and there is no cell east of 179)
        self.near_180 = lon_b >= 180 or lon_a < -180 or lon_b - lon_a > 180
        self.cells = set() if self.near_180 else set(_cells((lat_a, lat_b, lon_a, lon_b)))

    def detail_grid(self, spacing):
        """The work grid of the detail shade. Its margin holds the widest of the level rule's smoothing, the sky view's
        radius plus the shade's smoothing, and the contours' smoothing: the horizon of every map node lies on it, at
        nodes that the Gaussian's edge does not reach."""
        reach = max(self.sigma, self.tex_radius / 4 + self.tex_sigma, self.contour_sigma)
        return _grid(self.extent, max((self.extent[1] - self.extent[0]) / self.shape[1], spacing / 2), reach)

    def probe_pad(self, spacing):
        """Padding of the probe box: the outermost node, plus two pixels for the bilinear samples."""
        return self.probe_grid.reach + 2 * spacing

    def detail_pad(self, spacing):
        """Padding of the detail box: the outermost node, the ground filter, two pixels for the cubic samples."""
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
    s = _shade(_gauss(raw, job.gate_sigma, g), g.cx, g.cy)
    inner = (slice(g.m, g.m + g.ny), slice(g.m, g.m + g.nx))
    land = np.abs(raw[inner]) > LAND_EPS_M  # unsmoothed, so the sea does not bleed into the land
    if land.mean() < LAND_MIN_FRAC:
        return None
    return float((np.abs(s[inner][land]) > GATE_S).mean())


def _detail_level(job, dataset, n_levels):
    """The coarsest level whose N-S spacing is within the map's smoothing (sigma), or 0."""
    return max([L for L in range(n_levels) if _spacing(dataset, L) <= job.sigma] or [0])


def _level_blocks(job, dataset, headers, L, pad):
    """[(cell, L, br, bc)]: the blocks of level L that cover the map padded by pad."""
    box, cells = job.box(pad)
    return [(cell, L, br, bc) for cell, br, bc in _blocks(dataset, headers, cells, L, box)]


def _uncached(dataset, blocks, root):
    return [b for b in blocks if not os.path.exists(_block_path(root, dataset, *b))]


def _plan(job, dataset, headers, n_levels, root):
    """(level, its blocks): the detail level, one level coarser while the blocks still to download exceed
    BUDGET_BYTES; the coarsest level is always within budget."""
    L = _detail_level(job, dataset, n_levels)
    while True:
        blocks = _level_blocks(job, dataset, headers, L, job.detail_pad(_spacing(dataset, L)))
        todo = _uncached(dataset, blocks, root)
        if L == n_levels - 1 or sum(_block_bytes(headers, *b) for b in todo) <= BUDGET_BYTES:
            return L, blocks
        L += 1


def _download(dataset, headers, blocks, root):
    """Every block of the set, in sorted order: S3 sees the requests in order, and the maps of a print come in an
    order that says something (an atlas puts home first, then the trips by date), so the order is the set's own.
    Cached blocks are validated here too, so a damaged one is fetched again in its place in the order."""
    for cell, L, br, bc in sorted(set(blocks)):
        _block(dataset, cell, headers[cell], L, br, bc, root)


def _detail(job, dataset, headers, L, root):
    """(shade, smooth, z, z_smooth) of the map from level L: the mosaic and its city ground filter sampled on the map
    grid, the crisp shade of the surface, the smooth shade of the city ground, and both surfaces' heights smoothed for
    contour lines, each cropped to the map and resized (bicubic) to the output raster."""
    g = job.detail_grid(_spacing(dataset, L))
    box, cells = job.box(job.detail_pad(_spacing(dataset, L)))
    return _rasters(job, _mosaic(dataset, headers, cells, L, box, root), g)


def _rasters(job, mos, g):
    """(shade, smooth, z, z_smooth) of the map on the work grid g from the mosaic mos (see _detail)."""
    mos = Mosaic(np.asarray(mos.z, dtype=np.float64), mos.lon0, mos.dlon, mos.lat0, mos.dlat)
    raw, urb = _sample_cubic(mos, *_nodes(job.extent, job.crs, g), _ground_urban(mos).z)
    with ThreadPoolExecutor(4) as ex:  # four independent rasters; numpy lets go of the GIL in the heavy steps
        shade = ex.submit(lambda: _blend(_gauss(raw, job.tex_sigma, g), g, job.tex_radius))
        smooth = ex.submit(lambda: _blend(_gauss(urb, max(job.tex_sigma, URBAN_SHADE_SIGMA_M), g), g, job.tex_radius))
        z = ex.submit(_gauss, raw, job.contour_sigma, g)
        z_smooth = ex.submit(_gauss, urb, job.contour_sigma, g)
        shade, smooth, z, z_smooth = shade.result(), smooth.result(), z.result(), z_smooth.result()
    H, W = job.shape

    def out(a):
        a = np.asarray(a[g.m:g.m + g.ny, g.m:g.m + g.nx], np.float32)
        return np.asarray(Image.fromarray(a).resize((W, H), Image.Resampling.BICUBIC), np.float32)

    return (np.clip(out(shade), -1.0, 1.0), np.clip(out(smooth), -1.0, 1.0), out(z), out(z_smooth))


def prepare_many(maps, log=print, root=None):
    """One Relief (or None) per map of a print. maps: [(extent, crs, map_w_mm)]; nothing else about the print
    (tracks, zones, activities) reaches this function. Any network or decoding failure means no relief anywhere on
    the print and one log line; a map that is too large, flat or mostly water gets None and its own line."""
    log("[relief] terrain from the Copernicus elevation model (Amazon S3 learns the 1-degree cells around the map and, "
        "where the land is hilly, the blocks of about 30 km that cover it and about 2 km around it; never your routes)")
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
        # Blocks are requested in two sorted sweeps, the probe of every map and then the detail of the maps that
        # pass it; the steps of each map read only the cache.
        probe = n_levels - 1
        s = _spacing(dataset, probe)
        _download(dataset, headers, [b for j in jobs for b in _level_blocks(j, dataset, headers, probe, j.probe_pad(s))],
                  root)
        passed = []
        for job in jobs:  # the probe: the coarsest level only; detail blocks are requested after it passes
            share = _gate(job, dataset, headers, probe, root)
            if share is None:
                say(job.i, "mostly water here: no relief")
            elif share < GATE_SHARE:
                say(job.i, "flat here: no relief")
            else:
                passed.append((job, share))
        plans = [(job, share) + _plan(job, dataset, headers, n_levels, root) for job, share in passed]
        detail = {b for *_, blocks in plans for b in blocks}  # maps that share a block download it once
        todo = _uncached(dataset, detail, root)
        if todo:
            size = sum(_block_bytes(headers, *b) for b in todo)
            log(f"[relief] downloading {len(todo)} blocks, {size / 2 ** 20:.1f} MB (one time, cached)")
        _download(dataset, headers, detail, root)
        for job, share, L, _ in plans:
            shade, smooth, z, z_smooth = _detail(job, dataset, headers, L, root)
            out[job.i] = Relief(shade, dataset, L, share, smooth=smooth, z=z, z_smooth=z_smooth)
    except _Unsupported:  # the same on every run: no point in running it again
        log("[relief] skipped: the elevation data does not cover this map as it is; the poster is drawn without relief")
        return [None] * len(maps)
    except (ReliefError, OSError, zlib.error, ValueError) as ex:
        log(f"[relief] skipped ({type(ex).__name__}): the poster is drawn without relief; run the same command again "
            f"later to add it")
        return [None] * len(maps)
    return out
