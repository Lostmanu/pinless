"""Synthetic Copernicus-like Cloud-Optimized GeoTIFFs, written in memory (no committed binary), and a fake server.

write_cog writes a classic TIFF: the header, the IFDs and their arrays first, then the tiles, each predictor-3
encoded (byte planes, most significant first, then byte differences along the row) and zlib-compressed, the edge
tiles zero-padded. Every argument can be bent to build files the reader must refuse.
"""
import struct
import zlib

import numpy as np


def overview(a):
    """The next level: 2x2 means."""
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def levels_of(a0, n):
    out = [np.asarray(a0, dtype=np.float32)]
    for _ in range(n - 1):
        out.append(overview(out[-1]))
    return out


def _encode(tile):
    rows, cols = tile.shape
    planes = tile.astype(">f4").view(np.uint8).reshape(rows, cols, 4).transpose(0, 2, 1).reshape(rows, 4 * cols)
    diff = planes.copy()
    diff[:, 1:] = planes[:, 1:] - planes[:, :-1]  # uint8 arithmetic wraps, as the predictor does
    return zlib.compress(diff.tobytes())


def write_cog(cell=(40, -4), levels=None, tile=16, endian="<", compression=8, predictor=3, sample_format=3,
              raster_type=2, tiepoint=None, ifd_at_end=False, bits=32, scale=None, tags=None, ifd_loop=False):
    """bytes of a GeoTIFF for a cell (south, west) with one IFD per array in `levels` (full resolution first).

    tags={(level, tag): (type, values)} replaces (or adds) an entry of that level's IFD; ifd_loop points the last
    IFD back at the first."""
    N, W = cell
    levels = [np.asarray(a, dtype=np.float32) for a in levels]
    h0, w0 = levels[0].shape
    tiles = []  # per level: compressed tiles, row-major
    for a in levels:
        h, w = a.shape
        out = []
        for br in range(-(-h // tile)):
            for bc in range(-(-w // tile)):
                t = np.zeros((tile, tile), np.float32)
                part = a[br * tile:(br + 1) * tile, bc * tile:(bc + 1) * tile]
                t[:part.shape[0], :part.shape[1]] = part
                out.append(_encode(t))
        tiles.append(out)

    def entries(L, offsets, counts):
        a = levels[L]
        e = [(254, 4, [0 if L == 0 else 1]), (256, 4, [a.shape[1]]), (257, 4, [a.shape[0]]), (258, 3, [bits]),
             (259, 3, [compression]), (277, 3, [1]), (284, 3, [1]), (317, 3, [predictor]), (322, 3, [tile]),
             (323, 3, [tile]), (324, 4, offsets), (325, 4, counts), (339, 3, [sample_format])]
        if L == 0:
            sc = scale or (1 / w0, 1 / h0, 0.0)
            tp = tiepoint or (0.0, 0.0, 0.0, float(W), float(N + 1), 0.0)
            e += [(33550, 12, list(sc)), (33922, 12, list(tp)),
                  (34735, 3, [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, raster_type, 2048, 0, 1, 4326])]
        for (lv, tag), (typ, vals) in (tags or {}).items():
            if lv == L:
                e = [x for x in e if x[0] != tag] + [(tag, typ, list(vals))]
        return sorted(e)

    fmt = {3: "H", 4: "I", 12: "d"}
    size = {3: 2, 4: 4, 12: 8}

    def layout(start, tile_start):
        """([(entries, {tag: array offset})] per IFD, IFD offsets, end) laid out from `start`, tiles from tile_start."""
        out, pos = [], start
        ifd_pos = []
        offs = []
        t = tile_start
        for L in range(len(levels)):
            o = []
            for b in tiles[L]:
                o.append(t)
                t += len(b)
            offs.append(o)
        for L in range(len(levels)):
            e = entries(L, offs[L], [len(b) for b in tiles[L]])
            ifd_pos.append(pos)
            pos += 2 + 12 * len(e) + 4
            arrays = []
            for tag, typ, vals in e:
                n = len(vals) * size[typ]
                if n > 4:
                    arrays.append((tag, pos))
                    pos += n
            out.append((e, dict(arrays)))
        return out, ifd_pos, pos

    if not ifd_at_end:
        _, _, tile_start = layout(8, 0)  # the IFDs do not change size with the offsets they hold
        info, ifd_pos, end = layout(8, tile_start)
    else:
        tile_bytes = sum(len(b) for t in tiles for b in t)
        tile_start = 8
        start = max(8 + tile_bytes, 16384 + 64)
        info, ifd_pos, end = layout(start, tile_start)
    buf = bytearray(max(end, tile_start + sum(len(b) for t in tiles for b in t)))
    buf[:4] = (b"II*\x00" if endian == "<" else b"MM\x00*")
    struct.pack_into(endian + "I", buf, 4, ifd_pos[0])
    for L, ((e, arrays), at) in enumerate(zip(info, ifd_pos)):
        struct.pack_into(endian + "H", buf, at, len(e))
        for k, (tag, typ, vals) in enumerate(e):
            p = at + 2 + 12 * k
            data = struct.pack(endian + f"{len(vals)}{fmt[typ]}", *vals)
            if len(data) <= 4:
                struct.pack_into(endian + "HHI", buf, p, tag, typ, len(vals))
                buf[p + 8:p + 8 + len(data)] = data
            else:
                struct.pack_into(endian + "HHII", buf, p, tag, typ, len(vals), arrays[tag])
                buf[arrays[tag]:arrays[tag] + len(data)] = data
        nxt = ifd_pos[L + 1] if L + 1 < len(ifd_pos) else (ifd_pos[0] if ifd_loop else 0)
        struct.pack_into(endian + "I", buf, at + 2 + 12 * len(e), nxt)
    t = tile_start
    for lv in tiles:
        for b in lv:
            buf[t:t + len(b)] = b
            t += len(b)
    return bytes(buf)


def fake_server(files, calls):
    """A relief._fetch stand-in serving files {url: bytes}: unknown URLs are a 404 (None); every request is
    recorded as (url, first, last)."""
    def fetch(url, first, last):
        calls.append((url, first, last))
        data = files.get(url)
        return None if data is None else data[first:last + 1]
    return fetch
