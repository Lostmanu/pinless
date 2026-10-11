"""Terrain relief: the Copernicus DEM reader, the geometry that puts it on the map, the gate, the drawing and the
privacy contract (what may be requested, logged and cached). All offline: a fake server serves synthetic COGs."""
import io
import math

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from _cog_fixture import fake_server, levels_of, write_cog
from pinless import relief, render

H30, H90 = 480, 240  # small cells: 230 m and 460 m pixels at full resolution, so the files stay tiny


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """A test that forgets its fake server fails instead of reaching AWS; nothing is written outside tmp_path."""
    def no_network(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(relief, "_fetch", no_network)
    monkeypatch.setenv("PINLESS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": H30, "glo90": H90})


def _cell_grid(cell, h0, fn, dataset="glo30"):
    """fn(lon, lat) at the full-resolution pixel centres of a cell."""
    N, W = cell
    w0 = int(h0 / relief._band(N))
    lon = W + np.arange(w0) / w0
    lat = N + 1 - np.arange(h0) / h0
    LON, LAT = np.meshgrid(lon, lat)
    return fn(LON, LAT).astype(np.float32)


def _cog(cell, fn, h0=H30, n_levels=4, tile=64, **kw):
    return write_cog(cell=cell, levels=levels_of(_cell_grid(cell, h0, fn), n_levels), tile=tile, **kw)


def _serve(monkeypatch, files, dataset="glo30"):
    calls = []
    monkeypatch.setattr(relief, "_fetch", fake_server({relief.url(dataset, c): b for c, b in files.items()}, calls))
    return calls


def _utm_extent(lat, lon, w_m, h_m):
    crs = render.utm_crs(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return (x - w_m / 2, x + w_m / 2, y - h_m / 2, y + h_m / 2), crs


def test_predictor3_rows_decode_to_a_known_vector():
    # encoded by imagecodecs.floatpred_encode, independent of the fixture writer
    out = relief._unpredict(bytes.fromhex("3f01408000000000c0841c1a8610f000"), rows=2, cols=2)
    assert out.dtype == np.float32 and out.tolist() == [[1.0, 2.0], [-3.5, 1000.25]]


def test_reader_reads_every_level_of_a_synthetic_cog(monkeypatch, tmp_path):
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": 40, "glo90": 40})
    rng = np.random.default_rng(5)
    a0 = (-450.0 + rng.random((40, 40)) * 1950.0).astype(np.float32)  # negative, fractional, around 1e3
    a0[a0 == 0] = 0.5
    levels = levels_of(a0, 3)  # 40, 20, 10: edge tiles of 16 px are partial
    data = write_cog(cell=(40, -4), levels=levels, tile=16)
    _serve(monkeypatch, {(40, -4): data})
    hdr = relief.read_header(data[:relief.HEADER_BYTES], (40, -4))
    assert [(lv.w, lv.h) for lv in hdr.levels] == [(40, 40), (20, 20), (10, 10)]
    for L, want in enumerate(levels):
        lv = hdr.levels[L]
        got = np.full((lv.h, lv.w), np.nan, np.float32)
        for br in range(math.ceil(lv.h / lv.th)):
            for bc in range(math.ceil(lv.w / lv.tw)):
                z = relief._block("glo30", (40, -4), hdr, L, br, bc, str(tmp_path / "dem"))
                got[br * lv.th:br * lv.th + z.shape[0], bc * lv.tw:bc * lv.tw + z.shape[1]] = z
        assert np.array_equal(got, want), f"level {L} bit for bit"
        assert not (got == 0).any(), "no padding zero"
    im = Image.open(io.BytesIO(data))
    for L, want in enumerate(levels):
        im.seek(L)
        try:
            pil = np.asarray(im, dtype=np.float32)
        except OSError:
            pytest.skip("this Pillow cannot decode predictor 3")
        assert np.array_equal(pil, want), f"Pillow agrees on level {L}"


def _refusals():
    rng = np.random.default_rng(2)
    a0 = (100 + rng.random((40, 40)) * 500).astype(np.float32)
    good = levels_of(a0, 3)
    spike = [a.copy() for a in good]
    for a in spike:  # in every level, so the probe meets it too
        a[1, 1] = 1e5
    return {
        "big-endian": dict(levels=good, endian=">"),
        "compression 5": dict(levels=good, compression=5),
        "predictor 2": dict(levels=good, predictor=2),
        "int16 samples": dict(levels=good, bits=16, sample_format=2),
        "PixelIsArea": dict(levels=good, raster_type=1),
        "tiepoint off by 0.01": dict(levels=good, tiepoint=(0, 0, 0, -4.01, 41.0, 0)),
        "overview not half the size": dict(levels=[good[0], a0[:19, :19]]),
        "IFD after 16 KB": dict(levels=good, ifd_at_end=True),
        "a value of 1e5": dict(levels=spike),
    }


@pytest.mark.parametrize("case", sorted(_refusals()))
def test_reader_refuses_what_it_cannot_decode(monkeypatch, tmp_path, case):
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": 40, "glo90": 40})
    data = write_cog(cell=(40, -4), tile=16, **_refusals()[case])
    _serve(monkeypatch, {(40, -4): data})
    with pytest.raises(relief.ReliefError):
        hdr = relief.read_header(data[:relief.HEADER_BYTES], (40, -4))
        for br in range(3):
            for bc in range(3):
                relief._block("glo30", (40, -4), hdr, 0, br, bc, str(tmp_path / "dem"))
    assert not list(tmp_path.rglob("*.zz")), "nothing that failed is cached"
    logs = []
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    assert relief.prepare_many([(extent, crs, 184.5)], log=logs.append) == [None]
    assert "skipped (ReliefError)" in logs[-1]
    assert not list(tmp_path.rglob("*.zz")), "only validated files are cached"
    assert bool(list(tmp_path.rglob("header.bin"))) == (case == "a value of 1e5"), "a header is kept once it parses"


def test_cell_keys_and_urls():
    assert relief.url("glo30", relief.cell_of(40.4, -3.7)) == (
        "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N40_00_W004_00_DEM/"
        "Copernicus_DSM_COG_10_N40_00_W004_00_DEM.tif")
    assert relief.cell_name(relief.cell_of(-33.87, 151.21)) == "S34_00_E151_00"
    assert relief.cell_name(relief.cell_of(-0.5, -0.5)) == "S01_00_W001_00"
    assert relief.cell_name(relief.cell_of(0.5, 0.5)) == "N00_00_E000_00"
    u90 = relief.url("glo90", (40, -4))
    assert u90.startswith("https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N40_00_W004_00_DEM/")
    for u in (relief.url("glo30", (40, -4)), u90):
        assert "?" not in u and "#" not in u


def _sample_plane(monkeypatch, tmp_path, cells, fn, lat, lon, levels, size_m=(30000, 24000)):
    """The plane fn sampled through each level on a UTM grid around (lat, lon): (sampled, true) per level."""
    files = {c: _cog(c, fn) for c in cells}
    _serve(monkeypatch, files)
    root = str(tmp_path / "dem")
    headers = {c: relief.read_header(b[:relief.HEADER_BYTES], c) for c, b in files.items()}
    extent, crs = _utm_extent(lat, lon, *size_m)
    out = []
    for L in levels:
        spacing = relief.M_PER_DEG_LAT / H30 * 2 ** L
        g = relief._grid(extent, 400.0, 0.0)
        box = relief._lonlat_box(extent, crs, g.reach + 2 * spacing)
        assert set(relief._cells(box)) <= set(cells)
        mos = relief._mosaic("glo30", headers, relief._cells(box), L, box, root)
        lo, la = relief._nodes(extent, crs, g)
        out.append((relief._sample(mos, lo, la), fn(lo, la)))
    return out


def test_overview_pixels_are_centred_on_their_block(monkeypatch, tmp_path):
    def plane(lon, lat):
        return 1000 * (lon + 4) + 500 * (lat - 40)

    for L, (got, want) in enumerate(_sample_plane(monkeypatch, tmp_path, [(40, -4)], plane, 40.5, -3.5, (0, 1, 2))):
        assert np.abs(got - want).max() < 1e-3, f"level {L}: the overview is not where its block is"


def test_mosaic_across_cells_and_widths_has_no_seam(monkeypatch, tmp_path):
    def plane(lon, lat):
        return 1000 * (lon - 5) + 500 * (lat - 49) + 10

    cells = [(49, 5), (49, 6), (50, 5), (50, 6)]  # N50 has 2/3 the columns of N49
    assert relief._dims("glo30", (50, 5), 0)[0] * 1.5 == relief._dims("glo30", (49, 5), 0)[0]
    for L, (got, want) in enumerate(_sample_plane(monkeypatch, tmp_path, cells, plane, 50.0, 6.0, (0, 1))):
        assert np.abs(got - want).max() < 1e-3, f"level {L}: a seam between cells"


def _steep(lon, lat):
    """Hills about 7 km apart and 800 m high: well above the gate."""
    return 600 + 400 * np.sin(2 * np.pi * lon / 0.08) * np.cos(2 * np.pi * lat / 0.06)


A4_MAP = 210 * (1 - 36 / 297)  # the editorial map width on A4 paper, mm


def _levels_of_ranges(files, calls, h0=H30, dataset="glo30"):
    """{(url, level, br, bc)} of block requests, and the header requests, from recorded (url, first, last)."""
    by_offset = {}
    for cell, data in files.items():
        hdr = relief.read_header(data[:relief.HEADER_BYTES], cell, dataset)
        for L, lv in enumerate(hdr.levels):
            per_row = math.ceil(lv.w / lv.tw)
            for i, (o, n) in enumerate(zip(lv.offsets, lv.counts)):
                by_offset[(relief.url(dataset, cell), o)] = (L, i // per_row, i % per_row, o + n - 1)
    blocks, headers = [], []
    for u, first, last in calls:
        if first == 0:
            assert last == relief.HEADER_BYTES - 1
            headers.append(u)
        else:
            L, br, bc, end = by_offset[(u, first)]  # every range starts at a TileOffsets entry
            assert last == end
            blocks.append((u, L, br, bc))
    return blocks, headers


def test_requests_are_the_full_rectangle_and_ignore_the_tracks(monkeypatch, tmp_path):
    import inspect
    assert set(inspect.signature(relief.prepare_many).parameters) == {"maps", "log", "root"}
    files = {(40, -4): _cog((40, -4), _steep)}
    calls = _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 14000, 12000)
    logs = []
    out = relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append)
    assert out[0] is not None
    blocks, headers = _levels_of_ranges(files, calls)
    import re
    said = [m.group(1) for line in logs if (m := re.fullmatch(r"\[relief\] downloading (\d+) blocks, \d+\.\d MB "
                                                              r"\(one time, cached\)", line))]
    assert said == [str(len([b for b in blocks if b[1] == 0]))], "the log counts exactly the detail blocks requested"
    assert headers and blocks
    for u, L in {(u, L) for u, L, _, _ in blocks}:
        got = {(br, bc) for u_, L_, br, bc in blocks if (u_, L_) == (u, L)}
        rows, cols = [b[0] for b in got], [b[1] for b in got]
        want = {(r, c) for r in range(min(rows), max(rows) + 1) for c in range(min(cols), max(cols) + 1)}
        assert got == want, f"level {L}: the full rectangle, no holes"
    assert len({b for b in blocks if b[1] == 0}) > 1, "setup: the map spans several blocks"


def _hill(lat_p, lon_p, height=800.0, r_m=2500.0):
    def fn(lon, lat):
        dx = (lon - lon_p) * relief.M_PER_DEG_LON * math.cos(math.radians(lat_p))
        dy = (lat - lat_p) * relief.M_PER_DEG_LAT
        return 500 + height * np.exp(-(dx ** 2 + dy ** 2) / (2 * r_m ** 2))
    return fn


def test_relief_is_north_up_and_lit_from_the_northwest(monkeypatch, tmp_path):
    lat_p, lon_p = 40.52, -3.47
    files = {(40, -4): _cog((40, -4), _hill(lat_p, lon_p))}
    _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 16000, 13000)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert rel is not None and rel.shade.shape == relief.raster_shape(A4_MAP, extent) and rel.dataset == "glo30"
    px, py = Transformer.from_crs(4326, crs, always_xy=True).transform(lon_p, lat_p)
    # the resampled DEM peaks on the node of the peak
    job = relief._Map(0, extent, crs, A4_MAP)
    g = job.detail_grid(relief._spacing("glo30", rel.level))
    box, cells = job.box(job.detail_pad(relief._spacing("glo30", rel.level)))
    root = str(tmp_path / "cache" / "dem")
    headers = {c: relief._header("glo30", c, root) for c in cells}
    z = relief._sample(relief._mosaic("glo30", headers, cells, rel.level, box, root), *relief._nodes(extent, crs, g))
    r, c = np.unravel_index(np.argmax(z), z.shape)
    nx_, ny_ = extent[0] + (c - g.m + 0.5) * g.cx, extent[3] - (r - g.m + 0.5) * g.cy
    assert abs(nx_ - px) <= g.cx and abs(ny_ - py) <= g.cy
    # lit from the north-west: brightest on that flank, darkest on the opposite one; north is up
    H, W = rel.shade.shape
    to_xy = lambda rc: (extent[0] + (rc[1] + 0.5) / W * (extent[1] - extent[0]),
                        extent[3] - (rc[0] + 0.5) / H * (extent[3] - extent[2]))
    bx, by = to_xy(np.unravel_index(np.argmax(rel.shade), rel.shade.shape))
    dx_, dy_ = to_xy(np.unravel_index(np.argmin(rel.shade), rel.shade.shape))
    assert bx < px and by > py, "the most lit slope faces north-west"
    assert dx_ > px and dy_ < py, "the darkest slope faces south-east"
    assert rel.shade.max() > 0.1 and rel.shade.min() < -0.1


def test_flat_ground_has_shade_exactly_zero(monkeypatch, tmp_path):
    files = {(40, -4): _cog((40, -4), lambda lon, lat: np.full_like(lon, 650.0))}
    _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    job = relief._Map(0, extent, crs, A4_MAP)
    root = str(tmp_path / "dem")
    headers = {c: relief._header("glo30", c, root) for c in job.cells}
    shade = relief._detail(job, "glo30", headers, 0, root)
    assert shade.shape == job.shape and not shade.any(), "flat ground: exactly 0"


@pytest.mark.parametrize("terrain", ["flat", "gentle", "steep", "island"])
def test_flat_land_is_skipped_after_the_probe_only(monkeypatch, tmp_path, terrain):
    def island(lon, lat):  # 0 m sea and one steep island of one coarsest-level pixel: far less than 2% of the map
        z = np.zeros_like(lon)
        r, c = int((41 - 40.5) * H30) // 8 * 8, int((-3.5 + 4) * H30) // 8 * 8
        z[r:r + 8, c:c + 8] = 300.0
        return z

    fns = {"flat": lambda lon, lat: np.full_like(lon, 650.0), "gentle": lambda lon, lat: 650 + 20 * (lon + 4),
           "steep": _steep}
    if terrain == "island":
        a0 = island(*np.meshgrid(-4 + np.arange(H30) / H30, 41 - np.arange(H30) / H30))
        files = {(40, -4): write_cog(cell=(40, -4), levels=levels_of(a0, 4), tile=64)}
    else:
        files = {(40, -4): _cog((40, -4), fns[terrain])}
    calls = _serve(monkeypatch, files)
    logs = []
    size = (40000, 34000) if terrain == "island" else (12000, 10000)
    extent, crs = _utm_extent(40.5, -3.5, *size)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append)
    blocks, _ = _levels_of_ranges(files, calls)
    if terrain == "steep":
        assert rel is not None and rel.shade.shape == relief.raster_shape(A4_MAP, extent) and rel.share >= 0.02
        assert any(L == 0 for _, L, _, _ in blocks)
        return
    assert rel is None
    assert logs[-1] == ("[relief] mostly water here: no relief" if terrain == "island" else "[relief] flat here: no relief")
    assert {L for _, L, _, _ in blocks} == {3}, "only headers and the probe level, never a detail block"


REAL_FETCH = relief._fetch  # the real one: the autouse fixture replaces relief._fetch for every test


class _Response:
    def __init__(self, status, body, content_range):
        self.status, self._body = status, body
        self.headers = {"Content-Range": content_range} if content_range else {}

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    """S3 as relief._OPENER sees it: 206 with Content-Range for files {url: bytes}, 404 otherwise, and one kind of
    failure wherever when(url, first) says. Checks every request against the privacy contract."""

    def __init__(self, files, failure=None, when=lambda url, first: True):
        self.files, self.failure, self.when, self.calls = files, failure, when, []

    def open(self, req, timeout=None):
        import urllib.error
        from urllib.parse import urlparse
        from pinless import area
        u = req.full_url
        assert urlparse(u).hostname in ("copernicus-dem-30m.s3.amazonaws.com", "copernicus-dem-90m.s3.amazonaws.com")
        assert urlparse(u).scheme == "https" and not urlparse(u).query
        assert dict(req.header_items()).keys() == {"User-agent", "Range"}
        assert req.get_header("User-agent") == area.USER_AGENT and timeout == relief.TIMEOUT_S
        first, last = map(int, req.get_header("Range")[len("bytes="):].split("-"))
        self.calls.append((u, first, last))
        data = self.files.get(u)
        if data is None:
            raise urllib.error.HTTPError(u, 404, "Not Found", {}, None)
        end = min(last, len(data) - 1)
        body, cr = data[first:end + 1], f"bytes {first}-{end}/{len(data)}"
        fail = self.failure if self.failure and self.when(u, first) else None
        if fail == "URLError":
            raise urllib.error.URLError("unreachable")
        if fail == "TimeoutError":
            raise TimeoutError("timed out")
        if fail in ("HTTP 503", "HTTP 403", "HTTP 301"):
            raise urllib.error.HTTPError(u, int(fail[5:]), "no", {}, None)
        if fail == "200 instead of 206":
            return _Response(200, data, None)
        if fail == "wrong Content-Range":
            cr = f"bytes {first + 1}-{end}/{len(data)}"
        if fail == "short body":
            body = body[:-1]
        return _Response(206, body, cr)


def _detail_offsets(data, cell=(40, -4)):
    return set(relief.read_header(data[:relief.HEADER_BYTES], cell).levels[0].offsets)


def test_a_redirect_is_refused():
    import email.message
    import urllib.error
    import urllib.request
    import urllib.response

    class Moved(urllib.request.HTTPSHandler):  # replaces the real HTTPS handler: nothing leaves the machine
        def https_open(self, req):
            hdrs = email.message.Message()
            hdrs["Location"] = "https://elsewhere.example/x.tif"
            r = urllib.response.addinfourl(io.BytesIO(b""), hdrs, req.full_url, 301)
            r.msg = "Moved Permanently"
            return r

    opener = urllib.request.build_opener(relief._NoRedirect, Moved)
    with pytest.raises(urllib.error.HTTPError) as ex:
        opener.open(urllib.request.Request(relief.url("glo30", (40, -4))), timeout=1)
    assert ex.value.code == 301
    assert any(isinstance(h, relief._NoRedirect) for h in relief._OPENER.handlers)
    assert not any(type(h) is urllib.request.HTTPRedirectHandler for h in relief._OPENER.handlers)
    assert not any(isinstance(h, urllib.request.HTTPCookieProcessor) for h in relief._OPENER.handlers)


FAILURES = ["URLError", "TimeoutError", "HTTP 503", "HTTP 403", "HTTP 301", "200 instead of 206",
            "wrong Content-Range", "short body", "garbage zlib body", "out-of-range value"]


@pytest.mark.parametrize("failure", FAILURES)
def test_a_failing_server_draws_the_poster_without_relief(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)
    good = _cog((40, -4), _steep)
    detail = _detail_offsets(good)
    served = good
    if failure == "garbage zlib body":
        hdr = relief.read_header(good[:relief.HEADER_BYTES], (40, -4))
        served = bytearray(good)
        for o, n in zip(hdr.levels[0].offsets, hdr.levels[0].counts):
            served[o:o + n] = bytes((7 * i) % 251 for i in range(n))
        served = bytes(served)
    elif failure == "out-of-range value":
        a0 = _cell_grid((40, -4), H30, _steep)
        a0[150:340, 150:340] = 1e5  # under the whole map window; the probe level is kept clean
        lv = levels_of(a0, 4)
        lv[3] = levels_of(_cell_grid((40, -4), H30, _steep), 4)[3]
        served = write_cog(cell=(40, -4), levels=lv, tile=64)
    u = relief.url("glo30", (40, -4))
    opener = _Opener({u: served}, failure, when=lambda url, first: first in detail)
    monkeypatch.setattr(relief, "_OPENER", opener)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    logs = []
    assert relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append) == [None]
    want = {"URLError": "URLError", "TimeoutError": "TimeoutError"}.get(failure, "HTTPError" if "HTTP" in failure
                                                                       else "ReliefError")
    assert logs[-1].startswith(f"[relief] skipped ({want}):"), logs[-1]
    text = "\n".join(logs)
    for leak in ("Copernicus_DSM", ".tif", "N40", "W004", "40.", "-3.", "amazonaws"):
        assert leak not in text
    if failure == "HTTP 503":
        assert len([c for c in opener.calls if c[1] in detail]) == 2, "a 5xx is tried once more"
    root = tmp_path / "cache" / "dem"
    cached = {p.name for p in root.rglob("*.zz")}
    assert not any(n.startswith("L0_") for n in cached), "nothing is cached for the failed blocks"
    assert (root / "glo30" / "N40_00_W004_00" / "header.bin").exists()
    if failure == "out-of-range value":  # this fake file has another layout than the good one: start afresh
        import shutil
        shutil.rmtree(root)
    working = _Opener({u: good})
    monkeypatch.setattr(relief, "_OPENER", working)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert rel is not None and any(first in detail for _, first, _ in working.calls), "fetched later"


def _glo90_world(monkeypatch):
    """GLO-30 has (40, -4) only; GLO-90 has (40, -4) and (40, -3); (40, -5) is sea in both."""
    files = {relief.url("glo30", (40, -4)): _cog((40, -4), _steep),
             relief.url("glo90", (40, -4)): _cog((40, -4), _steep, h0=H90, n_levels=3),
             relief.url("glo90", (40, -3)): _cog((40, -3), _steep, h0=H90, n_levels=3)}
    calls = []
    monkeypatch.setattr(relief, "_fetch", fake_server(files, calls))
    return files, calls


def test_glo90_only_for_a_land_cell_missing_from_glo30(monkeypatch, tmp_path):
    _, calls = _glo90_world(monkeypatch)
    m1 = _utm_extent(40.5, -3.5, 12000, 10000)
    m2 = _utm_extent(40.5, -2.5, 12000, 10000)
    out = relief.prepare_many([(m1[0], m1[1], 60.0), (m2[0], m2[1], 60.0)], log=lambda *_: None)
    assert [r.dataset for r in out] == ["glo90", "glo90"], "one dataset, one notice for the whole print"
    assert "WorldDEM™-90" in relief.notice(out[0].dataset, "en")
    assert relief.notice("glo30", "es").startswith("Relieve: produced using Copernicus WorldDEM-30 © DLR")
    assert "™" not in relief.notice("glo30", "en")


def test_a_cell_missing_from_both_datasets_is_sea(monkeypatch, tmp_path):
    files = {relief.url("glo30", (40, -4)): _cog((40, -4), _steep)}
    calls = []
    monkeypatch.setattr(relief, "_fetch", fake_server(files, calls))
    extent, crs = _utm_extent(40.5, -3.93, 12000, 10000)  # its box reaches into (40, -5), absent from both
    job = relief._Map(0, extent, crs, A4_MAP)
    assert (40, -5) in job.cells
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert rel is not None and rel.dataset == "glo30"
    sea = [c for c in calls if "W005" in c[0]]
    assert sea and all(first == 0 for _, first, _ in sea), "only the header, on both datasets: no block requests"
    assert {u.split(".s3")[0] for u, _, _ in sea} == {"https://copernicus-dem-30m", "https://copernicus-dem-90m"}


def test_a_glo30_server_error_never_switches_to_glo90(monkeypatch, tmp_path):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)
    files = {relief.url("glo30", (40, -4)): _cog((40, -4), _steep),
             relief.url("glo90", (40, -4)): _cog((40, -4), _steep, h0=H90, n_levels=3)}
    opener = _Opener(files, "HTTP 503", when=lambda url, first: "30m" in url)
    monkeypatch.setattr(relief, "_OPENER", opener)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    logs = []
    assert relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append) == [None]
    assert "skipped (HTTPError)" in logs[-1]
    assert not any("90m" in u for u, _, _ in opener.calls), "GLO-90 is never asked because GLO-30 failed"


def test_cache_is_reused_offline_and_validated(monkeypatch, tmp_path):
    import re
    files = {(40, -4): _cog((40, -4), _steep)}
    calls = _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (first,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)

    def offline(*args):
        raise relief.ReliefError("offline")

    monkeypatch.setattr(relief, "_fetch", offline)
    (again,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert again is not None and np.array_equal(first.shade, again.shade), "the cache alone draws the same relief"
    root = tmp_path / "cache"
    blocks = sorted(root.rglob("L0_*.zz"))
    victim = blocks[0]
    victim.write_bytes(victim.read_bytes()[:-20])
    calls.clear()
    monkeypatch.setattr(relief, "_fetch", fake_server({relief.url("glo30", c): b for c, b in files.items()}, calls))
    (third,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert np.array_equal(first.shade, third.shade)
    assert len(calls) == 1, "only the truncated block is fetched again"
    for p in root.rglob("*"):
        if p.is_file():
            rel_ = p.relative_to(root).as_posix()
            assert re.fullmatch(r"dem/glo(30|90)/[NS]\d\d_00_[EW]\d{3}_00/(header\.bin|absent|L\d+_\d+_\d+\.zz)",
                                rel_), rel_


def test_level_budget_and_size_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": 3600, "glo90": 1200})  # real cells: the level rule
    for size in ("A4", "A3", "50x70"):
        wmm = render.SIZES[size][0] * (1 - 36 / 297)
        extent, crs = _utm_extent(40.42, -3.70, 12000, 12000 * 222 / 261)
        assert relief._detail_level(relief._Map(0, extent, crs, wmm), "glo30", 4) == 0, size
    extent, crs = _utm_extent(40.42, -3.70, 10000, 8000)
    assert relief._detail_level(relief._Map(0, extent, crs, 60.0), "glo30", 4) == 1, "a 60 mm atlas map"
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": H30, "glo90": H90})
    files = {(40, -4): _cog((40, -4), _steep)}
    _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (fine,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None, root=str(tmp_path / "a"))
    monkeypatch.setattr(relief, "BUDGET_BYTES", 2000)
    logs = []
    (coarse,) = relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append, root=str(tmp_path / "b"))
    assert fine.level == 0 and coarse.level > 0, "over the budget: a coarser level"
    assert coarse.shade.shape == fine.shade.shape
    big, crs = _utm_extent(40.5, -3.5, 400000, 400000)
    logs = []
    assert relief.prepare_many([(big, crs, A4_MAP)], log=logs.append) == [None]
    assert logs[-1] == "[relief] area too large for relief: drawn without it"


def test_a_sample_outside_the_elevation_data_raises_instead_of_clamping():
    mos = relief.Mosaic(np.zeros((4, 4), np.float32), lon0=-3.0, dlon=0.1, lat0=41.0, dlat=0.1)
    assert relief._sample(mos, np.array([-2.7]), np.array([40.7])).tolist() == [0.0]
    for lon, lat in ((-3.01, 40.9), (-2.69, 40.9), (-2.9, 41.01), (-2.9, 40.69)):
        with pytest.raises(relief.ReliefError):
            relief._sample(mos, np.array([lon]), np.array([lat]))
