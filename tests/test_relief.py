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


def _loop_act(i, lat, lon, r=0.012, n=150):
    import datetime as dt
    from pinless.ingest import Activity
    t = np.linspace(0, 2 * np.pi, n)
    return Activity(id=str(i), type="Run", start=dt.datetime(2025, 3, 1 + i, 8, tzinfo=dt.timezone.utc),
                    lonlat=list(zip(lon + r * 1.3 * np.cos(t), lat + r * np.sin(t))), source="x",
                    times=list(np.arange(n) * 10.0))


def _editorial_world(monkeypatch, acts, captured):
    """The editorial CLI without OSM, without stats and without drawing: render_editorial.render records its
    arguments."""
    from pinless import area, ingest, render_editorial, stats
    monkeypatch.setattr(ingest, "read_export", lambda path: (acts, ingest.IngestReport()))
    monkeypatch.setattr(area, "fetch_basemap", lambda bbox, root=None, log=print: "unused")
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (_empty(), _empty(), _empty()))
    monkeypatch.setattr(stats, "compute", lambda *args, **kw: dict(STATS, fav_street=None))

    def spy(*args, **kw):
        captured.update(kw)
        return {"W": 1, "H": 1, "cap": 3.0, "max": 1, "legend": {"mode": "heat"}}

    monkeypatch.setattr(render_editorial, "render", spy)


def _race_world(monkeypatch, captured, texts=None):
    """The race CLI for one public 6 km course in Madrid, without OSM; render_race.render draws for real."""
    import datetime as dt
    from pinless import area, ingest, race, render_race
    from pinless.ingest import Activity
    d = np.linspace(0, 6000, 601)
    xy = np.column_stack([d, d / 3])
    lonlat = [(-3.70 + x / (111320 * math.cos(math.radians(40.42))), 40.42 + y / 110540) for x, y in xy]
    run = Activity(id="r", type="Run", start=dt.datetime(2025, 3, 2, 8, tzinfo=dt.timezone.utc), lonlat=lonlat,
                   source="x", times=list(d * 0.3), ele=list(650 + 20 * np.sin(d / 900)))
    monkeypatch.setattr(ingest, "read_export", lambda path: ([run], ingest.IngestReport()))
    monkeypatch.setattr(area, "fetch_basemap", lambda bbox, root=None, log=print: "unused")
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (_empty(), _empty(), _empty()))
    monkeypatch.setattr(area, "place_names", lambda lat, lon, lang="en": ("Madrid", "Spain"))
    real = render_race.render

    def spy(*args, **kw):
        info = real(*args, **kw)
        captured.update(kw, info=info)
        return info

    monkeypatch.setattr(race.render_race, "render", spy)
    if texts is not None:
        from pinless import atlas, render_editorial
        real_text = render_editorial.place_text

        def spy_text(value, *args, **kw):
            texts.append(str(value))
            return real_text(value, *args, **kw)

        for mod in (render_editorial, atlas, render_race):
            monkeypatch.setattr(mod, "place_text", spy_text)


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
    # through the editorial CLI: other runs and other private zones, the very same requests
    from pinless import cli
    lists = []
    for k, (acts, zones) in enumerate((([_loop_act(i, 40.42, -3.70 + 0.01 * i) for i in range(3)], []),
                                       ([_loop_act(9, 40.45, -3.66, r=0.02)], ["40.45,-3.66,400", "40.43,-3.7,300"]))):
        calls.clear()
        monkeypatch.setenv("PINLESS_CACHE", str(tmp_path / f"cache{k}"))  # each run starts with nothing cached
        captured = {}
        _editorial_world(monkeypatch, acts, captured)
        args = ["x.zip", "--bbox", "40.38,-3.75,40.48,-3.62", "--title", "X", "--relief", "shade", "--out",
                str(tmp_path / "p.png")]
        assert cli.main(args + [z for zone in zones for z in ("--zone", zone)]) == 0
        assert isinstance(captured["relief"], relief.Relief)
        lists.append(list(calls))
    assert lists[0] and lists[0] == lists[1], "the requests depend on the map box only"
    assert any(first != 0 for _, first, _ in lists[0])


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


# --- drawing ----------------------------------------------------------------------------------------------------------

def _lstar(hex_or_rgb):
    """CIE L*, written again here so the test does not trust the code it checks."""
    from matplotlib.colors import to_rgb
    c = np.asarray(to_rgb(hex_or_rgb) if isinstance(hex_or_rgb, str) else hex_or_rgb, dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = float(lin @ [0.2126, 0.7152, 0.0722])
    return 116 * (y ** (1 / 3) if y > 216 / 24389 else (24389 / 27 * y + 16) / 116) - 16


def test_every_style_relief_stays_quieter_than_its_routes():
    from matplotlib.colors import to_rgb
    from pinless.render_editorial import STYLES, relief_colours
    assert len(STYLES) == 12
    for name, S in STYLES.items():
        bg = np.asarray(to_rgb(S["bg"]))
        lb = _lstar(bg)
        route = abs(lb - _lstar(S["heat"][0]))
        if lb >= 50:
            t_shadow, t_light = min(7, 0.30 * route), min(2.5, 0.15 * route)
        else:
            t_shadow, t_light = min(5, 0.5 * lb, 0.30 * route), min(6, 0.30 * route)
        (shadow, a_s), (light, a_l) = relief_colours(S)
        d_s = abs(_lstar((1 - a_s) * bg + a_s * np.asarray(shadow)) - lb)
        d_l = abs(_lstar((1 - a_l) * bg + a_l * np.asarray(light)) - lb)
        assert d_s <= t_shadow + 1e-9 and d_l <= t_light + 1e-9, f"{name}: louder than the rule"
        assert max(d_s, d_l) >= 1, f"{name}: the relief would not show"
        assert a_s <= 0.35 and a_l <= 0.60, name
        assert _lstar(shadow) < lb < _lstar(light), f"{name}: shadows darken, lights lighten"


def _synthetic_relief(shape, dataset="glo30"):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    shade = np.clip(np.sin(xx / 37.0) * np.cos(yy / 29.0) * 1.2, -1, 1).astype(np.float32)
    return relief.Relief(shade, dataset, 0, 0.5)


def _empty():
    import geopandas as gpd
    return gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])


STATS = {"km": 12.0, "n": 2, "hours": 1.0, "best5k_s": 1500, "best10k_s": 3100, "longest_km": 10.0, "streets": 7}


def _editorial_extent(size="A4"):
    from pinless import render_editorial
    aspect = render_editorial.map_aspect(size)
    return (0.0, 12000.0, 0.0, 12000.0 / aspect), render.SIZES[size][0] * (1 - 36 / 297)


def test_pdf_and_svg_hold_one_relief_image_the_size_of_the_map(tmp_path):
    import base64
    import re
    from shapely.geometry import LineString
    from pinless import render_editorial
    extent, map_w = _editorial_extent("A4")
    shape = relief.raster_shape(map_w, extent)
    assert shape == (927, 1090)
    pieces = [(0, LineString([(1000, 1000), (11000, 9000)]))]
    for with_relief in (True, False):
        pdf, svg = tmp_path / f"p{with_relief}.pdf", tmp_path / f"p{with_relief}.svg"
        info = render_editorial.render(pieces, (_empty(), _empty(), _empty()), extent, STATS,
                                       str(tmp_path / "p.png"), size="A4", dpi=30, city="X", year="2025",
                                       pdf_path=str(pdf), svg_path=str(svg),
                                       relief=_synthetic_relief(shape) if with_relief else None)
        assert info["relief"] is with_relief
        objects = re.split(rb"\d+ 0 obj", pdf.read_bytes())
        images = [o for o in objects if b"/Subtype /Image" in o]
        text = svg.read_text(encoding="utf-8")
        tags = re.findall(r"<image [^>]*>", text)
        if not with_relief:
            assert not images and not tags, "no relief: no image at all"
            continue
        assert len(images) == 2, "the image and its soft mask"
        for o in images:
            w, h = int(re.search(rb"/Width (\d+)", o).group(1)), int(re.search(rb"/Height (\d+)", o).group(1))
            assert (h, w) == shape
        assert len(tags) == 1
        data = re.search(r'data:image/png;base64,([^"]+)"', tags[0]).group(1)
        with Image.open(io.BytesIO(base64.b64decode(data))) as im:
            assert im.size[::-1] == shape, "the SVG embeds the array as it is"
    with pytest.raises(ValueError):
        render_editorial.render(pieces, (_empty(), _empty(), _empty()), extent, STATS, str(tmp_path / "x.png"),
                                size="A4", dpi=20, relief=_synthetic_relief((shape[0] + 1, shape[1])))


def test_flat_relief_is_transparent():
    from pinless.render_editorial import STYLES, relief_colours, relief_rgba
    rgba = relief_rgba(np.zeros((5, 7), np.float32), relief_colours(STYLES["huella"]))
    assert rgba.shape == (5, 7, 4) and rgba.dtype == np.uint8 and not rgba[..., 3].any()
    strong = relief_rgba(np.array([[-1.0, -0.3, 0.3, 1.0]], np.float32), relief_colours(STYLES["huella"]))
    (_, a_s), (_, a_l) = relief_colours(STYLES["huella"])
    assert strong[0, :, 3].tolist() == [round(255 * a_s), round(255 * a_s * 0.5), round(255 * a_l * 0.5),
                                        round(255 * a_l)]


def test_water_stays_flat_and_relief_sits_between_streets_and_routes(monkeypatch, tmp_path):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from shapely.geometry import LineString, Polygon
    from pinless import render_editorial
    extent, map_w = _editorial_extent("A4")
    shape = relief.raster_shape(map_w, extent)
    rel = relief.Relief(np.full(shape, -0.9, np.float32), "glo30", 0, 0.5)  # steep everywhere
    lake = Polygon([(4000, 3000), (8000, 3000), (8000, 7000), (4000, 7000)])
    water = gpd.GeoDataFrame({"natural": ["water"]}, geometry=[lake])
    streets = gpd.GeoDataFrame({"highway": ["primary"]}, geometry=[LineString([(0, 500), (12000, 600)])])
    figs, real_close = [], plt.close
    monkeypatch.setattr(plt, "close", lambda fig: figs.append(fig))
    out = tmp_path / "w.png"
    S = render_editorial.STYLES["huella"]
    render_editorial.render([(0, LineString([(500, 9000), (11500, 9500)]))], (streets, _empty(), water), extent,
                            STATS, str(out), size="A4", dpi=60, city="X", year="2025", relief=rel)
    fig = figs[0]
    ax = next(a for a in fig.axes if a.images)
    (im,) = ax.images
    assert im.get_zorder() == 3.5
    zs = sorted({p.get_zorder() for p in ax.patches})
    assert 3 in zs and 5 in zs and max(z for z in zs if z < 3.5) == 3 and min(z for z in zs if z > 3.5) >= 4.8
    real_close(fig)
    with Image.open(out) as img:
        px = np.asarray(img.convert("RGB"))
    # the map box on the page, at 60 dpi
    k = 210 / 297
    to_px = lambda mm: mm / 25.4 * 60
    x_mm = 18 * k + (6000 - extent[0]) / (extent[1] - extent[0]) * map_w
    y_mm = 94 * k + (extent[3] - 5000) / (extent[3] - extent[2]) * 222 * k
    centre = px[int(round(to_px(y_mm))), int(round(to_px(x_mm)))]
    assert tuple(centre) == tuple(round(c * 255) for c in to_rgb(S["water"])), "the water stays flat"
    x_mm = 18 * k + (1500 - extent[0]) / (extent[1] - extent[0]) * map_w
    y_mm = 94 * k + (extent[3] - 5000) / (extent[3] - extent[2]) * 222 * k
    slope = px[int(round(to_px(y_mm))), int(round(to_px(x_mm)))]
    assert tuple(slope) != tuple(round(c * 255) for c in to_rgb(S["bg"])), "the slope is shaded"


def _footer_spy(monkeypatch):
    from pinless import atlas, render_editorial, render_race
    boxes = []
    real = render_editorial.place_text

    def spy(value, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
        out = real(value, x, y, k, pt, font, align, max_w, height, clear)
        w, h = out[2], out[3]
        x0 = x - (w if align == "right" else w / 2 if align == "center" else 0)
        boxes.append((str(value), x0, y, x0 + w, y + h))
        return out

    for mod in (render_editorial, atlas, render_race):
        monkeypatch.setattr(mod, "place_text", spy)
    return boxes


def _check_footer(boxes, legend_y, osm_y, dataset, lang, more_note=None, mode="heat"):
    foot = [b for b in boxes if b[2] >= 395]
    osm = [b for b in foot if b[0].startswith("© OpenStreetMap")]
    notices = [b for b in foot if "Copernicus" in b[0]]
    assert len(osm) == 1
    if dataset is None:
        assert not notices and osm[0][2] == osm_y, "no relief: the footer of today"
    else:
        assert [b[0] for b in notices] == [relief.notice(dataset, lang)], "the notice exactly once"
        assert osm[0][2] == pytest.approx(osm_y - 3.4) and notices[0][2] == osm_y
    band = ("legend line", 217.0, legend_y + 0.4, 278.5 if mode == "month" else 268.0, legend_y + 1.6)
    for i, a in enumerate(foot + [band]):
        for b in foot[i + 1:]:
            overlap = a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]
            assert not overlap, f"{a[0][:30]!r} overlaps {b[0][:30]!r}"
    if more_note and notices:
        note = next(b for b in foot if b[0].startswith("+") and len(b[0]) > 1)  # not the legend's own +
        assert notices[0][3] < note[1], "the notice ends left of the +N note"
    assert all(b[4] <= 416 for b in foot), "the bottom margin of today"


def _acts_for_colours():
    import datetime as dt
    from pinless.ingest import Activity
    return [Activity(id=str(i), type="Run", start=dt.datetime(2025, 3 + i, 1, 8, tzinfo=dt.timezone.utc),
                     lonlat=[(-3.70, 40.41)] * 2, source="x", times=[0, 600], ele=[600, 650]) for i in range(2)]


FOOTER_CASES = [  # (lang, dataset, editorial legends, race legends); es + GLO-90 is the widest notice
    ("en", None, ("heat",), ("pace",)), ("es", None, ("month",), ("elevation",)),
    ("en", "glo30", ("pace",), ("pace",)), ("es", "glo30", ("month",), ("elevation",)),
    ("es", "glo90", ("heat", "month", "pace"), ("pace", "elevation")), ("en", "glo90", ("month",), ("elevation",))]


@pytest.mark.parametrize("lang,dataset,ed_modes,race_modes", FOOTER_CASES)
def test_footer_credits_fit_and_appear_only_with_relief(monkeypatch, tmp_path, lang, dataset, ed_modes, race_modes):
    import datetime as dt
    from shapely.geometry import LineString
    from pinless import atlas, render_editorial, render_race
    from matplotlib.figure import Figure
    from pinless.ingest import Activity
    boxes = _footer_spy(monkeypatch)
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: None)  # the boxes are all placed before saving
    extent, map_w = _editorial_extent("A4")
    rel = _synthetic_relief(relief.raster_shape(map_w, extent), dataset) if dataset else None
    pieces = [(0, LineString([(1000, 1000), (11000, 9000)])), (1, LineString([(1000, 9000), (11000, 1000)]))]
    tracks = [np.array([[1000, 1000], [11000, 9000]]), np.array([[1000, 9000], [11000, 1000]])]
    for mode in ed_modes:
        boxes.clear()
        render_editorial.render(pieces, (_empty(), _empty(), _empty()), extent, STATS, str(tmp_path / "e.png"),
                                size="A4", dpi=10, city="X", year="2025", lang=lang, color_by=mode,
                                activities=_acts_for_colours(), tracks_xy=tracks, relief=rel)
        _check_footer(boxes, 399, 409, dataset, lang, mode=mode)
    # the race print, pace and elevation legends
    xy = np.column_stack([np.linspace(0, 5000, 501), 800 * np.sin(np.linspace(0, 3, 501))])
    act = Activity(id="r", type="Run", start=dt.datetime(2025, 4, 6, 7, tzinfo=dt.timezone.utc),
                   lonlat=[(-3.70, 40.41)] * 501, source="x", times=list(np.linspace(0, 1500, 501)),
                   ele=list(600 + 40 * np.sin(np.linspace(0, 6, 501))))
    r_ext = (-500.0, 5500.0, -2500.0, -2500.0 + 6000.0 / render_race.map_aspect("A4"))
    r_rel = _synthetic_relief(relief.raster_shape(map_w, r_ext), dataset) if dataset else None
    for mode in race_modes:
        boxes.clear()
        render_race.render(act, xy, [(0, LineString(xy[30:-30]))], (_empty(), _empty(), _empty()), r_ext,
                           str(tmp_path / "r.png"), size="A4", dpi=10, title="T", city="Madrid", date=act.start,
                           color_by=mode, lang=lang, relief=r_rel)
        _check_footer(boxes, 404, 412, dataset, lang, mode=mode)
    # the atlas, with the +N note
    acts = [Activity(id=str(i), type="Run", start=dt.datetime(2025, 1 + i % 12, 3, 8),
                     lonlat=[(lon + 0.01 * np.cos(t), 40.41 + 0.01 * np.sin(t)) for t in np.linspace(0, 6.28, 60)],
                     source="x") for i, lon in enumerate([-3.70] * 4 + [2.35] * 2)]
    cities = atlas.group_by_city(acts)
    from pyproj import Transformer as _T
    for c in cities:
        c.crs = render.utm_crs(c.lon, c.lat)
        tr = _T.from_crs(4326, c.crs, always_xy=True)
        c.tracks = [np.column_stack(tr.transform(*np.asarray(a.lonlat).T)) for a in c.acts]
        c.pieces = [(i, LineString(t)) for i, t in enumerate(c.tracks)]
        c.name, c.country = "Somewhere", "Nowhere"
    cells = atlas.layout(len(cities))
    extents = [atlas.fit(c, w / h) for c, (_, _, w, h) in zip(cities, cells)]
    k = 210 / 297
    reliefs = [_synthetic_relief(relief.raster_shape(w * k, e), dataset) if dataset else None
               for (_, _, w, _), e in zip(cells, extents)]
    boxes.clear()
    atlas.render_atlas(cities, cells, extents, [None] * len(cities), str(tmp_path / "a.png"), size="A4", dpi=10,
                       lang=lang, more=3, reliefs=reliefs)
    _check_footer(boxes, 404, 412, dataset, lang, more_note=True)


# --- the command line -------------------------------------------------------------------------------------------------

RACE_ARGS = ["x.zip", "--layout", "race", "--dpi", "20", "--size", "A4"]


def test_a_failing_server_leaves_the_race_print_without_relief(monkeypatch, tmp_path, capsys):
    import urllib.error
    from pinless import cli

    def down(*args):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(relief, "_fetch", down)
    captured, texts = {}, []
    _race_world(monkeypatch, captured, texts)
    out = tmp_path / "race.png"
    assert cli.main(RACE_ARGS + ["--relief", "shade", "--out", str(out)]) == 0
    assert out.stat().st_size > 0 and captured["relief"] is None and captured["info"]["relief"] is False
    assert not any("Copernicus" in t for t in texts) and any(t.startswith("© OpenStreetMap") for t in texts)
    log = capsys.readouterr().out
    assert "[relief] skipped (URLError): the poster is drawn without relief" in log


def test_relief_flag_and_default(monkeypatch, tmp_path, capsys):
    import types
    from pinless import atlas, cli
    for layout in ("classic", "sport"):
        with pytest.raises(SystemExit) as ex:
            cli.main(["x.zip", "--layout", layout, "--relief", "shade"])
        assert ex.value.code == 2
    real_prepare = relief.prepare_many

    def never(*args, **kw):
        raise AssertionError("relief is off: nothing may be prepared or requested")

    monkeypatch.setattr(relief, "prepare_many", never)
    captured = {}
    _editorial_world(monkeypatch, [_loop_act(i, 40.42, -3.70 + 0.01 * i) for i in range(3)], captured)
    assert cli.main(["x.zip", "--bbox", "40.38,-3.75,40.48,-3.62", "--title", "X", "--out",
                     str(tmp_path / "e.png")]) == 0
    assert captured["relief"] is None
    captured = {}
    _race_world(monkeypatch, captured)
    assert cli.main(RACE_ARGS + ["--out", str(tmp_path / "r.png")]) == 0
    assert captured["relief"] is None and captured["info"]["relief"] is False
    # atlas.run with arguments built without a relief attribute (as other callers do)
    acts = [_loop_act(i, 40.42, -3.70) for i in range(4)]
    got = {}
    monkeypatch.setattr(atlas.area, "place_names", lambda lat, lon, lang="en": ("C", "K"))
    monkeypatch.setattr(atlas, "_basemap", lambda c, ext, log: None)
    monkeypatch.setattr(atlas, "render_atlas", lambda drawn, *a, **k: got.update(k) or {"cap": 3.0, "max": 1})
    args = types.SimpleNamespace(zone=[], trim=300.0, zone_radius=500.0, max_cities=2, lang="en", year=None,
                                 style=None, size="A3", dpi=40, name="", title="", out=str(tmp_path / "a.png"),
                                 pdf=None, svg=None, preview=None, color_by="heat")
    assert atlas.run(acts, args, log=lambda *_: None) == 0 and got["reliefs"] is None
    # --relief shade on the race print: a Relief reaches render_race.render, and the notice is drawn
    monkeypatch.setattr(relief, "prepare_many", real_prepare)
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep)})
    captured, texts = {}, []
    _race_world(monkeypatch, captured, texts)
    assert cli.main(RACE_ARGS + ["--relief", "shade", "--out", str(tmp_path / "r2.png")]) == 0
    assert isinstance(captured["relief"], relief.Relief) and captured["info"]["relief"] is True
    assert texts.count(relief.notice("glo30", "en")) == 1
    capsys.readouterr()
