"""Terrain relief: the Copernicus DEM reader, the geometry that puts it on the map, the gate, the drawing and the
privacy contract (what may be requested, logged and cached). All offline: a fake server serves synthetic COGs."""
import io
import math
import zlib

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from _cog_fixture import fake_server, levels_of, write_cog
from pinless import relief, render, render_editorial

H30, H90 = 480, 240  # small cells: 230 m and 460 m pixels at full resolution, so the files stay tiny
REAL = {name: getattr(relief, name) for name in ("HEIGHT0", "TILE", "NOMINAL_LEVELS")}  # before any test patches them


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """A test that forgets its fake server fails instead of reaching AWS; nothing is written outside tmp_path."""
    def no_network(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(relief, "_fetch", no_network)
    monkeypatch.setenv("PINLESS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": H30, "glo90": H90})
    monkeypatch.setattr(relief, "TILE", {"glo30": 64, "glo90": 64})  # the probe level is one block, as in real files


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


def _small_cells(monkeypatch):
    """Cells of 40 x 40 pixels in blocks of 16: four levels, the last one (5 x 5) a single block."""
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": 40, "glo90": 40})
    monkeypatch.setattr(relief, "TILE", {"glo30": 16, "glo90": 16})


@pytest.mark.parametrize("compression", [8, 32946])  # Deflate under its two TIFF codes
def test_reader_reads_every_level_of_a_synthetic_cog(monkeypatch, tmp_path, compression):
    _small_cells(monkeypatch)
    rng = np.random.default_rng(5)
    a0 = (-450.0 + rng.random((40, 40)) * 1950.0).astype(np.float32)  # negative, fractional, around 1e3
    a0[a0 == 0] = 0.5
    levels = levels_of(a0, 4)  # 40, 20, 10, 5: edge tiles of 16 px are partial
    data = write_cog(cell=(40, -4), levels=levels, tile=16, compression=compression)
    _serve(monkeypatch, {(40, -4): data})
    hdr = relief.read_header(data[:relief.HEADER_BYTES], (40, -4))
    assert [(lv.w, lv.h) for lv in hdr.levels] == [(40, 40), (20, 20), (10, 10), (5, 5)]
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
    good = levels_of(a0, 4)

    def spiked(value):
        out = [a.copy() for a in good]
        for a in out:  # in every level, so the probe meets it too
            a[1, 1] = value
        return out

    return {
        "big-endian": dict(levels=good, endian=">"),
        "compression 5": dict(levels=good, compression=5),
        "predictor 2": dict(levels=good, predictor=2),
        "int16 samples": dict(levels=good, bits=16, sample_format=2),
        "uint32 samples": dict(levels=good, sample_format=1),  # each of the two sample checks on its own
        "16-bit floats": dict(levels=good, bits=16),
        "two samples per pixel": dict(levels=good, tags={(0, 277): (3, [2])}),
        "separate planes": dict(levels=good, tags={(0, 284): (3, [2])}),
        "PixelIsArea": dict(levels=good, raster_type=1),
        "tiepoint off by 0.01": dict(levels=good, tiepoint=(0, 0, 0, -4.01, 41.0, 0)),
        "tiepoint latitude off by 0.01": dict(levels=good, tiepoint=(0, 0, 0, -4.0, 41.01, 0)),
        "tiepoint off by a micro-degree": dict(levels=good, tiepoint=(0, 0, 0, -4.0 + 1e-6, 41.0, 0)),
        "pixel scale off by 1%": dict(levels=good, scale=(1 / 40, 1.01 / 40, 0.0)),
        "a cell of 48 rows": dict(levels=levels_of(np.tile(a0, (2, 2))[:48, :48], 4)),  # consistent, but not 40
        "a cell narrower than its band": dict(levels=levels_of(a0[:, :24], 4)),  # at 40N a cell is as wide as high
        "width 0": dict(levels=good, tags={(0, 256): (4, [0])}),
        "a width in floating point": dict(levels=good, tags={(0, 256): (12, [40.0])}),
        "full resolution marked as an overview": dict(levels=good, tags={(0, 254): (4, [1])}),
        "overview not half the size": dict(levels=[good[0], a0[:19, :19], good[2], good[3]]),
        "an overview not marked as one": dict(levels=good, tags={(1, 254): (4, [0])}),
        "fewer tile offsets than tiles": dict(levels=good, tags={(0, 324): (4, [8, 8]), (0, 325): (4, [1, 1])}),
        "IFD chain loops back": dict(levels=good, ifd_loop=True),
        "IFD after 16 KB": dict(levels=good, ifd_at_end=True),
        "a value of 1e5": dict(levels=spiked(1e5)),
        "a value of -1e5": dict(levels=spiked(-1e5)),
    }


@pytest.mark.parametrize("case", sorted(_refusals()))
def test_reader_refuses_what_it_cannot_decode(monkeypatch, tmp_path, case):
    _small_cells(monkeypatch)
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
    # a header is cached only once it parses, and a cell whose blocks do not decode is forgotten, header included
    assert not list(tmp_path.rglob("header.bin"))


RETILED = {  # (tiling as write_cog takes it, levels): what a server could publish instead of the dataset's layout
    "smaller blocks at full resolution": ([32, 64, 64, 64], 4),
    "the probe level in small blocks": ([64, 64, 64, 16], 4),
    "strips one block wide": ([(8, 480), 64, 64, 64], 4),
    "strips one block high": ([(480, 8), 64, 64, 64], 4),
    "larger blocks": (128, 4),
    "one level less": (64, 3),
    "one level more": (64, 5),
    "a probe level of several blocks of the dataset's size": (16, 4),
}


@pytest.mark.parametrize("case", sorted(RETILED))
def test_a_cell_in_other_blocks_than_the_datasets_is_refused_after_its_header(monkeypatch, tmp_path, case):
    """S3 learns which blocks a print asks for, so their size is the real files' (1024 px for GLO-30; 64 here), never
    the server's choice: smaller blocks would pin the map down to a few km, and a probe level of several blocks would
    tell where even a flat map is. A cell in any other layout is refused once its header is read."""
    tile, n_levels = RETILED[case]
    assert len(relief.read_header(_cog((40, -4), _steep)[:relief.HEADER_BYTES], (40, -4)).levels) == 4, \
        "setup: the same cell in the dataset's own blocks is read"
    if case == "a probe level of several blocks of the dataset's size":
        monkeypatch.setattr(relief, "TILE", {"glo30": 16, "glo90": 16})  # 60 x 60 at the probe level: 4 x 4 blocks
    a0 = _cell_grid((40, -4), H30, _steep)
    data = write_cog(cell=(40, -4), levels=levels_of(a0, n_levels), tile=tile)
    with pytest.raises(relief.ReliefError):
        relief.read_header(data[:relief.HEADER_BYTES], (40, -4))
    calls = _serve(monkeypatch, {(40, -4): data})
    extent, crs = _utm_extent(40.5, -3.5, 14000, 12000)  # inside the cell, with every pad
    assert relief._Map(0, extent, crs, A4_MAP).cells == {(40, -4)}
    logs = []
    assert relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append) == [None]
    assert "skipped (ReliefError)" in logs[-1]
    assert calls == [(relief.url("glo30", (40, -4)), 0, relief.HEADER_BYTES - 1)], "the header and nothing else"
    assert not [p for p in tmp_path.rglob("*") if p.is_file()], "nothing cached"


def test_a_print_across_cells_with_different_levels_is_refused_not_crashed(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep), (40, -3): _cog((40, -3), _steep, n_levels=3)})
    extent, crs = _utm_extent(40.5, -3.0, 12000, 10000)  # across both cells
    logs = []
    assert relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append) == [None]
    assert "skipped (ReliefError)" in logs[-1]
    assert all(first == 0 for _, first, _ in calls), "refused at the headers: no block is asked for"


def test_the_sizes_the_requests_are_made_of():
    """The tests run on small cells; the real sizes, written out once (tools/check_relief.py reads real headers)."""
    assert relief.HEADER_BYTES == 16384 and relief.BUDGET_BYTES == 64 * 2 ** 20 and relief.MAX_CELLS == 16
    assert REAL["HEIGHT0"] == {"glo30": 3600, "glo90": 1200}
    assert REAL["NOMINAL_LEVELS"] == {"glo30": 4, "glo90": 3}
    assert REAL["TILE"] == {"glo30": 1024, "glo90": 2048}


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


def test_the_latitude_bands_of_the_copernicus_cells():
    """How many times fewer columns than rows a cell has, by the south latitude of the cell (written out, so the
    fixtures that derive their widths from relief._band cannot follow a wrong table)."""
    want = {0: 1, 49: 1, 50: 1.5, 59: 1.5, 60: 2, 69: 2, 70: 3, 79: 3, 80: 5, 84: 5, 85: 10, 89: 10,
            -1: 1, -50: 1, -51: 1.5, -60: 1.5, -61: 2, -70: 2, -71: 3, -80: 3, -81: 5, -85: 5, -86: 10, -90: 10}
    assert {N: relief._band(N) for N in want} == want
    assert relief._dims("glo30", (60, 24), 0)[:2] == (H30 // 2, H30), "a cell at 60N is half as wide as high"


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
            assert last == 16383, "a header is the first 16 KB"
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


REAL_EDITORIAL_RENDER = render_editorial.render  # before any test replaces it


def _editorial_world(monkeypatch, acts, captured, draw=False):
    """The editorial CLI without OSM and without stats: render_editorial.render records its arguments (the extent
    as captured["extent"]), and draws only if draw."""
    from pinless import area, ingest, render_editorial, stats
    monkeypatch.setattr(ingest, "read_export", lambda path: (acts, ingest.IngestReport()))
    monkeypatch.setattr(area, "fetch_basemap", lambda bbox, root=None, log=print: "unused")
    monkeypatch.setattr(render, "load_basemap", lambda folder, bbox, crs: (_empty(), _empty(), _empty()))
    monkeypatch.setattr(stats, "compute", lambda *args, **kw: dict(STATS, fav_street=None))

    def spy(*args, **kw):
        captured.update(kw, extent=args[2])
        if draw:
            captured["info"] = REAL_EDITORIAL_RENDER(*args, **kw)
            return captured["info"]
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
    assert "1-degree cells" in logs[0] and "blocks of about 30 km" in logs[0], "the log says what S3 learns"
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
        assert captured["relief"].shade.shape == relief.raster_shape(render.SIZES["A3"][0] * (1 - 36 / 297),
                                                                     captured["extent"]), "the editorial map's width"
        lists.append(list(calls))
    assert lists[0] and lists[0] == lists[1], "the requests depend on the map box only"
    assert any(first != 0 for _, first, _ in lists[0])
    # and the real renderer takes it (it refuses a raster of any other size), at another paper size
    captured = {}
    _editorial_world(monkeypatch, [_loop_act(i, 40.42, -3.70 + 0.01 * i) for i in range(3)], captured, draw=True)
    assert cli.main(["x.zip", "--bbox", "40.38,-3.75,40.48,-3.62", "--title", "X", "--relief", "shade", "--size", "A4",
                     "--dpi", "20", "--out", str(tmp_path / "a4.png")]) == 0
    assert captured["info"]["relief"] is True


THREE_CELLS = ((40, -4), (40, -3), (41, -4))


def test_the_order_of_the_maps_never_reorders_the_requests(monkeypatch, tmp_path):
    """S3 sees the requests in order. In an atlas the maps come home first, then the trips by the date of their first
    run: the requests must be a function of the set of maps alone, sorted, whatever their order."""
    files = {c: _cog(c, _steep) for c in THREE_CELLS}
    maps = [_utm_extent(41.5, -3.5, 12000, 10000), _utm_extent(40.45, -2.6, 12000, 10000),
            _utm_extent(40.42, -3.7, 12000, 10000)]
    lists = []
    for k, order in enumerate(((0, 1, 2), (2, 1, 0), (1, 2, 0))):
        calls = _serve(monkeypatch, files)
        out = relief.prepare_many([(maps[i][0], maps[i][1], 60.0) for i in order], log=lambda *_: None,
                                  root=str(tmp_path / f"dem{k}"))  # each time with nothing cached
        assert all(isinstance(r, relief.Relief) for r in out)
        lists.append(calls)
    assert lists[0] == lists[1] == lists[2], "the same requests in the same order"
    blocks = [(u, first) for u, first, _ in lists[0] if first != 0]
    assert len({u for u, _ in blocks}) == 3, "setup: blocks of the three cells"


def _dated_loop(i, lat, lon, day):
    import dataclasses
    import datetime as dt
    return dataclasses.replace(_loop_act(i, lat, lon), start=dt.datetime(day.year, day.month, day.day, 8,
                                                                           tzinfo=dt.timezone.utc))


def test_the_atlas_requests_do_not_tell_which_city_is_home_or_when_each_trip_was(monkeypatch, tmp_path):
    import datetime as dt
    import types
    from pinless import atlas
    files = {c: _cog(c, _steep) for c in THREE_CELLS}
    monkeypatch.setattr(atlas.area, "place_names", lambda lat, lon, lang="en": ("C", "K"))
    monkeypatch.setattr(atlas, "_basemap", lambda c, ext, log: (_empty(), _empty(), _empty()))
    monkeypatch.setattr(atlas, "render_atlas", lambda *a, **k: {"cap": 1.0, "max": 1})
    home, trip_a, trip_b = (41.50, -3.50), (40.45, -2.60), (40.42, -3.70)  # home sorts last of the three cells
    early, late = dt.date(2025, 3, 1), dt.date(2025, 7, 1)
    lists = []
    for k, a_first in enumerate((True, False)):  # the same runs; only the dates of the two trips swap
        acts = [_dated_loop(i, *home, dt.date(2025, 1, 1 + i)) for i in range(6)]  # home: the most km
        acts += [_dated_loop(10 + i, *trip_a, (early if a_first else late) + dt.timedelta(days=i)) for i in range(3)]
        acts += [_dated_loop(20 + i, *trip_b, (late if a_first else early) + dt.timedelta(days=i)) for i in range(3)]
        monkeypatch.setenv("PINLESS_CACHE", str(tmp_path / f"cache{k}"))  # each time with nothing cached
        calls = _serve(monkeypatch, files)
        args = types.SimpleNamespace(zone=[], trim=300.0, zone_radius=500.0, max_cities=3, lang="en", year=None,
                                     style=None, size="A4", dpi=20, name="", title="", out=str(tmp_path / "a.png"),
                                     pdf=None, svg=None, preview=None, color_by="heat", relief="shade")
        assert atlas.run(acts, args, log=lambda *_: None) == 0
        lists.append(calls)
    assert any(first != 0 for _, first, _ in lists[0]), "setup: blocks were requested"
    assert lists[0] == lists[1], "the very same requests, in the very same order"


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
    mos = relief._mosaic("glo30", headers, cells, rel.level, box, root)
    z = relief._sample_cubic(mos, *relief._nodes(extent, crs, g))
    r, c = np.unravel_index(np.argmax(z), z.shape)
    nx_, ny_ = extent[0] + (c - g.m + 0.5) * g.cx, extent[3] - (r - g.m + 0.5) * g.cy
    assert abs(nx_ - px) <= g.cx and abs(ny_ - py) <= g.cy
    # the rasters sit where the hill is: the heights for the contours peak within two output pixels (15 m each)
    H, W = rel.shade.shape
    to_xy = lambda rc: (extent[0] + (rc[1] + 0.5) / W * (extent[1] - extent[0]),
                        extent[3] - (rc[0] + 0.5) / H * (extent[3] - extent[2]))
    for heights in (rel.z, rel.z_smooth):
        zx, zy = to_xy(np.unravel_index(np.argmax(heights), heights.shape))
        assert math.hypot(zx - px, zy - py) <= 30

    def at(s, dx, dy):  # the mean shade of 5 x 5 output pixels around (px + dx, py + dy)
        row = int((extent[3] - (py + dy)) / (extent[3] - extent[2]) * H)
        col = int((px + dx - extent[0]) / (extent[1] - extent[0]) * W)
        return float(s[row - 2:row + 3, col - 2:col + 3].mean())

    # lit from the north-west, north up, both shades: of the hill's four diagonal flanks (at its sigma from the top,
    # where it is steepest) the north-west one is the brightest and the south-east one the darkest
    d = 2500 / math.sqrt(2)
    for name in ("shade", "smooth"):
        s = getattr(rel, name)
        nw, ne, sw, se = at(s, -d, d), at(s, d, d), at(s, -d, -d), at(s, d, -d)
        assert nw > 0.3 and se < 0, name  # at its sigma the hill's flank is still convex: the texture lifts it
        assert nw > max(ne, sw) and se < min(ne, sw), f"{name}: lit from the north-west, north up"


def test_the_shade_stays_between_minus_one_and_one(monkeypatch, tmp_path):
    def cliffs(lon, lat):  # slopes up to 2.8 m/m: full shadow next to bright light, where resizing overshoots
        return 3500 + 3000 * np.sin(2 * np.pi * lon / 0.08) * np.cos(2 * np.pi * lat / 0.06)

    _serve(monkeypatch, {(40, -4): _cog((40, -4), cliffs)})
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert rel.shade.dtype == np.float32 and rel.shade.min() == -1.0 and rel.shade.max() <= 1.0
    assert rel.smooth.dtype == np.float32 and rel.smooth.min() >= -1.0 and rel.smooth.max() <= 1.0


def test_flat_ground_has_shade_exactly_zero(monkeypatch, tmp_path):
    from pinless.render_editorial import STYLES, relief_colours, relief_rgba
    files = {(40, -4): _cog((40, -4), lambda lon, lat: np.full_like(lon, 650.0))}
    _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    job = relief._Map(0, extent, crs, A4_MAP)
    root = str(tmp_path / "dem")
    headers = {c: relief._header("glo30", c, root) for c in job.cells}
    shade, smooth, z, z_smooth = relief._detail(job, "glo30", headers, 0, root)
    assert shade.shape == smooth.shape == z.shape == z_smooth.shape == job.shape
    assert not shade.any() and not smooth.any(), "flat ground: exactly 0, hillshade, texture and sky view alike"
    assert not relief_rgba(smooth, relief_colours(STYLES["huella"]))[..., 3].any(), "and an overlay draws nothing"
    assert np.abs(z - 650).max() < 1e-3 and np.abs(z_smooth - 650).max() < 1e-3
    # the city ground filter: on a grid that counts as full resolution, a lone building of 2 x 2 pixels on flat ground
    # leaves no trace in the smooth shade (these test cells have 230 m pixels, so the filter is let run on them); the
    # crisp shade of the natural ground is not filtered and keeps it
    monkeypatch.setattr(relief, "OPEN_MAX_SPACING_M", 300.0)

    def town(lon, lat):
        z = np.full_like(lon, 650.0)
        z[240:242, 240:242] = 670.0  # at the centre of the map (40.5, -3.5)
        return z

    _serve(monkeypatch, {(40, -4): _cog((40, -4), town)})
    root = str(tmp_path / "town")
    headers = {c: relief._header("glo30", c, root) for c in job.cells}
    shade, smooth, z, z_smooth = relief._detail(job, "glo30", headers, 0, root)
    assert np.abs(smooth).max() <= 1e-6 and np.abs(z_smooth - 650).max() <= 1e-6, "the building is filtered off"
    assert np.abs(shade).max() > 0.05 and z.max() > 655, "the crisp surface keeps it"


def _plane(lat_c, lon_c, east, north, z0=4000.0):
    """A plane through z0 m at (lat_c, lon_c) rising `east` and `north` metres per metre. Far from the map it is
    clipped to valid elevations: no block that far is ever read."""
    def fn(lon, lat):
        dx = (lon - lon_c) * relief.M_PER_DEG_LON * math.cos(math.radians(lat_c))
        dy = (lat - lat_c) * relief.M_PER_DEG_LAT
        return np.clip(z0 + east * dx + north * dy, 10.0, 8900.0)
    return fn


def _hillshade(east, north):
    """The shade of a plane by the formula of the spec, written again here: light from azimuth 315, altitude 45."""
    lx, ly, lz = -0.5, 0.5, math.sqrt(0.5)
    hs = max(0.0, (-east * lx - north * ly + lz) / math.sqrt(east ** 2 + north ** 2 + 1))
    s = hs - lz
    return s / lz if s < 0 else s / (1 - lz)


def _island(height):
    """0 m sea and one island of one coarsest-level pixel (8 x 8 full-resolution pixels) at (40.5, -3.5)."""
    def fn(lon, lat):
        z = np.zeros_like(lon)
        r, c = int((41 - 40.5) * H30) // 8 * 8, int((-3.5 + 4) * H30) // 8 * 8
        z[r:r + 8, c:c + 8] = height
        return z
    return fn


def _coast_with_a_hill(lon, lat):
    """Sea west of -3.40; east of it flat land 2 m high with one hill 400 m high, 2 km wide."""
    dx = (lon + 3.36) * relief.M_PER_DEG_LON * math.cos(math.radians(40.5))
    dy = (lat - 40.5) * relief.M_PER_DEG_LAT
    return np.where(lon >= -3.40, 2 + 400 * np.exp(-(dx ** 2 + dy ** 2) / (2 * 2000.0 ** 2)), 0.0)


def _metres(lon, lat, lat_c=40.5, lon_c=-3.5):
    return (lon - lon_c) * relief.M_PER_DEG_LON * math.cos(math.radians(lat_c)), (lat - lat_c) * relief.M_PER_DEG_LAT


def _polder(lon, lat):
    """Land 1 m above the sea, all of it, with one hill 400 m high and 1.5 km wide (sigma)."""
    x, y = _metres(lon, lat)
    return 1.0 + 400 * np.exp(-(x ** 2 + y ** 2) / (2 * 1500.0 ** 2))


def _cone_island(lon, lat):
    """A steep island 2.5 km in radius in the sea: about 3% of a 24 km map."""
    r = np.hypot(*_metres(lon, lat))
    return np.where(r < 2500, 50 + 800 * (1 - r / 2500), 0.0)


def _ripples(lon, lat):
    """Ripples 30 m high and 900 m apart: a texture, not a landform."""
    x, y = _metres(lon, lat)
    return 600 + 30 * np.sin(2 * np.pi * x / 900) * np.sin(2 * np.pi * y / 900)


def _peninsula(height):
    """A strip of land 300 m wide running east to west between two seas, flat on top at height (in m), its sides
    cliffs: the Ponta de Sao Lourenco on Madeira."""
    def fn(lon, lat):
        _, y = _metres(lon, lat)
        return np.where(np.abs(y) < 150, height, 0.0)
    return fn


def _valley(lon, lat):
    """A flat valley floor the size of a 12 x 10 km map, its mountainsides rising 600 m beyond the map's edges."""
    x, y = _metres(lon, lat)
    d = np.maximum(np.abs(x) - 6000, np.abs(y) - 5000)
    return np.minimum(600 + np.where(d > 600, 0.5 * (d - 600), 0.0), 8000.0)


GATE_CASES = {  # terrain, map size (m), what the log says (None: the map gets relief)[, levels of the cell]
    "flat": (lambda lon, lat: np.full_like(lon, 650.0), (12000, 10000), "flat here"),
    "gentle": (lambda lon, lat: 650 + 20 * (lon + 4), (12000, 10000), "flat here"),
    "steep": (_steep, (12000, 10000), None),
    # planes facing the light: a shade of about 0.10 is not a slope (GATE_S is 0.15), one of about 0.20 is
    "plane at shade 0.10": (_plane(40.5, -3.5, 0.03, -0.03), (12000, 10000), "flat here"),
    "plane at shade 0.20": (_plane(40.5, -3.5, 0.0615, -0.0615), (12000, 10000), None),
    # a quarter of the map is land and 4% of that land slopes: 1% of the map, but the share is of the land
    "coast with a hill": (_coast_with_a_hill, (32000, 28000), None),
    # land is where the RAW elevation is: 1.3% and 0.6% of these maps, mostly water (smoothing would bleed the tall
    # island's cliffs over 2.2% of its map)
    "island near the threshold": (_island(1000.0), (28000, 28000), "mostly water here"),
    "island": (_island(300.0), (40000, 34000), "mostly water here"),
    # ... and a steep island of 3% of its map has enough land to show
    "steep island of 3%": (_cone_island, (24000, 24000), None),
    # land is anything 0.5 m off the sea: a polder 1 m up is land, all of it, and one small hill is too little of it
    "polder with one small hill": (_polder, (32000, 28000), "flat here"),
    # with the probe on the full-resolution level: its 250 m smoothing judges landforms, not texture; and the share is
    # of the map's own land, so mountainsides in the margin around a flat map do not count
    "short ripples": (_ripples, (12000, 10000), "flat here", 1),
    "flat valley floor among mountains": (_valley, (12000, 10000), "flat here", 1),
    # a hike along a peninsula 300 m wide with cliffs 100 m high between two seas, on a map of 2 km (the probe on
    # the full-resolution level of these small cells, 230 m): the probe's smoothing follows the paper (125 m here),
    # and 250 m into the sea no longer flattens it; the same land, flat, stays without relief
    "narrow peninsula with cliffs": (_peninsula(100.0), (2000, 1900), None, 1),
    "narrow flat peninsula": (_peninsula(3.0), (2000, 1900), "flat here", 1),
}


def _one_level(monkeypatch):
    """Cells of one level only (a single block), so the probe reads the full resolution."""
    monkeypatch.setattr(relief, "NOMINAL_LEVELS", {"glo30": 1, "glo90": 1})
    monkeypatch.setattr(relief, "TILE", {"glo30": 512, "glo90": 512})


@pytest.mark.parametrize("terrain", sorted(GATE_CASES))
def test_flat_land_is_skipped_after_the_probe_only(monkeypatch, tmp_path, terrain):
    fn, size, said, *levels = GATE_CASES[terrain]
    n_levels = levels[0] if levels else 4
    if n_levels == 1:
        _one_level(monkeypatch)
    files = {(40, -4): _cog((40, -4), fn, n_levels=n_levels, tile=512 if n_levels == 1 else 64)}
    calls = _serve(monkeypatch, files)
    logs = []
    extent, crs = _utm_extent(40.5, -3.5, *size)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append)
    blocks, _ = _levels_of_ranges(files, calls)
    if said is None:
        assert rel is not None and rel.shade.shape == relief.raster_shape(A4_MAP, extent) and rel.share >= 0.02
        assert any(L == 0 for _, L, _, _ in blocks)
        return
    assert rel is None
    assert logs[-1] == f"[relief] {said}: no relief"
    assert {L for _, L, _, _ in blocks} == {n_levels - 1}, "only headers and the probe level, never a detail block"


@pytest.mark.parametrize("east,north", [(0.15, -0.15), (-0.15, 0.15)])  # facing the light, facing away from it
def test_a_plane_is_shaded_evenly(monkeypatch, tmp_path, east, north):
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _plane(40.5, -3.5, east, north))})
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert abs(_hillshade(east, north)) > 0.2
    # every pixel, the borders too: the work grid's margin keeps the edges of the smoothing, of the sky view's horizon
    # and of the texture's mirrored grid off the map
    for name in ("shade", "smooth"):
        s = getattr(rel, name)
        assert np.ptp(s) < 0.01, name
        assert (s > 0).all() if east > 0 else (s < 0).all(), f"{name}: a plane facing the light is lit"


def test_the_smoothing_is_a_gaussian_of_sigma_map_units():
    g = relief.Grid(cx=10.0, cy=20.0, nx=61, ny=61, m=0)
    z = np.zeros((61, 61))
    z[30, 30] = 1.0
    out = relief._gauss(z, 30.0, g)
    r = np.arange(61) - 30
    assert out.sum() == pytest.approx(1.0)
    assert (out.sum(axis=0) * r ** 2).sum() == pytest.approx((30 / 10) ** 2, rel=0.02), "sigma 3 cells along x"
    assert (out.sum(axis=1) * r ** 2).sum() == pytest.approx((30 / 20) ** 2, rel=0.02), "sigma 1.5 cells along y"
    flat = relief._gauss(np.full((20, 30), 123.456), 30.0, g)
    assert np.ptp(flat) == 0 and flat[0, 0] == pytest.approx(123.456), "a constant stays exactly even"


def test_the_probe_grid_and_the_detail_smoothing_follow_the_map():
    """The probe works on cells of 125 m, or 1/400 of a larger map, and smooths by 6 mm of paper, between 125 and
    250 m; the level rule's sigma is 0.40 mm of paper, never less than 45 m; the shade is smoothed by 0.25 mm of paper,
    never less than 8 m; the sky view reaches 14 mm of paper, between 250 and 600 m; the contours' heights are
    smoothed by 0.20 mm of paper, never less than 20 m."""
    for w_m, map_w, sigma, cell, tex, radius, contour, gate in (
            (12000, A4_MAP, 45.0, 125.0, 0.25 * 12000 / A4_MAP, 600.0, 20.0, 250.0),
            (100000, 60.0, 0.40 * 100000 / 60.0, 250.0, 0.25 * 100000 / 60.0, 600.0, 0.20 * 100000 / 60.0, 250.0),
            (6000, A4_MAP, 45.0, 125.0, 0.25 * 6000 / A4_MAP, 14 * 6000 / A4_MAP, 20.0, 6 * 6000 / A4_MAP),
            (2000, A4_MAP, 45.0, 125.0, 8.0, 250.0, 20.0, 125.0)):
        extent, crs = _utm_extent(40.5, -3.5, w_m, 0.8 * w_m)
        job = relief._Map(0, extent, crs, map_w)
        assert job.sigma == pytest.approx(sigma), w_m
        assert (job.tex_sigma, job.tex_radius, job.contour_sigma) == pytest.approx((tex, radius, contour)), w_m
        g = job.probe_grid
        assert (g.cx, g.cy) == pytest.approx((w_m / math.ceil(w_m / cell), 0.8 * w_m / math.ceil(0.8 * w_m / cell))), w_m
        assert job.gate_sigma == pytest.approx(gate), w_m
        assert g.m == math.ceil(4 * gate / min(g.cx, g.cy)) + 1, "a margin of four times the probe's smoothing"


def test_the_ground_filter_removes_buildings_and_keeps_the_ground():
    ky, kx = 5, 7
    building = np.full((40, 50), 600.0)
    building[20:22, 24:26] += 20.0  # 2 x 2 pixels, 20 m
    assert np.array_equal(relief._opening(building, ky, kx), np.full((40, 50), 600.0)), "a building is removed"
    yy, xx = np.mgrid[0:40, 0:50].astype(float)
    plane = 600 + 0.8 * xx - 0.5 * yy
    inner = (slice(ky, -ky), slice(kx, -kx))  # away from the repeated edges
    assert np.array_equal(relief._opening(plane, ky, kx)[inner], plane[inner]), "sloping ground is kept"
    pit = np.full((40, 50), 600.0)
    pit[20:22, 24:26] -= 20.0
    assert np.array_equal(relief._opening(pit, ky, kx), pit), "a hollow is ground: kept"
    mesa = np.full((40, 50), 600.0)
    mesa[15:24, 20:31] += 20.0  # 9 x 11 pixels: wider than the window
    assert np.array_equal(relief._opening(mesa, ky, kx), mesa), "a landform wider than the window is kept"


def test_the_city_ground_filter_runs_at_full_resolution_only_with_a_150_m_window(monkeypatch):
    seen = []
    real = relief._opening
    monkeypatch.setattr(relief, "_opening", lambda z, ky, kx: seen.append((ky, kx)) or real(z, ky, kx))
    z = (600 + 50 * np.random.default_rng(1).random((30, 40))).astype(np.float32)
    glo30 = relief.Mosaic(z, lon0=-4.0, dlon=1 / 3600, lat0=40.6, dlat=1 / 3600)  # 30.7 m N-S, 23.5 m E-W here
    out = relief._ground_urban(glo30)
    assert seen == [(5, 7)], "the ground found with a window of 5 x 7 pixels at 40N"
    assert out.z.shape == z.shape and (out.lon0, out.dlon, out.lat0, out.dlat) == (-4.0, 1 / 3600, 40.6, 1 / 3600)
    assert np.isfinite(out.z).all() and out.z.min() >= z.min() - 1e-9 and out.z.max() <= z.max() + 1e-9
    coarse = relief.Mosaic(z, lon0=-4.0, dlon=61 / relief.M_PER_DEG_LAT, lat0=40.6, dlat=61 / relief.M_PER_DEG_LAT)
    assert relief._ground_urban(coarse) is coarse and seen == [(5, 7)], "pixels of 61 m: no filter"


def test_cubic_samples_keep_a_constant_exactly_and_a_ramp():
    rng = np.random.default_rng(3)
    lon, lat = -4.0 + rng.random(500) * 0.14, 41.0 - rng.random(500) * 0.11
    const = relief.Mosaic(np.full((12, 15), 650.123), lon0=-4.0, dlon=0.01, lat0=41.0, dlat=0.01)
    assert (relief._sample_cubic(const, lon, lat) == 650.123).all(), "a constant stays exactly constant"
    rows, cols = np.mgrid[0:12, 0:15]
    ramp = relief.Mosaic(100 + 3.0 * cols - 2.0 * rows, lon0=-4.0, dlon=0.01, lat0=41.0, dlat=0.01)
    # a pixel in from the edges, where the outer taps repeat the edge and bend the ramp
    lon, lat = -4.0 + (1 + rng.random(500) * 12) * 0.01, 41.0 - (1 + rng.random(500) * 9) * 0.01
    want = 100 + 3.0 * (lon + 4.0) / 0.01 - 2.0 * (41.0 - lat) / 0.01
    assert np.abs(relief._sample_cubic(ramp, lon, lat) - want).max() < 1e-9, "cubic convolution keeps a ramp"
    with pytest.raises(relief.ReliefError):  # outside the grid: raises, as the bilinear samples do
        relief._sample_cubic(ramp, np.array([-4.001]), np.array([40.95]))


def test_the_texture_is_zero_on_a_constant_positive_on_a_crest_and_negative_in_a_valley():
    assert not relief._texture(np.full((40, 50), 812.5), 30.0, 30.0, relief.TEX_ALPHA).any()
    x = (np.arange(90) - 45) * 30.0
    profile = 600 + 50 * np.exp(-(x + 450) ** 2 / (2 * 120.0 ** 2)) - 50 * np.exp(-(x - 450) ** 2 / (2 * 120.0 ** 2))
    z = np.tile(profile, (60, 1)) + 0.02 * np.arange(60)[:, None] * 30  # a crest and a valley on a gentle slope
    t = relief._texture(z, 30.0, 30.0, relief.TEX_ALPHA)
    assert t[30, 30] > 0 and t[30, 60] < 0, "the crest is a ridge, the trough a valley"
    assert t[30, 30] == pytest.approx(-t[30, 60], rel=0.05), "and the slope under them adds nothing"


def test_the_sky_view_is_one_on_flat_ground_and_lower_in_a_pit_than_on_a_peak():
    assert np.array_equal(relief._sky_view(np.full((40, 40), 300.0), 30.0, 30.0, 250.0), np.ones((40, 40)))
    yy, xx = np.mgrid[0:41, 0:41]
    bump = 40 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) * 30.0 ** 2 / (2 * 90.0 ** 2))
    peak = relief._sky_view(300 + bump, 30.0, 30.0, 250.0)[20, 20]
    pit = relief._sky_view(300 - bump, 30.0, 30.0, 250.0)[20, 20]
    assert peak == 1.0, "every horizon of a summit lies below it"
    assert pit < 0.95, "a hollow sees less of the sky"


def _full_resolution(fn, w_m=3000.0):
    """A map w_m wide on A4 over a synthetic mosaic at the full resolution of GLO-30 (1 arc-second: 30.7 x 23.5 m at
    40.5N): (job, its detail grid, the mosaic of fn(x, y), x and y in metres east and north of the map's centre)."""
    lat_c, lon_c = 40.5, -3.5
    extent, crs = _utm_extent(lat_c, lon_c, w_m, 0.8 * w_m)
    job = relief._Map(0, extent, crs, A4_MAP)
    spacing = relief.M_PER_DEG_LAT / 3600
    g = job.detail_grid(spacing)
    lat_a, lat_b, lon_a, lon_b = relief._lonlat_box(extent, crs, job.detail_pad(spacing))
    d = 1 / 3600
    lons, lats = np.arange(lon_a - 2 * d, lon_b + 2 * d, d), lat_b + 2 * d - np.arange(0, lat_b - lat_a + 4 * d, d)
    lon, lat = np.meshgrid(lons, lats)
    x, y = _metres(lon, lat, lat_c, lon_c)
    return job, g, relief.Mosaic(fn(x, y), float(lons[0]), d, float(lats[0]), d)


def test_the_natural_ground_is_not_opened():
    """A knife ridge 60 m wide and 40 m high: the crisp shade (no ground filter on natural ground) keeps its lee side
    in deep shadow; the smooth shade over the city ground filter takes it for a row of buildings."""
    job, g, mos = _full_resolution(lambda x, y: 600 + 40 * np.exp(-x ** 2 / (2 * 25.0 ** 2)) + 0 * y)
    shade, smooth, z, _ = relief._rasters(job, mos, g)
    H, W = shade.shape
    lee = (slice(H // 4, 3 * H // 4), slice(W // 2, W // 2 + 30))  # its east side: away from a north-west light
    assert shade[lee].min() < -0.3, "the lee side of the ridge in shadow"
    assert smooth[lee].min() > shade[lee].min() + 0.2, "weaker over the city ground"
    assert z.max() > 630, "the heights keep it too"


def test_the_city_ground_takes_out_buildings_without_terraces_and_keeps_hills():
    """Buildings 20 m high of 3 x 3 pixels on a slope leave less than 1 m in the city ground; a hill 40 m high and
    600 m across at half its height keeps its height within 10 % (the ground is found with a 150 m window: a hill
    half as wide is lowered, to about 60 % of its height, the price of a town's roofs)."""
    def plane(x, y):
        return 600 + 0.05 * x + 0.03 * y

    def town(x, y):
        z = plane(x, y)
        for r in range(4, z.shape[0] - 4, 12):
            for c in range(4, z.shape[1] - 4, 12):
                z[r:r + 3, c:c + 3] += 20.0
        return z

    job, g, mos = _full_resolution(town)
    _, _, z, z_smooth = relief._rasters(job, mos, g)
    _, _, _, ground = relief._rasters(job, relief.Mosaic(plane(*_metres(*np.meshgrid(
        mos.lon0 + np.arange(mos.z.shape[1]) * mos.dlon, mos.lat0 - np.arange(mos.z.shape[0]) * mos.dlat))),
        mos.lon0, mos.dlon, mos.lat0, mos.dlat), g)
    assert np.abs(z_smooth - ground).max() < 1.0, "the buildings are gone, and no terrace is left in their place"
    assert np.abs(z - ground).max() > 5.0, "setup: the crisp surface has them"
    sigma = 600 / 2.3548
    job, g, mos = _full_resolution(lambda x, y: 600 + 40 * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
    _, _, z, z_smooth = relief._rasters(job, mos, g)
    assert z_smooth.max() - 600 >= 0.9 * 40 and z.max() - 600 >= 0.95 * 40


DETAIL_MAPS = [(w_km, size) for w_km in (2.9, 5.6, 12, 30, 100) for size in ("A4", "A3", "50x70")]


@pytest.mark.parametrize("w_km,size", DETAIL_MAPS)
def test_the_detail_grid_holds_the_sky_view_radius_and_the_level_is_unchanged(monkeypatch, w_km, size):
    """The margin of the detail grid reaches at least the sky view's radius (so every horizon of a map node lies on
    the grid), at every level of both datasets; and the detail level, which fixes the blocks a map downloads, is the
    one of the level rule as it was: the coarsest whose spacing is within max(45 m, 0.40 mm of paper)."""
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": 3600, "glo90": 1200})
    map_w = render.SIZES[size][0] * (1 - 36 / 297)
    extent, crs = _utm_extent(40.5, -3.5, w_km * 1000, 0.85 * w_km * 1000)
    job = relief._Map(0, extent, crs, map_w)
    for ds in ("glo30", "glo90"):
        for L in range(relief.NOMINAL_LEVELS[ds]):
            assert job.detail_grid(relief._spacing(ds, L)).reach >= job.tex_radius, (ds, L)
    sigma = max(45.0, 0.40 * w_km * 1000 / map_w)
    for ds, n in (("glo30", 4), ("glo90", 3)):
        h0 = {"glo30": 3600, "glo90": 1200}[ds]
        want = max([L for L in range(n) if 110574.0 / h0 * 2 ** L <= sigma] or [0])
        assert relief._detail_level(job, ds, n) == want, ds


def test_the_relief_fields_are_float32_rasters_of_the_map(monkeypatch, tmp_path):
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep)})
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    for name in ("shade", "smooth", "z", "z_smooth"):
        a = getattr(rel, name)
        assert a.shape == relief.raster_shape(A4_MAP, extent) and a.dtype == np.float32, name
        assert np.isfinite(a).all(), name
    assert 200 < rel.z.min() < rel.z.max() < 1000, "heights in metres"
    assert relief.Relief(rel.shade, "glo30", 0, 0.5).smooth is None, "the old positional form still builds a Relief"


# three latitude bands of the Copernicus cells; the last two on their UTM zone's central meridian, where the top and
# bottom edges of a map bulge away from the pole in degrees at mid-edge (the corners alone would miss it)
GEO_PLACES = [(40.4, -3.7), (60.2, 10.7), (69.6, 18.9), (69.6, 21.0), (60.2, 9.0)]


@pytest.mark.parametrize("lat,lon", GEO_PLACES)
def test_every_node_and_its_ground_filter_lie_inside_the_requested_windows(monkeypatch, lat, lon):
    """Pure geometry at the real resolution (no data is read): the probe and detail nodes of a map, at every level of
    both datasets, fall inside the windows the request covers; at full resolution so does the 150 m reach of the
    ground filter around each of them."""
    monkeypatch.setattr(relief, "HEIGHT0", {"glo30": 3600, "glo90": 1200})
    for w_m, map_w in ((12000, A4_MAP), (60000, 60.0)):
        extent, crs = _utm_extent(lat, lon, w_m, 0.85 * w_m)
        job = relief._Map(0, extent, crs, map_w)
        for ds in ("glo30", "glo90"):
            for L in range(relief.NOMINAL_LEVELS[ds]):
                s = relief._spacing(ds, L)
                reach = relief.OPEN_M if s <= relief.OPEN_MAX_SPACING_M else 0.0
                for grid, pad, around in ((job.probe_grid, job.probe_pad(s), 0.0),
                                          (job.detail_grid(s), job.detail_pad(s), reach)):
                    box, cells = job.box(pad)
                    mos = relief._mosaic(ds, {}, cells, L, box, None)  # every cell as sea: the windows only
                    lo, la = relief._nodes(extent, crs, grid)
                    relief._sample(mos, lo, la)  # raises if a node falls outside
                    relief._sample_cubic(mos, lo, la)
                    if around:
                        dlat = around / relief.M_PER_DEG_LAT
                        dlon = around / (relief.M_PER_DEG_LON * np.cos(np.radians(la)))
                        relief._sample(mos, lo - dlon, la + dlat)
                        relief._sample(mos, lo + dlon, la - dlat)


REAL_FETCH = relief._fetch  # the real one: the autouse fixture replaces relief._fetch for every test


class _Response:
    def __init__(self, status, body, content_range, reads):
        self.status, self._body, self._reads = status, body, reads
        self.headers = {"Content-Range": content_range} if content_range else {}

    def read(self, n=-1):
        self._reads.append((self.status, n))
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    """S3 as relief._OPENER sees it: 206 with Content-Range for files {url: bytes}, 404 otherwise, and one kind of
    failure wherever when(url, first) says. Checks every request against the privacy contract."""

    def __init__(self, files, failure=None, when=lambda url, first: True):
        self.files, self.failure, self.when, self.calls, self.reads = files, failure, when, [], []

    def open(self, req, timeout=None):
        import http.client
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
        if fail == "URLError":  # the messages carry the URL, as real ones may: the log must not repeat them
            raise urllib.error.URLError(f"unreachable: {u}")
        if fail == "TimeoutError":
            raise TimeoutError(f"timed out: {u}")
        if fail == "ConnectionResetError":
            raise ConnectionResetError(f"reset by peer: {u}")
        if fail == "IncompleteRead":
            raise http.client.IncompleteRead(b"x")
        if fail in ("HTTP 503", "HTTP 403", "HTTP 301"):
            raise urllib.error.HTTPError(u, int(fail[5:]), "no", {}, None)
        if fail == "200 instead of 206":  # the whole file, with a Content-Range that would pass
            return _Response(200, data, cr, self.reads)
        if fail == "wrong Content-Range":
            cr = f"bytes {first + 1}-{end}/{len(data)}"
        if fail in ("short body", "short header"):
            body = body[:-1]
        return _Response(206, body, cr, self.reads)


def _detail_offsets(data, cell=(40, -4)):
    return set(relief.read_header(data[:relief.HEADER_BYTES], cell).levels[0].offsets)


def test_a_redirect_is_refused():
    import email.message
    import urllib.error
    import urllib.request
    import urllib.response

    asked = []

    class Moved(urllib.request.HTTPSHandler):  # replaces the real HTTPS handler: nothing leaves the machine
        def https_open(self, req):
            asked.append(req.full_url)
            hdrs = email.message.Message()
            hdrs["Location"] = "https://elsewhere.example/x.tif"
            r = urllib.response.addinfourl(io.BytesIO(b""), hdrs, req.full_url, 301)
            r.msg = "Moved Permanently"
            return r

    opener = urllib.request.build_opener(relief._NoRedirect, Moved)
    with pytest.raises(urllib.error.HTTPError) as ex:
        opener.open(urllib.request.Request(relief.url("glo30", (40, -4))), timeout=1)
    # a followed redirect would also end in an HTTPError 301 (urllib gives up on the loop): count the requests
    assert ex.value.code == 301 and asked == [relief.url("glo30", (40, -4))], "the redirect is never followed"
    req = urllib.request.Request(relief.url("glo30", (40, -4)))
    assert relief._NoRedirect().redirect_request(req, None, 301, "Moved", {}, "https://elsewhere.example/") is None
    assert any(isinstance(h, relief._NoRedirect) for h in relief._OPENER.handlers)
    assert not any(type(h) is urllib.request.HTTPRedirectHandler for h in relief._OPENER.handlers)
    assert not any(isinstance(h, urllib.request.HTTPCookieProcessor) for h in relief._OPENER.handlers)


def test_a_404_is_an_absent_cell_and_is_asked_once(monkeypatch):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)
    opener = _Opener({})
    monkeypatch.setattr(relief, "_OPENER", opener)
    assert relief._fetch(relief.url("glo90", (40, -4)), 0, 16383) is None
    assert len(opener.calls) == 1, "an answer, not a failure: never asked again"


class _Fixed:
    """An S3 that answers every request with the same 206 response."""

    def __init__(self, body, content_range):
        self.body, self.content_range, self.reads = body, content_range, []

    def open(self, req, timeout=None):
        return _Response(206, self.body, self.content_range, self.reads)


def test_a_range_answer_must_be_exactly_the_range(monkeypatch):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)
    data = bytes(range(256)) * 4
    u = relief.url("glo30", (40, -4))
    good = _Fixed(data[10:21], "bytes 10-20/1024")
    monkeypatch.setattr(relief, "_OPENER", good)
    assert relief._fetch(u, 10, 20) == data[10:21]
    assert good.reads == [(206, 12)], "one bounded read: at most one byte more than the range, so a longer body shows"
    monkeypatch.setattr(relief, "_OPENER", _Fixed(data[1020:], "bytes 1020-1023/1024"))
    assert relief._fetch(u, 1020, 1100) == data[1020:], "a range past the end of the file ends with the file"
    for body, content_range in ((data[10:21] + b"x", "bytes 10-20/1024"),  # one byte too many
                                (data[10:20], "bytes 10-20/1024"),  # one byte short
                                (data[10:21], "bytes 10-21/1024"),  # a range that ends elsewhere
                                (data[10:21], "bytes 11-20/1024"),  # ... or starts elsewhere
                                (data[10:21], "bytes 10-20/*"),  # a file of unknown length
                                (data[10:21], None)):  # no Content-Range at all
        monkeypatch.setattr(relief, "_OPENER", _Fixed(body, content_range))
        with pytest.raises(relief.ReliefError):
            relief._fetch(u, 10, 20)


FAILURES = ["URLError", "TimeoutError", "ConnectionResetError", "IncompleteRead", "HTTP 503", "HTTP 403", "HTTP 301",
            "200 instead of 206", "wrong Content-Range", "short body", "short header", "garbage zlib body",
            "a block of the wrong size", "out-of-range value"]
RETRIED = {"URLError", "TimeoutError", "ConnectionResetError", "IncompleteRead", "HTTP 503"}  # broken or 5xx: once more
FORGOTTEN = {"short header", "garbage zlib body", "a block of the wrong size", "out-of-range value"}


@pytest.mark.parametrize("failure", FAILURES)
def test_a_failing_server_draws_the_poster_without_relief(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)
    good = _cog((40, -4), _steep)
    detail = _detail_offsets(good)
    served = good
    if failure in ("garbage zlib body", "a block of the wrong size"):
        hdr = relief.read_header(good[:relief.HEADER_BYTES], (40, -4))
        served = bytearray(good)
        for o, n in zip(hdr.levels[0].offsets, hdr.levels[0].counts):
            if failure == "garbage zlib body":
                served[o:o + n] = bytes((7 * i) % 251 for i in range(n))
            else:  # a whole zlib stream (what follows it is ignored) that inflates to 12 bytes, not 64 x 64 x 4
                small = zlib.compress(b"\0" * 12)
                served[o:o + n] = small + bytes(n - len(small))
        served = bytes(served)
    elif failure == "out-of-range value":
        a0 = _cell_grid((40, -4), H30, _steep)
        a0[150:340, 150:340] = 1e5  # under the whole map window; the probe level is kept clean
        lv = levels_of(a0, 4)
        lv[3] = levels_of(_cell_grid((40, -4), H30, _steep), 4)[3]
        served = write_cog(cell=(40, -4), levels=lv, tile=64)
    failing = _detail_offsets(served)
    # a short header still parses (its IFDs end long before 16 KB): only the length check can refuse it
    target = (lambda url, first: first == 0) if failure == "short header" else (lambda url, first: first in failing)
    u = relief.url("glo30", (40, -4))
    opener = _Opener({u: served}, failure, when=target)
    monkeypatch.setattr(relief, "_OPENER", opener)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    logs = []
    assert relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append) == [None]
    want = failure if failure in ("URLError", "TimeoutError", "ConnectionResetError") else (
        "HTTPError" if "HTTP" in failure else "ReliefError")
    assert logs[-1] == (f"[relief] skipped ({want}): the poster is drawn without relief; run the same command again "
                        f"later to add it"), logs[-1]
    text = "\n".join(logs)
    for leak in ("Copernicus_DSM", ".tif", "N40", "W004", "40.", "-3.", "amazonaws", "unreachable", "timed out",
                 "reset by peer"):
        assert leak not in text
    failed = [c for c in opener.calls if target(c[0], c[1])]
    assert len(failed) == (2 if failure in RETRIED else 1), "only a broken connection or a 5xx is tried once more"
    if failure == "200 instead of 206":
        assert not [n for status, n in opener.reads if status == 200], "a 200 is refused before its body is read"
    root = tmp_path / "cache" / "dem"
    cached = {p.name for p in root.rglob("*.zz")}
    assert not any(n.startswith("L0_") for n in cached), "nothing is cached for the failed blocks"
    header = root / "glo30" / "N40_00_W004_00" / "header.bin"
    if failure in FORGOTTEN:
        # a header that failed is never cached; blocks that do not decode at the header's offsets may mean the file
        # changed on the server: the cell is forgotten, so the next run starts it afresh
        assert not header.exists()
    else:
        assert header.exists(), "a network failure keeps the header that parsed"
    working = _Opener({u: good})  # for "out-of-range value", a file with another layout than the failed one
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


def test_the_notices_are_the_licences_words():
    """Art. 6(b) of each licence, verbatim and in full (the GLO-30 notice has no trademark sign; the GLO-90 one has)."""
    rest = (" © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
            "the European Union and ESA; all rights reserved")
    assert relief.NOTICE == {"glo30": "produced using Copernicus WorldDEM-30" + rest,
                             "glo90": "produced using Copernicus WorldDEM™-90" + rest}
    assert relief.notice("glo90", "en") == "Relief: " + relief.NOTICE["glo90"]
    assert relief.notice("glo90", "es") == "Relieve: " + relief.NOTICE["glo90"], "the notice itself stays in English"


def test_glo90_is_never_asked_for_a_cell_glo30_has(monkeypatch, tmp_path):
    files = {relief.url("glo30", (40, -4)): _cog((40, -4), _steep),
             relief.url("glo90", (40, -4)): _cog((40, -4), _steep, h0=H90, n_levels=3)}  # GLO-90 has every land cell
    calls = []
    monkeypatch.setattr(relief, "_fetch", fake_server(files, calls))
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert rel is not None and rel.dataset == "glo30", "the GLO-30 notice, on every print GLO-30 covers"
    assert calls and not any("90m" in u for u, _, _ in calls), "and not one request to GLO-90"


def test_a_cell_missing_from_both_datasets_is_sea(monkeypatch, tmp_path):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)  # S3's 404 itself, through the real fetch
    opener = _Opener({relief.url("glo30", (40, -4)): _cog((40, -4), _steep)})
    monkeypatch.setattr(relief, "_OPENER", opener)
    extent, crs = _utm_extent(40.5, -3.93, 12000, 10000)  # its box reaches into (40, -5), absent from both
    job = relief._Map(0, extent, crs, A4_MAP)
    assert (40, -5) in job.cells
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert rel is not None and rel.dataset == "glo30"
    sea = [c for c in opener.calls if "W005" in c[0]]
    assert sea and all(first == 0 for _, first, _ in sea), "only the header, on both datasets: no block requests"
    assert {u.split(".s3")[0] for u, _, _ in sea} == {"https://copernicus-dem-30m", "https://copernicus-dem-90m"}
    # the elevation the relief was made from: exactly 0 m west of -4 degrees, the land east of it
    root = str(tmp_path / "cache" / "dem")
    box, cells = job.box(job.detail_pad(relief._spacing("glo30", rel.level)))
    mos = relief._mosaic("glo30", {c: relief._header("glo30", c, root) for c in cells}, cells, rel.level, box, root)
    lon = mos.lon0 + np.arange(mos.z.shape[1]) * mos.dlon
    assert (lon < -4).any() and not mos.z[:, lon < -4].any(), "the sea cell is 0 m"
    assert (mos.z[:, lon > -4] > 100).all()


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


def test_a_failure_on_one_map_leaves_the_whole_print_without_relief(monkeypatch, tmp_path):
    monkeypatch.setattr(relief, "_fetch", REAL_FETCH)
    files = {relief.url("glo30", c): _cog(c, _steep) for c in ((40, -4), (40, -3))}
    u2 = relief.url("glo30", (40, -3))
    second = _detail_offsets(files[u2], (40, -3))
    opener = _Opener(files, "HTTP 503", when=lambda url, first: url == u2 and first in second)
    monkeypatch.setattr(relief, "_OPENER", opener)
    m1, m2 = _utm_extent(40.5, -3.5, 12000, 10000), _utm_extent(40.5, -2.5, 12000, 10000)
    logs = []
    assert relief.prepare_many([(m1[0], m1[1], 60.0), (m2[0], m2[1], 60.0)], log=logs.append) == [None, None]
    u1 = relief.url("glo30", (40, -4))
    first_map = _detail_offsets(files[u1])
    assert any(u == u1 and first in first_map for u, first, _ in opener.calls), \
        "setup: the first map's blocks were fetched before the second one's failed"
    assert [line for line in logs if "skipped" in line] == [logs[-1]], "one line for the print"


def test_what_each_map_of_a_print_is_told(monkeypatch, tmp_path):
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep), (40, -3): _cog((40, -3), lambda lon, lat: 650 + 0 * lon)})
    m1, m2 = _utm_extent(40.5, -3.5, 12000, 10000), _utm_extent(40.5, -2.5, 12000, 10000)
    logs = []
    out = relief.prepare_many([(m1[0], m1[1], 60.0), (m2[0], m2[1], 60.0)], log=logs.append)
    assert isinstance(out[0], relief.Relief) and out[1] is None
    assert "[relief] map 2: flat here: no relief" in logs
    _serve(monkeypatch, {})  # no cell on either dataset: open sea
    logs = []
    assert relief.prepare_many([(m1[0], m1[1], 60.0)], log=logs.append, root=str(tmp_path / "sea")) == [None]
    assert logs[-1] == "[relief] only sea here: no relief", "a sea print is not a failure to retry"


def test_a_programming_error_is_not_taken_for_a_failed_download(monkeypatch, tmp_path):
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep)})

    def bug(*args):
        raise TypeError("a bug")

    monkeypatch.setattr(relief, "_shade", bug)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    with pytest.raises(TypeError):
        relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)


def test_a_map_the_data_cannot_cover_is_not_told_to_retry(monkeypatch, tmp_path):
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep)})

    def outside(*args):
        raise relief._Unsupported("a sample fell outside the elevation data")

    monkeypatch.setattr(relief, "_sample", outside)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    logs = []
    assert relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append) == [None]
    assert logs[-1] == ("[relief] skipped: the elevation data does not cover this map as it is; the poster is drawn "
                        "without relief")


def test_a_map_next_to_the_180th_meridian_gets_no_relief_and_asks_for_nothing(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, {(-17, 179): _cog((-17, 179), _steep)})
    lat = -16.6
    crs = render.utm_crs(179.9, lat)
    x180, y = Transformer.from_crs(4326, crs, always_xy=True).transform(180.0, lat)
    west = render.utm_crs(-179.9, lat)  # and from the other side, in UTM zone 1
    x_w, y_w = Transformer.from_crs(4326, west, always_xy=True).transform(-180.0, lat)
    for x0, x1, y_, crs_ in ((x180 - 12500, x180 - 500, y, crs), (x180 - 6000, x180 + 6000, y, crs),  # short, across
                             (x_w + 500, x_w + 12500, y_w, west)):  # 0.5 km east of it
        logs = []
        assert relief.prepare_many([((x0, x1, y_ - 5000, y_ + 5000), crs_, A4_MAP)], log=logs.append) == [None]
        assert logs[-1] == "[relief] no relief next to the 180th meridian: drawn without it"
    assert not calls, "no request, so none for a cell east of 179 or west of -180 (there is none)"
    extent, crs = _utm_extent(-16.5, 179.4, 12000, 10000)  # well clear of it
    assert isinstance(relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)[0], relief.Relief)
    assert all("E179" in u for u, _, _ in calls)


def test_a_cell_replaced_on_the_server_is_fetched_afresh_on_the_next_run(monkeypatch, tmp_path):
    release1 = _cog((40, -4), _steep)
    release2 = _cog((40, -4), lambda lon, lat: _steep(lon, lat) + 37.0 * np.sin(lat * 50))  # other tile sizes
    assert _detail_offsets(release1) != _detail_offsets(release2)
    _serve(monkeypatch, {(40, -4): release1})
    a = _utm_extent(40.8, -3.8, 12000, 10000)  # the north-west of the cell
    b = _utm_extent(40.2, -3.2, 12000, 10000)  # the south-east: other blocks
    assert relief.prepare_many([(a[0], a[1], A4_MAP)], log=lambda *_: None)[0] is not None
    _serve(monkeypatch, {(40, -4): release2})  # a new release replaces the file: the cached header is stale
    logs = []
    assert relief.prepare_many([(b[0], b[1], A4_MAP)], log=logs.append) == [None]
    assert logs[-1].endswith("run the same command again later to add it")
    (rel,) = relief.prepare_many([(b[0], b[1], A4_MAP)], log=lambda *_: None)
    assert rel is not None, "and running it again does add it"


def test_cache_is_reused_offline_and_validated(monkeypatch, tmp_path):
    import re
    import urllib.error
    files = {(40, -4): _cog((40, -4), _steep)}
    calls = _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.93, 12000, 10000)  # on the coast: its box reaches the sea cell (40, -5)
    assert (40, -5) in relief._Map(0, extent, crs, A4_MAP).cells
    (first,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert first is not None
    offline_calls = []

    def offline(*args):
        offline_calls.append(args)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(relief, "_fetch", offline)
    (again,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert not offline_calls, "the cache alone answers, the sea cell's 'absent' marker included"
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
    header = root / "dem" / "glo30" / "N40_00_W004_00" / "header.bin"
    whole = header.read_bytes()
    header.write_bytes(whole[:200])  # damaged: it no longer parses
    calls.clear()
    (fourth,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    assert fourth is not None and np.array_equal(first.shade, fourth.shade), "never taken for sea"
    assert [first for _, first, _ in calls] == [0], "only the damaged header is fetched again"
    assert header.read_bytes() == whole
    for p in root.rglob("*"):
        if p.is_file():
            rel_ = p.relative_to(root).as_posix()
            assert re.fullmatch(r"dem/glo(30|90)/[NS]\d\d_00_[EW]\d{3}_00/(header\.bin|absent|L\d+_\d+_\d+\.zz)",
                                rel_), rel_


def test_maps_that_share_blocks_download_and_count_them_once(monkeypatch, tmp_path):
    import re
    files = {(40, -4): _cog((40, -4), _steep)}
    calls = _serve(monkeypatch, files)
    m1, m2 = _utm_extent(40.5, -3.5, 12000, 10000), _utm_extent(40.52, -3.48, 12000, 10000)  # 2 km apart
    logs = []
    out = relief.prepare_many([(m1[0], m1[1], 60.0), (m2[0], m2[1], 60.0)], log=logs.append)
    assert all(isinstance(r, relief.Relief) for r in out) and out[0].level == out[1].level == 0
    blocks, _ = _levels_of_ranges(files, calls)
    detail = [b for b in blocks if b[1] == 0]
    assert len(detail) == len(set(detail)), "each block is downloaded once"
    said = [int(m.group(1)) for line in logs if (m := re.fullmatch(r"\[relief\] downloading (\d+) blocks, .*", line))]
    assert said == [len(detail)], "and counted once"
    alone = []
    for k, m in enumerate((m1, m2)):  # each map on a print of its own
        calls.clear()
        relief.prepare_many([(m[0], m[1], 60.0)], log=lambda *_: None, root=str(tmp_path / f"alone{k}"))
        alone.append({b for b in _levels_of_ranges(files, calls)[0] if b[1] == 0})
    assert set(detail) == alone[0] | alone[1] and len(detail) < len(alone[0]) + len(alone[1]), "setup: shared blocks"


def test_the_shade_is_resized_to_the_output_bicubic(monkeypatch, tmp_path):
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep)})
    shades = []
    real = relief._blend
    monkeypatch.setattr(relief, "_blend", lambda zs, g, radius: shades.append(real(zs, g, radius)) or shades[-1])
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (rel,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None)
    g = relief._Map(0, extent, crs, A4_MAP).detail_grid(relief._spacing("glo30", rel.level))
    H, W = rel.shade.shape
    resized = []
    for full in shades:  # the crisp and the smooth shade, in whatever order their threads ran
        s = full[g.m:g.m + g.ny, g.m:g.m + g.nx].astype(np.float32)  # the margin cropped
        assert s.shape != rel.shade.shape, "setup: a resize"
        resized.append(np.clip(np.asarray(Image.fromarray(s).resize((W, H), Image.Resampling.BICUBIC), np.float32),
                               -1, 1))
    assert len(resized) == 2
    assert any(np.array_equal(rel.shade, r) for r in resized) and any(np.array_equal(rel.smooth, r) for r in resized)


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
    calls = _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.5, 12000, 10000)
    (fine,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None, root=str(tmp_path / "a"))
    monkeypatch.setattr(relief, "BUDGET_BYTES", 2000)
    calls.clear()
    (cached,) = relief.prepare_many([(extent, crs, A4_MAP)], log=lambda *_: None, root=str(tmp_path / "a"))
    assert cached.level == 0 and not calls, "the budget counts only the blocks still to download"
    logs = []
    (coarse,) = relief.prepare_many([(extent, crs, A4_MAP)], log=logs.append, root=str(tmp_path / "b"))
    assert fine.level == 0 and coarse.level > 0, "over the budget: a coarser level"
    assert coarse.shade.shape == fine.shade.shape
    big, crs = _utm_extent(40.5, -3.5, 400000, 400000)
    logs = []
    assert relief.prepare_many([(big, crs, A4_MAP)], log=logs.append) == [None]
    assert logs[-1] == "[relief] area too large for relief: drawn without it"
    # the limit itself: 16 cells go ahead (to the headers: all sea here), 18 do not (17 cells cannot be a rectangle
    # around a map: 1 x 17)
    _serve(monkeypatch, {})
    for (w, h), n, last in (((200000, 210000), 16, "[relief] only sea here: no relief"),
                            ((340000, 200000), 18, "[relief] area too large for relief: drawn without it")):
        ext, crs = _utm_extent(42.0, -2.0, w, h)
        assert len(relief._Map(0, ext, crs, A4_MAP).cells) == n
        logs = []
        assert relief.prepare_many([(ext, crs, A4_MAP)], log=logs.append, root=str(tmp_path / "c")) == [None]
        assert logs[-1] == last


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
            t_shadow, t_light = min(16, 0.5 * route), min(4, 0.2 * route)
        else:
            t_shadow, t_light = min(8, 0.6 * lb, 0.5 * route), min(12, 0.35 * route)
        (shadow, a_s), (light, a_l) = relief_colours(S)
        d_s = abs(_lstar((1 - a_s) * bg + a_s * np.asarray(shadow)) - lb)
        d_l = abs(_lstar((1 - a_l) * bg + a_l * np.asarray(light)) - lb)
        assert d_s <= t_shadow + 1e-9 and d_l <= t_light + 1e-9, f"{name}: louder than the rule"
        assert max(d_s, d_l) >= 1, f"{name}: the relief would not show"
        assert a_s <= 0.60 and a_l <= 0.70, name
        assert _lstar(shadow) < lb < _lstar(light), f"{name}: shadows darken, lights lighten"
        # under full light the map stays readable: every street class, the parks and the water lines keep at least
        # 70% of their contrast with the paper (a white veil over light paper fades them far more than it lights)
        lit_bg = _lstar((1 - a_l) * bg + a_l * np.asarray(light))
        kept = []
        for c in [S["street"][k][0] for k in ("major", "minor", "path")] + [S["green"], S["water_line"]]:
            c = np.asarray(to_rgb(c))
            lit = abs(_lstar((1 - a_l) * c + a_l * np.asarray(light)) - lit_bg)
            assert lit >= 0.70 * abs(_lstar(c) - lb) - 1e-9, f"{name}: the light washes the map out"
            if abs(_lstar(c) - lb) > 0:
                kept.append(lit / abs(_lstar(c) - lb))
        # and each side as strong as its rule lets it be: at its target, or at its cap, or (the light) where the map
        # under it keeps just 70% of its contrast
        assert abs(d_s - t_shadow) < 1e-3 or a_s == 0.60, f"{name}: a shadow quieter than its rule"
        assert abs(d_l - t_light) < 1e-3 or a_l == 0.70 or abs(min(kept) - 0.70) < 1e-3, f"{name}: a dim light"
    # the shadow alphas of the spec's table
    assert {name: round(relief_colours(S)[0][1], 3) for name, S in STYLES.items()} == {
        "huella": 0.223, "carbon": 0.485, "midnight": 0.512, "blueprint": 0.242, "ink": 0.195, "graphite": 0.471,
        "neon": 0.593, "lavender": 0.128, "sunset": 0.116, "forest": 0.136, "blush": 0.128, "ocean": 0.109}


def test_a_style_can_set_its_own_relief_colours():
    from pinless.render_editorial import STYLES, relief_colours
    custom = (((0.1, 0.2, 0.3), 0.2), ((0.9, 0.9, 0.9), 0.1))
    assert relief_colours(dict(STYLES["huella"], relief=custom)) == custom


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
    strong = relief_rgba(np.array([[-1.0, -0.3, 0.3, 0.9, 1.0]], np.float32), relief_colours(STYLES["huella"]))
    (shadow, a_s), (light, a_l) = relief_colours(STYLES["huella"])
    ramp = (0.3 / 0.9) ** 0.85  # alpha = side alpha * min(1, |s| / 0.9) ** 0.85
    assert strong[0, :, 3].tolist() == [round(255 * a_s), round(255 * a_s * ramp), round(255 * a_l * ramp),
                                        round(255 * a_l), round(255 * a_l)]
    assert strong[0, 0, :3].tolist() == strong[0, 1, :3].tolist() == [round(255 * c) for c in shadow], "shadow"
    assert strong[0, 2, :3].tolist() == strong[0, 4, :3].tolist() == [round(255 * c) for c in light], "light"
    assert [round(255 * c) for c in shadow] == [34, 40, 32] and light == (1.0, 1.0, 1.0), "huella: its ink, white"


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


def _map_pixel(png, extent, map_w, x, y, dpi=60):
    """The colour of the A4 editorial page at map point (x, y)."""
    with Image.open(png) as img:
        px = np.asarray(img.convert("RGB"))
    k = 210 / 297
    x_mm = 18 * k + (x - extent[0]) / (extent[1] - extent[0]) * map_w
    y_mm = 94 * k + (extent[3] - y) / (extent[3] - extent[2]) * 222 * k
    return tuple(px[int(round(y_mm / 25.4 * dpi)), int(round(x_mm / 25.4 * dpi))])


def _render_over_water(tmp_path, water, name, dpi=60):
    """An A4 editorial page with a steep relief everywhere and this water; (png, extent, map width)."""
    import geopandas as gpd
    from shapely.geometry import LineString
    extent, map_w = _editorial_extent("A4")
    rel = relief.Relief(np.full(relief.raster_shape(map_w, extent), -0.9, np.float32), "glo30", 0, 0.5)
    out = tmp_path / name
    render_editorial.render([(0, LineString([(500, 9000), (900, 9400)]))],
                            (_empty(), _empty(), gpd.GeoDataFrame({"natural": ["water"]}, geometry=[water])), extent,
                            STATS, str(out), size="A4", dpi=dpi, city="X", year="2025", relief=rel)
    return out, extent, map_w


def _water_rgb():
    from matplotlib.colors import to_rgb
    return tuple(round(c * 255) for c in to_rgb(render_editorial.STYLES["huella"]["water"]))


def test_multipolygon_water_stays_flat(tmp_path):
    from shapely.geometry import MultiPolygon, Polygon
    lakes = MultiPolygon([Polygon([(2000, 2000), (5000, 2000), (5000, 5000), (2000, 5000)]),
                          Polygon([(7000, 2000), (10000, 2000), (10000, 5000), (7000, 5000)])])  # one OSM relation
    out, extent, map_w = _render_over_water(tmp_path, lakes, "m.png")
    assert _map_pixel(out, extent, map_w, 3500, 3500) == _map_pixel(out, extent, map_w, 8500, 3500) == _water_rgb()
    assert _map_pixel(out, extent, map_w, 6000, 3500) != _water_rgb(), "setup: the land between them is shaded"


def test_the_relief_stops_at_the_drawn_shore(tmp_path):
    from shapely.geometry import Point
    lake = Point(6000, 5000).buffer(2000, quad_segs=64)  # a round lake: a clip that cuts corners shows on its shore
    out, extent, map_w = _render_over_water(tmp_path, lake, "c.png", dpi=100)
    shaded = [a for a in np.arange(0, 360, 2.5)
              if _map_pixel(out, extent, map_w, 6000 + 1950 * math.cos(math.radians(a)),
                            5000 + 1950 * math.sin(math.radians(a)), dpi=100) != _water_rgb()]
    assert not shaded, f"relief on the lake 50 m inside its shore at {shaded[:5]} degrees"
    outside = [_map_pixel(out, extent, map_w, 6000 + 2100 * math.cos(math.radians(a)),
                          5000 + 2100 * math.sin(math.radians(a)), dpi=100) for a in np.arange(0, 360, 15)]
    assert all(p != _water_rgb() for p in outside), "setup: 100 m outside it, the land is shaded"


def test_the_drawn_relief_is_north_up(tmp_path):
    from matplotlib.colors import to_rgb
    from shapely.geometry import LineString
    from pinless import render_editorial
    extent, map_w = _editorial_extent("A4")
    shade = np.zeros(relief.raster_shape(map_w, extent), np.float32)
    shade[: shade.shape[0] // 2] = -0.9  # row 0 is north: the north half is in shadow, the south half flat
    out = tmp_path / "n.png"
    info = render_editorial.render([(0, LineString([(500, 300), (900, 400)]))], (_empty(), _empty(), _empty()), extent,
                                   STATS, str(out), size="A4", dpi=60, city="X", year="2025",
                                   relief=relief.Relief(shade, "glo30", 0, 0.5))
    assert info["relief"] is True
    bg = tuple(round(c * 255) for c in to_rgb(render_editorial.STYLES["huella"]["bg"]))
    north, south = extent[2] + 0.75 * (extent[3] - extent[2]), extent[2] + 0.25 * (extent[3] - extent[2])
    assert _map_pixel(out, extent, map_w, 6000, north) != bg, "the north of the map is shaded"
    assert _map_pixel(out, extent, map_w, 6000, south) == bg, "the south of the map is untouched paper"


def test_odd_water_never_stops_the_poster(monkeypatch, tmp_path):
    import geopandas as gpd
    from matplotlib.colors import to_rgb
    from shapely.geometry import LineString, Polygon
    from pinless import render_editorial
    extent, map_w = _editorial_extent("A4")
    rel = relief.Relief(np.full(relief.raster_shape(map_w, extent), -0.9, np.float32), "glo30", 0, 0.5)
    S = render_editorial.STYLES["huella"]
    route = [(0, LineString([(500, 300), (900, 400)]))]
    # a self-crossing lake (OSM has some; draw_basemap fills both lobes): the relief stays off it, and no crash
    bowtie = Polygon([(3000, 3000), (8000, 7000), (8000, 3000), (3000, 7000)])
    assert not bowtie.is_valid
    water = gpd.GeoDataFrame({"natural": ["water"]}, geometry=[bowtie])
    out = tmp_path / "b.png"
    info = render_editorial.render(route, (_empty(), _empty(), water), extent, STATS, str(out), size="A4", dpi=60,
                                   city="X", year="2025", relief=rel)
    assert info["relief"] is True
    assert _map_pixel(out, extent, map_w, 3800, 5000) == tuple(round(c * 255) for c in to_rgb(S["water"]))
    assert _map_pixel(out, extent, map_w, 1500, 5000) != tuple(round(c * 255) for c in to_rgb(S["bg"]))
    # a map that is all water: nothing to shade, so no relief and no relief notice
    boxes = _footer_spy(monkeypatch)
    sea = Polygon([(-100, -100), (12100, -100), (12100, 12100), (-100, 12100)])
    info = render_editorial.render(route, (_empty(), _empty(), gpd.GeoDataFrame({"natural": ["water"]}, geometry=[sea])),
                                   extent, STATS, str(tmp_path / "s.png"), size="A4", dpi=20, city="X", year="2025",
                                   relief=rel)
    assert info["relief"] is False and not any("Copernicus" in b[0] for b in boxes)
    assert any(b[0].startswith("© OpenStreetMap") for b in boxes)


def _svg_images(svg):
    import re
    return re.findall(r'<image [^>]*?xlink:href="([^"]*)"', svg.read_text(encoding="utf-8"))


def test_an_svg_holds_its_relief_whatever_the_user_matplotlibrc(monkeypatch, tmp_path):
    import matplotlib
    from shapely.geometry import LineString
    from pinless import render_editorial
    monkeypatch.setitem(matplotlib.rcParams, "svg.image_inline", False)  # as a user's matplotlibrc may say
    extent, map_w = _editorial_extent("A4")
    folder = tmp_path / "Users" / "someone" / "Desktop"
    folder.mkdir(parents=True)
    svg = folder / "poster.svg"
    info = render_editorial.render([(0, LineString([(1000, 1000), (11000, 9000)]))], (_empty(), _empty(), _empty()),
                                   extent, STATS, str(folder / "poster.png"), size="A4", dpi=20, city="X",
                                   year="2025", svg_path=str(svg),
                                   relief=_synthetic_relief(relief.raster_shape(map_w, extent)))
    assert info["relief"] is True
    hrefs = _svg_images(svg)
    assert len(hrefs) == 1 and hrefs[0].startswith("data:image/png;base64,"), "the image is inside the SVG"
    assert "someone" not in svg.read_text(encoding="utf-8"), "no local path"
    assert sorted(p.name for p in folder.iterdir()) == ["poster.png", "poster.svg"], "no PNG written next to it"


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
    from shapely.geometry import LineString
    from matplotlib.figure import Figure
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
    for mode in race_modes:
        boxes.clear()
        _race_page(tmp_path, (_empty(), _empty(), _empty()), dataset, color_by=mode, lang=lang)
        _check_footer(boxes, 404, 412, dataset, lang, mode=mode)
    # the atlas, with the +N note
    boxes.clear()
    _atlas_page(tmp_path, [None, None], dataset, lang=lang, more=3)
    _check_footer(boxes, 404, 412, dataset, lang, more_note=True)


def _race_page(tmp_path, basemap, dataset="glo30", make=None, **kw):
    """render_race.render of a 5 km course on A4 with this basemap and a synthetic relief (none if dataset is None;
    make(shape, dataset) builds another): its info."""
    import datetime as dt
    from shapely.geometry import LineString
    from pinless import render_race
    from pinless.ingest import Activity
    xy = np.column_stack([np.linspace(0, 5000, 501), 800 * np.sin(np.linspace(0, 3, 501))])
    act = Activity(id="r", type="Run", start=dt.datetime(2025, 4, 6, 7, tzinfo=dt.timezone.utc),
                   lonlat=[(-3.70, 40.41)] * 501, source="x", times=list(np.linspace(0, 1500, 501)),
                   ele=list(600 + 40 * np.sin(np.linspace(0, 6, 501))))
    ext = _race_extent()
    shape = relief.raster_shape(_editorial_extent("A4")[1], ext)
    rel = (make or _synthetic_relief)(shape, dataset) if dataset else None
    return render_race.render(act, xy, [(0, LineString(xy[30:-30]))], basemap, ext, str(tmp_path / "r.png"),
                              size="A4", dpi=10, title="T", city="Madrid", date=act.start, relief=rel, **kw)


def _race_extent():
    from pinless import render_race
    return (-500.0, 5500.0, -2500.0, -2500.0 + 6000.0 / render_race.map_aspect("A4"))


def _atlas_world():
    """Two cities (Madrid, Paris) ready for atlas.render_atlas: (cities, cells, extents)."""
    import datetime as dt
    from shapely.geometry import LineString
    from pinless import atlas
    from pinless.ingest import Activity
    acts = [Activity(id=str(i), type="Run", start=dt.datetime(2025, 1 + i % 12, 3, 8),
                     lonlat=[(lon + 0.01 * np.cos(t), 40.41 + 0.01 * np.sin(t)) for t in np.linspace(0, 6.28, 60)],
                     source="x") for i, lon in enumerate([-3.70] * 4 + [2.35] * 2)]
    cities = atlas.group_by_city(acts)
    for c in cities:
        c.crs = render.utm_crs(c.lon, c.lat)
        tr = Transformer.from_crs(4326, c.crs, always_xy=True)
        c.tracks = [np.column_stack(tr.transform(*np.asarray(a.lonlat).T)) for a in c.acts]
        c.pieces = [(i, LineString(t)) for i, t in enumerate(c.tracks)]
        c.name, c.country = "Somewhere", "Nowhere"
    cells = atlas.layout(len(cities))
    return cities, cells, [atlas.fit(c, w / h) for c, (_, _, w, h) in zip(cities, cells)]


def _atlas_page(tmp_path, basemaps, dataset="glo30", make=None, **kw):
    """atlas.render_atlas on A4 of the two cities, with these basemaps and a synthetic relief per map (make(shape,
    dataset) builds another): its info."""
    from pinless import atlas
    cities, cells, extents = _atlas_world()
    k = 210 / 297
    reliefs = [(make or _synthetic_relief)(relief.raster_shape(w * k, e), dataset) if dataset else None
               for (_, _, w, _), e in zip(cells, extents)]
    return atlas.render_atlas(cities, cells, extents, basemaps, str(tmp_path / "a.png"), size="A4", dpi=10,
                              reliefs=reliefs, **kw)


def _all_water(extent):
    import geopandas as gpd
    from shapely.geometry import box
    x0, x1, y0, y1 = extent
    return gpd.GeoDataFrame({"natural": ["water"]}, geometry=[box(x0 - 100, y0 - 100, x1 + 100, y1 + 100)])


def test_the_race_page_shades_no_water_and_credits_only_what_it_drew(monkeypatch, tmp_path):
    from pinless import render_race
    boxes = _footer_spy(monkeypatch)  # the race page writes through atlas.place_text
    seen, real = [], render_race.draw_relief
    monkeypatch.setattr(render_race, "draw_relief", lambda ax, rel, ext, S, water, *a: seen.append(water) or
                        real(ax, rel, ext, S, water, *a))
    sea = _all_water(_race_extent())
    info = _race_page(tmp_path, (_empty(), _empty(), sea))  # a course along a lake shore, all of its map water
    assert len(seen) == 1 and seen[0] is sea, "the relief is clipped off the water the page draws"
    assert info["relief"] is False, "nothing shaded"
    texts = [b[0] for b in boxes]
    assert not any("Copernicus" in t for t in texts), "so no relief notice"
    assert [b[2] for b in boxes if b[0].startswith("© OpenStreetMap")] == [412], "the footer of today"


def test_each_atlas_map_shades_no_water_and_the_print_credits_only_what_it_drew(monkeypatch, tmp_path):
    from pinless import atlas
    boxes = _footer_spy(monkeypatch)
    seen, real = [], atlas.draw_relief
    monkeypatch.setattr(atlas, "draw_relief", lambda ax, rel, ext, S, water, *a: seen.append(water) or
                        real(ax, rel, ext, S, water, *a))
    _, _, extents = _atlas_world()
    seas = [_all_water(e) for e in extents]
    info = _atlas_page(tmp_path, [(_empty(), _empty(), w) for w in seas])  # every map all water
    assert len(seen) == len(seas) == 2 and all(s is w for s, w in zip(seen, seas)), "each map's own water"
    assert not any("Copernicus" in b[0] for b in boxes), "nothing shaded, so no relief notice"
    assert [b[2] for b in boxes if b[0].startswith("© OpenStreetMap")] == [412], "the footer of today"
    assert info is not None
    boxes.clear()
    _atlas_page(tmp_path, [(_empty(), _empty(), seas[0]), (_empty(), _empty(), _empty())])  # the second map on land
    assert [b[0] for b in boxes].count(relief.notice("glo30", "en")) == 1, "one map drawn: the notice"


def _two_shades(shape, dataset="glo30"):
    """A relief whose crisp shade (lit everywhere) and smooth shade (in shadow everywhere) tell apart which is drawn."""
    return relief.Relief(np.full(shape, 0.8, np.float32), dataset, 0, 0.5, smooth=np.full(shape, -0.8, np.float32))


def test_the_city_couple_race_and_atlas_prints_draw_the_smooth_shade(monkeypatch, tmp_path):
    """Every print but the trail shades over the city ground filter (relief.smooth), where a town's roofs do not print
    as speckle; the crisp shade is the trail print's, and shade= still picks the raster to draw."""
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    from test_couple_poster import _couple
    seen = []
    real = render_editorial.relief_rgba
    monkeypatch.setattr(render_editorial, "relief_rgba", lambda shade, colours: seen.append(np.array(shade))
                        or real(shade, colours))

    def drew_smooth(what):
        assert seen and all((a == -0.8).all() for a in seen), f"{what}: the smooth shade"
        seen.clear()

    extent, map_w = _editorial_extent("A4")
    shape = relief.raster_shape(map_w, extent)
    pieces = [(0, LineString([(1000, 1000), (11000, 9000)])), (1, LineString([(1000, 9000), (11000, 1000)]))]
    for couple in (None, _couple(owner=(0, 1))):
        info = render_editorial.render(pieces, (_empty(), _empty(), _empty()), extent, STATS, str(tmp_path / "e.png"),
                                       size="A4", dpi=10, city="X", year="2025", relief=_two_shades(shape),
                                       couple=couple)
        assert info["relief"] is True
        drew_smooth("the poster for two" if couple else "the city poster")
    _race_page(tmp_path, (_empty(), _empty(), _empty()), make=_two_shades)
    drew_smooth("the race print")
    _atlas_page(tmp_path, [None, None], make=_two_shades)
    assert len(seen) == 2
    drew_smooth("the atlas")
    fig, ax = plt.subplots()
    rel = _two_shades(shape)
    render_editorial.draw_relief(ax, rel, extent, render_editorial.STYLES["huella"], None, 1.0, map_w)
    drew_smooth("draw_relief by default")
    render_editorial.draw_relief(ax, rel, extent, render_editorial.STYLES["huella"], None, 1.0, map_w,
                                 shade=rel.shade)
    assert len(seen) == 1 and (seen[0] == 0.8).all(), "shade= draws the raster it is given"
    plain = relief.Relief(rel.shade, "glo30", 0, 0.5)
    seen.clear()
    render_editorial.draw_relief(ax, plain, extent, render_editorial.STYLES["huella"], None, 1.0, map_w)
    assert len(seen) == 1 and (seen[0] == 0.8).all(), "a relief without a smooth shade draws its own"
    plt.close(fig)


def test_both_credit_lines_are_set_as_the_osm_line_of_today():
    S = render_editorial.STYLES["huella"]
    osm = "© OpenStreetMap contributors · ODbL · openstreetmap.org/copyright"
    for dataset, want in ((None, [(409, osm, 238)]),
                          ("glo30", [(405.6, osm, 238), (409, relief.notice("glo30", "en"), 220)])):
        calls = []
        render_editorial.footer_credits(lambda *a, **k: calls.append((a, k)), S, "en", 409, 238, 220, dataset)
        assert [(a[0], a[2], k["max_w"]) for a, k in calls] == [(18, text, w) for _, text, w in want]
        assert [a[1] for a, _ in calls] == pytest.approx([y for y, _, _ in want])
        assert [(k["pt"], k["color"]) for _, k in calls] == [(6.2, S["sub"])] * len(want), "6.2 pt, the sub colour"


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
    # race.run and atlas.run with arguments built without a relief attribute (as other callers do)
    from pinless import race
    real_run = race.run

    def bare(sel, pool, a, log=print):
        delattr(a, "relief")
        return real_run(sel, pool, a, log)

    monkeypatch.setattr(race, "run", bare)
    captured = {}
    _race_world(monkeypatch, captured)
    assert cli.main(RACE_ARGS + ["--out", str(tmp_path / "r0.png")]) == 0
    assert captured["relief"] is None
    monkeypatch.setattr(race, "run", real_run)
    acts = [_loop_act(i, 40.42, -3.70) for i in range(4)]
    got = {}
    monkeypatch.setattr(atlas.area, "place_names", lambda lat, lon, lang="en": ("C", "K"))
    monkeypatch.setattr(atlas, "_basemap", lambda c, ext, log: None)
    monkeypatch.setattr(atlas, "render_atlas", lambda drawn, *a, **k: got.update(k) or {"cap": 3.0, "max": 1})
    args = types.SimpleNamespace(zone=[], trim=300.0, zone_radius=500.0, max_cities=2, lang="en", year=None,
                                 style=None, size="A3", dpi=40, name="", title="", out=str(tmp_path / "a.png"),
                                 pdf=None, svg=None, preview=None, color_by="heat")
    assert atlas.run(acts, args, log=lambda *_: None) == 0 and got["reliefs"] is None
    # --relief shade on the race print: a Relief reaches render_race.render, and the notice is drawn; its SVG holds
    # the relief whatever the user's matplotlibrc says
    import matplotlib
    monkeypatch.setitem(matplotlib.rcParams, "svg.image_inline", False)
    monkeypatch.setattr(relief, "prepare_many", real_prepare)
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep)})
    captured, texts = {}, []
    _race_world(monkeypatch, captured, texts)
    folder = tmp_path / "race"
    folder.mkdir()
    assert cli.main(RACE_ARGS + ["--relief", "shade", "--out", str(folder / "r2.png"), "--svg",
                                 str(folder / "r2.svg")]) == 0
    assert isinstance(captured["relief"], relief.Relief) and captured["info"]["relief"] is True
    assert texts.count(relief.notice("glo30", "en")) == 1
    hrefs = _svg_images(folder / "r2.svg")
    assert len(hrefs) == 1 and hrefs[0].startswith("data:image/png;base64,")
    assert sorted(p.name for p in folder.iterdir()) == ["r2.png", "r2.svg"]
    capsys.readouterr()


def test_atlas_run_draws_each_map_its_relief_at_its_own_size(monkeypatch, tmp_path):
    import types
    import matplotlib
    from pinless import atlas
    monkeypatch.setitem(matplotlib.rcParams, "svg.image_inline", False)
    _serve(monkeypatch, {(40, -4): _cog((40, -4), _steep), (40, -3): _cog((40, -3), _steep)})
    acts = [_loop_act(i, 40.42, -3.70) for i in range(3)] + [_loop_act(i, 40.45, -2.60) for i in range(3, 6)]
    monkeypatch.setattr(atlas.area, "place_names", lambda lat, lon, lang="en": ("C", "K"))
    monkeypatch.setattr(atlas, "_basemap", lambda c, ext, log: (_empty(), _empty(), _empty()))
    got, real = {}, atlas.render_atlas

    def spy(drawn, cells, extents, *a, **k):
        got.update(k, cells=cells, extents=extents)
        return real(drawn, cells, extents, *a, **k)

    monkeypatch.setattr(atlas, "render_atlas", spy)
    boxes = _footer_spy(monkeypatch)
    folder = tmp_path / "atlas"
    folder.mkdir()
    args = types.SimpleNamespace(zone=[], trim=300.0, zone_radius=500.0, max_cities=2, lang="en", year=None,
                                 style=None, size="A4", dpi=20, name="", title="", out=str(folder / "a.png"),
                                 pdf=None, svg=str(folder / "a.svg"), preview=None, color_by="heat", relief="shade")
    assert atlas.run(acts, args, log=lambda *_: None) == 0
    k = render.SIZES["A4"][0] / 297  # A4: the atlas cells are laid out on the A3 grid and scaled
    assert len(got["reliefs"]) == len(got["cells"]) == 2
    for rel, (_, _, w, _), ext in zip(got["reliefs"], got["cells"], got["extents"]):
        assert isinstance(rel, relief.Relief) and rel.shade.shape == relief.raster_shape(w * k, ext)
    assert [b[0] for b in boxes].count(relief.notice("glo30", "en")) == 1, "one notice for the print"
    hrefs = _svg_images(folder / "a.svg")
    assert len(hrefs) == 2 and all(h.startswith("data:image/png;base64,") for h in hrefs)
    assert sorted(p.name for p in folder.iterdir()) == ["a.png", "a.svg"]
