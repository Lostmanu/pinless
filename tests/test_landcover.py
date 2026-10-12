"""The ground of the trail print: the ESA WorldCover reader, the tiles and what may be requested, the resampling and
the majority filter, the colours and the course ink over them, the credit line and the command line. All offline: a
fake server serves synthetic WorldCover-like files (conftest.py forbids the network)."""
import io
import math
import re

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from _cog_fixture import fake_server, landcover_levels, write_landcover_cog
from pinless import landcover, relief, render, render_editorial

W0, BLK, NLV = 192, 16, 4  # small tiles: 1/64 degree (1.7 km) at full resolution, 192 to 24 pixels
REAL = {name: getattr(landcover, name) for name in ("WIDTH0", "BLOCK", "NOMINAL_LEVELS")}  # before any patch
REAL_FETCH = landcover._fetch
REAL_RELIEF_FETCH = relief._fetch
LAND = [c for c in landcover.CLASSES if c not in (landcover.NO_DATA, landcover.WATER)]


@pytest.fixture(autouse=True)
def _small_tiles(monkeypatch):
    monkeypatch.setattr(landcover, "WIDTH0", W0)
    monkeypatch.setattr(landcover, "BLOCK", BLK)
    monkeypatch.setattr(landcover, "NOMINAL_LEVELS", NLV)


def _checker(lon, lat):
    """A class per full-resolution pixel (1/64 degree), different from its four neighbours."""
    i = np.floor(np.asarray(lon) * 64).astype(int) + 3 * np.floor(np.asarray(lat) * 64).astype(int)
    return np.asarray(LAND, dtype=np.uint8)[i % len(LAND)]


def _tile_cog(tile, fn=_checker, n=None, **bent):
    """A WorldCover-like file for the tile, its classes fn(lon, lat) at the full-resolution pixel centres."""
    S, W = tile
    n = n or W0
    lon = W + (np.arange(n) + 0.5) * 3 / n
    lat = S + 3 - (np.arange(n) + 0.5) * 3 / n
    LON, LAT = np.meshgrid(lon, lat)
    return write_landcover_cog(tile, landcover_levels(fn(LON, LAT), NLV), block=BLK, **bent)


def _serve(monkeypatch, files):
    calls = []
    monkeypatch.setattr(landcover, "_fetch", fake_server({landcover.url(t): b for t, b in files.items()}, calls))
    return calls


def _utm_extent(lat, lon, w_m, h_m):
    crs = render.utm_crs(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return (x - w_m / 2, x + w_m / 2, y - h_m / 2, y + h_m / 2), crs


# --- the reader ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("compression", [8, 32946])  # Deflate under its two TIFF codes
def test_the_reader_reads_every_level_bit_for_bit(monkeypatch, tmp_path, compression):
    monkeypatch.setattr(landcover, "WIDTH0", 200)  # 200, 100, 50, 25: partial edge blocks at every level
    rng = np.random.default_rng(4)
    a0 = np.asarray(landcover.CLASSES, dtype=np.uint8)[rng.integers(0, len(landcover.CLASSES), (200, 200))]
    levels = landcover_levels(a0, NLV)
    data = write_landcover_cog((39, -6), levels, block=BLK, compression=compression,
                               tags={(L, 262): (3, [1]) for L in range(NLV)})  # min-is-black, for Pillow
    _serve(monkeypatch, {(39, -6): data})
    hdr = landcover.read_header(data[:landcover.HEADER_BYTES], (39, -6))
    assert [(lv.w, lv.h) for lv in hdr.levels] == [(200, 200), (100, 100), (50, 50), (25, 25)]
    for L, want in enumerate(levels):
        lv = hdr.levels[L]
        got = np.full((lv.h, lv.w), 255, np.uint8)
        for br in range(math.ceil(lv.h / lv.th)):
            for bc in range(math.ceil(lv.w / lv.tw)):
                a = landcover._block((39, -6), hdr, L, br, bc, str(tmp_path / "lc"))
                got[br * lv.th:br * lv.th + a.shape[0], bc * lv.tw:bc * lv.tw + a.shape[1]] = a
        assert np.array_equal(got, want), f"level {L} bit for bit"
    im = Image.open(io.BytesIO(data))
    for L, want in enumerate(levels):
        im.seek(L)
        assert np.array_equal(np.asarray(im), want), f"Pillow agrees on level {L}"


def _refusals():
    good = landcover_levels(_checker(*np.meshgrid(-6 + (np.arange(W0) + 0.5) * 3 / W0,
                                                  42 - (np.arange(W0) + 0.5) * 3 / W0)), NLV)
    return {
        "big-endian": dict(levels=good, endian=">"),
        "compression 5": dict(levels=good, compression=5),
        "predictor 2": dict(levels=good, predictor=2),
        "16-bit samples": dict(levels=good, bits=16),
        "signed samples": dict(levels=good, sample_format=2),
        "two samples per pixel": dict(levels=good, tags={(0, 277): (3, [2])}),
        "separate planes": dict(levels=good, tags={(0, 284): (3, [2])}),
        "PixelIsPoint": dict(levels=good, raster_type=2),
        "tiepoint off by 0.01": dict(levels=good, tiepoint=(0, 0, 0, -6.01, 42.0, 0)),
        "another tile's tiepoint": dict(levels=good, tiepoint=(0, 0, 0, -3.0, 42.0, 0)),
        "another pixel size": dict(levels=good, scale=(3 / 190, 3 / W0, 0)),
        "a tile of another size": dict(levels=landcover_levels(good[0][:96, :96], NLV), scale=(3 / 96, 3 / 96, 0)),
        "blocks of another size": dict(levels=good, block=32),
        "one level missing": dict(levels=good[:3]),
        "an overview that is not a halving": dict(levels=good, tags={(1, 256): (4, [97])}),
        "an overview marked as a full image": dict(levels=good, tags={(2, 254): (4, [0])}),
        "a loop of IFDs": dict(levels=good, ifd_loop=True),
    }


@pytest.mark.parametrize("case", sorted(_refusals()))
def test_the_reader_refuses_what_it_cannot_decode(monkeypatch, case):
    kw = dict(block=BLK)
    kw.update(_refusals()[case])
    data = write_landcover_cog((39, -6), **kw)
    with pytest.raises(landcover.LandCoverError):
        landcover.read_header(data[:landcover.HEADER_BYTES], (39, -6))
    _serve(monkeypatch, {(39, -6): data})
    logs = []
    extent, crs = _utm_extent(40.5, -4.5, 30000, 25000)
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert logs[-1].startswith("[ground] skipped (LandCoverError): drawn with the style's ground")


def test_a_class_the_dataset_does_not_have_is_refused(monkeypatch, tmp_path):
    def odd(lon, lat):
        out = _checker(lon, lat)
        out[5, 5] = 7  # no such class
        return out

    _serve(monkeypatch, {(39, -6): _tile_cog((39, -6), odd)})
    extent, crs = _utm_extent(41.95, -5.95, 8000, 6000)  # the north-west corner of the tile: the odd pixel's block
    logs = []
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert "skipped (LandCoverError)" in logs[-1]
    assert not list((tmp_path / "cache" / "landcover").rglob("*.zz")), "a block that does not decode is not cached"


def test_tile_names_and_the_three_degree_grid_on_both_hemispheres():
    cases = {(40.85, -3.95): (39, -6), (37.02, -2.40): (36, -3), (46.0, 7.75): (45, 6), (-33.9, 151.2): (-36, 150),
             (-0.5, -0.5): (-3, -3), (0.0, 0.0): (0, 0), (-3.0, 3.0): (-3, 3), (2.999, -179.9): (0, -180)}
    for (lat, lon), tile in cases.items():
        assert landcover.tile_of(lat, lon) == tile, (lat, lon)
    names = {(39, -6): "N39W006", (36, -3): "N36W003", (-36, 150): "S36E150", (-3, -3): "S03W003", (0, 0): "N00E000",
             (0, -180): "N00W180", (-90, 177): "S90E177"}
    for tile, name in names.items():
        assert landcover.tile_name(tile) == name
    assert landcover.url((39, -6)) == ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                                       "ESA_WorldCover_10m_2021_v200_N39W006_Map.tif")
    assert landcover._tiles((39.9, 42.1, -3.5, -2.5)) == [(39, -6), (39, -3), (42, -6), (42, -3)], "a full rectangle"


def test_the_level_follows_the_map_pixel_with_the_real_sizes(monkeypatch):
    for name, value in REAL.items():
        monkeypatch.setattr(landcover, name, value)
    assert [round(landcover._pixel_m(L), 1) for L in range(7)] == [9.2, 18.4, 36.9, 73.7, 147.4, 294.9, 590.3]
    for w_km, want in ((10, 0), (13, 0), (30, 1), (60, 2), (300, 4), (900, 5), (2000, 6)):
        extent, crs = _utm_extent(40.85, -3.95, w_km * 1000, w_km * 1000)
        assert landcover._Map(0, extent, crs, 261.0).level() == want, w_km


def test_nearest_neighbour_on_the_map_grid_and_across_tiles(monkeypatch, tmp_path):
    files = {(39, -6): _tile_cog((39, -6)), (39, -3): _tile_cog((39, -3))}
    _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -3.0, 60000, 50000)  # across the -3 meridian: two tiles
    job = landcover._Map(0, extent, crs, 60.0)
    headers = landcover._headers(job.tiles, str(tmp_path / "lc"))
    L, blocks = landcover._plan(job, headers, str(tmp_path / "lc"))
    assert L == 0 and {b[0] for b in blocks} == {(39, -6), (39, -3)}
    got = landcover._classes(job, headers, L, str(tmp_path / "lc"))
    lon, lat = landcover._nodes(job)
    assert got.shape == job.shape == relief.raster_shape(60.0, extent)
    assert np.array_equal(got.ravel(), _checker(lon, lat)), "the pixel under every node's centre"


def test_the_full_rectangle_is_requested_and_logged(monkeypatch, tmp_path):
    import inspect
    assert set(inspect.signature(landcover.prepare_many).parameters) == {"maps", "log", "root"}
    files = {(39, -6): _tile_cog((39, -6))}
    calls = _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -4.5, 60000, 50000)
    logs = []
    out = landcover.prepare_many([(extent, crs, 60.0)], log=logs.append)
    assert isinstance(out[0], landcover.LandCover) and out[0].level == 0
    assert logs[0] == ("[ground] land cover from ESA WorldCover (Amazon S3; only the 3-degree tiles around the map "
                       "are requested)")
    hdr = landcover.read_header(files[(39, -6)], (39, -6))
    lv = hdr.levels[0]
    starts = {first: i for i, first in enumerate(lv.offsets)}
    assert calls[0][1:] == (0, landcover.HEADER_BYTES - 1), "the header first"
    got = {divmod(starts[first], math.ceil(lv.w / lv.tw)) for _, first, _ in calls[1:]}
    rows, cols = [b[0] for b in got], [b[1] for b in got]
    assert got == {(r, c) for r in range(min(rows), max(rows) + 1) for c in range(min(cols), max(cols) + 1)}, \
        "the full rectangle, no holes"
    assert len(got) > 1 and len(calls) == 1 + len(got), "setup: several blocks, each asked for once"
    assert re.fullmatch(rf"\[ground\] downloading {len(got)} blocks, \d+\.\d MB \(one time, cached\)", logs[1])
    assert all(u == landcover.url((39, -6)) for u, _, _ in calls), "only the WorldCover host, only that tile"
    for line in logs:  # no URL, tile name or coordinate in the log
        assert "http" not in line and "N39" not in line and not re.search(r"\d+\.\d{2,}", line), line


def test_a_missing_tile_is_water_and_is_not_asked_for_again(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, {(39, -6): _tile_cog((39, -6))})  # (39, -3) does not exist: the open sea
    extent, crs = _utm_extent(40.5, -3.0, 60000, 50000)
    out = landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)
    lon, _ = landcover._nodes(landcover._Map(0, extent, crs, 60.0))
    east = (lon >= -3.0).reshape(out[0].classes.shape)
    assert (out[0].classes[east] == landcover.WATER).mean() > 0.99, "the missing tile is water"
    assert (out[0].classes[~east] != landcover.WATER).mean() > 0.99, "the tile that exists keeps its land"
    sea = landcover.url((39, -3))
    assert [c for c in calls if c[0] == sea] == [(sea, 0, landcover.HEADER_BYTES - 1)], "asked once, its header"
    calls.clear()
    again = landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)
    assert calls == [] and np.array_equal(again[0].classes, out[0].classes), "the second run asks for nothing"


def test_a_print_where_every_tile_is_missing_is_not_taken_for_the_sea(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, {})  # a server, or a proxy, that has none of the tiles
    extent, crs = _utm_extent(40.5, -3.0, 60000, 50000)
    logs = []
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert logs[-1] == "[ground] no land cover here: drawn with the style's ground"
    assert not list(tmp_path.rglob("absent")), "nothing remembered as sea"
    n = len(calls)
    landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)
    assert len(calls) == 2 * n, "asked again next time"


def test_a_403_is_a_missing_tile_and_anything_else_fails(monkeypatch, tmp_path):
    import urllib.error
    from email.message import Message
    monkeypatch.setattr(landcover, "_fetch", REAL_FETCH)
    monkeypatch.setattr(relief, "_fetch", REAL_RELIEF_FETCH)
    for code in (403, 404, 400, 500):
        def opener_open(req, timeout=None, code=code):
            raise urllib.error.HTTPError(req.full_url, code, "no", Message(), None)
        monkeypatch.setattr(relief, "_OPENER", type("O", (), {"open": staticmethod(opener_open)}))
        if code in (403, 404):
            assert landcover._fetch(landcover.url((39, -6)), 0, 15) is None, code
        else:
            with pytest.raises(urllib.error.HTTPError):
                landcover._fetch(landcover.url((39, -6)), 0, 15)
        if code == 403:
            with pytest.raises(urllib.error.HTTPError):  # the elevation reader keeps a 403 an error
                relief._fetch(relief.url("glo30", (40, -4)), 0, 15)


def test_the_cache_is_reused_and_validated(monkeypatch, tmp_path):
    files = {(39, -6): _tile_cog((39, -6))}
    calls = _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -4.5, 60000, 50000)
    first = landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)[0]
    n = len(calls)
    calls.clear()
    assert np.array_equal(landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)[0].classes,
                          first.classes) and calls == [], "offline the second time"
    blocks = sorted((tmp_path / "cache" / "landcover").rglob("*.zz"))
    assert len(blocks) == n - 1 and all("N39W006" in str(b.parent) for b in blocks), "under landcover/, per tile"
    blocks[0].write_bytes(b"damaged")
    assert np.array_equal(landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)[0].classes,
                          first.classes) and len(calls) == 1, "a damaged block is fetched again, alone"


def test_a_replaced_file_is_forgotten_and_the_budget_takes_a_coarser_level(monkeypatch, tmp_path):
    files = {(39, -6): _tile_cog((39, -6))}
    _serve(monkeypatch, files)
    extent, crs = _utm_extent(40.5, -4.5, 60000, 50000)
    landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)
    for p in (tmp_path / "cache" / "landcover").rglob("*.zz"):
        p.unlink()
    files[(39, -6)] = _tile_cog((39, -6), lambda lon, lat: np.full(lon.shape, landcover.GRASS, np.uint8))[:5000]
    _serve(monkeypatch, files)  # the header cached, the file on the server since replaced (and cut short)
    logs = []
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert not (tmp_path / "cache" / "landcover" / "v200" / "N39W006").exists(), "the tile is forgotten"
    files[(39, -6)] = _tile_cog((39, -6))
    _serve(monkeypatch, files)
    monkeypatch.setattr(landcover, "BUDGET_BYTES", 1)
    out = landcover.prepare_many([(extent, crs, 60.0)], log=lambda *_: None)
    assert out[0].level == NLV - 1, "over budget: down to the coarsest level"


def test_maps_next_to_the_180th_meridian_or_too_large_ask_for_nothing(monkeypatch):
    calls = _serve(monkeypatch, {})
    logs = []
    extent, crs = _utm_extent(-16.8, 179.99, 6000, 5000)
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert "180th meridian" in logs[-1] and not calls
    extent, crs = _utm_extent(40.0, -3.0, 1_500_000, 1_500_000)
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert "area too large" in logs[-1] and not calls


def test_a_failing_server_is_one_line_and_no_ground(monkeypatch):
    import urllib.error

    def down(*args):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(landcover, "_fetch", down)
    logs = []
    extent, crs = _utm_extent(40.5, -4.5, 60000, 50000)
    assert landcover.prepare_many([(extent, crs, 60.0)], log=logs.append) == [None]
    assert logs[1:] == ["[ground] skipped (URLError): drawn with the style's ground; run the same command again later "
                        "to add the land cover"]


# --- the majority filter and the colours ------------------------------------------------------------------------------

def test_the_majority_filter_takes_out_speckle_and_keeps_edges_and_bands():
    T, G, B = landcover.TREE, landcover.GRASS, landcover.BARE
    a = np.full((20, 20), T, np.uint8)
    a[:, 12:] = G  # a straight edge
    a[3, 3] = B  # one pixel
    a[8, 3:5] = B  # two
    a[14:17, 2:9] = B  # a band three pixels wide
    out = landcover.majority(a)
    assert out[3, 3] == T and (out[8, 3:5] == T).all(), "speckle goes"
    rows = [r for r in range(20) if r not in (14, 15, 16)]
    assert (out[rows, :12] == T).all(), "the rest of the woods stays woods"
    assert (out[:, 12:] == G).all(), "a straight edge stays where it was"
    assert (out[14:17, 3:8] == B).all() and (out[15, 2:9] == B).all(), "a band three wide stays (but its corners)"
    tie = np.array([[T, T, T], [G, B, G], [B, B, G]], np.uint8)  # the centre: three of each class
    assert landcover.majority(tie)[1, 1] == B, "a tie with its own class keeps it"
    tie2 = np.array([[T, T, T], [G, B, G], [G, G, T]], np.uint8)  # four G, four T, its own B once
    assert landcover.majority(tie2)[1, 1] == T, "between two others, the smaller code"
    assert landcover.majority(a).dtype == np.uint8 and landcover.majority(a[:1, :1]).shape == (1, 1)


def test_the_colours_of_the_classes_on_light_and_dark_paper():
    from matplotlib.colors import to_rgb
    for name, S in render_editorial.STYLES.items():
        pal = landcover.palette(S)
        assert set(pal) == set(landcover.CLASSES)
        assert pal[landcover.WATER] == pytest.approx(to_rgb(S["water"])), f"{name}: the style's water"
        assert pal[landcover.NO_DATA] == pytest.approx(to_rgb(S["bg"])), f"{name}: no data takes the paper"
        lb = landcover._lstar(to_rgb(S["bg"]))
        for k, hex_ in landcover.NATURE.items():
            if lb >= 50:
                assert pal[k] == pytest.approx(to_rgb(hex_)), f"{name}: the nature palette itself"
            else:  # pulled towards the paper: darker than on light paper, and closer to the paper than to it
                lk, ln = landcover._lstar(pal[k]), landcover._lstar(to_rgb(hex_))
                assert lb < lk < ln and abs(lk - lb) < abs(ln - lk), f"{name} {landcover.NAMES[k]}"
                if _chroma(to_rgb(S["bg"])) < 20 and _chroma(to_rgb(hex_)) > 10:  # a near-grey paper (blueprint
                    # tints everything blue) and a colour that is not near white: quieter too
                    assert _chroma(pal[k]) < _chroma(to_rgb(hex_)), f"{name} {landcover.NAMES[k]}: quieter"
    S = render_editorial.STYLES["huella"]
    a = np.full((30, 40), landcover.TREE, np.uint8)
    a[:, 20:] = landcover.NO_DATA
    rgb = landcover.ground_rgb(a, S)
    assert rgb.shape == (30, 40, 3) and rgb.dtype == np.uint8
    assert (rgb[:, :18] == np.round(np.asarray(to_rgb("#7fa65a")) * 255)).all(), "flat inside a class"
    assert (rgb[:, 22:] == np.round(np.asarray(to_rgb(S["bg"])) * 255)).all(), "no data: the paper"
    edge = rgb[0, 17:23, 1].astype(int)
    assert len(set(edge.tolist())) >= 4 and (np.diff(edge) >= 0).all(), "the class edge is softened, not stepped"


def _chroma(rgb):
    _, a, b = landcover.lab(rgb)
    return math.hypot(a, b)


def test_ciede2000_on_the_reference_pairs():
    """Pairs 1, 7, 17 and 25 of Sharma, Wu and Dalal (2005), then sRGB colours through lab()."""
    pairs = [((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425), ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
             ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
             ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]
    for a, b, want in pairs:
        assert landcover.de2000(a, b) == pytest.approx(want, abs=1e-4)
        assert landcover.de2000(b, a) == pytest.approx(want, abs=1e-4)
    assert landcover.delta_e("#ffffff", "#ffffff") == 0
    assert landcover.delta_e("#000000", "#ffffff") == pytest.approx(100, abs=0.01)
    a, b = "#7fa65a", "#2f6b3a"
    assert landcover.delta_e(a, b) == pytest.approx(landcover.delta_e(b, a), abs=1e-9)
    assert landcover.lab((1, 1, 1)) == pytest.approx([100, 0, 0], abs=0.01)
    assert landcover.lab((1, 0, 0)) == pytest.approx([53.24, 80.09, 67.20], abs=0.02)
