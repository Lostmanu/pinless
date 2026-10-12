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
from shapely.geometry import LineString

from _cog_fixture import fake_server, landcover_levels, write_landcover_cog
from pinless import cli, landcover, race, relief, render, render_editorial, render_trail
from test_race_graphs import EMPTY
from test_trail import START, _hike, _lonlat, _loop, _trail_world

W0, BLK, NLV = 192, 16, 4  # small tiles: 1/64 degree (1.7 km) at full resolution, 192 to 24 pixels
REAL = {name: getattr(landcover, name) for name in ("WIDTH0", "BLOCK", "NOMINAL_LEVELS")}  # before any patch
REAL_FETCH = landcover._fetch
REAL_RELIEF_FETCH = relief._fetch
REAL_PREPARE = landcover.prepare_many
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


# --- the reader -------------------------------------------------------------------------------------------------------

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


def test_the_requests_depend_on_the_map_only_not_on_the_activities(monkeypatch, tmp_path):
    files = {(39, -6): _tile_cog((39, -6))}
    lists, extents = [], []
    far = _hike(_loop(30000, 30000, 2500, n=500), "far")  # 42 km from home: no private place touches it
    for k, (pool, zone) in enumerate((([far], []), ([far] + [_hike(_loop(0, 0, 1500 + 30 * i), i, kind="Run")
                                                              for i in range(6)],
                                                    ["--zone", "40.40,-3.72,400"]))):
        captured = {}
        _trail_world(monkeypatch, pool, captured, tmp_path)
        monkeypatch.setattr(landcover, "prepare_many", REAL_PREPARE)
        monkeypatch.setenv("PINLESS_CACHE", str(tmp_path / f"cache{k}"))  # each run starts with nothing cached
        calls = _serve(monkeypatch, files)
        assert cli.main(["x.zip", "--layout", "trail", "--activity", "far", "--dpi", "20", "--out",
                         str(tmp_path / f"t{k}.png")] + zone) == 0
        assert captured["info"]["ground"] is True and isinstance(captured["ground"], landcover.LandCover)
        lists.append(list(calls))
        extents.append(captured["extent"])
    assert extents[0] == extents[1] and lists[0] and lists[0] == lists[1], "the same requests, in the same order"


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
        lb = landcover._lstar(to_rgb(S["bg"]))
        for line, key in ((False, "water"), (True, "water_line")):
            own, got = np.asarray(to_rgb(S[key])), np.asarray(landcover.water(S, line=line))
            if lb < 50:
                assert got == pytest.approx(own), f"{name}: dark paper keeps its own {key}"
            else:  # pulled towards a lake (stream) blue: bluer than the style's own, and still from it
                blue = np.asarray(to_rgb(landcover.STREAM_BLUE if line else landcover.WATER_BLUE))
                assert got == pytest.approx(own + (blue - own) * landcover.WATER_MIX), f"{name} {key}"
                assert got[2] - got[0] > own[2] - own[0], f"{name}: bluer {key}"
        assert pal[landcover.WATER] == pytest.approx(landcover.water(S)), f"{name}: the water of the ground"
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
    r = math.ceil(3 * landcover.SMOOTH_PX)  # the reach of the smoothing
    assert 0.5 <= landcover.SMOOTH_PX <= 1.0, "light: a sixth of a millimetre at most"
    assert (rgb[:, :20 - r] == np.round(np.asarray(to_rgb("#7fa65a")) * 255)).all(), "flat inside a class"
    assert (rgb[:, 20 + r:] == np.round(np.asarray(to_rgb(S["bg"])) * 255)).all(), "no data: the paper"
    edge = rgb[0, 20 - r:20 + r, 1].astype(int)
    assert len(set(edge.tolist())) >= 4 and (np.diff(edge) >= 0).all(), "the class edge is softened, not stepped"


def _chroma(rgb):
    _, a, b = landcover.lab(rgb)
    return math.hypot(a, b)


# the class shares of a 10 km map (WorldCover 2021), measured by hand around public places
SURVEY = {"Penalara": (40.85, {10: .776, 20: .078, 30: .134, 60: .012}),
          "Tabernas": (37.02, {10: .092, 20: .124, 30: .493, 60: .273, 50: .017}),
          "Picos de Europa": (43.20, {10: .167, 30: .514, 60: .318}),
          "Aneto": (42.63, {10: .092, 30: .403, 60: .471, 70: .011, 100: .010, 80: .013}),
          "Ordesa": (42.65, {10: .408, 20: .010, 30: .358, 60: .220}),
          "Zermatt": (45.98, {10: .085, 30: .139, 60: .274, 70: .401, 100: .088}),
          "Mulhacen": (37.05, {10: .026, 20: .019, 30: .522, 60: .433}),
          "Teide": (28.27, {10: .049, 20: .034, 30: .311, 60: .606}),
          "Wadi Rum": (29.55, {30: .015, 60: .979}), "Sahara": (31.10, {60: .977, 50: .012}),
          "Gobi": (43.5, {60: .97}), "Bardenas": (42.19, {30: .308, 40: .330, 60: .331}),
          "Ebro delta": (40.66, {30: .049, 40: .589, 80: .290, 90: .052}),
          "Atacama": (-23.5, {60: .99}), "Patagonia": (-50.0, {30: .6, 60: .3})}


def test_a_dry_land_is_told_from_a_mountain_by_where_it_lies():
    dry = {name: landcover.dryness(sh, lat) for name, (lat, sh) in SURVEY.items()}
    for name in ("Tabernas", "Mulhacen", "Teide", "Wadi Rum", "Sahara", "Gobi", "Atacama"):
        assert dry[name] == pytest.approx(1.0, abs=0.01), name
    for name in ("Penalara", "Picos de Europa", "Aneto", "Ordesa", "Zermatt", "Bardenas", "Ebro delta", "Patagonia"):
        assert dry[name] == 0.0, name
    assert landcover.dryness({60: .25}, 39.5) == pytest.approx(0.5), "half way between 37 and 42 degrees"
    assert landcover.dryness({60: .175}, -30.0) == pytest.approx(0.5), "half way to a quarter of bare land, south too"
    assert landcover.dryness({60: .97, 70: .03}, 30.0) == 0.0, "snow and ice: a cold land"
    from matplotlib.colors import to_rgb
    S = render_editorial.STYLES["huella"]
    green, dry1, half = landcover.palette(S, 0.0), landcover.palette(S, 1.0), landcover.palette(S, 0.5)
    for k, c in landcover.DRY.items():
        assert dry1[k] == pytest.approx(to_rgb(c)) and green[k] == pytest.approx(to_rgb(landcover.NATURE[k]))
        assert half[k] == pytest.approx(tuple((np.asarray(green[k]) + dry1[k]) / 2))
    assert all(green[k] == dry1[k] for k in landcover.CLASSES if k not in landcover.DRY), "only grass, scrub, bare"


def test_the_map_learns_its_dryness_from_its_own_land(monkeypatch, tmp_path):
    bare = lambda lon, lat: np.full(np.shape(lon), landcover.BARE, np.uint8)  # noqa: E731
    _serve(monkeypatch, {(36, -3): _tile_cog((36, -3), bare), (42, -6): _tile_cog((42, -6), bare)})
    for lat, lon, want in ((37.0, -2.4, 1.0), (43.2, -4.85, 1.0)):  # all bare: a desert anywhere
        logs = []
        extent, crs = _utm_extent(lat, lon, 30000, 25000)
        out = landcover.prepare_many([(extent, crs, 60.0)], log=logs.append, root=str(tmp_path / f"all{lat}"))
        assert out[0].dry == pytest.approx(want) and any("a dry land" in line for line in logs)
    half = lambda lon, lat: np.where(lon < -2.4, landcover.BARE, landcover.GRASS).astype(np.uint8)  # noqa: E731
    for lat, lon, want in ((37.0, -2.4, 1.0), (43.2, -4.85, 0.0)):  # half bare: dry in the south only
        tile = landcover.tile_of(lat, lon)
        west = (lambda lo, la: np.where(lo < -4.89, 60, 30).astype(np.uint8))  # noqa: E731 (40 % bare)
        _serve(monkeypatch, {tile: _tile_cog(tile, half if lat < 40 else west)})
        logs = []
        extent, crs = _utm_extent(lat, lon, 30000, 25000)
        out = landcover.prepare_many([(extent, crs, 60.0)], log=logs.append, root=str(tmp_path / f"half{lat}"))
        assert out[0].dry == pytest.approx(want), lat
        assert any("a dry land" in line for line in logs) == (want > 0)
        assert not any(re.search(r"\d+\.\d{2,}|[NS]\d\d[EW]\d", line) for line in logs), "no coordinates or tiles"


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


# --- the trail print over the ground ----------------------------------------------------------------------------------

def _patchwork(shape):
    """Every class of the dataset in vertical bands, west to east (water and land without data too)."""
    H, W = shape
    bands = np.asarray(landcover.CLASSES, dtype=np.uint8)
    return np.repeat(bands[np.arange(W) * len(bands) // W][None, :], H, axis=0)


def _trail_page(tmp_path, style="huella", size="A3", classes=None, name="g.png", rel=None, dry=0.0, **kw):
    """render_trail.render of a 12 km loop over the given land cover (a class code, an array, the patchwork when
    None, or no ground when False): its info. rel: the strongest shade of a relief that slopes west to east."""
    xy = _loop(0, 0, 2000, n=600)
    extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(size), pad_frac=0.1)
    k = render.SIZES[size][0] / 297
    shape = relief.raster_shape((render_trail.RIGHT - render_trail.LEFT) * k, extent)
    if classes is None:
        classes = _patchwork(shape)
    elif isinstance(classes, int) and classes is not False:
        classes = np.full(shape, classes, np.uint8)
    ground = None if classes is False else landcover.LandCover(np.asarray(classes, dtype=np.uint8), 0, dry)
    if rel is not None:
        rel = relief.Relief(np.tile(np.linspace(-rel, rel, shape[1], dtype=np.float32), (shape[0], 1)), "glo30", 0,
                            0.5)
    args = dict(dpi=20, title="Peñalara", place="Comunidad de Madrid, Spain", name="Lucia", date=START, style=style,
                size=size, ground=ground, relief=rel)
    args.update(kw)
    return render_trail.render(_hike(xy), xy, [(0, LineString(xy[30:-30]))], (EMPTY, EMPTY, EMPTY), extent,
                               str(tmp_path / name), **args)


def test_the_course_takes_a_route_ink_only_where_the_accent_would_sink(monkeypatch, tmp_path):
    from matplotlib.colors import to_hex
    cases = [("forest", landcover.TREE, "deep red", 0.0), ("huella", landcover.TREE, None, 0.0),
             ("forest", landcover.GRASS, None, 0.0), ("sunset", landcover.BARE, "deep red", 1.0),
             ("sunset", landcover.BARE, None, 0.0), ("sunset", landcover.TREE, None, 1.0),
             ("carbon", landcover.TREE, None, 0.0), ("ink", landcover.SNOW, None, 0.0)]
    dark_green = dict(render_editorial.STYLES["carbon"])
    dark_green["accent"] = to_hex(landcover.palette(dark_green)[landcover.TREE])
    monkeypatch.setitem(render_editorial.STYLES, "dark-green", dark_green)
    cases.append(("dark-green", landcover.TREE, "red", 0.0))
    for style, cls, ink, dry in cases:
        logs = []
        info = _trail_page(tmp_path, style=style, classes=cls, log=logs.append, dry=dry)
        S = render_editorial.STYLES[style]
        if ink is None:
            assert info["course_ink"] == S["accent"] and not logs, (style, cls)
        else:
            want = dict(render_trail.ROUTE_INKS[landcover.is_dark(S)])[ink]
            assert info["course_ink"] == want, (style, cls)
            assert landcover.delta_e(want, landcover.palette(S, dry)[cls]) >= render_trail.COURSE_DE
            assert len(logs) == 1 and logs[0].startswith(f"[ground] the course is drawn in {ink}: the style's accent "
                                                         f"would sink into the {landcover.NAMES[cls]}"), logs
            assert not re.search(r"\d+\.\d", logs[0]), "no coordinates, no decimals"
    logs = []
    info = _trail_page(tmp_path, style="forest", classes=landcover.TREE, color_by="elevation", log=logs.append)
    assert info["legend"]["mode"] == "elevation" and info["course_ink"] is None and not logs, "the heat inks as ever"
    info = _trail_page(tmp_path, style="forest", classes=False)
    assert info["course_ink"] == render_editorial.STYLES["forest"]["accent"] and info["ground"] is False


def test_the_ground_under_the_course_is_measured_along_it():
    classes = np.full((100, 200), landcover.TREE, np.uint8)
    classes[:, 100:] = landcover.BARE
    extent = (0.0, 2000.0, 0.0, 1000.0)
    course = np.array([[[200.0, 500.0], [1400.0, 500.0]], [[1400.0, 500.0], [1400.0, 900.0]]])
    under = render_trail.ground_under(classes, extent, course)
    assert under[landcover.TREE] == pytest.approx(800 / 1600, abs=0.01)
    assert under[landcover.BARE] == pytest.approx(800 / 1600, abs=0.01)
    assert render_trail.ground_under(classes, extent, np.empty((0, 2, 2))) == {}


def test_paths_and_roads_keep_their_contrast_against_the_ground(monkeypatch, tmp_path):
    from matplotlib.colors import to_rgb

    def lstar(c):
        return landcover._lstar(to_rgb(c))

    for name, S in render_editorial.STYLES.items():
        TS = render_trail.trail_style(S)
        classes = _patchwork((60, 120))
        pal = landcover.palette(S)
        painted = {"colours": {k: pal[k] for k in landcover.CLASSES}, "shares": landcover.shares(classes),
                   "mean": tuple(np.asarray(landcover.ground_rgb(classes, S)).reshape(-1, 3).mean(0) / 255)}
        out = render_trail.ground_style(TS, S, painted)
        dark = lstar(S["bg"]) < 50
        ref = (max if dark else min)((lstar(c) for c in painted["colours"].values()))
        for key in ("trail", "road_minor", "road_major"):
            gap = abs(lstar(TS[key]) - lstar(S["bg"]))
            ends = [lstar(S["sub"]), lstar(S["ink"]), 100.0 if dark else 0.0]
            if all(abs(e - ref) < gap for e in ends):
                continue  # no ink of the style gets there: the end colour itself
            assert abs(lstar(out[key]) - ref) >= gap - 0.6, f"{name} {key}: against the ground as against the paper"
            assert (lstar(out[key]) > ref) == dark, f"{name} {key}: on the side of the style's inks"
        assert out["relief"] != TS["relief"], f"{name}: the relief measured on the ground"
        for key, line in (("water", False), ("water_line", True)):
            assert to_rgb(out[key]) == pytest.approx(landcover.water(S, line=line), abs=1 / 255), f"{name}: {key}"
    seen = []
    real = render_trail.draw_basemap
    monkeypatch.setattr(render_trail, "draw_basemap", lambda ax, basemap, *a, **k: seen.append(basemap)
                        or real(ax, basemap, *a, **k))
    import geopandas as gpd
    from shapely.geometry import box
    woods = gpd.GeoDataFrame({"landuse": ["forest"]}, geometry=[box(-3000, -1000, -1000, 1000)])
    xy = _loop(0, 0, 2000, n=600)
    extent = race.frame_course([(0, LineString(xy))], render_trail.map_aspect(), pad_frac=0.1)
    shape = relief.raster_shape(render_trail.RIGHT - render_trail.LEFT, extent)
    for ground, n_woods in ((None, 1), (landcover.LandCover(_patchwork(shape), 0), 0)):
        seen.clear()
        render_trail.render(_hike(xy), xy, [(0, LineString(xy))], (EMPTY, woods, EMPTY), extent,
                            str(tmp_path / "w.png"), dpi=20, ground=ground)
        assert len(seen[0][1]) == n_woods, "the land cover already says where the woods are"


def test_the_ground_is_one_image_of_the_map_box_and_nothing_spills(tmp_path):
    import base64
    from matplotlib.colors import to_rgb
    pdf, svg = tmp_path / "g.pdf", tmp_path / "g.svg"
    info = _trail_page(tmp_path, pdf_path=str(pdf), svg_path=str(svg), name="g.png", dpi=40)
    assert info["ground"] is True
    shape = relief.raster_shape(render_trail.RIGHT - render_trail.LEFT,
                                race.frame_course([(0, LineString(_loop(0, 0, 2000, n=600)))],
                                                  render_trail.map_aspect(), pad_frac=0.1))
    objects = re.split(rb"\d+ 0 obj", pdf.read_bytes())
    images = [o for o in objects if b"/Subtype /Image" in o]
    assert len(images) == 1, "one image, opaque: no soft mask"
    w, h = int(re.search(rb"/Width (\d+)", images[0]).group(1)), int(re.search(rb"/Height (\d+)", images[0]).group(1))
    assert (h, w) == shape
    tags = re.findall(r"<image [^>]*>", svg.read_text(encoding="utf-8"))
    assert len(tags) == 1
    data = re.search(r'data:image/png;base64,([^"]+)"', tags[0]).group(1)
    with Image.open(io.BytesIO(base64.b64decode(data))) as im:
        assert im.size[::-1] == shape, "the SVG embeds the array as it is"
    png = np.asarray(Image.open(tmp_path / "g.png").convert("RGB")).astype(int)
    k = png.shape[1] / 297  # pixels per A3-grid millimetre
    bg = np.round(np.asarray(to_rgb(render_editorial.STYLES["huella"]["bg"])) * 255)
    top, left = render_trail.MAP_TOP * k, render_trail.LEFT * k
    bottom, right = (render_trail.MAP_TOP + render_trail.MAP_H) * k, render_trail.RIGHT * k
    for strip in (png[int(top) - 4, int(left) + 4:int(right) - 4], png[int(bottom) + 3, int(left) + 4:int(right) - 4],
                  png[int(top) + 4:int(bottom) - 4, int(left) - 3], png[int(top) + 4:int(bottom) - 4, int(right) + 3]):
        assert (np.abs(strip - bg) <= 1).all(), "the paper all round the map"
    with pytest.raises(ValueError):
        _trail_page(tmp_path, classes=np.zeros((shape[0] + 1, shape[1]), np.uint8))


def _foot_spy(monkeypatch):
    from pinless import atlas
    boxes = []
    real = atlas.place_text

    def spy(value, x, y, k, pt=9, font=None, align="left", max_w=None, height=None, clear=None):
        out = real(value, x, y, k, pt, font, align, max_w, height, clear)
        w, h = out[2], out[3]
        x0 = x - (w if align == "right" else w / 2 if align == "center" else 0)
        boxes.append((str(value), x0, y, x0 + w, y + h, max_w))
        return out

    monkeypatch.setattr(atlas, "place_text", spy)
    return boxes


@pytest.mark.parametrize("size,lang", [("A4", "en"), ("A4", "es"), ("A3", "en"), ("A3", "es")])
def test_the_land_cover_credit_appears_only_with_the_ground_and_fits(monkeypatch, tmp_path, size, lang):
    from matplotlib.figure import Figure
    boxes = _foot_spy(monkeypatch)
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: None)  # the boxes are all placed before saving
    for ground in (True, False):
        for rel in (0.6, None):
            for mode in ("single", "elevation", "pace"):
                boxes.clear()
                info = _trail_page(tmp_path, size=size, lang=lang, classes=None if ground else False, rel=rel,
                                   color_by=mode)
                assert info["ground"] is ground and info["relief"] is (rel is not None)
                foot = [b for b in boxes if b[2] >= 395]
                cover = [b for b in foot if "WorldCover" in b[0]]
                assert [b[0] for b in cover] == ([landcover.notice(lang)] if ground else []), "once, only if drawn"
                credits = [b for b in foot if b[0].startswith(("©", "Land cover", "Cobertura", "Relie"))]
                assert len(credits) == 1 + ground + (rel is not None)
                assert credits[-1][2] == 412 and credits == sorted(credits, key=lambda b: b[2]), "bottom line kept"
                for b in credits:
                    assert b[3] - b[1] < b[5] - 0.5, f"{b[0][:30]!r}: set at its size, not squeezed to fit"
                top = min(b[2] for b in foot)
                band = ("legend line", 217.0, top + 0.4, 268.0, top + 1.6, None)
                for i, a in enumerate(foot + [band]):
                    for b in foot[i + 1:]:
                        overlap = a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]
                        assert not overlap, f"{a[0][:30]!r} overlaps {b[0][:30]!r}"
                left = sorted(b[2] for b in foot if b[1] < 100)  # the left column: the line above, then the credits
                assert all(b - a >= 2.9 for a, b in zip(left, left[1:])), "lines apart"
                assert all(b[4] <= 416 for b in foot), "the bottom margin of today"
                assert top >= 401.5, "clear of the rule at 398"


def test_the_ground_option_and_its_default(monkeypatch, tmp_path, capsys):
    for layout in ("editorial", "atlas", "race"):
        for value in ("nature", "style"):
            with pytest.raises(SystemExit) as ex:
                cli.main(["x.zip", "--layout", layout, "--ground", value])
            assert ex.value.code == 2
            assert "--ground is available for the trail layout only" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["x.zip", "--layout", "trail", "--ground", "satellite"])
    captured, calls = {}, []
    relief_calls = _trail_world(monkeypatch, [_hike(_loop(0, 0, 2500, n=500))], captured, tmp_path)
    monkeypatch.setattr(landcover, "prepare_many", lambda maps, log=print: calls.append(maps) or [None] * len(maps))
    args = ["h.gpx", "--layout", "trail", "--dpi", "20", "--size", "A4", "--out", str(tmp_path / "t.png")]
    assert cli.main(args) == 0 and len(calls) == 1, "nature by default"
    assert calls[0] == relief_calls[0], "the same map box and paper width as the relief, nothing else"
    assert cli.main(args + ["--ground", "style"]) == 0 and len(calls) == 1, "--ground style: nothing prepared"
    assert cli.main(args + ["--ground", "nature", "--relief", "off"]) == 0 and len(calls) == 2


def test_every_style_and_a_style_file_draw_over_the_nature_ground(monkeypatch, tmp_path):
    import json
    from pinless import styles as own_styles
    for style in list(render_editorial.STYLES):
        info = _trail_page(tmp_path, style=style, size="A4", name=f"{style}.png")
        assert info["ground"] is True and (tmp_path / f"{style}.png").stat().st_size > 0, style
    (tmp_path / "ground-night.json").write_text(json.dumps({"base": "midnight", "accent": "#ff8800", "bg": "#101010"}),
                                                encoding="utf-8")
    monkeypatch.setitem(render_editorial.STYLES, "file:ground-night", None)  # gone again after the test
    name, _ = own_styles.register(str(tmp_path / "ground-night.json"))
    assert name == "file:ground-night"
    assert _trail_page(tmp_path, style=name, size="A4", name="own.png")["ground"] is True


def test_a_failing_server_draws_the_style_ground_and_exits_0(monkeypatch, tmp_path, capsys):
    import urllib.error

    def down(*args):
        raise urllib.error.URLError("unreachable")

    captured = {}
    _trail_world(monkeypatch, [_hike(_loop(0, 0, 2500, n=500))], captured, tmp_path)
    monkeypatch.setattr(landcover, "prepare_many", REAL_PREPARE)
    monkeypatch.setattr(landcover, "_fetch", down)
    boxes = _foot_spy(monkeypatch)
    assert cli.main(["h.gpx", "--layout", "trail", "--dpi", "20", "--out", str(tmp_path / "t.png")]) == 0
    assert captured["ground"] is None and captured["info"]["ground"] is False
    assert not any("WorldCover" in b[0] for b in boxes), "no credit for what was not drawn"
    out = capsys.readouterr().out
    assert "[ground] skipped (URLError): drawn with the style's ground" in out
    assert (tmp_path / "t.png").stat().st_size > 0
