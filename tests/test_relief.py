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
    spike[0][5, 5] = 1e5
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


def test_a_sample_outside_the_elevation_data_raises_instead_of_clamping():
    mos = relief.Mosaic(np.zeros((4, 4), np.float32), lon0=-3.0, dlon=0.1, lat0=41.0, dlat=0.1)
    assert relief._sample(mos, np.array([-2.7]), np.array([40.7])).tolist() == [0.0]
    for lon, lat in ((-3.01, 40.9), (-2.69, 40.9), (-2.9, 41.01), (-2.9, 40.69)):
        with pytest.raises(relief.ReliefError):
            relief._sample(mos, np.array([lon]), np.array([lat]))
