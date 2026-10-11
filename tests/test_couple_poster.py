"""The poster for two: whose street each line is (five classes, exactly symmetric, measured on paper), five inks per
style that print and survive colour blindness, text that only uses colour to say who, and the numbers."""
import re

import geopandas as gpd
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from shapely.geometry import LineString

from _colour import contrast, de, de2000, rgb
from pinless import render_editorial as re_
from pinless.render import MM, SIZES

EMPTY = gpd.GeoDataFrame({"highway": [], "name": []}, geometry=[])
STYLES = sorted(re_.STYLES)


def stats_(n=184, km=1746.0, hours=160.0, b5=1425.0, b10=2874.0, longest=24.3):
    return {"n": n, "km": km, "hours": hours, "best5k_s": b5, "best10k_s": b10, "longest_km": longest}


def map_px(size="A3"):
    """W, H, k and the stroke width in pixels, as render draws them."""
    wmm, _ = SIZES[size]
    k = wmm / 297
    return (int(round((wmm - 36 * k) * MM * 220)), int(round(222 * k * MM * 220)), k,
            max(1, int(round(0.22 * 220 / 25.4 * k))))


def extent(width_m, cy=0.0, size="A3"):
    asp = re_.map_aspect(size)
    return (-width_m / 2, width_m / 2, cy - width_m / asp / 2, cy + width_m / asp / 2)


def street(y, x0=-1000.0, x1=1000.0, n=200):
    return LineString(np.column_stack([np.linspace(x0, x1, n), np.full(n, float(y))]))


class Scene:
    def __init__(self):
        self.pieces, self.owner = [], []

    def add(self, line, who, times=1):
        for _ in range(times):
            self.pieces.append((len(self.owner), line))
            self.owner.append(who)

    def classify(self, ext, size="A3"):
        W, H, k, wpx = map_px(size)
        segs, weight, values, classes = re_.who_values(self.pieces, np.array(self.owner), ext, W, H, k, wpx)
        _, ids = re_._segments(self.pieces)
        return segs, weight, classes, np.array(self.owner)[ids]


def test_who_classes_are_symmetric_and_exact_at_their_edges():
    a, b = np.meshgrid(np.arange(61), np.arange(61))
    a, b = a.ravel(), b.ravel()
    keep = (a + b) > 0
    for owner_is_b in (False, True):
        assert (re_.who_classes(b[keep], a[keep], owner_is_b) == 4 - re_.who_classes(a[keep], b[keep], owner_is_b)).all()
    for (x, y), c in {(10, 0): 0, (19, 1): 0, (9, 1): 1, (13, 7): 2, (7, 13): 2, (1, 9): 3, (0, 5): 4}.items():
        assert re_.who_classes(x, y, False) == c, (x, y)
    assert re_.who_classes(0, 0, False) == 0 and re_.who_classes(0, 0, True) == 4, "no count: the owner's ink"


def test_street_colour_and_weight_are_shared_by_both_lines():
    s = Scene()
    s.add(street(0), 0, 3)
    s.add(street(3), 1, 3)  # S1: three runs each, 3 m apart
    s.add(street(300), 0, 2)  # S2
    s.add(street(600), 1, 2)  # S3
    s.add(street(900), 0, 9)
    s.add(street(905), 1, 1)  # S4: 9 and 1, 5 m apart
    s.add(street(1200), 0, 20)
    s.add(street(1203), 1, 1)  # S5: one run of the second person on a street the first ran 20 times
    segs, weight, classes, owner = s.classify(extent(3000, cy=600))
    y = segs[:, :, 1].mean(axis=1)

    def on(y0, who=None):
        m = np.abs(y - y0) < 20
        return m if who is None else m & (owner == who)

    assert set(classes[on(0)]) == {2} and set(weight[on(0)]) == {6}, "both lines: both, weight 3 + 3"
    assert set(classes[on(300)]) == {0} and set(weight[on(300)]) == {2}
    assert set(classes[on(600)]) == {4}
    assert set(classes[on(900)]) == {1} and set(weight[on(900)]) == {10}, "both lines: mostly first, weight 9 + 1"
    assert set(classes[on(1200, 1)]) == {0}, "the colour belongs to the street, not the line"


def test_a_street_weighs_how_often_it_was_run_whoever_ran_it():
    """Each run's GPS line lies a few metres from the others. A street run 6 times by one person weighs what a street
    run 3 + 3 by two people does, and twice a street run 3 times: weight and colour count the activities within
    SHARED_MM of a line, the same way for one person and for two."""
    gen = np.random.default_rng(1)
    s, streets = Scene(), {}
    for j in range(6):
        y0 = -1500.0 + 600 * j
        for name, dy, runs in (("6+0", 0, (6, 0)), ("3+3", 200, (3, 3)), ("3+0", 400, (3, 0))):
            streets.setdefault(name, []).append(y0 + dy)
            for who, n in enumerate(runs):
                for _ in range(n):  # the whole run a few metres off, plus GPS noise along it
                    x = np.linspace(-2000, 2000, 400)
                    s.add(LineString(np.column_stack([x, y0 + dy + gen.normal(0, 3) + gen.normal(0, 1.5, 400)])), who)
    segs, weight, classes, owner = s.classify(extent(8000))
    y = segs[:, :, 1].mean(axis=1)

    def median(name):
        return np.median([np.median(weight[np.abs(y - y0) < 40]) for y0 in streets[name]])

    assert median("6+0") == median("3+3") == 6 and median("3+0") == 3
    shared = (np.abs(y[:, None] - np.array(streets["3+3"])) < 40).any(axis=1)
    assert set(classes[shared & (owner == 0)]) == set(classes[shared & (owner == 1)]) == {2}


@pytest.mark.parametrize("size", ["A3", "50x70"])
@pytest.mark.parametrize("width", [3000.0, 12000.0])
def test_half_a_millimetre_on_an_a3_print_is_one_street_at_every_size(size, width):
    """0.45 mm apart on an A3 print is one street and 0.8 mm is two, on any map and at every paper size (the same
    distance on the ground: 0.76 mm and 1.35 mm on a 50x70 print)."""
    for gap_mm, expect in ((0.45, ({2}, {2})), (0.8, ({0}, {4}))):
        gap = gap_mm * width / 261.0  # the A3 map is 261 mm wide
        s = Scene()
        s.add(street(0, -width / 3, width / 3), 0)
        s.add(street(gap, -width / 3, width / 3), 1)
        _, _, classes, owner = s.classify(extent(width, size=size), size)
        assert (set(classes[owner == 0]), set(classes[owner == 1])) == expect, (gap_mm, width, size)


@pytest.mark.parametrize("size", ["A3", "50x70"])
def test_shared_means_close_on_paper(size):
    for width, expect in ((3000, ({0}, {4})), (40000, ({2}, {2}))):
        s = Scene()
        s.add(street(0, -width / 3, width / 3), 0)
        s.add(street(30, -width / 3, width / 3), 1)  # 30 m apart: apart on a city map, one street on a region map
        _, _, classes, owner = s.classify(extent(width, size=size), size)
        assert (set(classes[owner == 0]), set(classes[owner == 1])) == expect, (width, size)


def test_the_colour_checker_is_ciede2000():
    # Sharma, Wu and Dalal (2005), pairs 1 and 7 of their test data
    assert de2000((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)) == pytest.approx(2.0425, abs=1e-4)
    assert de2000((50.0, 0.0, 0.0), (50.0, -1.0, 2.0)) == pytest.approx(2.3669, abs=1e-4)
    assert de(rgb("#d62728"), rgb("#2ca02c"), "deutan") < 15, "control: red and green look alike to a deuteranope"


@pytest.mark.parametrize("style", STYLES)
def test_every_style_has_five_who_inks_that_print_and_survive_colour_blindness(style):
    S = re_.STYLES[style]
    who = S["who"]
    assert len(who) == 5 and who[2] == S["ink"]
    c = [rgb(h) for h in who]
    bg, street_ = rgb(S["bg"]), rgb(S["street"]["major"][0])
    assert min(contrast(x, bg) for x in c) >= 3.0, "every class prints against the paper"
    first, second, both = c[0], c[4], c[2]
    assert de(first, second) >= 40
    assert de(first, second, "protan") >= 30 and de(first, second, "deutan") >= 30
    assert de(first, second, "tritan") >= 15
    assert min(de(first, both), de(second, both)) >= 20
    assert min(de(x, both, k) for x in (first, second) for k in ("protan", "deutan", "tritan")) >= 12
    assert min(de(c[i], c[i + 1]) for i in range(4)) >= 12, "neighbouring classes are told apart"
    assert min(de(first, street_), de(second, street_)) >= 30, "the inks stand out from the major streets"


def _couple(names=("Lucía", "Marcos"), owner=(0, 1), together=None, streets_both=63, st=None):
    st = st or (stats_(), stats_(n=96, km=702.0, hours=70.0, b5=1610.0, b10=None, longest=15.0))
    return re_.Couple(names=names, owner=np.array(owner), stats=st,
                      together=together or {"runs": 30, "km": 328.0, "known": True, "no_clock": (0, 0)},
                      streets_both=streets_both)


def _two_lines():
    pieces = [(0, LineString([(100, 100), (900, 700)])), (1, LineString([(100, 700), (900, 100)])),
              (2, LineString([(100, 400), (900, 400)])), (3, LineString([(100, 405), (900, 405)]))]
    return pieces, (0, 1, 0, 1)


@pytest.mark.parametrize("style", STYLES)
def test_couple_poster_draws_in_every_style(style, tmp_path):
    pieces, owner = _two_lines()
    out, pdf = tmp_path / "p.png", tmp_path / "p.pdf"
    info = re_.render(pieces, (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850), None, str(out), style=style, dpi=20,
                      name="Lucía", city="Madrid", year="2025", lang="es", pdf_path=str(pdf),
                      couple=_couple(owner=owner))
    assert out.stat().st_size > 0 and pdf.stat().st_size > 0
    assert info["legend"] == {"mode": "who"} and sum(info["who_frac"]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        re_.render(pieces, (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850), None, str(out), dpi=20, graphs=True,
                   couple=_couple(owner=owner))


def test_couple_poster_writes_only_who_inks_and_nothing_outside_its_map(monkeypatch, tmp_path):
    S = re_.STYLES["huella"]
    seen, real = [], re_.PathPatch

    def spy(path, *a, **k):
        if k.get("zorder") == 5:
            seen.append((np.asarray(path.vertices), k["edgecolor"]))
        return real(path, *a, **k)

    monkeypatch.setattr(re_, "PathPatch", spy)
    pieces = [(0, LineString([(-500, 400), (1500, 400)])), (1, LineString([(500, -300), (500, 1200)])),
              (2, LineString([(-200, 100), (1200, 120)]))]
    ext = (0, 1000, 0, 850)
    svg, pdf = tmp_path / "p.svg", tmp_path / "p.pdf"
    re_.render(pieces, (EMPTY, EMPTY, EMPTY), ext, None, str(tmp_path / "p.png"), dpi=20, name="Lucía", city="Madrid",
               year="2025", lang="en", svg_path=str(svg), pdf_path=str(pdf),
               couple=_couple(owner=(0, 1, 1)))
    pts = np.concatenate([v for v, _ in seen])
    assert ((pts[:, 0] >= -1e-6) & (pts[:, 0] <= 1000 + 1e-6) & (pts[:, 1] >= -1e-6) & (pts[:, 1] <= 850 + 1e-6)).all()
    inks = {tuple(np.round(to_rgba(h), 4)) for h in S["who"]}
    assert {tuple(np.round(to_rgba(c), 4)) for _, c in seen} <= inks
    text = svg.read_text(encoding="utf-8")
    assert "<text" not in text and not any(n in text for n in ("Lucía", "LUCÍA", "Marcos", "MARCOS", "Madrid"))
    raw = pdf.read_bytes()
    assert b"/Title" not in raw and b"/Author" not in raw


def test_in_a_couple_poster_colour_only_means_who(monkeypatch, tmp_path):
    S = re_.STYLES["blueprint"]
    assert S["accent"] == S["who"][0], "blueprint's accent is the first person's ink: it must not be used for text"
    values, colours, real_place, real_patch = {}, [], re_.place_text, re_.PathPatch

    def place(value, *a, **k):
        out = real_place(value, *a, **k)
        values[id(out[0])] = str(value)
        return out

    def patch(path, *a, **k):
        if "transform" in k:  # text; routes and the basemap are drawn in data coordinates
            colours.append((values.get(id(path)), k["color"]))
        return real_patch(path, *a, **k)

    monkeypatch.setattr(re_, "place_text", place)
    monkeypatch.setattr(re_, "PathPatch", patch)
    pieces, owner = _two_lines()
    for headline in ("city", "name"):
        colours.clear()
        re_.render(pieces, (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850), None, str(tmp_path / "p.png"), style="blueprint",
                   dpi=20, name="Lucía", city="Madrid", year="2025", lang="en", headline=headline,
                   couple=_couple(owner=owner))
        allowed = set(S["who"]) | {S["ink"], S["sub"]}
        assert len(colours) > 30 and {c for _, c in colours} <= allowed
        assert [c for v, c in colours if v == "2025"] == [S["ink"]], "the year is in the paper's ink"


def _same(c1, c2):
    return np.allclose(to_rgba(c1), to_rgba(c2), atol=1e-3)


def test_each_line_takes_its_person_s_ink_and_a_shared_street_the_paper_s_ink(monkeypatch, tmp_path):
    S = re_.STYLES["huella"]
    who = S["who"]
    seen, real = [], re_.PathPatch

    def spy(path, *a, **k):
        if k.get("zorder") == 5:
            seen.append((np.asarray(path.vertices), k["edgecolor"], k["linewidth"]))
        return real(path, *a, **k)

    monkeypatch.setattr(re_, "PathPatch", spy)
    pieces = [(0, street(100, 100, 900)), (1, street(700, 100, 500)), (2, street(400, 100, 900)),
              (3, street(401, 100, 900))]  # 1 m apart on a 1 km map: 0.26 mm on paper, one street
    info = re_.render(pieces, (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850), None, str(tmp_path / "p.png"), dpi=20,
                      couple=_couple(owner=(0, 1, 0, 1)))

    def inks(y0):
        return [c for v, c, _ in seen if np.abs(v[:, 1] - y0).min() < 2]

    assert inks(100) and all(_same(c, who[0]) for c in inks(100)), "the first person alone: their ink"
    assert inks(700) and all(_same(c, who[4]) for c in inks(700)), "the second person alone: theirs"
    assert inks(400) and all(_same(c, who[2]) for c in inks(400.5)), "a street both ran: the paper's ink"
    # drawn length by class, first to second: 800 m alone, 2 x 800 m both, 400 m alone
    assert info["who_frac"] == pytest.approx([800 / 2800, 0, 1600 / 2800, 0, 400 / 2800], abs=0.01)
    # weight 1 (cap 3) is the thinnest step: COUPLE_BASE plus the routes' gain at that step
    w1 = int(np.rint(np.log1p(1) / np.log1p(3) * 7) * 31 / 7) / 31
    assert min(lw for _, _, lw in seen) == pytest.approx(re_.COUPLE_BASE + re_.ROUTE_GAIN * w1)


def test_every_text_of_a_poster_for_two_takes_the_right_ink(monkeypatch, tmp_path):
    """Blueprint, whose accent is the first person's ink. The names, each person's line and the split bar are in
    their own ink; 'km' is in the paper's ink; the key shows each name on its side and strokes no heavier than the
    heaviest route."""
    import matplotlib.pyplot as plt

    S = re_.STYLES["blueprint"]
    who = S["who"]
    texts, colours, figs = {}, [], []
    real_place, real_runs, real_patch, real_close = re_.place_text, re_.place_runs, re_.PathPatch, plt.close
    calls = []

    def place(value, x, y, *a, **k):
        out = real_place(value, x, y, *a, **k)
        texts[id(out[0])] = str(value)
        calls.append((str(value), x, y, k.get("align", a[3] if len(a) > 3 else "left")))
        return out

    def runs(parts, *a, **k):
        items, w = real_runs(parts, *a, **k)
        for (p, _, _), t in zip(items, [t for t, _ in parts if str(t).strip()]):
            texts[id(p)] = str(t)
        return items, w

    def patch(path, *a, **k):
        if "transform" in k:
            colours.append((texts.get(id(path)), k["color"]))
        return real_patch(path, *a, **k)

    monkeypatch.setattr(re_, "place_text", place)
    monkeypatch.setattr(re_, "place_runs", runs)
    monkeypatch.setattr(re_, "PathPatch", patch)
    monkeypatch.setattr(plt, "close", lambda fig=None: figs.append(fig))
    pieces, owner = _two_lines()
    c = _couple(owner=owner)
    cc = re_.couple_cells(c, "en")
    for headline in ("name", "city"):
        colours.clear()
        calls.clear()
        re_.render(pieces, (EMPTY, EMPTY, EMPTY), (0, 1000, 0, 850), None, str(tmp_path / "p.png"), style="blueprint",
                   dpi=20, name="Lucía", city="Madrid", year="2025", lang="en", headline=headline, couple=c)
        ink = dict(colours)
        assert ink["km"] == S["ink"], "the unit is not the accent: colour only says who"
        assert ink[cc["first"]] == who[0] and ink[cc["second"]] == who[4]
        names = [(t, col) for t, col in colours if t in ("LUCÍA", " & ", "MARCOS")]
        assert names[:3] == [("LUCÍA", who[0]), (" & ", S["sub"]), ("MARCOS", who[4])], headline
        assert ("LUCÍA", 218, 399 + 3.6, "left") in calls and ("MARCOS", 266.8, 399 + 3.6, "right") in calls
    page = figs[-1].axes[0]
    bar = sorted((ln.get_xdata()[0], ln.get_color()) for ln in page.lines if np.allclose(ln.get_ydata(), 362.2))
    assert len(bar) == 2 and bar[0][0] == pytest.approx(18) and _same(bar[0][1], who[0]) and _same(bar[1][1], who[4])
    key = sorted(ln.get_linewidth() for ln in page.lines if np.allclose(ln.get_ydata(), 400.0)
                 and _same(ln.get_color(), S["sub"]))
    assert key == pytest.approx([re_.COUPLE_BASE + re_.ROUTE_GAIN * v for v in (0, 1 / 3, 2 / 3, 1)])
    assert max(key) == pytest.approx(re_.COUPLE_BASE + re_.ROUTE_GAIN), "the heaviest key stroke is the heaviest route"
    for fig in figs:
        real_close(fig)


def test_place_runs_sets_parts_like_one_string():
    from matplotlib.transforms import Bbox
    S = re_.STYLES["huella"]
    parts = [("LUCÍA", S["who"][0]), (" & ", S["sub"]), ("MARCOS", S["who"][4])]
    for font, kw in ((re_.DISPLAY, dict(pt=235, max_w=215, height=57, clear=(15, 36, 17.0))),
                     (re_.TEXT, dict(pt=13.2, align="right", max_w=105)), (re_.DISPLAY, dict(pt=25, max_w=20))):
        x = 279 if kw.get("align") == "right" else 18
        items, width = re_.place_runs(parts, x, 24, 1.0, font=font, **kw)
        path, trans, w, _ = re_.place_text("LUCÍA & MARCOS", x, 24, 1.0, font=font, **kw)
        whole = path.transformed(trans).get_extents()
        union = Bbox.union([p.transformed(t).get_extents() for p, t, _ in items])
        assert abs(union.x0 - whole.x0) <= 0.005 * whole.width and abs(union.x1 - whole.x1) <= 0.005 * whole.width
        assert [c for _, _, c in items] == [S["who"][0], S["sub"], S["who"][4]]
        assert len({id(t) for _, t, _ in items}) == 1 and width == pytest.approx(w)
        assert width <= kw["max_w"] + 1e-6
    items, _ = re_.place_runs([("ÁLVARO", "r"), (" & ", "g"), ("ÍÑIGO", "b")], 18, 24, 1.0, pt=235, font=re_.DISPLAY,
                              max_w=215, height=57, clear=(15, 36, 17.0))
    for p, t, _ in items:
        v = re_._outline(p, t, 0.3)
        under = (v[:, 0] >= 15) & (v[:, 0] <= 36)
        assert not under.any() or v[under, 1].min() >= 17.0 - 1e-6, "no accent reaches the PINLESS label"


def test_couple_numbers_and_fallbacks():
    c = _couple()
    es = re_.couple_cells(c, "es")
    assert es["total"] == "2.448" and es["first"] == "LUCÍA  1.746 KM · 184 SALIDAS"
    assert es["second"] == "702 KM · 96 SALIDAS  MARCOS"
    assert es["wa"] == pytest.approx(1746 / 2448 * 138.8)
    assert es["cells"] == [("30", "SALIDAS JUNTAS"), ("328", "KM CODO CON CODO")]
    assert es["records"][0] == (("23:45", "26:50"), "MEJOR 5K")
    assert es["records"][1][0][1] == re_.MISSING, "no 10K: the missing mark"
    assert es["records"][2][0] == ("24,3", "15,0") and es["streets"] == ("63", "CALLES DE LOS DOS")
    en = re_.couple_cells(_couple(streets_both=None), "en")
    assert en["total"] == "2,448" and en["first"].endswith("184 RUNS") and en["streets"][0] == re_.MISSING
    one = re_.couple_cells(_couple(st=(stats_(n=1, km=5000.0), stats_(n=1, km=1.0))), "es")
    assert one["first"].endswith("1 SALIDA") and one["wa"] == pytest.approx(135.8), "at least 3 mm for each"
    assert re_.couple_cells(_couple(st=(stats_(km=0.5), stats_(km=9000.0))), "en")["wa"] == pytest.approx(3.0)
    for tog in ({"runs": 0, "km": 0.0, "known": True, "no_clock": (0, 0)},
                {"runs": 0, "km": 0.0, "known": False, "no_clock": (184, 0)}):
        cells = re_.couple_cells(_couple(together=tog), "en")["cells"]
        assert cells == [("280", "RUNS"), ("230", "RECORDED HOURS")], "no run together: total runs and hours"
    odd = re_.couple_cells(_couple(st=(stats_(km=630.6, n=1234), stats_(km=7.6, n=1)), streets_both=1234), "es")
    assert odd["total"] == "639" and odd["first"] == "LUCÍA  631 KM · 1.234 SALIDAS", "the parts add up to the total"
    assert odd["second"] == "8 KM · 1 SALIDA  MARCOS" and odd["streets"][0] == "1.234"
    big = re_.couple_cells(_couple(together={"runs": 1234, "km": 4567.0, "known": True, "no_clock": (0, 0)}), "en")
    assert big["cells"] == [("1,234", "RUNS TOGETHER"), ("4,567", "KM SIDE BY SIDE")], "counts formatted like km"
    printed = [es["total"], es["first"], es["second"], *map(str, es["cells"]), *map(str, es["records"])]
    assert not any(re.search(r"\d[.,]\d{4}", x) for x in printed), "only rounded numbers are printed"
