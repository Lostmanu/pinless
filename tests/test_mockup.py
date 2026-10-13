"""The wall mockup tool renders a framed poster at any requested size, for A3 (with a mat) and 50x70."""
import importlib.util
import os

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("mockup", os.path.join(ROOT, "tools", "mockup.py"))
mockup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mockup)


@pytest.mark.parametrize("paper,frame", [("A3", "oak"), ("50x70", "black")])
def test_room_mockup(tmp_path, paper, frame):
    poster = tmp_path / "poster.png"
    Image.new("RGB", (297, 420), (200, 60, 40)).save(poster)
    out = tmp_path / "wall.jpg"
    assert mockup.room(str(poster), str(out), frame=frame, paper=paper, width=400, height=320, spp=2,
                       workers=1) == (400, 320)
    with Image.open(out) as im:
        r, g, b = im.convert("RGB").getpixel((200, 110))  # the middle of the frame
    assert r > g + 60 and r > b + 60, "the (red) poster is shown inside the frame, at the centre of the wall"


@pytest.mark.parametrize("paper", ["A3", "50x70"])
def test_the_room_is_modelled_at_true_size(paper):
    """The print, its frame and the sideboard in centimetres: A3 in a 40x50 frame with a mat, 50x70 without one, the
    frame's bottom 20 cm above the sideboard's top, 80 cm off the floor."""
    S = mockup.build("oak", paper)
    (pw, ph), (fw, fh) = mockup.PAPER[paper]
    x0, x1, y0, y1 = S.print_rect
    assert (x1 - x0, y1 - y0) == pytest.approx((pw, ph))
    g = S.geom
    assert (g["x1"] - g["x0"], g["y1"] - g["y0"]) == pytest.approx((fw + 2 * g["prof"], fh + 2 * g["prof"]))
    assert g["y0"] - mockup.TOP_Y == pytest.approx(20.0) and mockup.TOP_Y == 80.0
    assert (S.cut is not None) == (paper == "A3"), "a mat only round an A3 print"
    if S.cut is not None:
        cx0, cx1, cy0, cy1, _ = S.cut
        assert (cx1 - cx0, cy1 - cy0) == pytest.approx((pw - 0.6, ph - 0.6)), "the mat overlaps the print by 3 mm"
