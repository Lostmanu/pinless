"""The wall mockup tool renders a framed poster at any requested size, for A3 (with a mat) and 50x70."""
import importlib.util
import os

import numpy as np
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


spec_p = importlib.util.spec_from_file_location("mockup_photo", os.path.join(ROOT, "tools", "mockup_photo.py"))
mockup_photo = importlib.util.module_from_spec(spec_p)
spec_p.loader.exec_module(mockup_photo)


@pytest.mark.parametrize("scene", sorted(mockup_photo.SCENES))
def test_every_room_photo_hangs_its_posters_in_every_frame(tmp_path, scene):
    poster = tmp_path / "poster.png"
    Image.new("RGB", (297, 420), (200, 60, 40)).save(poster)
    n = len(mockup_photo.SCENES[scene]["frames"])
    frames = sorted(mockup_photo.FRAMES)[:n]
    out = tmp_path / "wall.jpg"
    w, h = mockup_photo.compose([str(poster)] * n, str(out), scene, frames, width=600)
    assert w == 600
    x0, y0, x1, y1 = mockup_photo.SCENES[scene]["crop"]
    s = 600 / (x1 - x0)
    with Image.open(out) as im:
        rgb = im.convert("RGB")
        for cx, top, hpx in mockup_photo.SCENES[scene]["frames"]:
            r, g, b = rgb.getpixel((int((cx - x0) * s), int((top + hpx * 0.45 - y0) * s)))
            assert r > g + 25 and r > b + 25, "the (red) poster shows in the middle of each frame, in the room's light"


def test_a_picture_on_the_wall_is_painted_out_from_the_bare_wall_round_it():
    """The new rooms have pictures of their own where the posters hang. erase fills their box from the wall round it:
    the wall's gradient runs on through the box, a sofa of another colour just under it stays out of the fill, and
    nothing outside the box changes."""
    H, W = 300, 400
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lit = np.dstack([200 + 0.10 * xx - 0.05 * yy, 196 + 0.10 * xx - 0.05 * yy, 188 + 0.08 * xx - 0.05 * yy])
    spot = np.dstack([215 - 0.0006 * ((xx - 180) ** 2 + (yy - 40) ** 2)] * 3)  # a downlight's pool on the wall
    hole = (slice(54, 181), slice(134, 266))
    for name, wall in (("a light from one side", lit), ("a downlight", spot)):
        photo = wall.copy()
        photo[60:180, 140:260] = (40, 38, 35)  # a dark picture
        photo[182:300, 60:340] = (90, 70, 60)  # a sofa just under it
        out = mockup_photo.erase(photo, [(134, 54, 266, 181)], noise=0.0, rng=np.random.default_rng(1))
        # measured: 0.0 on both; filled by spreading the wall in from its edges alone, 4.5 next to the sofa
        assert np.abs(out[hole] - wall[hole]).max() < 1.0, f"{name}: the wall's own light, through the box"
        outside = np.ones((H, W), bool)
        outside[hole] = False
        assert (out[outside] == photo[outside]).all(), "nothing outside the box is touched"


def test_the_room_photos_are_credited():
    credits = open(os.path.join(ROOT, "tools", "rooms", "CREDITS.md"), encoding="utf-8").read()
    for scene in mockup_photo.SCENES.values():
        assert os.path.exists(os.path.join(mockup_photo.ROOMS, scene["photo"]))
        assert f"`{scene['photo']}`" in credits
