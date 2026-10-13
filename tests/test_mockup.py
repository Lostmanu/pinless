"""The wall mockup tool draws a framed poster at any requested size, for A3 (with a mat) and 50x70."""
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
    assert mockup.room(str(poster), str(out), frame=frame, paper=paper, width=400, height=320) == (400, 320)
    with Image.open(out) as im:
        r, g, b = im.convert("RGB").getpixel((200, 110))  # the middle of the frame
    assert r > g + 60 and r > b + 60, "the (red) poster is shown inside the frame, at the centre of the wall"


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


def test_the_room_photos_are_credited():
    credits = open(os.path.join(ROOT, "tools", "rooms", "CREDITS.md"), encoding="utf-8").read()
    for scene in mockup_photo.SCENES.values():
        assert os.path.exists(os.path.join(mockup_photo.ROOMS, scene["photo"]))
        assert f"`{scene['photo']}`" in credits
