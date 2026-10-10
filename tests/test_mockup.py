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
