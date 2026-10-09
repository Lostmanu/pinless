"""Framed-on-a-wall mockups to show the poster on a phone.

Usage: python tools/mockup.py out/pinless_madrid_nightrun_A3.png out/mockup_nightrun.jpg --frame black
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

p = argparse.ArgumentParser()
p.add_argument("poster")
p.add_argument("out")
p.add_argument("--frame", default="black", choices=["black", "oak", "white"])
p.add_argument("--wall", default="#e9e4dc")
a = p.parse_args()

W, H = 1500, 1900
wall = Image.new("RGB", (W, H), a.wall)
# soft light from the top-left
yy, xx = np.mgrid[0:H, 0:W]
light = 1.06 - 0.16 * (np.hypot(xx - W * 0.3, yy - H * 0.15) / np.hypot(W, H))
arr = np.clip(np.asarray(wall, dtype=float) * light[..., None], 0, 255).astype(np.uint8)
wall = Image.fromarray(arr)

poster = Image.open(a.poster).convert("RGB")
ph = 1180
pw = int(poster.width * ph / poster.height)
poster = poster.resize((pw, ph), Image.LANCZOS)
mat = int(pw * 0.085)
frame = 22
fw, fh = pw + 2 * mat + 2 * frame, ph + 2 * mat + 2 * frame
x0, y0 = (W - fw) // 2, int(H * 0.12)

shadow = Image.new("L", (W, H), 0)
ImageDraw.Draw(shadow).rectangle([x0 + 18, y0 + 30, x0 + fw + 18, y0 + fh + 34], fill=120)
shadow = shadow.filter(ImageFilter.GaussianBlur(28))
wall = Image.composite(Image.new("RGB", (W, H), (60, 55, 50)), wall, shadow)

colors = {"black": (22, 22, 24), "oak": (176, 138, 92), "white": (244, 244, 242)}
d = ImageDraw.Draw(wall)
d.rectangle([x0, y0, x0 + fw, y0 + fh], fill=colors[a.frame])
d.rectangle([x0 + frame, y0 + frame, x0 + fw - frame, y0 + fh - frame], fill=(250, 250, 248))
# inner bevel line of the mat
d.rectangle([x0 + frame + mat - 3, y0 + frame + mat - 3, x0 + frame + mat + pw + 2, y0 + frame + mat + ph + 2], outline=(228, 226, 222), width=3)
wall.paste(poster, (x0 + frame + mat, y0 + frame + mat))
# a hint of glass reflection
gl = Image.new("L", (W, H), 0)
ImageDraw.Draw(gl).polygon([(x0 + frame, y0 + frame), (x0 + fw * 0.45, y0 + frame), (x0 + frame, y0 + fh * 0.55)], fill=22)
gl = gl.filter(ImageFilter.GaussianBlur(40))
wall = Image.composite(Image.new("RGB", (W, H), (255, 255, 255)), wall, gl)
wall.save(a.out, quality=92)
print(a.out, wall.size)
