"""The poster framed and hung in a room, at real scale, to show how it looks on a wall.

A3 goes in a standard 40x50 cm frame with a mat; 50x70 in a 50x70 frame without one. Everything (sideboard, vase,
books, frame profile) is drawn in centimetres, so the poster is shown at its true size against the furniture.

Usage: python tools/mockup.py docs/poster-huella.jpg docs/wall-huella.jpg --frame oak
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 2  # supersampling: drawn at twice the size, then reduced, for clean edges
FRAMES = {"oak": ((198, 163, 122), (158, 122, 86)), "black": ((34, 33, 32), (16, 16, 16)),
          "white": ((240, 238, 233), (214, 211, 204))}
PAPER = {"A3": ((29.7, 42.0), (40.0, 50.0)), "50x70": ((50.0, 70.0), (50.0, 70.0))}  # paper, frame window (cm)


def noise(shape, scale, rng):
    """Smooth noise in [-1, 1]: random at 1/scale resolution, enlarged."""
    h, w = shape
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    img = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-6) * 255).astype(np.uint8))
    return np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 127.5 - 1


def shadow(base, box, offset, blur, strength, radius=0):
    """Darken base under a blurred copy of box (x0, y0, x1, y1) moved by offset."""
    m = Image.new("L", (base.shape[1], base.shape[0]), 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(m).rounded_rectangle([x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]],
                                        radius=radius, fill=255)
    a = np.asarray(m.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32)[..., None] / 255
    base *= 1 - strength * a


def paint(base, mask_img, color):
    """Composite a flat colour (or an HxWx3 array) through an 'L' mask image."""
    a = np.asarray(mask_img, dtype=np.float32)[..., None] / 255
    base[:] = base * (1 - a) + np.asarray(color, dtype=np.float32) * a


def room(poster, out, frame="oak", paper="A3", width=2000, height=1600, seed=7):
    rng = np.random.default_rng(seed)
    W, H = width * SS, height * SS
    (pw_cm, ph_cm), (fw_cm, fh_cm) = PAPER[paper]
    # ~105 cm of wall height for an A3 frame; more for a bigger frame, so everything stays at true scale
    cm = H / max(105.0, fh_cm + 55.0)  # frame + profile + 24 cm of sideboard + 20 cm gap + ~6.6 cm above
    img = np.zeros((H, W, 3), np.float32)

    # wall: warm plaster, light from the upper left, a slow vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.05 - 0.13 * np.hypot((xx - 0.15 * W) / W, (yy + 0.1 * H) / H) - 0.04 * (yy / H) ** 2
    plaster = 1 + 0.012 * noise((H, W), 3 * SS, rng) + 0.006 * noise((H, W), 40 * SS, rng)
    img[:] = np.array([233, 227, 217], np.float32) * (light * plaster)[..., None]
    window = Image.new("L", (W, H), 0)
    ImageDraw.Draw(window).polygon([(0, int(0.02 * H)), (int(0.30 * W), int(0.10 * H)), (int(0.24 * W), int(0.80 * H)),
                                    (0, int(0.72 * H))], fill=255)
    img *= 1 + 0.045 * np.asarray(window.filter(ImageFilter.GaussianBlur(70 * SS)), np.float32)[..., None] / 255

    # sideboard: its top 24 cm above the bottom edge, wider than the view
    top = H - int(24 * cm)
    lip = int(1.6 * cm)
    shadow(img, (-W, top - int(0.6 * cm), 2 * W, H), (int(0.8 * cm), int(0.4 * cm)), 14 * SS, 0.25)
    streaks = rng.normal(0, 1, (H - top, W // 40)).astype(np.float32)  # long horizontal wood streaks
    grain = np.asarray(Image.fromarray(((np.clip(streaks, -3, 3) + 3) / 6 * 255).astype(np.uint8))
                       .resize((W, H - top), Image.BICUBIC).filter(ImageFilter.GaussianBlur(SS)), np.float32) / 127.5 - 1
    oak = np.array([190, 152, 110], np.float32)
    body = oak * (0.90 + 0.07 * grain[..., None]) * (1 - 0.16 * (np.arange(H - top)[:, None, None] / (H - top)))
    body[:lip] = oak * 1.12 * (1 + 0.03 * grain[:lip, :, None])  # the top surface catches the light
    body[lip:lip + max(2, SS)] *= 0.72  # the edge under the top
    img[top:] = body
    for fx in (0.04, 0.37, 0.70):  # door joints
        x = int(fx * W)
        img[top + lip + int(2 * cm):, x:x + max(2, SS)] *= 0.62

    # frame on the wall, its bottom 20 cm above the sideboard
    prof = 2.2 * cm
    win_w, win_h = fw_cm * cm, fh_cm * cm
    ow, oh = win_w + 2 * prof, win_h + 2 * prof
    fx0 = (W - ow) / 2
    fy1 = top - 20 * cm
    fy0 = fy1 - oh
    box = (int(fx0), int(fy0), int(fx0 + ow), int(fy1))
    shadow(img, box, (int(1.0 * cm), int(1.8 * cm)), 22 * SS, 0.32)  # soft, light from the upper left
    shadow(img, box, (int(0.2 * cm), int(0.35 * cm)), 4 * SS, 0.30)  # contact with the wall
    face, side = FRAMES[frame]
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    paint(img, m, face)
    x0, y0, x1, y1 = box
    p = int(prof)
    bev = [((x0, y0), (x1, y0), (x1 - p, y0 + p), (x0 + p, y0 + p), 1.06),   # top: lit
           ((x0, y0), (x0 + p, y0 + p), (x0 + p, y1 - p), (x0, y1), 1.02),   # left
           ((x1, y0), (x1, y1), (x1 - p, y1 - p), (x1 - p, y0 + p), 0.93),   # right: shade
           ((x0, y1), (x0 + p, y1 - p), (x1 - p, y1 - p), (x1, y1), 0.89)]   # bottom: shade
    for *pts, k in bev:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(pts, fill=255)
        paint(img, m, np.clip(np.asarray(face, np.float32) * k, 0, 255))
    m = Image.new("L", (W, H), 0)  # thin inner lip
    ImageDraw.Draw(m).rectangle([x0 + p - SS, y0 + p - SS, x1 - p + SS, y1 - p + SS], outline=255, width=2 * SS)
    paint(img, m, side)

    # mat and poster
    wx0, wy0, wx1, wy1 = x0 + p, y0 + p, x1 - p, y1 - p
    mat = np.array([246, 244, 238], np.float32) * (light[wy0:wy1, wx0:wx1, None] * 0.6 + 0.42)
    img[wy0:wy1, wx0:wx1] = mat
    shadow(img, (wx0, wy0, wx1, wy0 + int(0.4 * cm)), (0, int(0.15 * cm)), 3 * SS, 0.18)  # the frame's lip shades the mat
    pw, ph = int(pw_cm * cm), int(ph_cm * cm)
    px0, py0 = wx0 + (wx1 - wx0 - pw) // 2, wy0 + (wy1 - wy0 - ph) // 2
    art = np.asarray(Image.open(poster).convert("RGB").resize((pw, ph), Image.LANCZOS), np.float32)
    if paper == "A3":  # bevelled mat cut: a light rim around the window
        rim = int(0.35 * cm)
        img[py0 - rim:py0 + ph + rim, px0 - rim:px0 + pw + rim] = np.array([251, 250, 246], np.float32)
        img[py0 - rim:py0 - rim + SS, px0 - rim:px0 + pw + rim] *= 0.90
    img[py0:py0 + ph, px0:px0 + pw] = art * (light[py0:py0 + ph, px0:px0 + pw, None] * 0.35 + 0.66)
    # glass: a faint diagonal reflection
    gy, gx = np.mgrid[wy0:wy1, wx0:wx1].astype(np.float32)
    t = ((gx - wx0) / (wx1 - wx0) + (gy - wy0) / (wy1 - wy0) * 0.6)
    glare = np.clip(0.07 - 0.11 * np.abs(t - 0.45), 0, None) + 0.015
    img[wy0:wy1, wx0:wx1] = img[wy0:wy1, wx0:wx1] * (1 - glare[..., None]) + 255 * glare[..., None]

    base = top + int(0.85 * lip)  # objects stand on the top surface, near its front edge

    def contact(x0_, x1_, depth):
        """A flat soft shadow on the wood under an object, pushed right by the light."""
        shadow(img, (x0_, base - int(0.35 * cm), x1_, base + int(0.12 * cm)), (int(0.9 * cm), 0), 3 * SS, depth,
               radius=int(0.2 * cm))

    # books, left on the sideboard
    bx = int(fx0 - 4 * cm)
    y = base
    for h_cm, w_cm, color in ((3.2, 30, (92, 104, 96)), (2.4, 27, (196, 178, 150)), (2.8, 24, (140, 70, 52))):
        hh, ww = int(h_cm * cm), int(w_cm * cm)
        x = bx - ww
        if y == base:
            contact(x, x + ww, 0.35)
        shadow(img, (x, y - hh, x + ww, y), (int(1.5 * cm), int(0.4 * cm)), 10 * SS, 0.10)  # soft, on the wall
        img[y - hh:y, x:x + ww] = np.array(color, np.float32) * np.linspace(1.08, 0.88, ww)[None, :, None]
        pages = int(0.35 * cm)  # page block on the visible spine side
        img[y - hh + pages:y - pages, x + ww - int(0.5 * cm):x + ww] = np.array([238, 230, 214], np.float32)
        y -= hh
        bx -= int(1.2 * cm)

    # two stoneware vases and a dried stem, right on the sideboard
    def vase(cx, height, width, color, neck, belly_at):
        ys = np.linspace(0, 1, 240)  # 0 = mouth, 1 = foot
        body = np.sin(np.pi * np.clip((ys - 0.05) / 0.98, 0, 1)) ** 0.8  # round belly
        r = width / 2 * np.maximum(neck, body * (1 - 0.35 * (ys > belly_at) * (ys - belly_at) / (1 - belly_at)))
        r[ys > 0.96] *= 0.92  # foot
        pts = [(cx + q, base - height * (1 - y_)) for y_, q in zip(ys, r)]
        pts += [(cx - q, base - height * (1 - y_)) for y_, q in zip(ys[::-1], r[::-1])]
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(pts, fill=255)
        contact(int(cx - r[-1]), int(cx + r[-1]), 0.38)
        shadow(img, (int(cx - width / 2), int(base - height), int(cx + width / 2), int(top)),
               (int(2.6 * cm), int(0.8 * cm)), 20 * SS, 0.10, radius=int(width / 2))  # its soft shadow on the wall
        gx = np.clip((xx - (cx - width / 2)) / width, 0, 1)
        shade = (1.10 - 0.45 * gx ** 1.4)[..., None] * (1 + 0.02 * noise((H, W), 2, rng))[..., None]
        paint(img, m.filter(ImageFilter.GaussianBlur(SS * 0.5)), np.array(color, np.float32) * shade)
        return cx, base - height, r[0]

    mx, my, mr = vase(fx0 + ow + 9 * cm, 26 * cm, 14 * cm, (206, 196, 180), neck=0.32, belly_at=0.55)
    vase(fx0 + ow + 21 * cm, 13 * cm, 12 * cm, (120, 112, 100), neck=0.42, belly_at=0.5)
    stem = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(stem)
    for angle, length, bend, leaves in ((0.10, 46, 0.22, 7), (0.36, 38, 0.12, 5)):  # leaning away from the frame
        pts = [(mx + np.sin(angle + bend * t) * length * cm * t, my - np.cos(angle + bend * t) * length * cm * t)
               for t in np.linspace(0, 1, 50)]
        d.line(pts, fill=255, width=max(2, int(0.18 * cm)))
        for j, t in enumerate(np.linspace(0.45, 0.98, leaves)):
            x_, y_ = pts[int(t * 49)]
            size = (0.55 + 0.35 * rng.random()) * cm
            off = (1 if j % 2 else -1) * 0.8 * cm
            d.ellipse([x_ + off - size, y_ - 0.6 * size, x_ + off + size, y_ + 0.6 * size], fill=255)
    img *= 1 - 0.07 * np.roll(np.roll(np.asarray(stem.filter(ImageFilter.GaussianBlur(10 * SS)), np.float32)[..., None]
                                     / 255, int(2.0 * cm), 1), int(1.0 * cm), 0)
    paint(img, stem.filter(ImageFilter.GaussianBlur(SS * 0.5)), (112, 104, 86))

    # grain and output
    img *= 1 + 0.008 * rng.normal(0, 1, (H, W, 1)).astype(np.float32)
    out_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((width, height), Image.LANCZOS)
    out_img.save(out, quality=90, optimize=True)
    return out_img.size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("poster")
    ap.add_argument("out")
    ap.add_argument("--frame", default="oak", choices=sorted(FRAMES))
    ap.add_argument("--paper", default="A3", choices=sorted(PAPER))
    a = ap.parse_args()
    print(a.out, room(a.poster, a.out, frame=a.frame, paper=a.paper))
