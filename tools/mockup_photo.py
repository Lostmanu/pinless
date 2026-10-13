"""A poster framed and hung in a real room photo: the frame in a real material, a bevelled mat, glass, the room's own
light and colour on the print, the shadow the frame casts on the wall and the photo's grain.

    python tools/mockup_photo.py docs/poster-huella.jpg docs/wall-huella.jpg --scene sofa --frame walnut
    python tools/mockup_photo.py a.jpg b.jpg docs/wall-pair.jpg --scene sunny --frame oak --frame black

The room photos are in tools/rooms/ (free Unsplash photos; their authors are in tools/rooms/CREDITS.md). A scene says
where each frame hangs, in photo pixels, how big a 40x50 cm frame is there, and where the light comes from.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rooms")
SS = 2  # the framed print is drawn at twice its size, then reduced

# frame face colour and its darker side; wood grain strength; a metal has a sheen instead of a grain
FRAMES = {
    "oak": ((196, 161, 120), (150, 116, 82), 0.10, False),
    "walnut": ((98, 66, 46), (64, 42, 30), 0.14, False),
    "natural": ((224, 199, 162), (186, 158, 120), 0.07, False),
    "black": ((30, 29, 28), (14, 14, 14), 0.03, False),
    "white": ((240, 238, 233), (210, 207, 200), 0.02, False),
    "brass": ((214, 178, 104), (128, 94, 42), 0.0, True),
}
FRAME_CM = (40.0, 50.0)  # outer size of the frame of an A3 print
MOULDING_CM = 2.1
WINDOW_CM = (28.7, 41.0)  # the mat's window: an A3 print with 5 mm under the mat

# the scenes: frames as (centre x, top y, frame height in px), light direction (towards the shadow), shadow strength,
# glass glare, a final crop (x0, y0, x1, y1)
SCENES = {
    "sofa": {"photo": "sofa.jpg", "frames": [(1225, 118, 455)], "light": (0.35, 0.55), "shadow": 0.42,
             "glare": 0.05, "crop": (180, 0, 2280, 1582)},
    "dusk": {"photo": "dusk.jpg", "frames": [(1180, 38, 600)], "light": (-0.75, 0.45), "shadow": 0.55,
             "glare": 0.035, "crop": (150, 0, 2250, 1600)},
    "nordic": {"photo": "nordic.jpg", "frames": [(1318, 92, 520)], "light": (0.45, 0.6), "shadow": 0.30,
               "glare": 0.04, "crop": (200, 0, 2400, 1600)},
    "sunny": {"photo": "sunny.jpg", "frames": [(1150, 520, 400), (1530, 520, 400)], "light": (-0.65, 0.4),
              "shadow": 0.38, "glare": 0.06, "crop": (300, 160, 2400, 1600)},
}


def _noise(shape, scale, rng):
    h, w = shape
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    img = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-6) * 255).astype(np.uint8))
    return np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 127.5 - 1


def _grain(h, w, along, rng):
    """Wood grain: fine streaks along one axis (0: horizontal rails, 1: vertical stiles)."""
    if along == 0:
        g = _noise((max(2, h), max(2, w // 40)), 1, rng)
        g = np.asarray(Image.fromarray(((g + 1) * 127.5).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                       dtype=np.float32) / 127.5 - 1
    else:
        g = _noise((max(2, h // 40), max(2, w)), 1, rng)
        g = np.asarray(Image.fromarray(((g + 1) * 127.5).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                       dtype=np.float32) / 127.5 - 1
    return 0.7 * g + 0.3 * _noise((h, w), 6, rng)


def framed(poster, frame, height_px, light, seed=7):
    """RGBA image of the framed print, height_px tall (outer frame), lit from the side opposite to `light`."""
    face, side, grain, metal = FRAMES[frame]
    rng = np.random.default_rng(seed)
    H = int(round(height_px * SS))
    k = H / FRAME_CM[1]  # px per cm
    W = int(round(FRAME_CM[0] * k))
    m = int(round(MOULDING_CM * k))
    img = np.zeros((H, W, 3), np.float32)
    # the mat, with paper texture
    img[:] = np.array([244, 241, 234], np.float32) * (1 + 0.012 * _noise((H, W), 3, rng)[..., None])
    # the print in the window
    ww, wh = WINDOW_CM[0] * k, WINDOW_CM[1] * k
    x0, y0 = (W - ww) / 2, (H - wh) / 2 - 0.4 * k  # a little more mat under the print, as framers do
    p = poster.convert("RGB")
    pw, ph = p.size
    s = max(ww / (pw * 0.976), wh / (ph * 0.976))  # 5 mm under the mat on every side
    pr = p.resize((int(round(pw * s)), int(round(ph * s))), Image.LANCZOS)
    px0, py0 = int(round(x0 + (ww - pr.width) / 2)), int(round(y0 + (wh - pr.height) / 2))
    win = (int(round(x0)), int(round(y0)), int(round(x0 + ww)), int(round(y0 + wh)))
    crop = pr.crop((win[0] - px0, win[1] - py0, win[2] - px0, win[3] - py0))
    img[win[1]:win[3], win[0]:win[2]] = np.asarray(crop, np.float32)
    # the mat's bevel: 3 mm of white core, lit on the sides that face the light
    lx, ly = light
    b = max(2, int(round(0.3 * k)))
    bev = np.array([250, 249, 245], np.float32)
    shade_top, shade_left = (0.86 if ly > 0 else 1.0), (0.88 if lx > 0 else 1.0)
    shade_bottom, shade_right = (1.0 if ly > 0 else 0.86), (1.0 if lx > 0 else 0.88)
    img[win[1] - b:win[1], win[0] - b:win[2] + b] = bev * shade_bottom  # bevel above the window faces down
    img[win[3]:win[3] + b, win[0] - b:win[2] + b] = bev * shade_top
    img[win[1] - b:win[3] + b, win[0] - b:win[0]] = bev * shade_right
    img[win[1] - b:win[3] + b, win[2]:win[2] + b] = bev * shade_left
    # the shadow the moulding casts on the mat (the mat sits deeper)
    sh = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(sh)
    off = (int(round(lx * 0.45 * k)), int(round(ly * 0.45 * k)))
    d.rectangle([0, 0, W, H], fill=255)
    d.rectangle([m + off[0], m + off[1], W - m + off[0], H - m + off[1]], fill=0)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(0.35 * k)), np.float32)[..., None] / 255
    img *= 1 - 0.38 * sh
    # the moulding: four mitred pieces with their grain, a rounded profile and a sheen for metal
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]).astype(np.float32)
    on = dist < m
    top = on & (yy <= xx) & (yy <= W - 1 - xx) & (yy < m)
    bottom = on & (H - 1 - yy <= xx) & (H - 1 - yy <= W - 1 - xx) & (H - 1 - yy < m)
    left = on & ~top & ~bottom & (xx < m)
    right = on & ~top & ~bottom & ~left
    profile = np.clip(dist / m, 0, 1)
    round_ = 0.82 + 0.30 * np.sin(np.pi * np.clip(profile * 0.95 + 0.05, 0, 1))  # a rounded moulding
    col = np.array(face, np.float32)
    dark = np.array(side, np.float32)
    for mask, along, lit in ((top, 0, ly > 0), (bottom, 0, ly <= 0), (left, 1, lx > 0), (right, 1, lx <= 0)):
        if not mask.any():
            continue
        g = _grain(H, W, along, rng)
        base = col if lit else (0.55 * col + 0.45 * dark)
        piece = base[None, None, :] * (round_[..., None]) * (1 + grain * g[..., None])
        if metal:  # a polished profile: dark in the hollow, a bright ridge, a brushed streak along the piece
            ridge = np.exp(-((profile - 0.30) / 0.10) ** 2)
            hollow = np.exp(-((profile - 0.72) / 0.16) ** 2)
            brushed = 1 + 0.05 * _grain(H, W, along, rng)
            piece = piece * (0.80 + (0.55 if lit else 0.35) * ridge - 0.25 * hollow)[..., None] * brushed[..., None]
        img[mask] = piece[mask]
    # mitre lines
    for a, b2 in (((0, 0), (m, m)), ((W - 1, 0), (W - 1 - m, m)), ((0, H - 1), (m, H - 1 - m)),
                  ((W - 1, H - 1), (W - 1 - m, H - 1 - m))):
        n = int(np.hypot(b2[0] - a[0], b2[1] - a[1]))
        for t in np.linspace(0, 1, n):
            x, y = int(a[0] + (b2[0] - a[0]) * t), int(a[1] + (b2[1] - a[1]) * t)
            img[y, x] *= 0.80
    # the outer edge: a thin dark arris all round, lighter on the sides that face the light
    edge = (dist < max(1.0, 0.07 * k))[..., None]
    img = np.where(edge, img * 0.62, img)
    lit_edge = ((yy < max(1.0, 0.07 * k)) & (ly > 0)) | ((xx < max(1.0, 0.07 * k)) & (lx > 0)) |         ((H - 1 - yy < max(1.0, 0.07 * k)) & (ly <= 0)) | ((W - 1 - xx < max(1.0, 0.07 * k)) & (lx <= 0))
    img[lit_edge] = np.clip(img[lit_edge] * 1.25 + 12, 0, 255)
    # the rebate: a dark step where the moulding meets the mat
    lip = (dist >= m) & (dist < m + max(1.0, 0.12 * k))
    img[lip] *= 0.55
    # the glass: a soft diagonal reflection over the window and the mat
    gl = np.clip(1 - np.abs((xx / W - yy / H) * 1.6 + 0.25) * 1.4, 0, 1) ** 2
    inside = (dist >= m)[..., None]
    img = np.where(inside, img + 255 * 0.05 * gl[..., None], img)
    img = np.clip(img, 0, 255)
    rgba = np.dstack([img, np.full((H, W), 255, np.float32)]).astype(np.uint8)
    out = Image.fromarray(rgba, "RGBA")
    return out.resize((max(1, W // SS), max(1, H // SS)), Image.LANCZOS)


def _wall_light(photo, box, blur):
    """The wall's own light around box, normalised: the gradient the print must take on."""
    x0, y0, x1, y1 = box
    pad = int(blur * 2)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(photo.shape[1], x1 + pad), min(photo.shape[0], y1 + pad)
    region = Image.fromarray(photo[Y0:Y1, X0:X1].astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    reg = np.asarray(region, np.float32)
    lum = reg.mean(axis=2, keepdims=True) + 1e-3
    cut = reg[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    lcut = lum[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    tint = cut.reshape(-1, 3).mean(axis=0) / cut.reshape(-1, 3).mean()  # the light's colour on the wall
    return lcut / np.median(lcut), tint


def _photo_noise(photo):
    """Noise level of the photo: the high frequencies of its flattest region."""
    g = photo.mean(axis=2)
    hp = g - np.asarray(Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32)
    h, w = g.shape
    best = None
    for y in range(0, h - 64, 64):
        for x in range(0, w - 64, 64):
            s = float(hp[y:y + 64, x:x + 64].std())
            best = s if best is None else min(best, s)
    return max(0.6, best or 1.0)


def compose(posters, out, scene="sofa", frames=None, width=2000, seed=7):
    S = SCENES[scene]
    photo = np.asarray(Image.open(os.path.join(ROOMS, S["photo"])).convert("RGB"), np.float32)
    frames = frames or ["oak"] * len(S["frames"])
    lx, ly = S["light"]
    rng = np.random.default_rng(seed)
    noise = _photo_noise(photo)
    for poster, frame, (cx, top, hpx) in zip(posters, frames, S["frames"]):
        art = framed(Image.open(poster), frame, hpx, (lx, ly), seed=seed)
        aw, ah = art.size
        x0, y0 = int(round(cx - aw / 2)), int(round(top))
        box = (x0, y0, x0 + aw, y0 + ah)
        k = ah / FRAME_CM[1]
        # the room's light and colour where the frame hangs, measured on the bare wall (before any shadow)
        light, tint = _wall_light(photo, box, blur=max(8, ah * 0.35))
        wall_level = float(np.median(photo[box[1]:box[3], box[0]:box[2]].mean(axis=2)))
        # the shadow on the wall: a soft one from the frame's depth, and a tight contact one
        for off_cm, blur_cm, strength in ((2.4, 1.6, S["shadow"]), (0.35, 0.35, 0.45 * S["shadow"])):
            m = Image.new("L", (photo.shape[1], photo.shape[0]), 0)
            dx, dy = lx * off_cm * k, ly * off_cm * k
            ImageDraw.Draw(m).rectangle([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy], fill=255)
            a = np.asarray(m.filter(ImageFilter.GaussianBlur(blur_cm * k)), np.float32)[..., None] / 255
            photo = photo * (1 - strength * a)
        # the room's light and colour on the framed print: its gradient, its tint and its level (a white mat
        # reads a touch darker than the wall, being under glass and a step back)
        rgb = np.asarray(art, np.float32)[..., :3]
        rgb = rgb * (light ** 0.85) * (1 + 0.65 * (tint - 1))[None, None, :]
        rgb = rgb * (wall_level / 244.0 * 0.97)
        rgb += S["glare"] * 255 * np.clip(1 - np.abs(np.linspace(-1, 1, rgb.shape[1]))[None, :, None] * 1.5, 0, 1)
        rgb += rng.normal(0, noise, rgb.shape)
        rgb = np.clip(rgb, 0, 255)
        soft = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.45))
        photo[box[1]:box[3], box[0]:box[2]] = np.asarray(soft, np.float32)
    img = Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8))
    img = img.crop(S["crop"])
    img = img.resize((width, int(round(img.height * width / img.width))), Image.LANCZOS)
    img.save(out, quality=88, optimize=True)
    return img.size


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="poster images, then the output .jpg")
    ap.add_argument("--scene", default="sofa", choices=sorted(SCENES))
    ap.add_argument("--frame", action="append", choices=sorted(FRAMES), help="one per poster (default oak)")
    ap.add_argument("--width", type=int, default=2000)
    a = ap.parse_args()
    *posters, out = a.files
    if len(posters) != len(SCENES[a.scene]["frames"]):
        ap.error(f"scene {a.scene} hangs {len(SCENES[a.scene]['frames'])} poster(s)")
    print(out, compose(posters, out, a.scene, a.frame, a.width))


if __name__ == "__main__":
    main()
