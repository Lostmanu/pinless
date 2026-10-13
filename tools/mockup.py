"""The poster framed and hung in a room, at real scale, rendered as light shows it rather than drawn.

Everything is modelled in centimetres at its true size: the frame (its four rails, the bevelled mat of an A3 print, the
print and the glass), the sideboard under it, a stack of books, two stoneware vases and a dried branch. The room is lit
as a room is: daylight through a window on the left, the sun through some of its panes (dappled by a tree outside)
and the sky in all of them, bouncing off the walls, the floor and the ceiling. A small path tracer renders it in
numpy, in parallel bands; the noise of the light is filtered apart from the textures (the print stays sharp), and the
image is developed as a camera would: exposure, a filmic curve, a slight vignette and grain.

A3 goes in a standard 40x50 cm frame with a mat; 50x70 in a 50x70 frame without one.
Usage: python tools/mockup.py docs/poster-huella.jpg docs/wall-huella.jpg --frame oak [--spp 64]
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

EPS = 1e-3
# frame finishes: face colour (sRGB), wood grain strength, coat (Fresnel at normal incidence), roughness
FRAMES = {"oak": ((198, 160, 116), 1.0, 0.04, 0.32), "black": ((30, 29, 28), 0.35, 0.05, 0.22),
          "white": ((238, 236, 230), 0.0, 0.04, 0.3)}
PAPER = {"A3": ((29.7, 42.0), (40.0, 50.0)), "50x70": ((50.0, 70.0), (50.0, 70.0))}  # paper, frame window (cm)
TOP_Y, GAP = 80.0, 20.0  # the sideboard's top above the floor, and the gap from it to the frame (cm)
DIFFUSE, GLOSSY, METAL, GLASS = 0, 1, 2, 3
FIREFLY = 3.0  # a sample brighter than this (linear) is clipped: a rare path to the sun is noise, not light


# --- textures -----------------------------------------------------------------------------------------------------

def _lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _noise(shape, scale, rng):
    """Smooth noise of unit spread: random at 1/scale resolution, enlarged bicubically."""
    h, w = shape
    small = rng.normal(0, 1, (max(2, int(h / scale) + 2), max(2, int(w / scale) + 2))).astype(np.float32)
    a = np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), np.float32)
    return a / (a.std() + 1e-6)


def fbm(shape, scale, rng, octaves=4, gain=0.5):
    out, amp, total = np.zeros(shape, np.float32), 1.0, 0.0
    for i in range(octaves):
        out += amp * _noise(shape, max(1.0, scale / 2 ** i), rng)
        total += amp
        amp *= gain
    return out / total


def wood(rng, size, base, strength=1.0):
    """A veneer whose grain runs along the second axis: growth rings warped by noise, long streaks and fine pores."""
    h, w = size
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    warp = fbm(size, 150, rng, 4) * 24 + fbm(size, 35, rng, 3) * 4
    rings = np.sin((yy + warp) * 2 * np.pi / 31.0)
    rings = np.sign(rings) * np.abs(rings) ** 0.4
    streak = np.asarray(Image.fromarray(fbm((h, w // 24), 4, rng, 3)).resize((w, h), Image.BICUBIC))
    pores = np.asarray(Image.fromarray((rng.random((h, w // 8)) < 0.07).astype(np.float32))
                       .resize((w, h), Image.BILINEAR))
    v = (0.07 * rings + 0.08 * streak - 0.12 * pores + 0.03 * fbm(size, 10, rng, 2)) * strength
    col = _lin(np.asarray(base, np.float64) / 255).astype(np.float32)
    return np.clip(col * (1 + v[..., None] * np.array([1.0, 1.1, 1.3], np.float32)), 0, 1)


def sample(tex, u, v, wrap=True):
    """Bilinear lookup of an (H, W, C) texture: u along W, v along H, in texels."""
    H, W = tex.shape[:2]
    if wrap:
        u, v = np.mod(u, W - 1), np.mod(v, H - 1)
    else:
        u, v = np.clip(u, 0, W - 1.001), np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    a = tex[v0, u0] * (1 - fu) + tex[v0, u0 + 1] * fu
    b = tex[v0 + 1, u0] * (1 - fu) + tex[v0 + 1, u0 + 1] * fu
    return a * (1 - fv) + b * fv


# --- primitives -----------------------------------------------------------------------------------------------------

def _t_box(lo, hi, o, d, inv):
    t1, t2 = (lo - o) * inv, (hi - o) * inv
    tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
    near, far = tn.max(1), tf.min(1)
    t = np.where(near > EPS, near, far)
    return np.where((far >= near) & (t > EPS), t, np.inf)


def _t_ellipsoid(c, r, R, o, d):
    oc = ((o - c) @ R) / r
    dd = (d @ R) / r
    a, b, cc = (dd * dd).sum(1), (oc * dd).sum(1), (oc * oc).sum(1) - 1
    disc = b * b - a * cc
    sq = np.sqrt(np.maximum(disc, 0))
    t0, t1 = (-b - sq) / a, (-b + sq) / a
    t = np.where(t0 > EPS, t0, t1)
    return np.where((disc > 0) & (t > EPS), t, np.inf)


def _t_capsule(pa, pb, r, o, d):
    """Inigo Quilez's capsule intersection."""
    ba, oa = pb - pa, o - pa
    baba, bard, baoa = ba @ ba, d @ ba, oa @ ba
    rdoa, oaoa = (d * oa).sum(1), (oa * oa).sum(1)
    a = baba - bard * bard
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa * baoa - r * r * baba
    h = b * b - a * c
    safe = np.where(np.abs(a) > 1e-12, a, 1.0)
    t = np.where((h >= 0) & (np.abs(a) > 1e-12), (-b - np.sqrt(np.maximum(h, 0))) / safe, np.inf)
    y = baoa + t * bard
    body = np.isfinite(t) & (y > 0) & (y < baba)
    out = np.where(body, t, np.inf)
    for end in (pa, pb):
        oc = o - end
        bb, cc = (d * oc).sum(1), (oc * oc).sum(1) - r * r
        hh = bb * bb - cc
        tc = -bb - np.sqrt(np.maximum(hh, 0))
        out = np.where((hh > 0) & ~body & (tc < out), tc, out)
    return np.where(out > EPS, out, np.inf)


def _t_cylinder(cx, cz, y0, y1, r, o, d):
    """A vertical cylinder with its top cap: (t, on the cap)."""
    ox, oz = o[:, 0] - cx, o[:, 2] - cz
    a = d[:, 0] ** 2 + d[:, 2] ** 2
    b = ox * d[:, 0] + oz * d[:, 2]
    c = ox * ox + oz * oz - r * r
    disc = b * b - a * c
    sq = np.sqrt(np.maximum(disc, 0))
    aa = np.where(a > 1e-12, a, 1.0)
    t = np.full(len(o), np.inf)
    for tt in ((-b - sq) / aa, (-b + sq) / aa):
        y = o[:, 1] + tt * d[:, 1]
        t = np.where((disc > 0) & (a > 1e-12) & (tt > EPS) & (y >= y0) & (y <= y1) & (tt < t), tt, t)
    dy = np.where(np.abs(d[:, 1]) > 1e-12, d[:, 1], 1e-12)
    tt = (y1 - o[:, 1]) / dy
    px, pz = ox + tt * d[:, 0], oz + tt * d[:, 2]
    top = (tt > EPS) & (px * px + pz * pz <= r * r) & (tt < t)
    return np.where(top, tt, t), top


def _t_quad(x0, x1, y0, y1, z, o, d):
    dz = np.where(np.abs(d[:, 2]) > 1e-12, d[:, 2], 1e-12)
    t = (z - o[:, 2]) / dz
    px, py = o[:, 0] + t * d[:, 0], o[:, 1] + t * d[:, 1]
    return np.where((t > EPS) & (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1), t, np.inf)


class Scene:
    """Primitives (box, ellipsoid, capsule, cylinder, quad), each with a material and a group: a group's bounding
    box is tested first, so a ray far from the branch never meets its forty pieces."""

    def __init__(self):
        self.prims, self.materials, self.groups = [], [], []
        self._group = -1

    def material(self, kind=DIFFUSE, albedo=(0.7, 0.7, 0.7), tex=None, coat=0.0, rough=1.0):
        self.materials.append(dict(kind=kind, albedo=np.asarray(albedo, float), tex=tex, coat=coat, rough=rough))
        return len(self.materials) - 1

    def group(self):
        self.groups.append([np.full(3, np.inf), np.full(3, -np.inf)])
        self._group = len(self.groups) - 1
        return self._group

    def end_group(self):
        self._group = -1

    def _add(self, kind, params, mat, lo, hi, **extra):
        g = self._group
        if g >= 0:
            self.groups[g][0] = np.minimum(self.groups[g][0], lo)
            self.groups[g][1] = np.maximum(self.groups[g][1], hi)
        self.prims.append(dict(kind=kind, p=params, mat=mat, group=g, **extra))

    def box(self, lo, hi, mat, rounding=0.0, round_axes=(0, 1, 2)):
        """An axis-aligned box; rounding: the radius of its edges across round_axes (the edges it rounds)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self._add("box", (lo, hi), mat, lo, hi, rounding=rounding, round_axes=tuple(round_axes))

    def ellipsoid(self, c, r, mat, rot=None):
        c, r = np.asarray(c, float), np.asarray(r, float)
        R = np.eye(3) if rot is None else np.asarray(rot, float)
        ext = np.abs(R) @ r
        self._add("ell", (c, r, R), mat, c - ext, c + ext)

    def capsule(self, a, b, r, mat):
        a, b = np.asarray(a, float), np.asarray(b, float)
        self._add("cap", (a, b, float(r)), mat, np.minimum(a, b) - r, np.maximum(a, b) + r)

    def cylinder(self, cx, cz, y0, y1, r, mat):
        self._add("cyl", (cx, cz, y0, y1, r), mat, np.array([cx - r, y0, cz - r]), np.array([cx + r, y1, cz + r]))

    def quad(self, x0, x1, y0, y1, z, mat):
        self._add("quad", (x0, x1, y0, y1, z), mat, np.array([x0, y0, z - 0.01]), np.array([x1, y1, z + 0.01]))

    def freeze(self):
        self.kind = np.array([m["kind"] for m in self.materials])
        self.coat = np.array([m["coat"] for m in self.materials])
        self.rough = np.array([m["rough"] for m in self.materials])
        self.mat = np.array([p["mat"] for p in self.prims])
        for g in self.groups:
            g[0], g[1] = g[0] - 0.05, g[1] + 0.05

    def intersect(self, o, d, tmax=None):
        """(t, the primitive hit or -1, extra): the nearest hit of every ray (extra: the box face's axis, or 1 on a
        cylinder's cap)."""
        n = len(o)
        best = np.full(n, np.inf) if tmax is None else np.asarray(tmax, float).copy()
        prim = np.full(n, -1)
        extra = np.zeros(n, np.int64)
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        rays = []
        for lo, hi in self.groups:
            t = _t_box(lo, hi, o, d, inv)
            inside = np.all((o > lo) & (o < hi), axis=1)
            rays.append(np.flatnonzero(inside | (t < best)))
        everyone = np.arange(n)
        for k, P in enumerate(self.prims):
            sel = everyone if P["group"] < 0 else rays[P["group"]]
            if not len(sel):
                continue
            oo, dd = o[sel], d[sel]
            ex = None
            if P["kind"] == "box":
                lo, hi = P["p"]
                ii = inv[sel]
                t = _t_box(lo, hi, oo, dd, ii)
                t1, t2 = (lo - oo) * ii, (hi - oo) * ii
                tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
                ex = np.where(tn.max(1) > EPS, tn.argmax(1), tf.argmin(1))
            elif P["kind"] == "ell":
                t = _t_ellipsoid(*P["p"], oo, dd)
            elif P["kind"] == "cap":
                t = _t_capsule(*P["p"], oo, dd)
            elif P["kind"] == "cyl":
                t, top = _t_cylinder(*P["p"], oo, dd)
                ex = top.astype(np.int64)
            else:
                t = _t_quad(*P["p"], oo, dd)
            win = t < best[sel]
            if win.any():
                s = sel[win]
                best[s], prim[s] = t[win], k
                extra[s] = 0 if ex is None else ex[win]
        return best, prim, extra

    def normals(self, p, d, prim, extra):
        """The geometric normal at each hit (facing wherever it faces) and its material."""
        n = np.zeros_like(p)
        for k in np.unique(prim[prim >= 0]):
            m = prim == k
            P, pp = self.prims[k], p[m]
            if P["kind"] == "box":
                lo, hi = P["p"]
                ax = extra[m]
                nn = np.zeros_like(pp)
                nn[np.arange(len(pp)), ax] = -np.sign(d[m][np.arange(len(pp)), ax])
                r = P["rounding"]
                if r > 0:  # rounded edges: near an edge the normal leans towards the neighbouring face
                    for b in P["round_axes"]:
                        other = ax != b
                        wl = np.where(other, np.clip(1 - (pp[:, b] - lo[b]) / r, 0, 1), 0)
                        wh = np.where(other, np.clip(1 - (hi[b] - pp[:, b]) / r, 0, 1), 0)
                        nn[:, b] += 1.3 * (wh - wl)
            elif P["kind"] == "ell":
                c, r, R = P["p"]
                nn = (((pp - c) @ R) / r ** 2) @ R.T
            elif P["kind"] == "cap":
                a, b, _ = P["p"]
                ba = b - a
                h = np.clip(((pp - a) @ ba) / (ba @ ba), 0, 1)
                nn = pp - (a + h[:, None] * ba)
            elif P["kind"] == "cyl":
                cx, cz = P["p"][0], P["p"][1]
                nn = np.stack([pp[:, 0] - cx, np.zeros(len(pp)), pp[:, 2] - cz], 1)
                nn[extra[m] == 1] = (0.0, 1.0, 0.0)
            else:
                nn = np.zeros_like(pp)
                nn[:, 2] = 1.0
            n[m] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        return n


def _rot(ax, a):
    c, s = math.cos(a), math.sin(a)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --- the room -------------------------------------------------------------------------------------------------------

class Room:
    """The room box we stand in (x right, y up, z out of the wall the frame hangs on), its window on the left wall,
    the sun and the sky."""
    LO, HI = np.array([-200.0, 0.0, 0.0]), np.array([240.0, 270.0, 420.0])
    WINDOW = (70.0, 230.0, 60.0, 250.0)  # y0, y1, z0, z1 on the wall x = -200
    SUN_PANES = (70.0, 230.0, 60.0, 104.0)  # the panes the sun reaches; a house next door shades the others
    MULLION = 2.4  # half-width of the mullions (cm)
    SUN_DIR = np.array([0.81, -0.364, -0.486]) / np.linalg.norm([0.81, -0.364, -0.486])  # into the room
    SUN = np.array([1.0, 0.88, 0.72]) * 7.0
    SKY = np.array([0.66, 0.76, 0.95]) * 1.6

    def __init__(self, seed):
        rng = np.random.default_rng(seed + 1)
        blobs = fbm((256, 512), 18, rng, 4)
        self.leaves = (0.15 + 0.85 * np.clip(0.5 + 1.4 * (blobs - 0.35), 0, 1) ** 1.5)[..., None]

    def exit(self, o, d):
        """How far a ray from inside goes to the room's walls, and which (0..5: -x, +x, -y, +y, -z, +z)."""
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        tf = np.maximum((self.LO - o) * inv, (self.HI - o) * inv)
        ax = tf.argmin(1)
        t = tf[np.arange(len(o)), ax]
        return t, ax * 2 + (np.take_along_axis(d, ax[:, None], 1)[:, 0] > 0)

    def pane(self, p):
        """True where a point of the left wall is glass of the window, not its frame or mullions."""
        y0, y1, z0, z1 = self.WINDOW
        y, z = p[:, 1], p[:, 2]
        inside = (y > y0) & (y < y1) & (z > z0) & (z < z1)
        bars = (np.abs(y - (y0 + y1) / 2 - 12) < self.MULLION) | (np.abs(z - (z0 + z1) / 2) < self.MULLION)
        return inside & ~bars

    def sunlit(self, p):
        """How much sun a point of the left wall lets in: the panes it reaches, dappled by the tree."""
        y0, y1, z0, z1 = self.SUN_PANES
        ok = self.pane(p) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] > z0) & (p[:, 2] < z1)
        wy0, wy1, wz0, wz1 = self.WINDOW
        u = (p[:, 2] - wz0) / (wz1 - wz0) * (self.leaves.shape[1] - 1)
        v = (wy1 - p[:, 1]) / (wy1 - wy0) * (self.leaves.shape[0] - 1)
        return np.where(ok, sample(self.leaves, u, v, wrap=False)[:, 0], 0.0)

    def surface(self, p, side):
        """Albedo and inward normal of the room's walls, floor (oak) and ceiling."""
        n = np.zeros_like(p)
        n[np.arange(len(p)), side // 2] = np.where(side % 2 == 1, -1.0, 1.0)
        alb = np.tile(_lin(np.array([226, 222, 214]) / 255), (len(p), 1))
        alb[side == 2] = _lin(np.array([150, 116, 84]) / 255)
        alb[side == 3] = _lin(np.array([240, 238, 234]) / 255)
        return alb, n


# --- the scene ------------------------------------------------------------------------------------------------------

def build(frame="oak", paper="A3", seed=7):
    rng = np.random.default_rng(seed)
    S = Scene()
    (pw, ph), (fw, fh) = PAPER[paper]
    prof, depth, off = 2.2, 2.4, 0.6  # the rail's face width and depth, and how far it stands off the wall
    ow, oh = fw + 2 * prof, fh + 2 * prof
    x0, y0 = -ow / 2, TOP_Y + GAP
    x1, y1 = x0 + ow, y0 + oh
    face = off + depth
    S.geom = dict(x0=x0, x1=x1, y0=y0, y1=y1, prof=prof, face=face, fw=fw, fh=fh)
    colour, grain, coat, rough = FRAMES[frame]
    S.frame_finish = (colour, grain)
    S.wall = S.material(tex="wall")
    m_frame = S.material(GLOSSY, tex="frame", coat=coat, rough=rough)
    m_mat = S.material(albedo=_lin(np.array([244, 242, 236]) / 255), tex="mat")
    m_print = S.material(tex="print")
    m_glass = S.material(GLASS)

    S.group()  # the frame and what it holds
    S.box((x0, y0, off), (x0 + prof, y1, face), m_frame, 0.3)
    S.box((x1 - prof, y0, off), (x1, y1, face), m_frame, 0.3)
    S.box((x0 + prof, y1 - prof, off), (x1 - prof, y1, face), m_frame, 0.3, round_axes=(1, 2))
    S.box((x0 + prof, y0, off), (x1 - prof, y0 + prof, face), m_frame, 0.3, round_axes=(1, 2))
    wx0, wx1, wy0, wy1 = x0 + prof, x1 - prof, y0 + prof, y1 - prof
    mat_front = face - 0.5
    px0, py0 = wx0 + (fw - pw) / 2, wy0 + (fh - ph) / 2
    if paper == "A3":  # the mat's window overlaps the print by 3 mm all round
        cx0, cx1, cy0, cy1 = px0 + 0.3, px0 + pw - 0.3, py0 + 0.3, py0 + ph - 0.3
        back = mat_front - 0.15
        S.box((wx0, wy0, back), (cx0, wy1, mat_front), m_mat)
        S.box((cx1, wy0, back), (wx1, wy1, mat_front), m_mat)
        S.box((cx0, cy1, back), (cx1, wy1, mat_front), m_mat)
        S.box((cx0, wy0, back), (cx1, cy0, mat_front), m_mat)
        S.cut = (cx0, cx1, cy0, cy1, mat_front)
        S.quad(px0, px0 + pw, py0, py0 + ph, back - 0.01, m_print)
    else:
        S.cut = None
        S.quad(px0, px0 + pw, py0, py0 + ph, mat_front, m_print)
    S.print_rect = (px0, px0 + pw, py0, py0 + ph)
    S.quad(wx0, wx1, wy0, wy1, face - 0.4, m_glass)
    S.box((wx0, wy0, off), (wx1, wy1, off + 0.3), S.material(albedo=(0.3, 0.27, 0.24)))  # the backing board
    S.end_group()

    # the sideboard: wider than the view, 45 cm deep, a top board 3 cm thick overhanging the doors by 1.2 cm
    m_front = S.material(GLOSSY, tex="oak_front", coat=0.04, rough=0.35)
    m_top = S.material(GLOSSY, tex="oak_top", coat=0.05, rough=0.2)
    m_brass = S.material(METAL, albedo=_lin(np.array([180, 146, 92]) / 255), rough=0.22)
    S.box((-170.0, 0.0, 0.5), (170.0, TOP_Y - 3.0, 45.0), m_front, 0.2)
    S.box((-171.0, TOP_Y - 3.0, 0.5), (171.0, TOP_Y, 46.2), m_top, 0.5)
    S.doors = (-170.0, -85.0, 0.0, 85.0, 170.0)
    for xh in (-5.0, 5.0):
        S.box((xh - 0.55, TOP_Y - 24.0, 45.0), (xh + 0.55, TOP_Y - 10.0, 46.6), m_brass, 0.45)

    # books lying flat, spines to the camera, left of the frame; a pebble on top
    S.group()
    y, bx = TOP_Y, x0 - 6.0
    for i, (col, h, w, dz) in enumerate((((84, 98, 90), 3.2, 30.0, 22.0), ((208, 194, 166), 2.4, 27.0, 19.5),
                                          ((156, 64, 46), 2.8, 24.5, 17.0))):
        mb = S.material(albedo=_lin(np.array(col) / 255), tex="cloth")
        xr, z0 = bx - i * 1.5, 8.0 + i * 0.9
        S.box((xr - w, y, z0), (xr, y + h, z0 + dz), mb, 0.3)
        y += h
    S.ellipsoid((bx - 13.0, y + 1.5, 17.5), (4.3, 1.7, 3.3), S.material(GLOSSY, albedo=_lin(np.array([74, 72, 68])
                                                                                              / 255), tex="speck",
                                                                         coat=0.03, rough=0.3))
    S.end_group()

    # two stoneware vases and a dried branch, right of the frame
    vx, vz = x1 + 13.0, 18.0
    m_vase = S.material(GLOSSY, albedo=_lin(np.array([216, 206, 188]) / 255), tex="speck", coat=0.03, rough=0.45)
    m_vase2 = S.material(GLOSSY, albedo=_lin(np.array([82, 74, 64]) / 255), tex="speck", coat=0.06, rough=0.2)
    S.group()
    S.ellipsoid((vx, TOP_Y + 11.5, vz), (8.5, 12.0, 8.5), m_vase)
    S.cylinder(vx, vz, TOP_Y + 18.0, TOP_Y + 29.6, 2.6, m_vase)
    S.ellipsoid((vx, TOP_Y + 29.8, vz), (3.3, 0.8, 3.3), m_vase)
    S.end_group()
    S.group()
    v2 = vx + 16.5
    S.ellipsoid((v2, TOP_Y + 6.6, 23.0), (6.8, 6.8, 6.8), m_vase2)
    S.cylinder(v2, 23.0, TOP_Y + 9.0, TOP_Y + 14.4, 2.1, m_vase2)
    S.end_group()
    S.group()
    m_stem = S.material(albedo=_lin(np.array([112, 96, 72]) / 255))
    leaf_mats = [S.material(albedo=_lin(np.array(c) / 255), tex="speck")
                 for c in ((132, 128, 96), (118, 118, 88), (146, 136, 100))]
    for ang, length, bend, n_leaves, dz in ((-0.08, 60.0, 0.24, 11, 0.0), (0.34, 47.0, 0.16, 8, 2.5),
                                           (0.14, 37.0, -0.12, 6, -2.0)):
        pts = [np.array([vx + math.sin(ang + bend * t) * length * t, TOP_Y + 27.0 + math.cos(ang + bend * t)
                         * length * t, vz + dz * t]) for t in np.linspace(0, 1, 8)]
        for j, (a_, b_) in enumerate(zip(pts, pts[1:])):
            S.capsule(a_, b_, 0.24 - 0.016 * j, m_stem)
        for j, t in enumerate(np.linspace(0.3, 0.98, n_leaves)):
            a = ang + bend * t
            c = np.array([vx + math.sin(a) * length * t, TOP_Y + 27.0 + math.cos(a) * length * t, vz + dz * t])
            side = 1 if j % 2 else -1
            spread = side * (0.8 + 0.4 * rng.random())
            size = 1.5 + 0.7 * rng.random() - 0.4 * t
            along = np.array([math.sin(a + spread), math.cos(a + spread), 0.25 * rng.normal()])
            along /= np.linalg.norm(along)
            R = _rot("z", -(a + spread)) @ _rot("y", 0.6 * rng.normal()) @ _rot("x", 0.5 * side)
            S.ellipsoid(c + along * (size * 0.9), (0.42 + 0.15 * rng.random(), size, 0.06), leaf_mats[j % 3], rot=R)
    S.end_group()
    S.freeze()
    return S


class Textures:
    """Each texture from its own seed: the room is the same whatever the frame."""

    def __init__(self, S, poster, seed):
        def rng(k):
            return np.random.default_rng([seed, k])

        colour, grain = S.frame_finish
        if grain > 0:
            self.frame = wood(rng(1), (256, 2048), colour, grain)
        else:
            self.frame = np.ones((8, 8, 3), np.float32) * _lin(np.array(colour) / 255).astype(np.float32)
        self.oak_top = wood(rng(2), (512, 4096), (190, 152, 110))
        self.oak_front = wood(rng(3), (512, 4096), (184, 146, 104), 0.9)
        self.wall = fbm((1024, 1024), 5, rng(4), 5)[..., None]
        self.wall_lo = fbm((256, 256), 30, rng(5), 3)[..., None]
        self.speck = fbm((512, 512), 2, rng(6), 2)[..., None]
        self.cloth = rng(7).normal(0, 1, (256, 256, 1)).astype(np.float32)
        self.print = _lin(np.asarray(Image.open(poster).convert("RGB"), np.float32) / 255).astype(np.float32)


# --- shading --------------------------------------------------------------------------------------------------------

def shade_surface(S, T, mats, p, n):
    """(albedo, normal) of every hit: the textures, the plaster's bump, the bevel of the mat's window."""
    alb = np.zeros_like(p)
    n = n.copy()
    g = S.geom
    for mid in np.unique(mats):
        m = mats == mid
        M, pp = S.materials[mid], p[m]
        tex = M["tex"]
        if tex == "print":
            x0, x1, y0, y1 = S.print_rect
            H, W = T.print.shape[:2]
            alb[m] = sample(T.print, (pp[:, 0] - x0) / (x1 - x0) * (W - 1), (y1 - pp[:, 1]) / (y1 - y0) * (H - 1),
                            wrap=False)
        elif tex == "wall":
            u, v = pp[:, 0] * 6.0, pp[:, 1] * 6.0
            a = 1 + 0.02 * sample(T.wall, u, v)[:, 0] + 0.025 * sample(T.wall_lo, pp[:, 0] * 2, pp[:, 1] * 2)[:, 0]
            alb[m] = _lin(np.array([234, 229, 220]) / 255) * a[:, None]
            du = sample(T.wall, u + 1, v)[:, 0] - sample(T.wall, u - 1, v)[:, 0]
            dv = sample(T.wall, u, v + 1)[:, 0] - sample(T.wall, u, v - 1)[:, 0]
            nn = n[m]
            nn[:, 0] -= 0.03 * du
            nn[:, 1] -= 0.03 * dv
            n[m] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
        elif tex == "frame":  # mitred rails: the grain runs along each, and turns at the diagonal of a corner
            dx = np.minimum(pp[:, 0] - g["x0"], g["x1"] - pp[:, 0])
            dy = np.minimum(pp[:, 1] - g["y0"], g["y1"] - pp[:, 1])
            vert = (dx < g["prof"]) & ((dy >= g["prof"]) | (dx < dy))
            along = np.where(vert, pp[:, 1], pp[:, 0]) * 18.0
            across = np.where(vert, pp[:, 0] + 50, pp[:, 1]) * 18.0 + pp[:, 2] * 6
            c = sample(T.frame, along, across)
            seam = (np.abs(dx - dy) < 0.02) & (dx < g["prof"]) & (dy < g["prof"])
            alb[m] = np.where(seam[:, None], c * 0.75, c)
        elif tex in ("oak_top", "oak_front"):
            if tex == "oak_top":
                c = sample(T.oak_top, pp[:, 0] * 14.0, pp[:, 2] * 14.0)
            else:
                c = sample(T.oak_front, pp[:, 0] * 14.0, pp[:, 1] * 14.0 + 300)
                joint = np.zeros(len(pp), bool)
                for xj in S.doors:
                    joint |= np.abs(pp[:, 0] - xj) < 0.15
                c = np.where((joint & (pp[:, 2] > 44.9))[:, None], c * 0.18, c)
            alb[m] = c
        elif tex == "mat":
            a = 1 + 0.012 * sample(T.speck, pp[:, 0] * 40, pp[:, 1] * 40)[:, 0]
            alb[m] = M["albedo"] * a[:, None]
            if S.cut is not None:  # the bevel: 3.5 mm round the window, its white core, facing the window at 45 deg
                cx0, cx1, cy0, cy1, zf = S.cut
                front = pp[:, 2] > zf - 0.01
                dist = np.stack([pp[:, 0] - cx0, cx1 - pp[:, 0], pp[:, 1] - cy0, cy1 - pp[:, 1]], 1)
                inside_band = (dist > -0.35) & (dist < 0)
                nn = n[m]
                for j, vec in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    on = front & inside_band[:, j] & np.all(dist[:, [i for i in range(4) if i != j]] > -0.35, axis=1)
                    nn[on, 0] += vec[0]
                    nn[on, 1] += vec[1]
                n[m] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
                band = front & np.any(inside_band, axis=1)
                core = M["albedo"] * 1.04
                alb[m] = np.where(band[:, None], core, alb[m])
        elif tex == "cloth":
            a = 1 + 0.07 * sample(T.cloth, pp[:, 0] * 30, pp[:, 1] * 30 + pp[:, 2] * 30)[:, 0]
            alb[m] = M["albedo"] * a[:, None]
        elif tex == "speck":
            a = 1 + 0.08 * sample(T.speck, pp[:, 0] * 25 + pp[:, 2] * 11, pp[:, 1] * 25)[:, 0]
            alb[m] = M["albedo"] * a[:, None]
        else:
            alb[m] = M["albedo"]
    return alb, n


def _onb(n):
    sign = np.where(n[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    return (np.stack([1 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], 1),
            np.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], 1))


def _cosine(n, rng):
    u1, u2 = rng.random(len(n)), rng.random(len(n))
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    t, b = _onb(n)
    return t * (r * np.cos(phi))[:, None] + b * (r * np.sin(phi))[:, None] + n * np.sqrt(1 - u1)[:, None]


def _lobe(axis, expo, rng):
    u1, u2 = rng.random(len(axis)), rng.random(len(axis))
    cos_t = u1 ** (1 / (expo + 1))
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = 2 * np.pi * u2
    t, b = _onb(axis)
    return t * (sin_t * np.cos(phi))[:, None] + b * (sin_t * np.sin(phi))[:, None] + axis * cos_t[:, None]


def _schlick(cos, f0):
    return f0 + (1 - f0) * (1 - np.clip(cos, 0, 1)) ** 5


def _transmission(S, p, dirs, dist):
    """How much of a straight segment is clear: 0 behind any object, 0.97 through the glass of the frame."""
    t, prim, _ = S.intersect(p, dirs, tmax=dist - 2 * EPS)
    tr = np.ones(len(p))
    hit = prim >= 0
    if hit.any():
        glass = np.zeros(len(p), bool)
        glass[hit] = S.kind[S.mat[prim[hit]]] == GLASS
        if glass.any():
            gi = np.flatnonzero(glass)
            tr[gi] = 0.97 * _transmission(S, p[gi] + dirs[gi] * (t[gi] + 2 * EPS)[:, None], dirs[gi],
                                          dist[gi] - t[gi] - 2 * EPS)
        tr[hit & ~glass] = 0.0
    return tr


def _brdf(alb, spec, expo, n, ls, view):
    """Lambert under the coat, and a normalised Blinn-Phong lobe for the coat."""
    h = ls - view
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    ndh = np.clip((n * h).sum(1), 0, 1)
    return alb / np.pi * (1 - spec)[:, None] + (spec * (expo + 8) / (8 * np.pi) * ndh ** expo)[:, None]


def _direct(S, room, p, n, alb, spec, expo, view, rng):
    """The sun through the panes it reaches and the sky in all of them (one sample of each)."""
    out = np.zeros_like(p)
    ls = -room.SUN_DIR + rng.normal(0, 0.0045, p.shape)
    ls /= np.linalg.norm(ls, axis=1, keepdims=True)
    cos = (n * ls).sum(1)
    ok = np.flatnonzero(cos > 0)
    if len(ok):
        t_wall, side = room.exit(p[ok], ls[ok])
        sun = np.where(side == 0, room.sunlit(p[ok] + ls[ok] * t_wall[:, None]), 0.0)
        lit = sun > 0
        if lit.any():
            sun[lit] *= _transmission(S, p[ok][lit], ls[ok][lit], t_wall[lit])
            out[ok] += _brdf(alb[ok], spec[ok], expo[ok], n[ok], ls[ok], view[ok]) * room.SUN * (cos[ok] * sun)[:, None]
    y0, y1, z0, z1 = room.WINDOW
    q = np.stack([np.full(len(p), room.LO[0]), y0 + rng.random(len(p)) * (y1 - y0),
                  z0 + rng.random(len(p)) * (z1 - z0)], 1)
    v = q - p
    dist = np.linalg.norm(v, axis=1)
    ls = v / dist[:, None]
    cos, cos_l = (n * ls).sum(1), -ls[:, 0]
    ok = np.flatnonzero(room.pane(q) & (cos > 0) & (cos_l > 0))
    if len(ok):
        g = cos[ok] * cos_l[ok] * (y1 - y0) * (z1 - z0) / dist[ok] ** 2 * _transmission(S, p[ok], ls[ok], dist[ok])
        out[ok] += _brdf(alb[ok], spec[ok], expo[ok], n[ok], ls[ok], view[ok]) * room.SKY * g[:, None]
    return out


def trace(S, room, T, o, d, rng, bounces=3):
    """Radiance along each camera ray; the albedo, normal and depth of what it first sees (for the filter); and
    whether it looked through the glass (for its reflection, laid on in the development)."""
    N = len(o)
    o, d = o.copy(), d.copy()
    L, thr = np.zeros((N, 3)), np.ones((N, 3))
    alive = np.ones(N, bool)
    seen = np.zeros(N, bool)  # the first surface (past the glass) is recorded
    sharp = np.ones(N, bool)  # the camera ray (through the glass too): the only one that sees the window's sky
    aux_a, aux_n, aux_z = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros(N)
    glazed = np.zeros(N)  # 1 where the camera ray went through the glass
    for bounce in range(bounces + 1):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        oo, dd = o[idx], d[idx]
        t, prim, extra = S.intersect(oo, dd)
        t_room, side = room.exit(oo, dd)
        obj = (prim >= 0) & (t < t_room)
        p = oo + dd * np.where(obj, t, t_room)[:, None]
        mats = np.where(obj, S.mat[np.maximum(prim, 0)], -1)
        n = np.zeros_like(p)
        if obj.any():
            n[obj] = S.normals(p[obj], dd[obj], prim[obj], extra[obj])
        alb = np.zeros_like(p)
        rs = ~obj
        if rs.any():
            alb[rs], n[rs] = room.surface(p[rs], side[rs])
        plaster = rs & (side == 4)
        mats[plaster] = S.wall
        tex = mats >= 0
        if tex.any():
            alb[tex], n[tex] = shade_surface(S, T, mats[tex], p[tex], n[tex])
        window = rs & (side == 0) & room.pane(p)
        kind = np.where(mats >= 0, S.kind[np.maximum(mats, 0)], DIFFUSE)
        # first surfaces, for the filter
        rec = ~seen[idx] & (kind != GLASS)
        if rec.any():
            r = idx[rec]
            aux_a[r] = np.where(window[rec][:, None], 1.0, alb[rec])
            aux_n[r] = n[rec]
            aux_z[r] = np.linalg.norm(p[rec] - oo[rec], axis=1)
            seen[r] = True
        if window.any():
            w = idx[window]
            L[w] += np.where(sharp[w][:, None], thr[w] * room.SKY, 0.0)
            alive[w] = False
        glass = kind == GLASS
        if glass.any():  # anti-reflective picture glass: straight through, 98.5 % of the light
            gi = np.flatnonzero(glass)
            if bounce == 0:
                glazed[idx[gi]] = 1.0
            o[idx[gi]] = p[gi] + dd[gi] * 2 * EPS
            thr[idx[gi]] *= 0.985
        surf = np.flatnonzero(~window & ~glass)
        if not len(surf):
            continue
        si = idx[surf]
        view = dd[surf]
        nn = n[surf]
        nn[(nn * view).sum(1) > 0] *= -1
        a, k_ = alb[surf], kind[surf]
        mm = mats[surf]
        coat = np.where(mm >= 0, S.coat[np.maximum(mm, 0)], 0.0)
        rough = np.where(mm >= 0, S.rough[np.maximum(mm, 0)], 1.0)
        expo = 2.0 / np.maximum(rough, 1e-3) ** 2
        metal = k_ == METAL
        cos_v = np.clip(-(view * nn).sum(1), 0, 1)
        spec = np.where(metal, 1.0, np.where(coat > 0, _schlick(cos_v, coat), 0.0))
        diff = np.where(metal[:, None], 0.0, a)
        pts = p[surf] + nn * 2 * EPS
        L[si] += thr[si] * _direct(S, room, pts, nn, diff, spec, expo, view, rng) * np.where(
            metal[:, None], a / np.maximum(a.max(1, keepdims=True), 1e-3), 1.0)
        go_spec = rng.random(len(surf)) < spec
        newd = _cosine(nn, rng)
        if go_spec.any():
            r_dir = view[go_spec] - 2 * (view[go_spec] * nn[go_spec]).sum(1)[:, None] * nn[go_spec]
            gd = _lobe(r_dir, expo[go_spec], rng)
            below = (gd * nn[go_spec]).sum(1) <= 0
            gd[below] = r_dir[below]
            newd[go_spec] = gd
        o[si], d[si] = pts, newd
        thr[si] *= np.where(go_spec[:, None], np.where(metal[:, None], a, 1.0), diff)
        sharp[si] = False
        if bounce >= 1:  # Russian roulette
            q = np.clip(thr[si].max(1), 0.05, 0.95)
            live = rng.random(len(si)) < q
            thr[si] /= q[:, None]
            alive[si[~live]] = False
    miss = ~seen
    aux_a[miss] = 1.0
    return L, aux_a, aux_n, aux_z, glazed


# --- camera, bands, filter and development ---------------------------------------------------------------------------

def view_of(S, width, height):
    """The camera: square to the wall (no keystone), framed on the wall by a shifted lens, focused on it."""
    view_h = max(112.0, S.geom["fh"] + 62.0)
    bottom = TOP_Y - 24.0
    view_w = view_h * width / height
    return dict(view=(-view_w / 2, view_w / 2, bottom, bottom + view_h), pos=np.array([6.0, TOP_Y + 40.0, 320.0]),
                aperture=0.35)


def camera_rays(cam, px, py, W, H, rng):
    x0, x1, y0, y1 = cam["view"]
    u, v = (px + rng.random(len(px))) / W, (py + rng.random(len(py))) / H
    target = np.stack([x0 + u * (x1 - x0), y1 - v * (y1 - y0), np.zeros(len(px))], 1)
    r, phi = np.sqrt(rng.random(len(px))) * cam["aperture"], rng.random(len(px)) * 2 * np.pi
    lens = cam["pos"] + np.stack([r * np.cos(phi), r * np.sin(phi), np.zeros(len(px))], 1)
    d = target - lens
    return lens, d / np.linalg.norm(d, axis=1, keepdims=True)


_W = {}


def _init(poster, frame, paper, seed, width, height):
    S = build(frame, paper, seed)
    _W.update(S=S, T=Textures(S, poster, seed), room=Room(seed), cam=view_of(S, width, height))


def _band(args):
    y0, y1, W, H, spp, seed = args
    S, T, room, cam = _W["S"], _W["T"], _W["room"], _W["cam"]
    rng = np.random.default_rng(seed)
    py, px = np.mgrid[y0:y1, 0:W]
    px, py = px.ravel().astype(float), py.ravel().astype(float)
    acc = np.zeros((len(px), 3))
    a_acc, n_acc, z_acc, g_acc = np.zeros((len(px), 3)), np.zeros((len(px), 3)), np.zeros(len(px)), np.zeros(len(px))
    for _ in range(spp):
        o, d = camera_rays(cam, px, py, W, H, rng)
        L, a, n, z, g = trace(S, room, T, o, d, rng)
        acc += np.minimum(L, FIREFLY)
        a_acc += a
        n_acc += n
        z_acc += z
        g_acc += g
    shape = (y1 - y0, W)
    return (y0, (acc / spp).reshape(*shape, 3).astype(np.float32), (a_acc / spp).reshape(*shape, 3).astype(np.float32),
            (n_acc / spp).reshape(*shape, 3).astype(np.float32), (z_acc / spp).reshape(shape).astype(np.float32),
            (g_acc / spp).reshape(shape).astype(np.float32))


def denoise(L, A, N, Z, iterations=5, sigma_z=0.6, sigma_n=64.0, sigma_l=6.0):
    """The light without the textures (L / A) filtered by an edge-avoiding a-trous wavelet (Dammertz et al. 2010):
    it does not cross edges of depth, normal or brightness, then the textures go back on. The print stays sharp."""
    A = np.maximum(A, 0.02)
    E = L / A
    k = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
    Nn = N / np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-6)
    H, W = Z.shape
    for it in range(iterations):
        step = 2 ** it
        lum = E @ np.array([0.2126, 0.7152, 0.0722])
        acc, wsum = np.zeros_like(E), np.zeros((H, W))
        pad = 2 * step
        Ep = np.pad(E, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        Np = np.pad(Nn, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        Zp = np.pad(Z, pad, mode="edge")
        Lp = np.pad(lum, pad, mode="edge")
        scale = sigma_l * (np.mean(lum) + 1e-6)
        for i, ki in enumerate(k):
            for j, kj in enumerate(k):
                dy, dx = (i - 2) * step, (j - 2) * step
                sl = (slice(pad + dy, pad + dy + H), slice(pad + dx, pad + dx + W))
                w = ki * kj
                w = w * np.exp(-np.abs(Zp[sl] - Z) / (sigma_z * step))
                w = w * np.clip((Np[sl] * Nn).sum(2), 0, 1) ** sigma_n
                w = w * np.exp(-np.abs(Lp[sl] - lum) / scale)
                acc += Ep[sl] * w[..., None]
                wsum += w
        E = acc / np.maximum(wsum, 1e-9)[..., None]
    return E * A


def develop(img, out, A=None, G=None, seed=7, exposure=1.0):
    """White balance on the light itself (the image over its albedo: white paper prints white), a touch warm;
    exposure set on the wall; the faint reflection of the window on the glass (G: where the camera looks through
    it), a broad soft band; a filmic curve (ACES, Narkowicz's fit), a slight vignette, fine grain."""
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    if A is not None:
        light = np.median((img / np.maximum(A, 0.05)).reshape(-1, 3), axis=0)
        img = img * (light.mean() / np.maximum(light, 1e-6)) * np.array([1.0, 0.98, 0.94])
    key = float(np.percentile(img.mean(2), 55))
    if G is not None and G.any():
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ys, xs = np.nonzero(G > 0.5)
        t = (xx - xs.min()) / max(1, np.ptp(xs)) + 0.55 * (yy - ys.min()) / max(1, np.ptp(ys))
        band = 0.010 + 0.035 * np.exp(-((t - 0.42) / 0.16) ** 2)
        img = img + (G * band)[..., None] * key * 1.6
    x = img * (0.46 / max(key, 1e-6)) * exposure
    yy, xx = np.mgrid[0:H, 0:W]
    x = x * (1 - 0.2 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.5)[..., None]
    y = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    s = _srgb(y) + rng.normal(0, 0.005, (H, W))[..., None] + rng.normal(0, 0.002, (H, W, 3))
    Image.fromarray(np.clip(s * 255 + 0.5, 0, 255).astype(np.uint8)).save(out, quality=92, optimize=True)


def room(poster, out, frame="oak", paper="A3", width=2000, height=1600, seed=7, spp=64, workers=None, band=24,
         raw=None, filtered=True):
    """Render the poster framed on the wall to out (JPEG); returns its size. workers: processes (1 renders here)."""
    jobs = [(y, min(height, y + band), width, height, spp, seed * 100003 + y) for y in range(0, height, band)]
    shape = (height, width)
    L, A, N = (np.zeros((*shape, 3), np.float32) for _ in range(3))
    Z, G = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    if workers is None:  # a small render is done here; a large one in parallel bands (more than 8 do not help)
        workers = 1 if width * height * spp < 4e6 else max(1, min(8, (os.cpu_count() or 2) - 1))

    def put(parts):
        for y0, *planes in parts:
            for full, part in zip((L, A, N, Z, G), planes):
                full[y0:y0 + len(part)] = part

    if workers == 1:
        _init(poster, frame, paper, seed, width, height)
        put(map(_band, jobs))
    else:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per process
            os.environ[var] = "1"
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(poster, frame, paper, seed, width, height)) as ex:
            put(ex.map(_band, jobs))
    if raw:
        np.savez_compressed(raw, L=L, A=A, N=N, Z=Z, G=G)
    develop(denoise(L, A, N, Z) if filtered else L, out, A, G, seed)
    return width, height


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("poster")
    ap.add_argument("out")
    ap.add_argument("--frame", default="oak", choices=sorted(FRAMES))
    ap.add_argument("--paper", default="A3", choices=sorted(PAPER))
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    print(a.out, room(a.poster, a.out, frame=a.frame, paper=a.paper, width=a.width, height=a.height, spp=a.spp,
                      raw=a.raw, workers=a.workers))
