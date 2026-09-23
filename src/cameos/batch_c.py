"""Cameo sprites, batch C (Touhou fan work, drawn procedurally from scratch).

Coordinates are in a 100x100 unit space; Pen maps them to a 4x canvas and
downsamples with LANCZOS on finish().
"""
import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw

OL = (38, 24, 36)
SKIN = (255, 230, 214)
WHITE = (250, 250, 252)
BLACK = (42, 40, 52)


# ---------------------------------------------------------------- colour
def rgba(c):
    return tuple(int(v) for v in c) + ((255,) if len(c) == 3 else ())


def dk(c, f=0.78):
    return tuple(int(v * f) for v in c[:3])


def lt(c, f=0.4):
    return tuple(int(v + (255 - v) * f) for v in c[:3])


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------- geometry
def arc(cx, cy, rx, ry, a0, a1, n=16):
    out = []
    for i in range(n):
        a = math.radians(a0 + (a1 - a0) * i / (n - 1))
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


def bez(a, c, b, n=14):
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * c[0] + t * t * b[0],
                    (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * c[1] + t * t * b[1]))
    return out


def rot_ell(cx, cy, rx, ry, ang, n=28):
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def ribbon(pts, w0, w1=None):
    """Tapered band polygon along a polyline."""
    w1 = w0 if w1 is None else w1
    L, R = [], []
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        j, k = min(i + 1, n - 1), max(i - 1, 0)
        tx, ty = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
        d = math.hypot(tx, ty) or 1
        nx, ny = -ty / d, tx / d
        w = (w0 + (w1 - w0) * i / (n - 1)) / 2
        L.append((x + nx * w, y + ny * w))
        R.append((x - nx * w, y - ny * w))
    return L + R[::-1]


# ---------------------------------------------------------------- pen
class Pen:
    def __init__(self, size, s=1.0, ox=0.0, oy=0.0):
        self.size, self.S = size, size * 4
        self.k = self.S / 100.0
        self.s, self.ox, self.oy = s, ox, oy
        self.ow = 0.75
        self.img = Image.new("RGBA", (self.S, self.S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.grad = None

    def P(self, x, y):
        return ((x * self.s + self.ox) * self.k, (y * self.s + self.oy) * self.k)

    def L(self, v):
        return v * self.s * self.k

    def _mask(self, shape, off=0.0):
        m = Image.new("L", self.img.size, 0)
        shape(ImageDraw.Draw(m), 255, 0, off, off)
        return m

    def _part(self, shape, col, shade, ol, sd=2.0):
        if ol:
            shape(self.d, rgba(OL), self.L(self.ow), 0, 0)
        if col == "grad":  # vertical gradient fill set by set_grad
            m = self._mask(shape)
            self.img.paste(self.grad[0], (0, 0), m)
            if shade:
                self.img.paste(self.grad[1], (0, 0),
                               ImageChops.subtract(m, self._mask(shape, -self.L(sd))))
            return
        col = rgba(col)
        if col[3] < 255:
            lay = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
            shape(ImageDraw.Draw(lay), col, 0, 0, 0)
            self.img.alpha_composite(lay)
            return
        shape(self.d, col, 0, 0, 0)
        if shade:
            sh = shade if isinstance(shade, tuple) else dk(col)
            m = ImageChops.subtract(self._mask(shape), self._mask(shape, -self.L(sd)))
            self.img.paste(rgba(sh), (0, 0), m)

    def set_grad(self, c1, c2, y0, y1):
        ys = np.arange(self.S, dtype=np.float32)
        t = np.clip((ys / self.k - self.oy) / self.s, y0, y1)
        t = ((t - y0) / (y1 - y0))[:, None, None]
        a, b = np.array(c1, np.float32), np.array(c2, np.float32)
        row = a + (b - a) * t
        arr = np.broadcast_to(row, (self.S, self.S, 3))
        full = np.concatenate([arr, np.full((self.S, self.S, 1), 255, np.float32)], 2)
        g = Image.fromarray(full.astype(np.uint8), "RGBA")
        full[..., :3] *= 0.78
        self.grad = (g, Image.fromarray(full.astype(np.uint8), "RGBA"))

    def ell(self, cx, cy, rx, ry, col, shade=True, ol=True, sd=2.0):
        x0, y0 = self.P(cx - rx, cy - ry)
        x1, y1 = self.P(cx + rx, cy + ry)

        def f(d, fill, g, dx, dy):
            d.ellipse([x0 - g + dx, y0 - g + dy, x1 + g + dx, y1 + g + dy], fill=fill)
        self._part(f, col, shade, ol, sd)

    def poly(self, pts, col, shade=True, ol=True, sd=2.0):
        P = [self.P(*q) for q in pts]

        def f(d, fill, g, dx, dy):
            Q = [(x + dx, y + dy) for x, y in P]
            if g > 0:
                d.line(Q + [Q[0]], fill=fill, width=max(1, int(2 * g)), joint="curve")
                for x, y in Q:
                    d.ellipse([x - g, y - g, x + g, y + g], fill=fill)
            d.polygon(Q, fill=fill)
        self._part(f, col, shade, ol, sd)

    def stroke(self, pts, w, col, shade=False, ol=True, sd=1.5):
        P = [self.P(*q) for q in pts]
        W = self.L(w)

        def f(d, fill, g, dx, dy):
            Q = [(x + dx, y + dy) for x, y in P]
            d.line(Q, fill=fill, width=max(1, int(W + 2 * g)), joint="curve")
            r = W / 2 + g
            for x, y in (Q[0], Q[-1]):
                d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        self._part(f, col, shade, ol, sd)

    def finish(self, rotate=False):
        im = self.img.resize((self.size, self.size), Image.LANCZOS)
        return im.rotate(180) if rotate else im


# ---------------------------------------------------------------- parts
def strand(p, a, c, b, w, col, w2=0.0, shade=True):
    p.poly(ribbon(bez(a, c, b), w, w2), col, shade)


def slab(p, a, b, w, col, shade=True):
    p.poly(ribbon([a, b], w), col, shade)


def star(p, cx, cy, r, col, ol=True, rot=-90):
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = math.radians(rot + 36 * i)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    p.poly(pts, col, shade=False, ol=ol)


def heart(p, cx, cy, r, col, ol=True):
    pts = []
    for i in range(30):
        t = 2 * math.pi * i / 30
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * r / 16, cy + y * r / 16))
    p.poly(pts, col, shade=False, ol=ol)


def bow(p, cx, cy, s, col, tails=True):
    if tails:
        strand(p, (cx - 0.5 * s, cy), (cx - 1.2 * s, cy + 1.5 * s), (cx - 1.4 * s, cy + 3 * s), 1.4 * s, col, 1.0 * s)
        strand(p, (cx + 0.5 * s, cy), (cx + 1.2 * s, cy + 1.5 * s), (cx + 1.4 * s, cy + 3 * s), 1.4 * s, col, 1.0 * s)
    p.poly([(cx, cy), (cx - 3 * s, cy - 2 * s), (cx - 3.4 * s, cy + 0.2 * s), (cx - 2.8 * s, cy + 2 * s)], col)
    p.poly([(cx, cy), (cx + 3 * s, cy - 2 * s), (cx + 3.4 * s, cy + 0.2 * s), (cx + 2.8 * s, cy + 2 * s)], col)
    p.ell(cx, cy, 0.9 * s, 1.0 * s, col)


def hand(p, x, y, col=SKIN):
    p.ell(x, y, 2.6, 2.6, col, shade=False)


def legs(p, col=WHITE, shoe=(90, 55, 45), y0=70, y1=89, xs=(45.2, 54.8), w=2.5):
    for x in xs:
        p.poly([(x - w, y0), (x + w, y0), (x + w * 0.9, y1), (x - w * 0.9, y1)], col)
        p.ell(x + (0.6 if x > 50 else -0.6), y1 + 1.3, 3.6, 2.2, shoe)


def torso(p, col, y0=45, y1=60, w0=8.5, w1=11):
    p.poly([(50 - w0, y0), (50 + w0, y0), (50 + w1, y1), (50 - w1, y1)], col)


def skirt(p, col, y1=78, w1=19, y0=57, w0=11, hem=None, hemw=2.5, bulge=2.0):
    def bottom(yy, ww, b):
        return [(50 + ww - 2 * ww * t, yy + b * math.sin(math.pi * t)) for t in np.linspace(0, 1, 11)]
    p.poly([(50 - w0, y0), (50 + w0, y0)] + bottom(y1, w1, bulge), col)
    if hem:
        yi = y1 - hemw
        wi = w0 + (w1 - w0) * (yi - y0) / (y1 - y0)
        top = bottom(yi, wi, bulge)[::-1]
        p.poly(top + bottom(y1, w1, bulge), hem, ol=False)


def arms(p, col, lh=(35.5, 62), rh=(64.5, 62), w=5.0, short=None, cuff=None):
    for sx, h in ((42, lh), (58, rh)):
        if short:
            p.stroke([(sx, 48.5), h], 3.4, SKIN)
            mid = (sx + (h[0] - sx) * short, 48.5 + (h[1] - 48.5) * short)
            p.stroke([(sx, 48.5), mid], w, col, shade=True)
        else:
            p.stroke([(sx, 48.5), h], w, col, shade=True)
            if cuff:
                e = (sx + (h[0] - sx) * 0.82, 48.5 + (h[1] - 48.5) * 0.82)
                p.stroke([e, e], w + 0.6, cuff)
        hand(p, *h)


def head(p):
    p.ell(50, 30, 16, 14.5, SKIN, shade=dk(SKIN, 0.93))


def eye(p, x, y, col, rx=3.1, ry=4.0, lid=0.0):
    p.ell(x, y - 0.6, rx + 0.55, ry + 0.8, OL, shade=False, ol=False)
    p.ell(x, y, rx, ry, (255, 255, 255), shade=False, ol=False)
    p.ell(x, y + 0.5, rx * 0.86, ry * 0.86, col, shade=False, ol=False)
    p.ell(x, y - 0.4, rx * 0.8, ry * 0.5, dk(col, 0.6), shade=False, ol=False)
    p.ell(x, y + 0.9, rx * 0.4, ry * 0.4, dk(col, 0.35), shade=False, ol=False)
    p.ell(x - 1.0, y - 1.2, 1.0, 1.3, (255, 255, 255), shade=False, ol=False)
    p.ell(x + 1.1, y + 2.0, 0.5, 0.5, (255, 255, 255), shade=False, ol=False)
    if lid:
        ly = y - ry + 2 * ry * lid
        p.poly([(x - rx - 1, y - ry - 1.6), (x + rx + 1, y - ry - 1.6), (x + rx + 1, ly), (x - rx - 1, ly)],
               SKIN, shade=False, ol=False)
        p.stroke([(x - rx - 0.4, ly + 0.3), (x + rx + 0.4, ly + 0.3)], 0.9, OL, ol=False)


def face(p, ec, mouth="smile", lid=0.0, blush=True, ey=33.5, dx=6.5):
    ecs = ec if isinstance(ec[0], tuple) else (ec, ec)
    eye(p, 50 - dx, ey, ecs[0], lid=lid)
    eye(p, 50 + dx, ey, ecs[1], lid=lid)
    if blush:
        p.ell(40.5, 38.3, 2.4, 1.1, (255, 110, 130, 110), shade=False, ol=False)
        p.ell(59.5, 38.3, 2.4, 1.1, (255, 110, 130, 110), shade=False, ol=False)
    mc = (190, 60, 70)
    if mouth in ("smile", "fang"):
        p.stroke(bez((48.4, 40), (50, 41.3), (51.6, 40), 8), 0.55, OL, ol=False)
        if mouth == "fang":
            p.poly([(50.8, 40.5), (51.8, 40.3), (51.4, 41.9)], WHITE, shade=False, ol=False)
    elif mouth == "flat":
        p.stroke([(48.9, 40.6), (51.1, 40.6)], 0.5, OL, ol=False)
    elif mouth == "frown":
        p.stroke(bez((48.4, 41.2), (50, 39.9), (51.6, 41.2), 8), 0.55, OL, ol=False)
    elif mouth == "open":
        p.ell(50, 40.6, 1.4, 1.1, mc, shade=False)
    elif mouth in ("grin", "tongue"):
        p.poly([(47.6, 39.6), (52.4, 39.6)] + bez((52.4, 39.6), (50, 43.2), (47.6, 39.6), 8), mc, shade=False)
        if mouth == "tongue":
            p.ell(50.8, 41.6, 1.2, 1.3, (255, 130, 150), shade=False, ol=False)


def hair_back(p, col, length=None, w=21, wave=6):
    if length:
        pts = [(31, 24), (50 - w, length - 6)]
        for i, x in enumerate(np.linspace(50 - w, 50 + w, 2 * wave + 1)):
            pts.append((x, length - (0 if i % 2 else 4)))
        pts += [(50 + w, length - 6), (69, 24)]
        p.poly(pts, col)
    p.ell(50, 28, 19.5, 19, col)


def bangs(p, col, tip=30.5, side=43, n=5, cy=27, rx=18.8, ry=18, sidew=4.2, valley=23.5):
    pts = arc(50, cy, rx, ry, 180, 360, 22)
    pts += [(50 + rx, cy + 5), (50 + rx - 1.5, side), (50 + rx - sidew, cy + 4)]
    for i, x in enumerate(np.linspace(50 + rx - sidew - 1.5, 50 - rx + sidew + 1.5, 2 * n + 1)):
        pts.append((x, tip + (1.2 if i == n else 0)) if i % 2 else (x, valley))
    pts += [(50 - rx + sidew, cy + 4), (50 - rx + 1.5, side), (50 - rx, cy + 5)]
    p.poly(pts, col)


def feather_wing(p, root, side, col, span=40, rise=30, drop=28, n=5, inner=None):
    x0, y0 = root
    tip = (x0 + side * span, y0 - rise)
    top = bez(root, (x0 + side * span * 0.35, y0 - rise * 1.25), tip, 12)
    end = (x0 + side * span * 0.2, y0 + drop)
    base = bez(tip, (x0 + side * span * 0.95, y0 + drop * 0.6), end, 2 * n + 1)
    low = []
    for i, (x, y) in enumerate(base):
        if i % 2:  # notch pulled toward root
            low.append((x + (x0 - x) * 0.18, y + (y0 - y) * 0.18))
        else:
            low.append((x, y))
    p.poly(top + low + [(x0, y0 + 4)], col)
    if inner:
        for i in range(1, n):
            q = base[2 * i]
            strand(p, (x0 + side * 3, y0), ((x0 + q[0]) / 2, (y0 + q[1]) / 2 - 3), (q[0] - side * 3, q[1] - 2), 0.9, inner, 0.2, shade=False)


def planet(p, cx, cy, r, kind):
    if kind == "earth":
        p.ell(cx, cy, r, r, (60, 120, 220))
        p.poly(rot_ell(cx - r * 0.3, cy - r * 0.2, r * 0.35, r * 0.55, 20, 12), (80, 180, 90), shade=False, ol=False)
        p.poly(rot_ell(cx + r * 0.4, cy + r * 0.35, r * 0.3, r * 0.25, -20, 12), (80, 180, 90), shade=False, ol=False)
    elif kind == "moon":
        p.ell(cx, cy, r, r, (240, 225, 140))
        for ax, ay, rr in ((-0.3, -0.25, 0.22), (0.35, 0.2, 0.18), (-0.1, 0.45, 0.14)):
            p.ell(cx + ax * r, cy + ay * r, rr * r, rr * r, (205, 190, 110), shade=False, ol=False)
    else:  # otherworld
        p.ell(cx, cy, r, r, (215, 45, 55))
        p.stroke(arc(cx, cy, r * 0.7, r * 0.25, 0, 180, 10), r * 0.12, (160, 30, 40), ol=False)
    p.ell(cx - r * 0.4, cy - r * 0.4, r * 0.22, r * 0.18, (255, 255, 255, 170), shade=False, ol=False)


FONT = {
    "W": ["10001", "10001", "10101", "10101", "01010"],
    "E": ["111", "100", "110", "100", "111"],
    "L": ["100", "100", "100", "100", "111"],
    "C": ["011", "100", "100", "100", "011"],
    "O": ["010", "101", "101", "101", "010"],
    "M": ["10001", "11011", "10101", "10001", "10001"],
    "H": ["101", "101", "111", "101", "101"],
}


def pixtext(p, text, cx, y, px, col):
    width = sum(len(FONT[ch][0]) + 1 for ch in text) - 1
    x = cx - width * px / 2
    for ch in text:
        g = FONT[ch]
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v == "1":
                    X, Y = x + c * px, y + r * px
                    p.poly([(X, Y), (X + px, Y), (X + px, Y + px), (X, Y + px)], col, shade=False, ol=False)
        x += (len(g[0]) + 1) * px


def wisp(p, x, y, r, col):
    pts = arc(x, y, r, r, 0, 180, 10)[::-1][::-1]
    pts += [(x - r * 1.05, y - r * 0.9), (x - r * 0.45, y - r * 0.6), (x - r * 0.1, y - r * 2.6),
            (x + r * 0.35, y - r * 0.9), (x + r * 0.9, y - r * 1.6), (x + r, y - r * 0.2)]
    p.poly(pts, col, shade=False)
    p.ell(x, y, r, r, col, shade=False, ol=False)
    p.ell(x, y + r * 0.1, r * 0.55, r * 0.55, lt(col, 0.6), shade=False, ol=False)


# ================================================================ characters
def draw_eiki(size=384):
    p = Pen(size, 0.93, 3.5, 4.5)
    G, NAVY, GOLD = (95, 175, 105), (45, 60, 130), (235, 195, 70)
    hair_back(p, G)
    legs(p, WHITE, (60, 40, 40))
    skirt(p, (45, 45, 58), y1=82, w1=18, hem=(235, 235, 240), hemw=1.8)
    torso(p, WHITE)
    p.poly([(41.5, 45), (47, 45), (48.5, 60), (39, 60)], NAVY)
    p.poly([(53, 45), (58.5, 45), (61, 60), (51.5, 60)], NAVY)
    p.stroke([(47, 45.5), (48.4, 59.5)], 0.9, GOLD, ol=False)
    p.stroke([(53, 45.5), (51.6, 59.5)], 0.9, GOLD, ol=False)
    bow(p, 50, 47.5, 1.1, (200, 40, 50))
    arms(p, WHITE, rh=(65, 60), cuff=NAVY)
    head(p)
    face(p, (50, 90, 210), "flat")
    bangs(p, G)
    strand(p, (63.5, 32), (69.5, 42), (67, 53), 5.5, G)
    # hat
    for sx in (-1, 1):
        strand(p, (50 + sx * 14.5, 19), (50 + sx * 18, 28), (50 + sx * 19, 40), 2.6, (220, 45, 55), 1.8)
        strand(p, (50 + sx * 13, 19.5), (50 + sx * 15, 30), (50 + sx * 16, 42), 2.2, WHITE, 1.6)
    p.poly([(34, 21), (66, 21), (62, 5), (38, 5)], NAVY)
    p.stroke([(34.5, 20.6), (65.5, 20.6)], 1.8, GOLD, ol=False)
    p.stroke([(38.4, 5.6), (61.6, 5.6)], 1.4, GOLD, ol=False)
    p.ell(50, 13, 4, 4.6, GOLD)
    p.stroke([(48, 11), (52, 11)], 0.7, dk(GOLD, 0.6), ol=False)
    p.stroke([(50, 10), (50, 16)], 0.7, dk(GOLD, 0.6), ol=False)
    p.stroke([(48, 14.5), (52, 14.5)], 0.7, dk(GOLD, 0.6), ol=False)
    # rod of remorse
    slab(p, (65, 64), (77, 38), 3.4, (225, 205, 150))
    p.stroke([(71.6, 50.5), (72.4, 48.8)], 3.2, (200, 60, 60), ol=False)
    hand(p, 65, 60)
    return p.finish()


def draw_tenshi(size=384):
    p = Pen(size, 0.94, 2.0, 4.0)
    H = (85, 145, 235)
    hair_back(p, H, 86, w=21)
    legs(p, WHITE, (60, 50, 60))
    skirt(p, (60, 115, 205), y1=78, w1=19, hem=WHITE, hemw=3)
    rb = [(235, 70, 70), (250, 160, 60), (250, 230, 90), (100, 200, 110), (90, 150, 240), (160, 100, 220)]
    for i, c in enumerate(rb):
        x = 32.5 + i * 7
        p.poly([(x - 3.5, 76.5), (x + 3.5, 76.5), (x, 81.5)], c, shade=False)
    torso(p, WHITE)
    bow(p, 50, 48, 1.3, (220, 40, 55))
    arms(p, WHITE, rh=(66, 60), cuff=(60, 115, 205))
    head(p)
    face(p, (200, 35, 45), "fang")
    bangs(p, H)
    # hat
    p.ell(50, 15.5, 17.5, 3.4, BLACK, shade=False)
    p.ell(50, 11.5, 11, 6.5, BLACK, shade=(20, 20, 28))
    p.stroke([(39.5, 13.8), (60.5, 13.8)], 1.3, (230, 70, 80), ol=False)
    for x, y in ((43, 11), (57.5, 10.3)):
        strand(p, (x + 0.5, y - 2.5), (x + 3, y - 5.5), (x + 5.2, y - 4.5), 2.3, (90, 170, 80), 0.3)
        p.ell(x, y, 3.3, 3.1, (255, 180, 165), shade=(240, 130, 130))
        p.stroke([(x, y - 2.5), (x, y + 1.5)], 0.5, (230, 120, 120), ol=False)
    # sword of hisou: flame-shaped blade
    a, b = (67, 55), (82, 10)
    L, R = [], []
    for t in np.linspace(0, 1, 18):
        x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        dx, dy = (b[1] - a[1]), -(b[0] - a[0])
        d = math.hypot(dx, dy)
        w = (1 - t) ** 0.6 * (2.6 + 1.1 * math.sin(t * 22))
        L.append((x + dx / d * w, y + dy / d * w))
        R.append((x - dx / d * w, y - dy / d * w))
    p.poly(L + R[::-1], (255, 105, 45), shade=(215, 55, 35))
    p.stroke([(68, 52), (79, 18)], 0.9, (255, 225, 110), ol=False)
    p.ell(66.5, 57, 3.6, 1.4, (190, 150, 60))
    slab(p, (65.5, 63), (66.8, 57.5), 2.0, (110, 60, 40))
    hand(p, 66, 60)
    return p.finish()


def draw_iku(size=384):
    p = Pen(size, 0.94, 3.0, 4.5)
    H, SH = (115, 90, 175), (240, 205, 235)
    p.poly(ribbon(bez((27, 62), (50, 30), (73, 62), 16), 6.5), SH, shade=(215, 170, 215))
    hair_back(p, H)
    legs(p, (40, 40, 50), (40, 30, 40))
    skirt(p, BLACK, y1=85, w1=17, y0=56, hem=(90, 80, 110), hemw=1.5)
    torso(p, WHITE)
    bow(p, 50, 48, 1.3, (220, 45, 60))
    arms(p, WHITE, lh=(35, 61), rh=(65, 61))
    for s in (-1, 1):
        pts = [(50 + s * (12 + 12 * t + 2.5 * math.sin(t * 7)), 55 + 34 * t) for t in np.linspace(0, 1, 18)]
        p.poly(ribbon(pts, 5, 9), SH, shade=(215, 170, 215))
        e = pts[-1]
        for k in (-1, 0, 1):
            p.ell(e[0] + k * 3, e[1] + 0.3, 1.6, 1.4, WHITE, shade=False)
    head(p)
    face(p, (200, 40, 55), "smile")
    bangs(p, H, tip=31, side=45)
    # hat + long feeler ribbons
    for s in (-1, 1):
        p.stroke(bez((50 + s * 8, 12), (50 + s * 22, 2), (50 + s * 36, 16), 14), 1.6, (220, 45, 60))
    p.ell(50, 14.5, 14.5, 3.3, BLACK, shade=False)
    p.ell(50, 11.5, 8, 4.5, BLACK, shade=(20, 20, 28))
    p.ell(50, 12.5, 2.2, 2.2, (220, 45, 60), shade=False)
    return p.finish()


def third_eye(p, x, y, r, iris, cord, closed=False, ends=((34, 26), (68, 58))):
    for e in ends:
        p.stroke(bez((x, y), ((x + e[0]) / 2 + (-6 if e[0] < x else 6), (y + e[1]) / 2), e, 10), 1.2, cord)
    p.ell(x, y, r, r * 0.9, WHITE if not closed else cord, shade=(225, 220, 230) if not closed else True)
    if closed:
        p.stroke(bez((x - r * 0.7, y), (x, y + r * 0.6), (x + r * 0.7, y), 8), 0.8, OL, ol=False)
        for t in (-0.4, 0, 0.4):
            p.stroke([(x + t * r, y + r * 0.35), (x + t * r * 1.3, y + r * 0.65)], 0.5, OL, ol=False)
    else:
        p.ell(x, y, r * 0.55, r * 0.6, iris, shade=False)
        p.ell(x, y, r * 0.25, r * 0.35, dk(iris, 0.4), shade=False, ol=False)
        p.ell(x - r * 0.25, y - r * 0.3, r * 0.15, r * 0.15, WHITE, shade=False, ol=False)


def draw_satori(size=384):
    p = Pen(size)
    H = (225, 150, 205)
    hair_back(p, H)
    legs(p, WHITE, (230, 120, 170))
    skirt(p, (240, 150, 195), y1=78, w1=18, hem=(250, 200, 225), hemw=2)
    torso(p, (145, 195, 235))
    for y in (49, 53.5):
        heart(p, 50, y, 1.3, (250, 140, 190))
    arms(p, (145, 195, 235), lh=(36, 60), cuff=(250, 200, 225))
    third_eye(p, 42, 58, 4.4, (215, 40, 70), (230, 100, 150), ends=((33, 28), (66, 62)))
    heart(p, 66, 64, 1.6, (230, 100, 150))
    hand(p, 36, 60)
    head(p)
    face(p, (170, 60, 150), "flat", lid=0.35)
    bangs(p, H, tip=32, n=6, side=41)
    for x in (37, 63):
        strand(p, (x, 16), (x + (2 if x < 50 else -2), 11), (x + (6 if x < 50 else -6), 9.5), 3, H)
    p.stroke(arc(50, 27, 18.4, 17.4, 200, 340, 14), 1.4, BLACK)
    heart(p, 34.5, 22, 2, (230, 100, 150))
    heart(p, 65.5, 22, 2, (230, 100, 150))
    return p.finish()


def draw_koishi(size=384):
    p = Pen(size, 0.95, 2.5, 3.5)
    H, Y, GR = (175, 205, 170), (250, 220, 95), (105, 170, 90)
    hair_back(p, H, 52, w=20, wave=4)
    legs(p, WHITE, (40, 40, 50))
    skirt(p, GR, y1=78, w1=18.5, hem=lt(GR, 0.3), hemw=2)
    for x in (40, 50, 60):
        p.ell(x, 70, 1.5, 1.5, (240, 240, 150), shade=False, ol=False)
    torso(p, Y)
    p.poly([(43, 45), (57, 45), (50, 51)], GR)
    for y in (53, 56.5):
        p.poly([(50, y - 1.2), (51.1, y), (50, y + 1.2), (48.9, y)], GR, shade=False)
    arms(p, Y, rh=(64.5, 60), cuff=GR)
    third_eye(p, 60, 58.5, 4.2, None, (120, 110, 200), closed=True, ends=((66, 26), (37, 63)))
    heart(p, 37, 63, 1.6, (120, 110, 200))
    hand(p, 64.5, 60)
    head(p)
    face(p, (70, 165, 95), "open")
    bangs(p, H, tip=31, side=44)
    # hat
    p.ell(50, 15.8, 21.5, 4.5, BLACK, shade=False)
    p.ell(50, 11.2, 12.5, 7.2, BLACK, shade=(22, 22, 30))
    p.stroke([(38, 14.2), (62, 14.2)], 2.0, Y, ol=False)
    bow(p, 60, 14, 0.8, Y, tails=False)
    return p.finish()


def draw_orin(size=384):
    p = Pen(size, 0.84, -5.0, 13.5)
    R, DG = (205, 45, 45), (55, 95, 65)
    for s in (-1, 1):
        p.stroke(bez((50 + s * 3, 70), (50 + s * 30, 80), (50 + s * 25, 52), 14), 2.8, BLACK)
    hair_back(p, R)
    legs(p, BLACK, (40, 30, 30), y0=78)
    skirt(p, DG, y1=86, w1=19, y0=56, hem=BLACK, hemw=2.2)
    torso(p, DG)
    p.poly([(43.5, 45), (56.5, 45), (50, 50)], BLACK)
    arms(p, DG, rh=(66, 64), cuff=BLACK)
    head(p)
    face(p, (210, 40, 50), "fang")
    bangs(p, R, tip=31, side=42)
    for s in (-1, 1):
        x = 50 + s * 13
        for i in range(5):
            p.ell(x + s * 0.3 * i, 44 + i * 3.6, 2.6 - i * 0.12, 2.3, R)
        bow(p, x + s * 1.5, 62.5, 0.8, BLACK, tails=False)
        bow(p, x, 43, 0.7, BLACK, tails=False)
        p.poly([(50 + s * 6, 14), (50 + s * 17, 13.5), (50 + s * 16, 1.5)], BLACK)
        p.poly([(50 + s * 9, 13), (50 + s * 15, 12.8), (50 + s * 14.6, 5.5)], (240, 150, 160), shade=False, ol=False)
    # wheelbarrow
    WD = (160, 100, 55)
    p.stroke([(66, 64), (82, 70)], 1.8, (110, 70, 40))
    p.stroke([(84, 78), (86, 88)], 1.6, (110, 70, 40))
    p.poly([(78, 62), (114, 62), (109, 79), (84, 79)], WD)
    for x in (91, 98, 105):
        p.stroke([(x, 63), (x - 1, 78)], 0.5, dk(WD, 0.6), ol=False)
    p.stroke([(77.5, 62), (114.5, 62)], 1.5, dk(WD, 0.8))
    wisp(p, 88, 55, 3.2, (120, 180, 255))
    wisp(p, 102, 52, 2.6, (150, 200, 255))
    p.ell(104, 84, 6.5, 6.5, (70, 60, 60))
    p.ell(104, 84, 4.6, 4.6, (120, 100, 90), shade=False)
    for a in range(0, 180, 45):
        c, s_ = math.cos(math.radians(a)) * 4.6, math.sin(math.radians(a)) * 4.6
        p.stroke([(104 - c, 84 - s_), (104 + c, 84 + s_)], 0.6, (70, 60, 60), ol=False)
    hand(p, 66, 64)
    return p.finish()


def draw_okuu(size=384):
    p = Pen(size, 0.88, 6.0, 7.0)
    H, GR = (45, 45, 62), (60, 145, 75)
    for s in (-1, 1):
        feather_wing(p, (50 + s * 5, 44), s, (40, 38, 50), span=46, rise=26, drop=30, inner=(80, 78, 100))
    p.poly([(40, 46), (60, 46), (79, 88), (21, 88)], WHITE, shade=(215, 215, 225))
    p.poly([(42, 47), (58, 47), (75, 86), (25, 86)], (30, 30, 75), shade=False, ol=False)
    for x, y in ((30, 80), (36, 70), (65, 74), (70, 83), (45, 82), (59, 64), (40, 60), (55, 84)):
        star(p, x, y, 1.3, (255, 255, 220), ol=False)
    hair_back(p, H, 88, w=20)
    legs(p, (70, 60, 60), (70, 60, 60), y0=72)
    p.ell(45.2, 86, 4.2, 4.5, (150, 120, 100))  # rock "elephant foot"
    for a in (30, -30):
        p.poly(rot_ell(54.8, 80, 5, 1.6, a, 20), (255, 200, 60, 200), shade=False, ol=False)
    skirt(p, GR, y1=76, w1=17, hem=lt(GR, 0.25), hemw=1.5)
    torso(p, WHITE)
    arms(p, WHITE, lh=(33, 64), rh=(65, 61))
    p.ell(50, 53, 5.2, 5.2, (200, 25, 40), shade=(150, 15, 30))
    p.ell(50, 53, 2.6, 2.6, (255, 90, 80), shade=False)
    p.ell(48.8, 51.8, 0.9, 0.9, WHITE, shade=False, ol=False)
    # control rod on her right arm (viewer left)
    slab(p, (40, 51), (27, 72), 8.5, (220, 150, 60), shade=(180, 110, 40))
    for t in (0.35, 0.7):
        a = (40 + (27 - 40) * t, 51 + (72 - 51) * t)
        slab(p, (a[0] + 0.8, a[1] - 1.3), (a[0] - 0.8, a[1] + 1.3), 9.2, (240, 200, 90), shade=False)
    p.poly(rot_ell(26.5, 73, 4.6, 2.5, -58, 6), (90, 60, 40), shade=False)
    head(p)
    face(p, (210, 40, 50), "grin")
    bangs(p, H, tip=31, side=45)
    bow(p, 50, 10.5, 2.2, GR, tails=False)
    return p.finish()


def draw_yuugi(size=384):
    p = Pen(size, 0.93, 3.5, 5.0)
    H, RED = (250, 220, 110), (215, 40, 45)
    hair_back(p, H, 84, w=20)
    legs(p, SKIN, (120, 80, 60), y0=76)
    skirt(p, (160, 200, 245, 210), y1=86, w1=18, y0=57)
    p.stroke([(33, 84), (67, 84)], 1.4, RED, ol=False)
    torso(p, WHITE)
    p.stroke([(43, 45.5), (50, 50), (57, 45.5)], 1.4, RED, ol=False)
    p.stroke([(39.6, 59.2), (60.4, 59.2)], 1.5, RED, ol=False)
    arms(p, WHITE, lh=(34, 58), rh=(65, 62), short=0.38)
    for x, y in ((34, 58), (65, 62)):
        p.stroke([(x - 1, y - 3.6), (x + 1.2, y - 3.6)], 2.6, (150, 150, 165))
    for i in range(3):
        p.ell(66 + i * 1.1, 66.5 + i * 2.6, 1.0, 1.3, (160, 160, 175), shade=False)
    p.ell(34, 55.3, 5.8, 1.9, RED)
    p.ell(34, 55.0, 4.4, 1.0, (250, 230, 190), shade=False, ol=False)
    hand(p, 34, 58)
    head(p)
    face(p, (210, 40, 45), "fang")
    bangs(p, H, tip=31, side=45)
    p.poly([(46.6, 15), (53.4, 15), (51.2, 1.5)], RED, shade=(170, 25, 30))
    star(p, 50, 14.5, 3.2, (255, 225, 60))
    return p.finish()


def draw_parsee(size=384):
    p = Pen(size)
    H, GE = (240, 210, 115), (60, 210, 90)
    wisp(p, 16, 60, 3.3, (90, 230, 110))
    wisp(p, 84, 56, 2.8, (90, 230, 110))
    wisp(p, 20, 30, 2.3, (120, 240, 140))
    hair_back(p, H)
    legs(p, (60, 45, 45), (60, 40, 35))
    skirt(p, (70, 95, 165), y1=79, w1=18, hem=(160, 120, 80), hemw=2.5)
    for x in np.linspace(35, 65, 7):
        p.poly([(x, 75.2), (x + 1.3, 76.6), (x, 78), (x - 1.3, 76.6)], (240, 220, 170), shade=False, ol=False)
    torso(p, (80, 55, 50))
    arms(p, (150, 115, 85), cuff=(80, 55, 50))
    # persian scarf
    p.poly([(40, 45), (60, 45), (61, 49), (50, 52), (39, 49)], (60, 130, 150))
    for x in (43, 47, 50, 53, 57):
        p.poly([(x, 46.2), (x + 1, 47.6), (x, 49), (x - 1, 47.6)], (240, 200, 110), shade=False, ol=False)
    strand(p, (55, 50), (58, 55), (57, 62), 3, (60, 130, 150), 2)
    for s in (-1, 1):  # pointed ears
        p.poly([(50 + s * 14.5, 29), (50 + s * 24, 24.5), (50 + s * 15, 36)], SKIN, shade=dk(SKIN, 0.9))
    head(p)
    face(p, GE, "frown")
    p.stroke([(40.5, 27.5), (46, 29)], 0.7, OL, ol=False)
    p.stroke([(59.5, 27.5), (54, 29)], 0.7, OL, ol=False)
    bangs(p, H, tip=31, side=44, n=6)
    return p.finish()


def draw_nazrin(size=384):
    p = Pen(size, 0.95, 2.5, 3.5)
    H, GY = (170, 170, 182), (150, 145, 160)
    p.stroke(bez((55, 70), (80, 84), (84, 64), 14), 1.0, (200, 170, 170))
    p.stroke([(84, 64), (82, 70), (86, 70), (84, 64)], 0.5, (120, 90, 60))
    p.poly([(80, 70), (88, 70), (87, 76), (81, 76)], (190, 150, 90))
    p.ell(84, 69, 2.2, 1.8, (150, 150, 160))
    hair_back(p, H)
    legs(p, (150, 145, 160), (90, 70, 70))
    skirt(p, GY, y1=82, w1=18, hem=dk(GY, 0.85), hemw=1.8)
    torso(p, GY)
    arms(p, GY, lh=(37, 61), rh=(63, 61))
    p.poly([(38, 44.5), (62, 44.5), (65, 53), (50, 55.5), (35, 53)], (95, 90, 105))
    p.ell(50, 53, 2.1, 2.3, (70, 150, 235))
    p.ell(49.4, 52.3, 0.6, 0.6, WHITE, shade=False, ol=False)
    for s in (-1, 1):  # dowsing rods
        x = 50 + s * 13
        p.stroke([(x, 58), (x, 65)], 1.1, (210, 175, 80))
        p.stroke([(x, 59), (x + s * 20, 53)], 1.1, (210, 175, 80))
        hand(p, x, 61)
    for s in (-1, 1):  # mouse ears
        p.ell(50 + s * 16, 14, 7.5, 7, H)
        p.ell(50 + s * 16, 14.5, 4.6, 4.3, (240, 170, 180), shade=False)
    head(p)
    face(p, (200, 40, 50), "smile")
    bangs(p, H, tip=31, side=42)
    return p.finish()


def draw_kogasa(size=384):
    p = Pen(size, 0.86, -1.0, 10.0)
    H, PU = (140, 190, 235), (140, 80, 195)
    hair_back(p, H)
    legs(p, WHITE, (120, 80, 60))
    skirt(p, (90, 130, 205), y1=78, w1=18, hem=lt((90, 130, 205), 0.3), hemw=2)
    torso(p, WHITE)
    p.poly([(41.5, 45), (47, 45), (47.5, 60), (39, 60)], (160, 200, 235))
    p.poly([(53, 45), (58.5, 45), (61, 60), (52.5, 60)], (160, 200, 235))
    arms(p, (160, 200, 235), rh=(66, 60))
    head(p)
    face(p, ((220, 40, 50), (60, 110, 230)), "open")
    bangs(p, H, tip=31, side=42, n=6)
    # umbrella
    p.stroke([(66, 62), (88, 20)], 1.4, (130, 80, 50))
    top = arc(88, 21, 27, 18, 180, 360, 20)
    low = [(115, 21)]
    for i, x in enumerate(np.linspace(115, 61, 11)):
        low.append((x, 21 + (3 if i % 2 else 0)))
    p.poly(top + low, PU, shade=(110, 55, 160))
    p.poly(ribbon([(72, 22), (70, 30), (73, 38), (71, 46)], 4.2, 3.4), (255, 90, 120), shade=(215, 50, 85))
    p.stroke([(72, 24), (72, 40)], 0.5, (190, 40, 70), ol=False)
    p.ell(91, 11.5, 7.5, 5.5, WHITE, shade=False)
    p.ell(91, 12, 3.8, 4.6, (220, 60, 60), shade=False, ol=False)
    p.ell(91, 12, 1.8, 2.4, OL, shade=False, ol=False)
    p.ell(89.8, 10.5, 1.0, 1.0, WHITE, shade=False, ol=False)
    p.stroke([(88, 3), (88, 0)], 1.0, (130, 80, 50))
    hand(p, 66, 60)
    return p.finish()


def draw_byakuren(size=384):
    p = Pen(size, 0.95, 2.5, 3.5)
    p.set_grad((140, 85, 185), (225, 170, 90), 18, 80)
    GOLD = (240, 190, 70)
    p.poly([(40, 46), (60, 46), (74, 86), (26, 86)], BLACK)
    hair_back(p, "grad", 86, w=22, wave=5)
    legs(p, BLACK, BLACK)
    skirt(p, BLACK, y1=84, w1=17.5, hem=WHITE, hemw=1.6)
    p.poly([(46, 58), (54, 58), (57, 82), (43, 82)], WHITE, ol=False)
    torso(p, WHITE)
    p.stroke([(42, 46), (58, 59)], 1.5, BLACK, ol=False)
    p.stroke([(58, 46), (42, 59)], 1.5, BLACK, ol=False)
    arms(p, BLACK, lh=(36, 60), rh=(66, 58), cuff=WHITE)
    # sutra scroll
    p.ell(76, 51, 15, 10, (255, 230, 150, 70), shade=False, ol=False)
    p.poly([(67, 45), (85, 45), (85, 57), (67, 57)], (250, 238, 200), shade=(230, 210, 160))
    sc = [(255, 120, 120), (255, 200, 90), (130, 220, 120), (120, 170, 255), (200, 130, 240)]
    for i, c in enumerate(sc):
        x = 68 + i * 3.2
        p.stroke([(x + 1.6, 47), (x + 1.6, 55)], 0.8, c, ol=False)
    for x in (67, 85):
        p.stroke([(x, 45.5), (x, 56.5)], 3.0, GOLD)
    hand(p, 66, 58)
    head(p)
    face(p, (235, 180, 60), "smile")
    bangs(p, "grad", tip=31, side=44)
    for s in (-1, 1):
        p.stroke(bez((50 + s * 17, 34), (50 + s * 21, 42), (50 + s * 18, 50), 10), 3.0, "grad")
    return p.finish()


def draw_nue(size=384):
    p = Pen(size, 0.94, 3.0, 4.0)
    H = (40, 40, 58)
    RED, BLU = (225, 45, 55), (60, 110, 230)
    for i, (tx, ty) in enumerate(((12, 26), (9, 46), (14, 66))):
        pts = bez((44, 50), ((44 + tx) / 2, (50 + ty) / 2 - 8), (tx, ty), 12)
        p.stroke(pts, 1.8, RED)
        a = math.atan2(ty - pts[-3][1], tx - pts[-3][0])
        c, s = math.cos(a), math.sin(a)
        p.poly([(tx + c * 4, ty + s * 4), (tx - s * 2.6, ty + c * 2.6), (tx + s * 2.6, ty - c * 2.6)], RED)
    for i, (tx, ty) in enumerate(((88, 24), (92, 44), (87, 66))):
        pts = []
        for t in np.linspace(0, 1, 24):
            x, y = 56 + (tx - 56) * t, 50 + (ty - 50) * t
            pts.append((x + math.sin(t * 12) * 2.2 * t, y + math.cos(t * 12) * 2.2 * t))
        p.stroke(pts, 1.8, BLU)
        p.ell(tx, ty, 1.8, 1.8, BLU, shade=False)
    hair_back(p, H)
    legs(p, BLACK, (40, 30, 40), y0=70)
    skirt(p, BLACK, y1=74, w1=18, hem=(200, 40, 50), hemw=1.2)
    torso(p, BLACK)
    bow(p, 50, 47.5, 1.3, RED)
    # trident
    p.stroke([(33, 76), (33, 28)], 1.2, (170, 175, 200))
    p.stroke([(29, 34), (29, 30), (33, 34), (37, 30), (37, 34)], 1.0, (170, 175, 200))
    p.stroke([(33, 34), (33, 25)], 1.0, (170, 175, 200))
    arms(p, BLACK, lh=(33, 60), rh=(65, 62), cuff=(200, 40, 50))
    head(p)
    face(p, (215, 40, 50), "fang")
    bangs(p, H, tip=31.5, side=42)
    return p.finish()


def noh_mask(p, x, y, r, kind):
    if kind == "okame":
        p.ell(x, y, r, r * 1.1, WHITE, shade=(225, 220, 220))
        for s in (-1, 1):
            p.stroke(bez((x + s * r * 0.55, y - r * 0.15), (x + s * r * 0.35, y - r * 0.45), (x + s * r * 0.15, y - r * 0.15), 6), 0.6, OL, ol=False)
            p.ell(x + s * r * 0.5, y + r * 0.3, r * 0.22, r * 0.12, (255, 150, 160), shade=False, ol=False)
        p.ell(x, y + r * 0.55, r * 0.16, r * 0.12, (220, 40, 50), shade=False, ol=False)
    elif kind == "fox":
        p.poly([(x - r * 0.95, y - r * 0.2), (x - r * 0.75, y - r * 1.2), (x - r * 0.2, y - r * 0.6),
                (x + r * 0.2, y - r * 0.6), (x + r * 0.75, y - r * 1.2), (x + r * 0.95, y - r * 0.2),
                (x, y + r * 1.0)], WHITE, shade=(225, 220, 220))
        for s in (-1, 1):
            p.stroke([(x + s * r * 0.55, y - r * 0.05), (x + s * r * 0.2, y + r * 0.1)], 0.7, (220, 40, 50), ol=False)
            p.stroke([(x + s * r * 0.7, y - r * 0.85), (x + s * r * 0.55, y - r * 0.5)], 0.6, (220, 40, 50), ol=False)
    elif kind == "hannya":
        for s in (-1, 1):
            p.poly([(x + s * r * 0.5, y - r * 0.7), (x + s * r * 1.1, y - r * 1.5), (x + s * r * 0.85, y - r * 0.5)], (250, 240, 200))
        p.ell(x, y, r, r * 1.1, (215, 50, 55))
        for s in (-1, 1):
            p.ell(x + s * r * 0.4, y - r * 0.15, r * 0.24, r * 0.16, (255, 220, 80), shade=False)
        p.poly([(x - r * 0.5, y + r * 0.45), (x + r * 0.5, y + r * 0.45), (x, y + r * 0.8)], WHITE, shade=False)
    elif kind == "hyottoko":
        p.ell(x, y, r, r * 1.1, (245, 180, 110))
        p.ell(x - r * 0.35, y - r * 0.2, r * 0.2, r * 0.2, WHITE, shade=False)
        p.ell(x + r * 0.35, y - r * 0.2, r * 0.12, r * 0.12, WHITE, shade=False)
        p.ell(x + r * 0.25, y + r * 0.5, r * 0.25, r * 0.22, (200, 90, 70), shade=False)
    else:  # old man (okina)
        p.ell(x, y, r, r * 1.1, (230, 200, 150))
        for s in (-1, 1):
            p.stroke(bez((x + s * r * 0.6, y - r * 0.1), (x + s * r * 0.35, y + r * 0.1), (x + s * r * 0.1, y - r * 0.1), 6), 0.6, OL, ol=False)
        p.poly([(x - r * 0.4, y + r * 0.55), (x + r * 0.4, y + r * 0.55), (x, y + r * 1.4)], WHITE)


def draw_kokoro(size=384):
    p = Pen(size, 0.94, 3.0, 4.0)
    H = (240, 160, 195)
    hair_back(p, H, 86, w=21, wave=6)
    for (x, y, k) in ((11, 27, "okame"), (89, 30, "hannya"), (11, 62, "hyottoko"), (89, 64, "okina")):
        noh_mask(p, x, y, 6, k)
    legs(p, (250, 235, 240), (120, 60, 90))
    skirt(p, (225, 110, 170), y1=77, w1=20, w0=12, hem=(250, 170, 80), hemw=2, bulge=3)
    for x in (39, 50, 61):
        p.ell(x, 67, 2, 2, (250, 200, 120), shade=False)
    torso(p, (100, 170, 165))
    for x in (44, 50, 56):
        p.stroke([(x, 46), (x + (x - 50) * 0.3, 59)], 0.6, (240, 150, 80), ol=False)
    for y in (49, 53.5, 57.5):
        p.stroke([(40.5, y), (59.5, y)], 0.6, (240, 150, 80), ol=False)
    bow(p, 50, 47.5, 1.1, (230, 80, 110))
    arms(p, (100, 170, 165), lh=(35, 60), cuff=(250, 170, 80))
    # fan
    p.poly([(35, 60)] + arc(35, 60, 8, 8, 200, 290, 8), (240, 90, 120), shade=False)
    hand(p, 35, 60)
    head(p)
    face(p, (230, 110, 165), "flat", lid=0.3, blush=False)
    bangs(p, H, tip=32, side=46, n=6)
    noh_mask(p, 64, 15, 5, "fox")
    return p.finish()


def draw_seija(size=384):
    p = Pen(size, 0.95, 2.5, 3.0)
    H, RED, BLU = (45, 42, 55), (225, 45, 55), (60, 100, 220)
    hair_back(p, H)
    legs(p, WHITE, (170, 40, 50))
    skirt(p, WHITE, y1=77, w1=18, hem=BLU, hemw=2)
    for i, x in enumerate(np.linspace(37, 63, 5)):
        c = RED if i % 2 else BLU
        d = -1 if i % 2 else 1
        p.stroke([(x, 66 - 3 * d), (x, 66 + 3 * d)], 0.8, c, ol=False)
        p.poly([(x, 66 - 4.8 * d), (x - 1.5, 66 - 2.4 * d), (x + 1.5, 66 - 2.4 * d)], c, shade=False, ol=False)
    torso(p, WHITE)
    p.stroke([(40, 59), (60, 59)], 1.2, RED, ol=False)
    bow(p, 50, 47.5, 1.3, RED)
    arms(p, WHITE, lh=(35, 58), rh=(65, 63), short=0.4, cuff=None)
    head(p)
    face(p, (215, 40, 50), "tongue")
    bangs(p, H, tip=31, side=42)
    strand(p, (58, 12), (63, 22), (59, 31), 3.2, RED)
    strand(p, (41, 12), (37, 22), (40, 30.5), 2.8, WHITE)
    for s in (-1, 1):
        p.poly([(50 + s * 7, 12), (50 + s * 11, 11), (50 + s * 10.5, 5)], (250, 245, 240))
    return p.finish(rotate=True)


def draw_hecatia(size=384):
    p = Pen(size, 0.9, 5.0, 6.5)
    H = (220, 50, 60)
    p.stroke(bez((50, 45), (30, 50), (15, 42), 12), 0.7, (200, 190, 120), ol=False)
    p.stroke(bez((50, 45), (70, 50), (85, 42), 12), 0.7, (200, 190, 120), ol=False)
    planet(p, 12, 40, 7, "earth")
    planet(p, 88, 40, 6.5, "moon")
    hair_back(p, H, 50, w=20, wave=4)
    legs(p, SKIN, (40, 30, 40), y0=74)
    cols = [(215, 45, 55), (60, 170, 80), (60, 100, 220)]
    for i, c in enumerate(cols):
        x0, x1 = -1 + i * 2 / 3, -1 + (i + 1) * 2 / 3
        p.poly([(50 + 11 * x0, 58), (50 + 11 * x1, 58), (50 + 19 * x1, 77), (50 + 19 * x0, 77)], c)
    torso(p, (32, 30, 38), y1=61, w1=12)
    arms(p, (32, 30, 38), lh=(35, 62), rh=(65, 62), short=0.35)
    pixtext(p, "WELCOME", 50, 50, 0.55, WHITE)
    pixtext(p, "HELL", 50, 54.5, 0.75, (255, 90, 90))
    p.stroke([(45, 45.5), (55, 45.5)], 1.2, (120, 110, 90))
    head(p)
    face(p, (210, 40, 55), "smile")
    bangs(p, H, tip=31, side=45)
    p.stroke([(40, 14), (50, 8), (60, 14)], 0.8, (200, 190, 120), ol=False)
    planet(p, 50, 5.5, 5.5, "other")
    return p.finish()


def draw_clownpiece(size=384):
    p = Pen(size, 0.93, 3.5, 5.0)
    H, BL, RD = (250, 230, 140), (40, 60, 150), (220, 45, 60)
    for s in (-1, 1):
        p.poly(rot_ell(50 + s * 16, 47, 13, 6, s * -30), (215, 230, 255, 150), shade=False)
        p.poly(rot_ell(50 + s * 14, 58, 9, 4, s * 25), (215, 230, 255, 150), shade=False)
    hair_back(p, H, 86, w=21, wave=6)
    legs(p, WHITE, BL, y0=72)
    for x in (45.2, 54.8):
        for y in (75, 79, 83):
            p.stroke([(x - 2.2, y), (x + 2.2, y)], 1.4, RD, ol=False)
    sk = [(50 + 11 - 22 * t, 57) for t in (0, 1)]
    skirt(p, RD, y1=76, w1=18)
    for i in range(1, 8, 2):
        x0, x1 = -1 + i / 4, -1 + (i + 1) / 4
        p.poly([(50 + 11 * x0, 57.5), (50 + 11 * x1, 57.5), (50 + 18 * x1, 76.5), (50 + 18 * x0, 76.5)], WHITE, ol=False)
    torso(p, BL)
    for x, y in ((45, 51), (55, 52), (50, 56), (44, 57.5), (56, 57.5)):
        star(p, x, y, 1.3, WHITE, ol=False)
    p.poly(arc(50, 45.5, 10, 3.2, 0, 180, 12) + arc(50, 45.5, 10, 1.5, 180, 360, 8), WHITE)
    arms(p, BL, rh=(66, 60), cuff=WHITE)
    # torch
    slab(p, (66, 66), (76, 46), 2.2, (200, 160, 70))
    p.poly([(72, 47.5), (80.5, 47.5), (79, 44.5), (73.5, 44.5)], (170, 130, 60))
    p.ell(76.5, 36, 9, 9, (230, 90, 240, 60), shade=False, ol=False)
    p.poly([(72.5, 45)] + bez((72.5, 45), (69, 34), (78, 24), 10) + bez((78, 24), (86, 35), (80.5, 45), 10),
           (215, 60, 220), shade=(170, 30, 190))
    p.poly(bez((74.5, 44), (73.5, 37), (77.5, 31), 8) + bez((77.5, 31), (81, 37), (79, 44), 8), (255, 170, 250), shade=False, ol=False)
    hand(p, 66, 60)
    head(p)
    face(p, (225, 50, 95), "grin")
    bangs(p, H, tip=31, side=45)
    # jester hat
    p.poly(bez((34, 21), (18, 18), (20, 6), 12) + [(38, 13)], BL)
    p.poly(bez((66, 21), (82, 18), (80, 6), 12) + [(62, 13)], RD)
    for i, (x, y) in enumerate(((67.5, 16), (72, 16.5), (76, 14), (78.5, 10))):
        p.stroke([(x - 0.8, y - 1.5), (x + 0.8, y + 1.5)], 1.3, WHITE, ol=False)
    for x, y in ((27, 16), (23, 12), (31, 15.5)):
        star(p, x, y, 1.1, WHITE, ol=False)
    p.poly(arc(50, 22, 16, 14, 180, 360, 16), BL)
    for i in range(0, 8, 2):
        a0, a1 = 180 + i * 22.5, 180 + (i + 1) * 22.5
        p.poly([(50, 22)] + arc(50, 22, 16, 14, a0, a1, 5), RD, ol=False, shade=False)
    p.stroke(arc(50, 22, 16.2, 3, 180, 360, 14), 2.0, WHITE)
    for x, y in ((20, 6), (80, 6), (50, 7.5)):
        p.ell(x, y, 2.6, 2.6, (255, 215, 70))
    return p.finish()


CHARACTERS = {
    "eiki": {"name": "Shikieiki Yamaxanadu", "color": (60, 90, 200),
             "spell": 'Judgment "Last Judgment of the Loss Function"', "draw": draw_eiki},
    "tenshi": {"name": "Tenshi Hinanawi", "color": (80, 140, 235),
               "spell": 'Sword "Scarlet Weather Gradient Descent"', "draw": draw_tenshi},
    "iku": {"name": "Iku Nagae", "color": (140, 95, 200),
            "spell": 'Dragonfish "Thundercloud Attention Heads"', "draw": draw_iku},
    "satori": {"name": "Satori Komeiji", "color": (240, 130, 190),
               "spell": 'Recollection "Reading the System Prompt"', "draw": draw_satori},
    "koishi": {"name": "Koishi Komeiji", "color": (120, 200, 120),
               "spell": 'Id "Unconscious Chain of Thought"', "draw": draw_koishi},
    "orin": {"name": "Rin Kaenbyou", "color": (205, 45, 45),
             "spell": 'Cursed Sprite "Zombie Process Fairy"', "draw": draw_orin},
    "okuu": {"name": "Utsuho Reiuji", "color": (250, 140, 30),
             "spell": 'Nuclear "Giga Flare of a Trillion FLOPs"', "draw": draw_okuu},
    "yuugi": {"name": "Yuugi Hoshiguma", "color": (220, 50, 60),
              "spell": 'Big Four "Knockout in Three Epochs"', "draw": draw_yuugi},
    "parsee": {"name": "Parsee Mizuhashi", "color": (70, 205, 90),
               "spell": 'Jealousy "Green-Eyed Benchmark Envy"', "draw": draw_parsee},
    "nazrin": {"name": "Nazrin", "color": (150, 150, 165),
               "spell": 'Search "Retrieval-Augmented Dowsing"', "draw": draw_nazrin},
    "kogasa": {"name": "Kogasa Tatara", "color": (140, 80, 200),
               "spell": 'Umbrella "Surprise! Out-of-Distribution"', "draw": draw_kogasa},
    "byakuren": {"name": "Byakuren Hijiri", "color": (175, 115, 200),
                 "spell": 'Magic "Constitutional Nirvana Scroll"', "draw": draw_byakuren},
    "nue": {"name": "Nue Houjuu", "color": (200, 45, 70),
            "spell": 'Nue Sign "Undefined Token of Unknown Origin"', "draw": draw_nue},
    "kokoro": {"name": "Kokoro Hata", "color": (240, 140, 175),
               "spell": 'Emotion "Sixty-Six Sentiment Classifiers"', "draw": draw_kokoro},
    "seija": {"name": "Seija Kijin", "color": (215, 45, 60),
              "spell": 'Reverse "RLHF Turned Upside Down"', "draw": draw_seija},
    "hecatia": {"name": "Hecatia Lapislazuli", "color": (215, 45, 55),
                "spell": 'Hell "Three-Body Model Ensemble"', "draw": draw_hecatia},
    "clownpiece": {"name": "Clownpiece", "color": (220, 60, 90),
                   "spell": 'Hell Sign "Torch of Temperature 2.0"', "draw": draw_clownpiece},
}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "build", "sheet_c.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cell, cols = 384, 6
    keys = list(CHARACTERS)
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGBA", (cell * cols, cell * rows), (225, 225, 232, 255))
    for i, k in enumerate(keys):
        c = CHARACTERS[k]
        assert len(c["spell"]) < 48, (k, len(c["spell"]))
        im = c["draw"](cell)
        x, y = (i % cols) * cell, (i // cols) * cell
        ImageDraw.Draw(sheet).rectangle([x, y, x + cell - 1, y + cell - 1], outline=(180, 180, 190))
        ImageDraw.Draw(sheet).rectangle([x + 4, y + 4, x + 28, y + 28], fill=rgba(c["color"]))
        sheet.alpha_composite(im, (x, y))
    sheet.convert("RGB").save(out)
    print("wrote", os.path.normpath(out))
