"""Cameo sprites, batch B: Eientei, Youkai Mountain, Moriya and friends.

Procedural chibi sprites (numpy + Pillow only). See SPEC.md.
Coordinates are in a 0..100 unit square; each canvas is drawn at 4x and
downsampled with LANCZOS.
"""
import math
import os

from PIL import Image, ImageChops, ImageDraw, ImageFilter

# ---------------------------------------------------------------- helpers
OL = (40, 24, 38)
SKIN = (255, 230, 214)
WHITE = (250, 250, 252)
BLUSH = (250, 168, 170)


def dk(c, f=0.78):
    return tuple(int(v * f) for v in c[:3])


def lt(c, f=0.35):
    return tuple(int(v + (255 - v) * f) for v in c[:3])


def rgba(c):
    return tuple(c[:3]) + (255,)


def E(cx, cy, rx, ry, rot=0.0, n=40):
    """Rotated ellipse as polygon prim (rot in degrees)."""
    r = math.radians(rot)
    cr, sr = math.cos(r), math.sin(r)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    return ('p', pts)


def arc(cx, cy, rx, ry, a0, a1, n=24):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def strand(x0, y0, x1, y1, w, bend=0.0, w1=0.0, n=14):
    """Tapered curved lock of hair (quadratic bezier), root width w -> tip w1."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    L = math.hypot(x1 - x0, y1 - y0) or 1
    px, py = -(y1 - y0) / L, (x1 - x0) / L
    cx, cy = mx + px * bend, my + py * bend
    left, right = [], []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1
        tx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
        ty = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
        tl = math.hypot(tx, ty) or 1
        nx, ny = -ty / tl, tx / tl
        ww = (w * (1 - t) + w1 * t) / 2
        left.append((x + nx * ww, y + ny * ww))
        right.append((x - nx * ww, y - ny * ww))
    return ('p', left + right[::-1])


def star(cx, cy, r, n=5, inner=0.45, rot=-90):
    pts = []
    for i in range(2 * n):
        rr = r if i % 2 == 0 else r * inner
        a = math.radians(rot + 180 * i / n)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return ('p', pts)


class C:
    """4x canvas with outlined, shaded flat parts."""

    def __init__(self, size=384, sc=1.0, oy=0.0):
        self.size, self.S = size, size * 4
        self.k = self.S / 100
        self.sc, self.oy = sc, oy
        self.img = Image.new('RGBA', (self.S, self.S), (0, 0, 0, 0))

    def P(self, x, y):
        return ((50 + (x - 50) * self.sc) * self.k,
                (50 + self.oy + (y - 50) * self.sc) * self.k)

    def _mask(self, prims):
        m = Image.new('L', (self.S, self.S), 0)
        d = ImageDraw.Draw(m)
        u = self.k * self.sc
        for pr in prims:
            val = 255
            if pr[0] == '-':
                val, pr = 0, pr[1]
            t = pr[0]
            if t == 'e':
                _, cx, cy, rx, ry = pr
                x, y = self.P(cx, cy)
                d.ellipse([x - rx * u, y - ry * u, x + rx * u, y + ry * u], fill=val)
            elif t == 'p':
                d.polygon([self.P(x, y) for x, y in pr[1]], fill=val)
            elif t == 'l':
                pts = [self.P(x, y) for x, y in pr[1]]
                w = pr[2] * u
                d.line(pts, fill=val, width=max(1, int(w)), joint='curve')
                for x, y in (pts[0], pts[-1]):
                    d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=val)
            elif t == 'r':
                _, x0, y0, x1, y1, rad = pr
                a, b = self.P(x0, y0), self.P(x1, y1)
                d.rounded_rectangle([a[0], a[1], b[0], b[1]], radius=rad * u, fill=val)
        return m

    def part(self, prims, fill, shade=True, ol=True, olw=0.55, olc=OL, sd=1.5, clip=None):
        m = self._mask(prims)
        if clip is not None:
            m = ImageChops.multiply(m, self._mask(clip))
        bb = m.getbbox()
        if not bb:
            return
        pad = int(4 * self.k)
        box = (max(0, bb[0] - pad), max(0, bb[1] - pad),
               min(self.S, bb[2] + pad), min(self.S, bb[3] + pad))
        sub = m.crop(box)
        if ol:
            sig = olw * self.k * self.sc / 1.86
            dil = sub.filter(ImageFilter.GaussianBlur(sig)).point(lambda v: 255 if v > 8 else 0)
            self.img.paste(rgba(olc), box, dil)
        self.img.paste(rgba(fill), box, sub)
        if shade:
            sc = dk(fill) if shade is True else shade
            o = int(-sd * self.k * self.sc)
            sh = ImageChops.subtract(sub, ImageChops.offset(sub, o, o))
            self.img.paste(rgba(sc), box, sh)

    def det(self, prims, fill):
        self.part(prims, fill, shade=False, ol=False)

    def glow(self, prims, col, blur=2.0, alpha=0.8):
        m = self._mask(prims).filter(ImageFilter.GaussianBlur(blur * self.k))
        m = m.point(lambda v: int(v * alpha))
        layer = Image.new('RGBA', self.img.size, rgba(col))
        layer.putalpha(m)
        self.img = Image.alpha_composite(self.img, layer)

    def done(self):
        return self.img.resize((self.size, self.size), Image.LANCZOS)


# ---- body parts ------------------------------------------------------------
def face(c, eye, skin=SKIN, mouth='smile', closed=False, fang=False):
    c.part([('e', 50, 33.5, 19.2, 17.2)], skin, shade=False)
    c.det([('e', 39.5, 42.5, 3.2, 1.6), ('e', 60.5, 42.5, 3.2, 1.6)], BLUSH)
    eyes(c, eye, closed=closed)
    mc = (150, 50, 60)
    if mouth == 'smile':
        c.det([('l', [(48.2, 45.2), (50, 46.4), (51.8, 45.2)], 0.8)], mc)
    elif mouth == 'open':
        c.part([('p', arc(50, 45, 2.2, 2.2, 0, 180))], (220, 90, 100), shade=False, olw=0.35)
    elif mouth == 'cat':
        c.det([('l', [(47.3, 45), (48.7, 46.2), (50, 45.2), (51.3, 46.2), (52.7, 45)], 0.75)], mc)
    elif mouth == 'flat':
        c.det([('l', [(48.5, 45.6), (51.5, 45.6)], 0.8)], mc)
    if fang:
        c.det([('p', [(51, 45.6), (52.4, 45.6), (51.7, 47.3)])], WHITE)


def eyes(c, col, y=37.8, dx=7.8, rx=3.9, ry=5.3, closed=False):
    for s in (-1, 1):
        x = 50 + s * dx
        if closed:
            c.det([('l', [(x - rx, y), (x, y + 1.6), (x + rx, y)], 1.1)], OL)
            continue
        c.det([('e', x, y, rx + 0.5, ry + 0.5)], OL)
        c.det([('e', x, y + 0.3, rx, ry - 0.1)], dk(col, 0.72))
        c.det([('e', x, y + 1.6, rx * 0.85, ry * 0.6)], col)
        c.det([('e', x, y + 0.3, rx * 0.42, ry * 0.5)], dk(col, 0.35))
        c.det([('e', x - rx * 0.33, y - ry * 0.38, rx * 0.4, ry * 0.3)], WHITE)
        c.det([('e', x + rx * 0.35, y + ry * 0.45, rx * 0.2, ry * 0.15)], WHITE)
        c.det([('l', arc(x, y + 0.6, rx + 0.7, ry + 0.9, 195, 345, n=10), 1.5)], OL)
        ox = x + s * (rx + 0.6)
        c.det([('l', [(ox - s * 0.8, y - ry + 1.6), (ox + s * 0.9, y - ry + 1.0)], 1.0)], OL)


def hair_back(c, col, yend=None, wb=26.0, top=31, n=6):
    prims = [('e', 50, top, 23.8, 21)]
    if yend:
        prims.append(('p', [(26.5, top + 1), (73.5, top + 1), (50 + wb, yend), (50 - wb, yend)]))
        for i in range(n):
            x = 50 - wb + (i + 0.5) * 2 * wb / n
            prims.append(('e', x, yend, wb / n + 0.4, 2.4))
    c.part(prims, col)


DEFAULT_TIPS = [(71, 41), (64.5, 33.5), (57.5, 32.3), (50, 32), (42.5, 32.3), (35.5, 33.5), (29, 41)]


def bangs(c, col, tips=None, hime=False, side=50, side_w=5.5, extra=(), cy=31, notch=25.5):
    pts = arc(50, cy, 23.8, 19.5, 180, 360)
    if hime:
        pts += [(73.8, 34), (70, 35), (68.5, 32.3), (31.5, 32.3), (30, 35), (26.2, 34)]
    else:
        tips = tips or DEFAULT_TIPS
        fr = [(73.8, cy + 2)]
        for i, (tx, ty) in enumerate(tips):
            fr.append((tx, ty))
            if i < len(tips) - 1:
                nx = (tx + tips[i + 1][0]) / 2
                fr.append((nx, notch + abs(nx - 50) * 0.08))
        fr.append((26.2, cy + 2))
        pts += fr
    prims = [('p', pts)]
    if side:
        prims += [strand(28.5, 29, 30, side, side_w, bend=-1.5, w1=1.5),
                  strand(71.5, 29, 70, side, side_w, bend=1.5, w1=1.5)]
    prims += list(extra)
    c.part(prims, col)
    # hair shine
    c.det([('p', arc(50, 20.5, 13, 3.4, 200, 340) + arc(50, 21.8, 11, 2.2, 340, 200))], lt(col, 0.35))


ARMS = [((41.8, 51), (36, 64.5)), ((58.2, 51), (64, 64.5))]


def body(c, top, skirt=None, skin=SKIN, sleeve=None, legs=None, shoes=(110, 60, 50),
         skirt_y=79, skirt_w=17, pants=None, arms=None, sleeve_w=5.2, hands=True,
         torso_y=64, cuff=None, shorts_y=84):
    sleeve = sleeve or top
    legs = legs or skin
    c.part([('l', [(45.2, 74), (45.2, 89)], 5.2), ('l', [(54.8, 74), (54.8, 89)], 5.2)], legs)
    c.part([('e', 44.5, 91.2, 4.3, 2.7)], shoes)
    c.part([('e', 55.5, 91.2, 4.3, 2.7)], shoes)
    if pants:
        c.part([('p', [(40.5, 60), (59.5, 60), (60.8, shorts_y), (51.2, shorts_y), (50, 72),
                       (48.8, shorts_y), (39.2, shorts_y)])], pants)
    if skirt:
        c.part([('p', [(41, 58), (59, 58), (50 + skirt_w, skirt_y - 2.5)] +
                arc(50, skirt_y - 2.5, skirt_w, 2.5, 0, 180, n=16) + [(50 - skirt_w, skirt_y - 2.5)])], skirt)
    c.part([('p', [(41.5, 48), (58.5, 48), (60.8, torso_y), (39.2, torso_y)])], top)
    arms = arms or ARMS
    for sh, hd in arms:
        c.part([('l', [sh, hd], sleeve_w)], sleeve)
        if cuff:
            c.part([('e', hd[0], hd[1], sleeve_w * 0.58, sleeve_w * 0.58)], cuff, shade=False)
    if hands:
        draw_hands(c, arms, skin)
    return arms


def draw_hands(c, arms, skin=SKIN):
    for sh, hd in arms:
        dx, dy = hd[0] - sh[0], hd[1] - sh[1]
        L = math.hypot(dx, dy) or 1
        c.part([('e', hd[0] + dx / L * 1.8, hd[1] + dy / L * 1.8, 2.6, 2.6)], skin)


def bow(c, x, y, s, col, tails=True, rot=0):
    r = math.radians(rot)

    def R(px, py):
        return (x + (px * math.cos(r) - py * math.sin(r)) * s, y + (px * math.sin(r) + py * math.cos(r)) * s)
    pr = [('p', [R(0, 0), R(-1.1, -0.75), R(-1.25, 0.6)]), ('p', [R(0, 0), R(1.1, -0.75), R(1.25, 0.6)]),
          E(*R(-0.75, -0.05), 0.55 * s, 0.6 * s, rot), E(*R(0.75, -0.05), 0.55 * s, 0.6 * s, rot)]
    if tails:
        pr += [('p', [R(-0.15, 0), R(-0.75, 1.5), R(-0.35, 1.6), R(0.1, 0.2)]),
               ('p', [R(0.15, 0), R(0.75, 1.5), R(0.35, 1.6), R(-0.1, 0.2)])]
    c.part(pr, col, sd=0.5)
    c.part([E(x, y, 0.35 * s, 0.4 * s, rot)], dk(col, 0.85), shade=False, olw=0.35)


def feather_wing(c, side, col, x=57, y=54, length=27, n=6, a0=-75, a1=15, width=5.0, inner=None):
    """Fan of feathers; side=+1 right, -1 left."""
    prims = []
    for i in range(n):
        a = a0 + (a1 - a0) * i / (n - 1)
        L = length * (1 - 0.25 * i / (n - 1))
        ar = math.radians(a)
        ex, ey = math.cos(ar) * L, math.sin(ar) * L
        prims.append(strand(x * 1 if side > 0 else 100 - x, y,
                            (x + ex) if side > 0 else 100 - x - ex, y + ey, width, bend=-2 * side, w1=1.4))
    prims.append(('e', x if side > 0 else 100 - x, y, 6, 6))
    c.part(prims, col)
    if inner:
        c.det([strand(x if side > 0 else 100 - x, y,
                      (x + math.cos(math.radians(a0 + 20)) * length * 0.5) * (1 if side > 0 else 1) if side > 0 else
                      100 - x - math.cos(math.radians(a0 + 20)) * length * 0.5,
                      y + math.sin(math.radians(a0 + 20)) * length * 0.5, 4, w1=0.5)], inner)


def wings(c, col, **kw):
    for s in (-1, 1):
        feather_wing(c, s, col, **kw)


def tokin(c, x, y, s, col):
    c.part([('p', [(x - 3 * s, y + 1.5 * s), (x - 2.4 * s, y - 1.8 * s), (x, y - 3 * s), (x + 2.4 * s, y - 1.8 * s),
                   (x + 3 * s, y + 1.5 * s), (x, y + 2.4 * s)])], col, sd=0.6)
    c.det([('l', [(x, y - 2.8 * s), (x, y + 2.2 * s)], 0.35 * s)], dk(col, 0.7))


def ofuda(c, x, y, w, h, rot=0, col=WHITE, ink=(200, 40, 50)):
    r = math.radians(rot)
    cr, sr = math.cos(r), math.sin(r)
    pts = [(x + px * cr - py * sr, y + px * sr + py * cr) for px, py in ((-w, -h), (w, -h), (w, h), (-w, h))]
    c.part([('p', pts)], col, shade=False, olw=0.35)
    c.det([('l', [(x, y - h * 0.6), (x, y + h * 0.6)], 0.45)], ink)


def gohei(c, x0, y0, x1, y1, paper=WHITE):
    c.part([('l', [(x0, y0), (x1, y1)], 1.6)], (170, 120, 70))
    for s in (-1, 1):
        zx = x1 + s * 1.2
        pts = [(zx, y1)]
        for i in range(4):
            pts += [(zx + s * 2.6, y1 + 2 + i * 3.2), (zx + s * 0.4, y1 + 3.2 + i * 3.2)]
        pts += [(zx + s * 1.6, y1 + 13.5), (zx - s * 0.2, y1 + 13)]
        c.part([('p', pts)], paper, shade=(210, 210, 225), sd=0.6)


def ears_animal(c, col, inner, pts_l, pts_r):
    c.part([('p', pts_l), ('p', pts_r)], col)
    for pts in (pts_l, pts_r):
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        c.det([('p', [(cx + (px - cx) * 0.55, cy + (py - cy) * 0.55 + 0.6) for px, py in pts])], inner)


# ---------------------------------------------------------------- characters
def draw_reisen(size=384):
    c = C(size, sc=0.86, oy=6)
    hair = (205, 175, 238)
    hair_back(c, hair, yend=90, wb=25, n=7)
    body(c, WHITE, skirt=(245, 170, 205), sleeve=(40, 36, 58), skirt_y=76, legs=(250, 250, 250),
         shoes=(120, 70, 60), cuff=WHITE)
    # blazer
    bz = (40, 36, 58)
    c.part([('p', [(41.5, 48), (47.5, 48), (49, 62), (39, 64)]), ('p', [(52.5, 48), (58.5, 48), (61, 64), (51, 62)])], bz)
    c.part([('p', [(48.6, 49), (51.4, 49), (51.8, 57), (50, 59), (48.2, 57)])], (215, 40, 60), sd=0.5)
    c.det([('e', 44, 60, 0.7, 0.7)], (230, 200, 90))
    face(c, (230, 40, 70))
    bangs(c, hair, side=60, side_w=6)
    # bunny ears: one straight, one bent
    ear = (252, 244, 250)
    c.part([strand(42, 17, 36, -8, 7, bend=1.5, w1=3.5),
            strand(58, 17, 63, 2, 7, bend=-1, w1=6), strand(63, 2, 72, -4, 6, bend=-2, w1=3),
            ('e', 36, -7.5, 1.9, 1.9), ('e', 71.8, -3.9, 1.6, 1.6)], ear, sd=0.8)
    c.det([strand(41.5, 13, 36.5, -5, 3, bend=1.2, w1=1.2), strand(59, 13, 62.5, 3, 3, bend=-0.6, w1=2.5)],
          (245, 205, 225))
    c.det([('l', [(62, 5), (65, 1.5)], 0.6)], (190, 170, 200))
    c.part([('r', 38.5, 13.5, 43.5, 16.5, 0.8)], (230, 225, 235), sd=0.5)
    return c.done()


def draw_tewi(size=384):
    c = C(size, sc=0.9, oy=4)
    hair = (48, 40, 50)
    hair_back(c, hair, yend=46, wb=25, n=6)
    pink = (250, 175, 200)
    body(c, pink, skirt=pink, skirt_y=78, skirt_w=16, sleeve=pink, shoes=(250, 180, 190), cuff=(230, 90, 110))
    c.det([('l', [(35, 77), (65, 77)], 1.0)], (230, 90, 110))
    # carrot necklace
    c.det([('l', [(44, 49), (50, 54), (56, 49)], 0.5)], (120, 80, 60))
    c.part([('p', [(48.2, 54), (51.8, 54), (50, 61)])], (250, 140, 40), sd=0.5, olw=0.4)
    c.part([strand(50, 54.5, 48, 51.5, 1.4), strand(50, 54.5, 52, 51.5, 1.4)], (90, 170, 80), shade=False, olw=0.35)
    face(c, (230, 50, 70), mouth='cat')
    tips = [(71, 38), (65, 34), (58, 33), (50, 32.5), (42, 33), (35, 34), (29, 38)]
    bangs(c, hair, tips=tips, side=46, side_w=6,
          extra=[('e', 27.5, 42, 3.5, 4), ('e', 72.5, 42, 3.5, 4), ('e', 26.5, 34, 3, 3.5), ('e', 73.5, 34, 3, 3.5)])
    # floppy ears
    ear = (252, 250, 252)
    c.part([strand(40, 16, 22, 4, 7, bend=4, w1=5), strand(22, 4, 16, 14, 5, bend=2, w1=2.5),
            strand(60, 16, 76, 6, 7, bend=-4, w1=5), strand(76, 6, 83, 15, 5, bend=-2, w1=2.5),
            ('e', 16, 14, 1.4, 1.4), ('e', 83, 15, 1.4, 1.4)], ear, sd=0.8)
    c.det([strand(38, 14, 25, 6, 3, bend=3, w1=2), strand(62, 14, 74, 7.5, 3, bend=-3, w1=2)], (248, 205, 220))
    return c.done()


def draw_eirin(size=384):
    c = C(size, sc=0.94, oy=2.5)
    hair = (218, 222, 236)
    red, blue = (205, 45, 60), (45, 70, 160)
    hair_back(c, hair, yend=70, wb=24, n=6)
    body(c, red, skirt=blue, sleeve=red, skirt_y=89, skirt_w=19, shoes=(60, 50, 60), cuff=WHITE)
    # split colours: right half of top blue, right half of skirt red
    c.part([('p', [(50, 48), (58.5, 48), (60.8, 64), (50, 64)])], blue, ol=False)
    c.part([('p', [(50, 64), (59, 58), (69, 87), (66.5, 89), (50, 89)])], red, ol=False)
    c.part([('l', [(58.2, 51), (64, 64.5)], 5.2)], blue)
    c.part([('e', 64, 64.5, 3, 3)], WHITE, shade=False)
    draw_hands(c, [ARMS[1]])
    c.det([('l', [(50, 48), (50, 89)], 0.5)], OL)
    # constellation dots
    for x, y in [(44, 70), (39, 78), (46, 83), (56, 70), (61, 80), (55, 85), (45, 55), (56, 57)]:
        c.det([star(x, y, 1.3)], (250, 220, 90))
    c.det([('l', [(44, 70), (39, 78), (46, 83)], 0.3), ('l', [(56, 70), (61, 80), (55, 85)], 0.3)], (250, 220, 90))
    face(c, (110, 120, 170))
    bangs(c, hair, side=58, side_w=5)
    # braid over left shoulder
    for i in range(7):
        c.part([('e', 32 - i * 0.25, 52 + i * 3.8, 3.2 - i * 0.12, 2.6)], hair, sd=0.8)
    bow(c, 30.5, 79, 3, blue, tails=False)
    # nurse cap
    c.part([('p', arc(50, 16, 12, 6.5, 180, 360) + [(62, 16), (38, 16)])], blue, sd=0.8)
    c.part([('p', arc(50, 16, 12, 6.5, 270, 360) + [(62, 16), (50, 16)])], red, ol=False, sd=0.8)
    c.part([('r', 48.5, 10, 51.5, 17, 0.2), ('r', 46.5, 12, 53.5, 15, 0.2)], WHITE, shade=False, olw=0.35)
    c.det([('r', 49.2, 10.8, 50.8, 16.2, 0.1), ('r', 47.3, 12.7, 52.7, 14.3, 0.1)], (220, 40, 60))
    return c.done()


def draw_kaguya(size=384):
    c = C(size, sc=0.94, oy=2.5)
    hair = (30, 26, 40)
    pink, maroon = (248, 175, 205), (150, 30, 55)
    hair_back(c, hair, yend=92, wb=31, n=8)
    arms = [((41.8, 51), (35, 64)), ((58.2, 51), (66, 60))]
    # jewel branch (behind right hand)
    c.part([('l', [(66, 62), (70, 48), (73, 38)], 1.3), ('l', [(70, 48), (77, 44)], 0.9),
            ('l', [(71.5, 43), (66, 36)], 0.9)], (120, 80, 50))
    for (x, y), col in zip([(73, 37), (77.5, 43.5), (66, 35.5), (74.5, 44), (68.5, 40.5)],
                           [(240, 80, 90), (80, 170, 240), (250, 220, 90), (130, 220, 140), (220, 130, 240)]):
        c.part([('e', x, y, 1.9, 1.9)], col, sd=0.5, olw=0.35)
        c.det([('e', x - 0.6, y - 0.6, 0.6, 0.6)], WHITE)
    body(c, pink, skirt=maroon, sleeve=pink, skirt_y=90, skirt_w=21, arms=arms, sleeve_w=7,
         shoes=(90, 30, 40))
    # skirt patterns
    for x, y in [(40, 76), (50, 84), (60, 75), (45, 68), (56, 66)]:
        c.det([star(x, y, 2, n=6, inner=0.55)], (240, 190, 90))
    c.det([('l', [(31, 87), (69, 87)], 1.0)], (240, 190, 90))
    c.part([('p', [(43, 48.5), (57, 48.5), (50, 55)])], WHITE, sd=0.5)
    bow(c, 50, 55.5, 4.2, WHITE)
    face(c, (170, 70, 90))
    bangs(c, hair, hime=True, side=56, side_w=6.5)
    return c.done()


def draw_mokou(size=384):
    c = C(size, sc=0.92, oy=3)
    hair = (242, 242, 248)
    # fire wings
    for s in (-1, 1):
        fl = []
        for i, (a, L) in enumerate([(-80, 44), (-58, 42), (-36, 38), (-14, 34), (8, 28)]):
            ar = math.radians(a)
            x0 = 50 + s * 12
            fl.append(strand(x0, 58, x0 + s * math.cos(ar) * L, 58 + math.sin(ar) * L,
                             14, bend=-7 * s, w1=0))
        fl.append(('e', 50 + s * 16, 56, 10, 9))
        c.part(fl, (245, 110, 40), olc=(190, 40, 30), shade=(230, 70, 30), sd=1.5)
        inner = []
        for i, (a, L) in enumerate([(-66, 32), (-44, 30), (-22, 26), (0, 20)]):
            ar = math.radians(a)
            x0 = 50 + s * 14
            inner.append(strand(x0, 58, x0 + s * math.cos(ar) * L, 58 + math.sin(ar) * L, 7, bend=-5 * s))
        c.det(inner, (255, 215, 90))
    hair_back(c, hair, yend=88, wb=27, n=7)
    arms = [((41.8, 51), (40, 64)), ((58.2, 51), (60, 64))]
    red = (205, 40, 50)
    body(c, WHITE, pants=red, sleeve=WHITE, arms=arms, hands=False, shoes=(80, 50, 45), shorts_y=89)
    c.part([('p', [(40.5, 58), (59.5, 58), (59.7, 63), (40.3, 63)])], red, sd=0.8)
    c.part([('l', [(44, 49), (43.5, 59)], 1.5), ('l', [(56, 49), (56.5, 59)], 1.5)], red, shade=False, olw=0.35)
    for x, y in [(44, 75), (56, 70), (55.5, 80)]:
        ofuda(c, x, y, 1.6, 3)
    face(c, (210, 40, 50), mouth='flat')
    bangs(c, hair, side=58, side_w=5.5)
    bow(c, 50, 13, 6, WHITE, tails=False)
    c.det([('l', [(45, 12.5), (48, 13)], 0.5), ('l', [(52, 13), (55, 12.5)], 0.5)], (210, 40, 50))
    for x, y in [(27, 70), (73, 70), (33, 88), (67, 88)]:
        bow(c, x, y, 2.3, WHITE, tails=False)
        c.det([('e', x - 1.6, y, 0.7, 0.7), ('e', x + 1.6, y, 0.7, 0.7)], (210, 40, 50))
    return c.done()


def draw_keine(size=384):
    c = C(size, sc=0.9, oy=4.5)
    hair = (205, 218, 238)
    blue = (60, 90, 190)
    hair_back(c, hair, yend=82, wb=26, n=7)
    body(c, blue, skirt=blue, sleeve=WHITE, skirt_y=88, skirt_w=19, shoes=(60, 50, 60))
    c.part([('p', [(41.5, 48), (58.5, 48), (59.2, 54), (40.8, 54)])], WHITE, sd=0.6)
    for s in (-1, 1):
        c.part([('e', 50 + s * 9.5, 52, 4.2, 3.6)], WHITE, sd=0.6)
    c.det([('l', [(32, 86), (68, 86)], 1.1)], WHITE)
    bow(c, 50, 55, 3.6, (215, 40, 60))
    face(c, (170, 60, 70))
    bangs(c, hair, side=60, side_w=6)
    # box hat with red ribbon
    c.part([('p', [(37, 19), (63, 19), (64, 6), (50, -1), (36, 6)])], blue, sd=1.2)
    c.part([('p', [(37, 15), (63, 15), (63.4, 19), (36.6, 19)])], WHITE, shade=False, olw=0.35)
    c.det([('p', [(47, 4.5), (53, 4.5), (53, 10), (47, 10)])], (215, 40, 60))
    c.det([('p', [(48.4, 5.8), (51.6, 5.8), (51.6, 8.7), (48.4, 8.7)])], blue)
    bow(c, 63, 17, 3.8, (215, 40, 60))
    return c.done()


def draw_mystia(size=384):
    c = C(size, sc=0.92, oy=3.5)
    hair = (238, 140, 180)
    wings(c, (150, 90, 120), x=56, y=56, length=27, n=6, a0=-65, a1=25, width=8.5)
    hair_back(c, hair, yend=50, wb=25, n=6)
    dress = (125, 60, 75)
    body(c, WHITE, skirt=dress, sleeve=WHITE, skirt_y=80, skirt_w=17, shoes=(90, 50, 55), cuff=dress)
    c.part([('p', [(43, 52), (57, 52), (58.8, 63), (41.2, 63)])], dress, sd=0.8)
    c.det([('l', [(33, 78), (67, 78)], 0.9)], (240, 220, 180))
    bow(c, 50, 50.5, 3.2, (90, 170, 110))
    face(c, (200, 110, 150), mouth='open')
    bangs(c, hair, side=48, side_w=6)
    # feather ears
    for s in (-1, 1):
        c.part([strand(50 + s * 22, 32, 50 + s * 31, 24 + i * 3, 3.5, w1=1) for i in range(3)],
               (250, 240, 230), sd=0.5)
    # winged mob cap
    cap = (140, 80, 110)
    c.part([('p', arc(50, 17, 18, 8, 180, 360)), ('e', 50, 17, 19, 2.6)], cap, sd=0.8)
    for s in (-1, 1):
        c.part([strand(50 + s * 14, 14, 50 + s * 22, 6 + i * 2.5, 3, bend=-s, w1=0.8) for i in range(3)],
               (180, 130, 150), sd=0.5)
    c.det([('l', [(76, 15), (76, 8), (79, 7)], 0.6), ('e', 75.3, 15, 1.2, 0.9)], (80, 50, 90))
    return c.done()


def draw_wriggle(size=384):
    c = C(size, sc=0.92, oy=3.5)
    hair = (110, 200, 90)
    cape, cape_in = (35, 35, 65), (200, 40, 60)
    c.part([('p', [(41, 48), (59, 48), (73, 86), (50, 84), (27, 86)])], cape_in, sd=1)
    c.part([('p', [(41, 48), (59, 48), (69, 84), (50, 82), (31, 84)])], cape, ol=False)
    hair_back(c, hair, yend=None)
    navy = (40, 50, 110)
    body(c, WHITE, pants=navy, sleeve=WHITE, shoes=(40, 40, 60), shorts_y=73, cuff=navy)
    c.part([('p', [(40.5, 58), (59.5, 58), (59.7, 62), (40.3, 62)])], navy, sd=0.5)
    c.part([('p', [(41.5, 47.5), (58.5, 47.5), (55, 52), (45, 52)])], cape_in, sd=0.5)
    face(c, (60, 170, 80), mouth='open')
    tips = [(71, 40), (65, 33), (58, 32), (50, 31.5), (42, 32), (35, 33), (29, 40)]
    bangs(c, hair, tips=tips, side=42, side_w=5)
    c.part([('l', [(45, 14), (41, 6), (36, 4)], 0.9), ('l', [(55, 14), (59, 6), (64, 4)], 0.9)], hair, shade=False)
    for x, y in [(20, 30), (82, 40), (17, 62), (85, 70), (24, 84)]:
        c.glow([('e', x, y, 2.2, 2.2)], (220, 255, 120), blur=1.2)
        c.det([('e', x, y, 1, 1)], (250, 255, 190))
    return c.done()


def draw_suika(size=384):
    c = C(size, sc=0.9, oy=4.5)
    hair = (238, 165, 95)
    hair_back(c, hair, yend=86, wb=24, n=6)
    purple = (125, 70, 160)
    arms = [((41.8, 51), (36, 64.5)), ((58.2, 51), (65, 58))]
    body(c, WHITE, skirt=purple, sleeve=WHITE, skirt_y=78, skirt_w=17, arms=arms, hands=False,
         shoes=(80, 50, 60))
    c.det([('l', [(41, 58), (59, 58)], 1.2)], (200, 60, 70))
    # chains with shapes
    c.det([('l', [(35, 66), (33, 73), (34, 79)], 0.5), ('l', [(38, 66), (40, 72)], 0.5)], (140, 140, 150))
    c.part([('p', [(34, 77.5), (36.5, 82), (31.5, 82)])], (250, 210, 60), sd=0.4, olw=0.35)
    c.part([('e', 40, 73.5, 1.8, 1.8)], (220, 60, 70), sd=0.4, olw=0.35)
    # gourd
    gr = (145, 80, 170)
    c.part([('e', 69, 64, 4.5, 4.8), ('e', 69, 57.5, 3, 3.2), ('r', 68, 52.5, 70, 55, 0.3)], gr, sd=1)
    c.det([('l', [(65.2, 60.5), (72.8, 60.5)], 0.8)], (220, 60, 70))
    draw_hands(c, arms)
    face(c, (190, 70, 60), mouth='open', fang=True)
    bangs(c, hair, side=56, side_w=6)
    # horns
    hc = (170, 115, 80)
    for s in (-1, 1):
        c.part([strand(50 + s * 18, 22, 50 + s * 30, -1, 5.5, bend=s * 3, w1=1)], hc, sd=0.8)
        c.det([('l', [(50 + s * 20, 16), (50 + s * 23.5, 15.5)], 0.6), ('l', [(50 + s * 22.8, 10), (50 + s * 25.5, 9.5)], 0.6)],
              dk(hc, 0.7))
    bow(c, 50, 12, 4.5, (215, 45, 60), tails=False)
    bow(c, 70, 14, 2.4, (80, 110, 200), tails=True)
    return c.done()


def draw_aya(size=384):
    c = C(size, sc=0.94, oy=2)
    hair = (38, 32, 42)
    wings(c, (30, 28, 40), x=57, y=55, length=28, n=6, a0=-65, a1=20, width=9)
    hair_back(c, hair, yend=46, wb=25, n=6)
    arms = [((41.8, 51), (45, 60)), ((58.2, 51), (55, 60))]
    body(c, WHITE, skirt=(35, 32, 40), sleeve=WHITE, skirt_y=77, skirt_w=16, arms=arms, hands=False,
         shoes=(200, 50, 50))
    c.part([('r', 42.5, 90.5, 46.5, 94, 0.3), ('r', 53.5, 90.5, 57.5, 94, 0.3)], (200, 50, 50), shade=False)
    # camera
    c.part([('r', 42, 55, 58, 64, 1.2), ('r', 44, 53, 49, 56, 0.5)], (45, 45, 50), sd=0.8)
    c.part([('e', 51, 59.5, 3.4, 3.4)], (80, 80, 90), sd=0.6, olw=0.4)
    c.det([('e', 51, 59.5, 2, 2)], (40, 60, 100))
    c.det([('e', 50.3, 58.7, 0.7, 0.7)], WHITE)
    draw_hands(c, [((40, 55), (41.5, 60)), ((60, 55), (58.5, 60))])
    face(c, (210, 40, 50))
    tips = [(71, 42), (65, 34), (58, 33), (50, 32), (42, 33), (35, 34), (29, 42)]
    bangs(c, hair, tips=tips, side=46, side_w=6)
    tokin(c, 50, 11.5, 1.9, (220, 40, 50))
    c.det([('l', [(45, 15), (40, 18)], 0.6), ('l', [(55, 15), (60, 18)], 0.6)], (220, 40, 50))
    c.part([('e', 39.5, 19, 1.4, 1.4), ('e', 60.5, 19, 1.4, 1.4)], (240, 80, 80), sd=0.4, olw=0.35)
    return c.done()


def draw_momiji(size=384):
    c = C(size, sc=0.9, oy=4.5)
    hair = (244, 244, 246)
    # tail
    c.part([strand(55, 72, 80, 80, 13, bend=-6, w1=4), E(74, 80, 6.5, 4.5, 20), ('e', 80, 80, 2.4, 2.4)],
           (240, 240, 244), shade=(205, 205, 215), sd=1.4)
    c.det([strand(78, 82, 81.5, 79, 2.5, w1=1)], (215, 215, 225))
    # broad sword behind left hand
    c.part([('p', [(32, 62), (27, 42), (24, 28), (29, 32), (33, 44), (36.5, 62)])], (225, 230, 240),
           shade=(180, 190, 205), sd=1.2)
    c.part([('l', [(33.5, 63), (35, 70)], 1.8)], (180, 50, 50), shade=False)
    c.part([('l', [(31.5, 62), (36.5, 61)], 1.3)], (220, 180, 70), shade=False)
    hair_back(c, hair, yend=None)
    arms = [((41.8, 51), (35, 64)), ((58.2, 51), (65, 62))]
    body(c, WHITE, skirt=(35, 32, 40), sleeve=WHITE, skirt_y=78, skirt_w=17, arms=arms, hands=False,
         shoes=(40, 35, 40), cuff=(200, 50, 60))
    c.det([('l', [(34, 76), (66, 76)], 1.4)], (200, 50, 60))
    draw_hands(c, arms)
    # shield with maple leaf
    c.part([('e', 69, 64, 7, 7)], WHITE, shade=(210, 210, 220))
    c.det([star(69, 64, 4.8, n=7, inner=0.5), ('l', [(69, 64), (70.5, 68.5)], 0.6)], (210, 50, 50))
    face(c, (210, 40, 50), mouth='open', fang=True)
    tips = [(71, 40), (65, 34), (58, 33), (50, 32), (42, 33), (35, 34), (29, 40)]
    bangs(c, hair, tips=tips, side=46, side_w=6)
    ears_animal(c, hair, (250, 200, 210), [(30, 20), (29, 5), (40, 14)], [(70, 20), (71, 5), (60, 14)])
    tokin(c, 50, 11.5, 1.6, (220, 40, 50))
    return c.done()


def draw_nitori(size=384):
    c = C(size, sc=0.92, oy=3.5)
    hair = (85, 145, 225)
    green = (80, 165, 105)
    # backpack
    c.part([('r', 30, 44, 70, 76, 5)], green, sd=1.2)
    c.det([('r', 30, 70, 70, 73, 0.5)], dk(green, 0.8))
    hair_back(c, hair, yend=None)
    blue = (85, 150, 230)
    body(c, blue, skirt=blue, sleeve=blue, skirt_y=78, skirt_w=16, shoes=(70, 110, 190), cuff=lt(blue, 0.3))
    c.part([('l', [(44, 49), (43, 63)], 1.8), ('l', [(56, 49), (57, 63)], 1.8)], green, shade=False, olw=0.35)
    for x, y in [(41, 71), (59, 71)]:
        c.part([('r', x - 3, y - 2.5, x + 3, y + 2.5, 0.8)], lt(blue, 0.3), shade=False, olw=0.35)
    # key
    c.part([('e', 50, 53, 2, 2), ('l', [(50, 54), (50, 60)], 1), ('l', [(50, 58.5), (52, 58.5)], 0.8)],
           (250, 210, 70), shade=False, olw=0.35)
    c.det([('e', 50, 53, 0.8, 0.8)], blue)
    face(c, (60, 110, 200), mouth='open')
    bangs(c, hair, side=44, side_w=5)
    # twintails
    for s in (-1, 1):
        c.part([strand(50 + s * 22, 28, 50 + s * 29, 46, 5.5, bend=-s * 2, w1=1.5)], hair, sd=0.8)
        c.part([('e', 50 + s * 22.5, 28.5, 1.6, 1.6)], (220, 50, 60), sd=0.4, olw=0.35)
    # cap
    c.part([('p', arc(50, 20, 21, 12.5, 180, 360)), ('e', 50, 20, 22.5, 3)], green, sd=1)
    c.det([('e', 50, 11, 1.5, 1.5)], dk(green, 0.8))
    return c.done()


def draw_sanae(size=384):
    c = C(size, sc=0.94, oy=2.5)
    hair = (110, 195, 135)
    blue = (70, 105, 215)
    hair_back(c, hair, yend=84, wb=25, n=7)
    arms = [((41.8, 51), (36, 64.5)), ((58.2, 51), (65, 60))]
    gohei(c, 67, 67, 72, 30)
    body(c, WHITE, skirt=blue, sleeve=WHITE, skirt_y=80, skirt_w=18, arms=arms, sleeve_w=7,
         shoes=(90, 60, 50), cuff=blue)
    c.det([('l', [(41.5, 48.5), (50, 55), (58.5, 48.5)], 1.0)], blue)
    bow(c, 50, 55, 2.8, blue)
    face(c, (80, 170, 120))
    bangs(c, hair, side=62, side_w=5.5)
    # snake ornament (right) and frog clip (left)
    c.part([strand(69, 22, 72, 40, 2.4, bend=4, w1=1), ('e', 68.5, 21.5, 1.8, 1.4)], WHITE,
           shade=(215, 225, 235), olw=0.35)
    fg = (100, 200, 90)
    c.part([('e', 31, 24, 3.2, 2.6), ('e', 29.3, 21.8, 1.3, 1.3), ('e', 32.7, 21.8, 1.3, 1.3)], fg, sd=0.5, olw=0.4)
    c.det([('e', 29.3, 21.6, 0.5, 0.5), ('e', 32.7, 21.6, 0.5, 0.5)], OL)
    return c.done()


def draw_suwako(size=384):
    c = C(size, sc=0.9, oy=5)
    hair = (248, 222, 125)
    hair_back(c, hair, yend=48, wb=25, n=6)
    purple = (125, 90, 175)
    body(c, purple, skirt=purple, sleeve=WHITE, skirt_y=79, skirt_w=17, legs=WHITE, shoes=(120, 90, 60))
    c.part([('p', [(41.5, 48), (58.5, 48), (56, 53), (44, 53)])], WHITE, sd=0.5)
    for x, y in [(40, 72), (58, 68), (50, 76)]:
        c.det([('e', x, y, 1.8, 1.3), ('e', x - 1, y - 1, 0.7, 0.7), ('e', x + 1, y - 1, 0.7, 0.7)], (90, 200, 90))
    face(c, (190, 150, 60), mouth='cat')
    tips = [(71, 40), (64.5, 33.5), (57.5, 32.3), (50, 32), (42.5, 32.3), (35.5, 33.5), (29, 40)]
    bangs(c, hair, tips=tips, side=47, side_w=5.5)
    # big frog-eye hat
    hat = (225, 205, 150)
    c.part([('e', 50, 20, 31, 4.5)], hat, sd=0.8)
    c.part([('p', arc(50, 20, 19, 13, 180, 360))], hat, sd=1)
    c.det([('l', arc(50, 20.5, 19, 2, 0, 180), 1.2)], (125, 90, 175))
    for s in (-1, 1):
        x = 50 + s * 9
        c.part([('e', x, 5.5, 5, 5)], WHITE, shade=(220, 220, 225))
        c.part([('e', x + s * 0.6, 6, 2.4, 2.8)], (40, 30, 30), shade=False, olw=0.3)
        c.det([('e', x + s * 0.2, 4.8, 0.9, 0.9)], WHITE)
    return c.done()


def draw_kanako(size=384):
    c = C(size, sc=0.9, oy=4)
    # shimenawa ring
    rope = (195, 160, 110)
    m = [('e', 50, 50, 43, 30), ('-', ('e', 50, 50, 37, 24))]
    c.part(m, rope, sd=1.2)
    for i in range(20):
        a = math.radians(i * 18)
        x, y = 50 + 40 * math.cos(a), 50 + 27 * math.sin(a)
        c.det([('l', [(x - 1.5, y - 1.5), (x + 1.5, y + 1.5)], 0.7)], dk(rope, 0.7))
    # shide papers hanging off the ring
    for x in (14, 86):
        zz = [(x - 1, 55), (x + 2.5, 58), (x - 0.5, 61), (x + 2.5, 65), (x - 0.5, 68), (x + 1.5, 70), (x - 2, 67),
              (x + 1, 64), (x - 2, 60), (x + 0.5, 58), (x - 2, 56)]
        c.part([('p', zz)], WHITE, shade=False, olw=0.35)
    # onbashira pillars
    wood = (160, 100, 60)
    for x in (13, 87):
        c.part([('r', x - 4, 62, x + 4, 94, 1.2)], wood, sd=1)
        c.part([('e', x, 62, 4, 1.8)], (215, 170, 120), shade=False, olw=0.35)
        c.det([('l', [(x - 4, 72), (x + 4, 72)], 1.2), ('l', [(x - 4, 83), (x + 4, 83)], 1.2)], (80, 60, 50))
    hair = (80, 90, 175)
    hair_back(c, hair, yend=62, wb=28, n=7)
    red, skirt = (215, 45, 55), (140, 60, 55)
    body(c, red, skirt=skirt, sleeve=WHITE, skirt_y=88, skirt_w=19, sleeve_w=6.5, shoes=(60, 45, 45), cuff=red)
    c.det([('l', [(32, 85.5), (68, 85.5)], 1.2)], (230, 190, 90))
    c.part([('e', 50, 56, 3.6, 3.6)], (230, 190, 90), sd=0.6, olw=0.4)
    c.det([('e', 50, 56, 2.4, 2.4)], (210, 225, 240))
    c.det([('e', 49.2, 55.2, 0.8, 0.8)], WHITE)
    face(c, (200, 50, 60))
    bangs(c, hair, side=52, side_w=7, extra=[('e', 27, 38, 4, 7), ('e', 73, 38, 4, 7)])
    # leaf headband
    for i in range(5):
        x = 38 + i * 6
        c.part([E(x, 15.5 + abs(i - 2) * 0.8, 1.7, 3.2, -30 + i * 15)], (190, 40, 50), sd=0.4, olw=0.35)
    return c.done()


def draw_yuuka(size=384):
    c = C(size, sc=0.92, oy=3)
    hair = (110, 185, 100)
    red = (205, 45, 55)
    # parasol behind, handle to right hand
    c.part([('l', [(66, 64), (72, 26)], 1.3)], (120, 80, 50), shade=False)
    pts = arc(73, 26, 24, 16, 200, 340)
    scal = []
    for i in range(6):
        a0 = 340 - i * 23.3
        scal += arc(73 + 24 * math.cos(math.radians(a0 - 11.6)) * 0.93,
                    26 + 16 * math.sin(math.radians(a0 - 11.6)) * 0.93 + 1.5, 4.2, 2.5, 0, 180, n=6)
    c.part([('p', pts)], (252, 215, 225), shade=(235, 185, 200), sd=2)
    c.det([('l', [(73, 11), (73, 25)], 0.4), ('l', [(73, 11), (60, 22)], 0.4), ('l', [(73, 11), (86, 22)], 0.4)],
          (225, 170, 190))
    c.part([('l', [(73, 10), (73, 8)], 0.9)], (120, 80, 50), shade=False, olw=0.3)
    hair_back(c, hair, yend=None)
    arms = [((41.8, 51), (36, 64)), ((58.2, 51), (65, 62))]
    body(c, WHITE, skirt=red, sleeve=WHITE, skirt_y=88, skirt_w=19, arms=arms, shoes=(90, 50, 40))
    c.part([('p', [(41.5, 48), (46.5, 48), (49, 64), (39.2, 64)]), ('p', [(53.5, 48), (58.5, 48), (60.8, 64), (51, 64)])],
           red, sd=0.8)
    plaid = dk(red, 0.72)
    for x in (37, 44, 51, 58, 64):
        c.det([('l', [(x, 64), (x + (x - 50) * 0.3, 87)], 0.5)], plaid)
    for y in (70, 78):
        c.det([('l', [(35, y), (65, y)], 0.5)], plaid)
    c.part([('p', [(47.5, 49), (52.5, 49), (50, 53)])], (250, 210, 70), sd=0.4, olw=0.35)
    # sunflower in left hand
    c.part([('l', [(34, 68), (31, 80)], 1)], (80, 150, 60), shade=False, olw=0.35)
    petals = [E(34 + 3.6 * math.cos(math.radians(a)), 64 + 3.6 * math.sin(math.radians(a)), 2.2, 1.1, a) for a in range(0, 360, 36)]
    c.part(petals, (250, 205, 50), sd=0.5, olw=0.4)
    c.part([('e', 34, 64, 2.4, 2.4)], (120, 70, 40), sd=0.5, olw=0.35)
    face(c, (210, 40, 50))
    tips = [(72, 41), (64.5, 34), (57.5, 33), (50, 32.5), (42.5, 33), (35.5, 34), (28, 41)]
    bangs(c, hair, tips=tips, side=46, side_w=6,
          extra=[('e', 27, 44, 3.5, 3.5), ('e', 73, 44, 3.5, 3.5)])
    return c.done()


def draw_komachi(size=384):
    c = C(size, sc=0.9, oy=5)
    hair = (225, 75, 75)
    # scythe: bent handle + curved blade
    c.part([('l', [(71, 90), (68, 70), (72, 50), (69, 28), (73, 8)], 2.2)], (130, 90, 60))
    c.part([('p', [(74, 3), (62, -1), (46, 0), (32, 6), (22, 16), (34, 11), (48, 8), (61, 8.5), (73, 11)])],
           (205, 212, 228), shade=(150, 160, 185), sd=1.2)
    c.part([('r', 70.5, 2, 75.5, 12, 0.6)], (90, 70, 60), shade=False)
    hair_back(c, hair, yend=None)
    blue = (70, 110, 195)
    arms = [((41.8, 51), (36, 64.5)), ((58.2, 51), (66, 58))]
    body(c, blue, skirt=blue, sleeve=WHITE, skirt_y=82, skirt_w=18, arms=arms, shoes=(70, 50, 45))
    c.part([('p', [(41.5, 48), (58.5, 48), (55, 54), (45, 54)])], WHITE, sd=0.5)
    c.part([('r', 39.5, 59, 60.5, 63, 0.5)], (240, 225, 180), shade=False, olw=0.4)
    c.part([('e', 50, 61, 2.6, 2.6)], (230, 185, 70), sd=0.5, olw=0.4)
    c.det([('r', 49.2, 60.2, 50.8, 61.8, 0.1)], (240, 225, 180))
    face(c, (200, 50, 60), mouth='open')
    bangs(c, hair, side=46, side_w=5)
    for s in (-1, 1):
        c.part([strand(50 + s * 18, 18, 50 + s * 28, 30, 6.5, bend=s * 2, w1=2)], hair, sd=0.8)
        c.part([('e', 50 + s * 19, 18.5, 1.8, 1.8)], (240, 200, 90), sd=0.4, olw=0.35)
    return c.done()


CHARACTERS = {
    "reisen": {"name": "Reisen Udongein Inaba", "color": (200, 160, 235),
               "spell": 'Eyes "Lunatic Red Hallucination"', "draw": draw_reisen},
    "tewi": {"name": "Tewi Inaba", "color": (250, 170, 200),
             "spell": 'Prank "Ancient Duper: Fake Benchmark"', "draw": draw_tewi},
    "eirin": {"name": "Eirin Yagokoro", "color": (60, 80, 170),
              "spell": 'Medicine "Hourai Elixir of Fine-Tuning"', "draw": draw_eirin},
    "kaguya": {"name": "Kaguya Houraisan", "color": (245, 170, 205),
               "spell": 'Impossible Request "Infinite Context"', "draw": draw_kaguya},
    "mokou": {"name": "Fujiwara no Mokou", "color": (240, 100, 40),
              "spell": 'Undying "Phoenix of the Last Checkpoint"', "draw": draw_mokou},
    "keine": {"name": "Keine Kamishirasawa", "color": (70, 100, 200),
              "spell": 'Old History "Rewrite the Training Data"', "draw": draw_keine},
    "mystia": {"name": "Mystia Lorelei", "color": (235, 135, 180),
               "spell": 'Night Song "Hallucination Serenade"', "draw": draw_mystia},
    "wriggle": {"name": "Wriggle Nightbug", "color": (110, 200, 90),
                "spell": 'Firefly "Bug Report of a Thousand Lights"', "draw": draw_wriggle},
    "suika": {"name": "Suika Ibuki", "color": (238, 160, 90),
              "spell": 'Oni Sign "Mixture of a Hundred Experts"', "draw": draw_suika},
    "aya": {"name": "Aya Shameimaru", "color": (215, 40, 50),
            "spell": 'Wind God "Tengu Web Search Scoop"', "draw": draw_aya},
    "momiji": {"name": "Momiji Inubashiri", "color": (240, 240, 240),
               "spell": 'Fang Sign "Guardrail Watchdog Slash"', "draw": draw_momiji},
    "nitori": {"name": "Nitori Kawashiro", "color": (85, 150, 230),
               "spell": 'Kappa "Tool-Use Toolbox Overflow"', "draw": draw_nitori},
    "sanae": {"name": "Sanae Kochiya", "color": (110, 195, 135),
              "spell": 'Miracle "Emergent Abilities at Scale"', "draw": draw_sanae},
    "suwako": {"name": "Suwako Moriya", "color": (245, 215, 120),
               "spell": 'Native God "Frog-Pond Reward Model"', "draw": draw_suwako},
    "kanako": {"name": "Kanako Yasaka", "color": (140, 60, 150),
               "spell": 'Divine Virtue "Onbashira Constitution"', "draw": draw_kanako},
    "yuuka": {"name": "Yuuka Kazami", "color": (205, 45, 55),
              "spell": 'Flower Sign "Gradient Garden Descent"', "draw": draw_yuuka},
    "komachi": {"name": "Komachi Onozuka", "color": (225, 75, 75),
                "spell": 'Death Toll "Ferry Past the Context Limit"', "draw": draw_komachi},
}


if __name__ == "__main__":
    cell, cols = 384, 6
    keys = list(CHARACTERS)
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new('RGBA', (cell * cols, (cell + 20) * rows), (70, 74, 88, 255))
    d = ImageDraw.Draw(sheet)
    for i, key in enumerate(keys):
        ch = CHARACTERS[key]
        assert len(ch["spell"]) < 48, (key, len(ch["spell"]))
        x, y = (i % cols) * cell, (i // cols) * (cell + 20)
        bg = Image.new('RGBA', (cell, cell), (205, 208, 218, 255) if (i % 2) else (190, 196, 210, 255))
        sheet.alpha_composite(bg, (x, y))
        sheet.alpha_composite(ch["draw"](cell), (x, y))
        d.text((x + 6, y + cell + 4), key, fill=(255, 255, 255, 255))
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')
    os.makedirs(root, exist_ok=True)
    out = os.path.normpath(os.path.join(root, 'sheet_b.png'))
    sheet.convert('RGB').save(out)
    print(out)
