"""Cameo batch A: procedural chibi sprites (Yakumo family, EoSD cast, PCB cast).

Everything is drawn from scratch with Pillow polygons on a 100x100 unit grid,
rendered at 4x and downsampled with LANCZOS. See SPEC.md.
"""
import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw

SS = 4
SKIN = (255, 230, 214)
WHITE = (250, 250, 253)
BLACK = (52, 46, 62)
DARK = (30, 18, 34)
RED = (220, 38, 52)
GOLD = (245, 200, 70)


# ----------------------------------------------------------------- colour --
def mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def shade_of(c):
    return (int(c[0] * .80), int(c[1] * .76), int(min(255, c[2] * .86 + 14)))


def line_of(c):
    return mix(c, DARK, .80)


# --------------------------------------------------------------- geometry --
def ell_pts(cx, cy, rx, ry, a0=0, a1=360, n=56, rot=0):
    out = []
    r = math.radians(rot)
    for i in range(n + 1 if a1 - a0 < 360 else n):
        a = math.radians(a0 + (a1 - a0) * i / n)
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append((cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)))
    return out


def bez(ctrl, t):
    p = list(ctrl)
    while len(p) > 1:
        p = [(p[i][0] + (p[i + 1][0] - p[i][0]) * t, p[i][1] + (p[i + 1][1] - p[i][1]) * t)
             for i in range(len(p) - 1)]
    return p[0]


def chaikin(pts, it=2, closed=True):
    for _ in range(it):
        out = []
        m = len(pts) if closed else len(pts) - 1
        if not closed:
            out.append(pts[0])
        for i in range(m):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            out += [(a[0] * .75 + b[0] * .25, a[1] * .75 + b[1] * .25),
                    (a[0] * .25 + b[0] * .75, a[1] * .25 + b[1] * .75)]
        if not closed:
            out.append(pts[-1])
        pts = out
    return pts


def rot(pts, ang, cx, cy):
    r = math.radians(ang)
    co, si = math.cos(r), math.sin(r)
    return [(cx + (x - cx) * co - (y - cy) * si, cy + (x - cx) * si + (y - cy) * co) for x, y in pts]


def mir(pts, cx=50):
    return [(2 * cx - x, y) for x, y in pts]


def tr(pts, dx, dy, s=1.0):
    return [(x * s + dx, y * s + dy) for x, y in pts]


def strand_pts(ctrl, w0, w1=0.0, bulge=0.0, t0=0.0, t1=1.0, n=26, peak=0.5):
    """Tapered ribbon along a bezier; half-width goes w0->w1 plus sin bulge."""
    L, R = [], []
    for i in range(n + 1):
        t = t0 + (t1 - t0) * i / n
        x, y = bez(ctrl, t)
        x2, y2 = bez(ctrl, min(1, t + 1e-3))
        x1, y1 = bez(ctrl, max(0, t - 1e-3))
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        nx, ny = -dy / d, dx / d
        w = w0 * (1 - t) + w1 * t + bulge * math.sin(math.pi * t ** (math.log(.5) / math.log(peak)))
        L.append((x + nx * w, y + ny * w))
        R.append((x - nx * w, y - ny * w))
    return L + R[::-1]


def star_pts(cx, cy, r1, r2, n=5, a=-90):
    return [(cx + (r1 if i % 2 == 0 else r2) * math.cos(math.radians(a + 180 * i / n)),
             cy + (r1 if i % 2 == 0 else r2) * math.sin(math.radians(a + 180 * i / n)))
            for i in range(2 * n)]


# ----------------------------------------------------------------- canvas --
class Cv:
    def __init__(s, size=384):
        s.size = size
        s.W = size * SS
        s.k = s.W / 100.0
        s.img = Image.new('RGBA', (s.W, s.W), (0, 0, 0, 0))
        s.d = ImageDraw.Draw(s.img)

    def px(s, pts):
        return [(x * s.k, y * s.k) for x, y in pts]

    def fill(s, pts, col, sh=1.3, ol=0.5, a=255, lc=None, sc=None, sd=(0.55, 1.0)):
        """Outlined, flat-filled polygon with a rim shade on the sd side."""
        P = s.px(pts)
        if ol:
            lc = lc or line_of(col)
            s.d.polygon(P, fill=lc + (255,))
            s.d.line(P + [P[0]], fill=lc + (255,), width=max(1, int(ol * 2 * s.k)), joint='curve')
        if a >= 255:
            s.d.polygon(P, fill=col + (255,))
        if sh or a < 255:
            xs, ys = [p[0] for p in P], [p[1] for p in P]
            pad = int(sh * s.k) + 4
            x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
            x1, y1 = min(s.W, int(max(xs)) + pad), min(s.W, int(max(ys)) + pad)
            if x1 <= x0 or y1 <= y0:
                return
            m = Image.new('L', (x1 - x0, y1 - y0), 0)
            ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in P], fill=255)
            if a < 255:
                s.img.paste(col + (255,), (x0, y0), m.point(lambda v: v * a // 255))
            if sh:
                arr = np.array(m) > 0
                L = math.hypot(*sd)
                dx, dy = int(round(sd[0] / L * sh * s.k)), int(round(sd[1] / L * sh * s.k))
                sft = np.zeros_like(arr)
                h, w = arr.shape
                sft[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] = \
                    arr[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
                band = arr & ~sft
                sa = 255 if a >= 255 else a
                s.img.paste((sc or shade_of(col)) + (255,), (x0, y0),
                            Image.fromarray((band * sa).astype(np.uint8)))

    def ell(s, cx, cy, rx, ry, col, rot=0, **kw):
        s.fill(ell_pts(cx, cy, rx, ry, rot=rot), col, **kw)

    def poly(s, pts, col, sm=0, **kw):
        s.fill(chaikin(pts, sm) if sm else pts, col, **kw)

    def strand(s, ctrl, w0, col, w1=0.0, bulge=0.0, **kw):
        s.fill(strand_pts(ctrl, w0, w1, bulge), col, **kw)

    def line(s, pts, col, w=0.5, a=255):
        P = s.px(pts)
        W = max(1, int(w * 2 * s.k))
        if a < 255:
            lay = Image.new('RGBA', s.img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
        else:
            d = s.d
        d.line(P, fill=col + (a,), width=W, joint='curve')
        for x, y in (P[0], P[-1]):
            d.ellipse((x - W / 2, y - W / 2, x + W / 2, y + W / 2), fill=col + (a,))
        if a < 255:
            s.img.alpha_composite(lay)

    def clip(s, pts, fn):
        """Draw fn(tmp_canvas) clipped to polygon pts."""
        tmp = Cv(s.size)
        fn(tmp)
        m = Image.new('L', (s.W, s.W), 0)
        ImageDraw.Draw(m).polygon(s.px(pts), fill=255)
        tmp.img.putalpha(ImageChops.multiply(tmp.img.getchannel('A'), m))
        s.img.alpha_composite(tmp.img)

    def done(s):
        return s.img.resize((s.size, s.size), Image.LANCZOS)


# ------------------------------------------------------------ body parts --
HX, HY, HRX, HRY = 50, 38, 19, 17.2
EYE_Y = 43.5


def head(c, skin=SKIN):
    c.ell(HX, HY, HRX, HRY, skin, sh=1.2)


def eye(c, x, y, col, rx=3.4, ry=4.5, style='open', side=1):
    if style == 'closed':
        c.line(ell_pts(x, y + 1, rx, ry * .55, 200, 340, n=14), DARK, .6)
        return
    c.ell(x, y, rx, ry, WHITE, sh=0, ol=.45, lc=DARK)
    c.ell(x, y + .5, rx * .86, ry * .86, col, sh=ry * .55, sd=(0, -1), sc=shade_of(shade_of(col)), ol=0)
    c.ell(x, y + .8, rx * .42, ry * .46, line_of(col), sh=0, ol=0)
    c.ell(x + rx * .30, y + ry * .45, rx * .55, ry * .22, mix(col, WHITE, .45), sh=0, ol=0)
    c.ell(x - rx * .32, y - ry * .32, rx * .36, rx * .36, WHITE, sh=0, ol=0)
    c.ell(x + rx * .30, y + ry * .05, rx * .16, rx * .16, WHITE, sh=0, ol=0)
    if style == 'half':
        c.poly([(x - rx - 1, y - ry - 1.5), (x + rx + 1, y - ry - 1.5), (x + rx + 1, y - ry * .15),
                (x - rx - 1, y - ry * .15)], SKIN, sh=0, ol=0)
        c.line([(x - rx - .3, y - ry * .15), (x + rx + .3, y - ry * .15)], DARK, .55)
        return
    # upper lash with an outward flick
    c.line(ell_pts(x, y + .2, rx * 1.02, ry * 1.02, 195, 345, n=16), DARK, .75)
    ox = x + side * rx * 1.0
    c.line([(ox, y - ry * .35), (ox + side * 1.3, y - ry * .7)], DARK, .45)


def mouth(c, kind='smile', x=50, y=51.5):
    if kind == 'smile':
        c.line(ell_pts(x, y - .6, 1.9, 1.3, 25, 155, n=10), line_of(SKIN), .38)
    elif kind == 'open':
        c.poly([(x - 2.0, y - .8), (x + 2.0, y - .8), (x + 1.2, y + 1.3), (x - 1.2, y + 1.3)],
               (190, 60, 80), sm=2, sh=0, ol=.35, lc=DARK)
        c.ell(x, y + .7, 1.0, .5, (240, 130, 140), sh=0, ol=0)
    elif kind == 'fang':
        c.line(ell_pts(x, y - .6, 2.2, 1.4, 20, 160, n=10), line_of(SKIN), .38)
        c.poly([(x + .5, y + .6), (x + 1.6, y + .5), (x + 1.1, y + 1.9)], WHITE, sh=0, ol=.22, lc=DARK)
    elif kind == 'cat':
        c.line(ell_pts(x - 1.1, y - .5, 1.1, 1.1, 10, 170, n=8), line_of(SKIN), .36)
        c.line(ell_pts(x + 1.1, y - .5, 1.1, 1.1, 10, 170, n=8), line_of(SKIN), .36)
    elif kind == 'sly':
        c.line([(x - 2.3, y - .6), (x - .5, y + .2), (x + 2.4, y - 1.0)], line_of(SKIN), .38)
    elif kind == 'flat':
        c.line([(x - 1.4, y), (x + 1.4, y)], line_of(SKIN), .36)


def face(c, eyecol, style='open', mo='smile', blush=True, ex=8.3):
    if blush:
        for x in (50 - 12.5, 50 + 12.5):
            c.ell(x, 49.2, 3.2, 1.6, (250, 140, 150), sh=0, ol=0, a=110)
    eye(c, 50 - ex, EYE_Y, eyecol, style=style, side=-1)
    eye(c, 50 + ex, EYE_Y, eyecol, style=style, side=1)
    mouth(c, mo)


def hair_back(c, col, bottom=60, spread=21, top=17.5, waves=6, amp=1.6, wobble=0.0):
    pts = ell_pts(50, 38, 21.5, 38 - top, 180, 360, n=24)
    pts += [(50 + spread * .92, (38 + bottom) / 2), (50 + spread, bottom - 3)]
    for i in range(waves + 1):
        t = i / waves
        x = 50 + spread - 2 * spread * t
        y = bottom + (amp if i % 2 else -amp * .3) + wobble * math.sin(t * 7)
        pts.append((x, y))
    pts += [(50 - spread, bottom - 3), (50 - spread * .92, (38 + bottom) / 2)]
    c.poly(pts, col, sm=2, sh=2.2)


def bangs(c, col, tips=None, top=17.3, sh=1.4):
    if tips is None:
        tips = [(70.5, 49), (64, 36.5), (58.5, 39), (53.5, 36), (49.5, 41.5), (45, 36),
                (40.5, 39), (35.5, 36.5), (29.5, 49)]
    pts = ell_pts(50, 38.5, 21.3, 38.5 - top, 172, 368, n=28)
    vy = 33.5
    for i, (x, y) in enumerate(tips):
        if i:
            px_ = tips[i - 1][0]
            pts.append(((x + px_) / 2, vy + (3 if i in (1, len(tips) - 1) else 0)))
        pts.append((x, y))
    c.poly(pts, col, sm=1, sh=sh)


def side_lock(c, col, side=-1, length=58, x=31.2, curl=0.0, w=2.6):
    x0 = x if side < 0 else 100 - x
    c.strand([(x0, 33), (x0 + side * (1.5 + curl), (33 + length) / 2), (x0 - side * curl * .4, length)],
             w, col, w1=.2, sh=1.0)


def bow(c, cx, cy, s, col, ang=0, tails=True, knot=None, frill=None):
    lp = [(0, 0), (-.55, -.62), (-1.05, -.55), (-1.12, .05), (-.95, .55), (-.5, .52)]
    tl = [(-.12, .12), (-.55, .95), (-.28, 1.1), (-.05, .85), (.05, .2)]
    parts = []
    if tails:
        parts += [(tl, col), (mir(tl, 0), col)]
    parts += [(lp, col), (mir(lp, 0), col)]
    for p, cl in parts:
        q = rot(tr(p, cx, cy, s), ang, cx, cy)
        c.poly(q, cl, sm=2 if p is not tl else 1, sh=s * .12)
    if frill:
        for p in (lp, mir(lp, 0)):
            c.poly(rot(tr([(x * 1.12, y * 1.12) for x, y in p], cx, cy, s), ang, cx, cy),
                   frill, sm=2, sh=0)
            c.poly(rot(tr(p, cx, cy, s * .92), ang, cx, cy), col, sm=2, sh=s * .1)
    c.ell(cx, cy, s * .2, s * .24, knot or shade_of(col), rot=ang, sh=0)


def legs(c, sock=WHITE, shoe=(95, 55, 45), top=81, bottom=90.3, skin=SKIN, sep=4.8):
    for x in (50 - sep, 50 + sep):
        c.poly([(x - 2.3, top), (x + 2.3, top), (x + 2.0, bottom), (x - 2.0, bottom)], sock, sh=.8)
        c.ell(x + (x - 50) * .12, bottom + 1.3, 3.3, 2.0, shoe, sh=.7)


def skirt(c, col, top=63, hem=84, wt=9.5, wb=19, waves=8, amp=1.1, sh=2.0):
    pts = [(50 - wt, top), (50 + wt, top)]
    for i in range(waves + 1):
        t = i / waves
        pts.append((50 + wb - 2 * wb * t, hem + (amp if i % 2 else 0) - 1.2 * math.sin(math.pi * t) * 0))
    c.poly(pts, col, sm=2, sh=sh)
    return pts


def torso(c, col, top=53.5, bot=66, wt=7.3, wb=8.8, **kw):
    pts = [(50 - wt, top), (50 + wt, top), (50 + wb, bot), (50 - wb, bot)]
    c.poly(pts, col, sm=1, **kw)
    return pts


def arm(c, sx, sy, hx, hy, col, w=2.6, bend=0.0, hand=True, skin=SKIN, hr=2.4):
    mx, my = (sx + hx) / 2 + bend, (sy + hy) / 2
    c.strand([(sx, sy), (mx, my), (hx, hy)], w, col, w1=w * .75, sh=.9)
    if hand:
        c.ell(hx, hy + .6, hr, hr, skin, sh=.6)


def hand(c, x, y, r=2.4):
    c.ell(x, y, r, r, SKIN, sh=.6)


def wide_sleeve(c, sx, sy, hx, hy, col, flare=5.0, root=2.8, hand=True, cuff=None):
    dx, dy = hx - sx, hy - sy
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    pts = [(sx + nx * root, sy + ny * root), (sx - nx * root, sy - ny * root),
           (hx - nx * flare, hy - ny * flare), (hx + ux * 1.2, hy + uy * 1.2),
           (hx + nx * flare, hy + ny * flare)]
    if hand:
        c.ell(hx + ux * 1.5, hy + uy * 1.5 + .4, 2.4, 2.4, SKIN, sh=.6)
    c.poly(pts, col, sm=2, sh=1.4)
    if cuff:
        c.line([(hx - nx * flare * .8, hy - ny * flare * .8 - uy),
                (hx + nx * flare * .8, hy + ny * flare * .8 - uy)], cuff, .5)


def puff(c, x, y, col, r=4.0):
    c.ell(x, y, r, r * .85, col, sh=1.0)


def mob_cap(c, col, band=RED, bowcol=None, cy=21.5, rx=23.5, ry=10.5, bowx=66, frill=None):
    c.ell(50, cy, rx, ry, col, sh=2.0)
    fr = frill or col
    for i in range(13):
        t = -1 + 2 * i / 12
        c.ell(50 + t * (rx + .5), cy + ry * .5 + 2.6 * t * t, 2.5, 2.2, fr, sh=.6)
    if band:
        c.strand([(50 - rx * .92, cy + 3.5), (50, cy + 1.2), (50 + rx * .92, cy + 3.5)], 1.2, band, w1=1.2, sh=.5)
        bow(c, bowx, cy + 3, 5.2, bowcol or band, ang=12)


def gap(c, cx, cy, rx, ry, ang=0, eyes=5):
    """Yukari's sukima: dark slit with bows at the ends and eyes inside."""
    pts = [(cx - rx, cy)] + [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
                             for a in range(-170, -9, 10)] + [(cx + rx, cy)] + \
          [(cx + rx * math.cos(math.radians(a)), cy + ry * .8 * math.sin(math.radians(a))) for a in range(10, 171, 10)]
    pts = rot(pts, ang, cx, cy)
    c.fill(pts, (48, 14, 70), sh=1.5, sc=(90, 30, 110), sd=(0, -1), lc=(20, 5, 30))

    def inner(t):
        for i in range(eyes):
            u = -0.7 + 1.4 * i / max(1, eyes - 1)
            ex, ey = cx + u * rx, cy + (0.15 if i % 2 else -0.25) * ry
            ex, ey = rot([(ex, ey)], ang, cx, cy)[0]
            t.ell(ex, ey, 2.2, 1.1, (230, 60, 60), rot=ang, sh=0, ol=.2, lc=(20, 0, 10))
            t.ell(ex, ey, .5, .9, (20, 0, 10), rot=ang, sh=0, ol=0)
    c.clip(pts, inner)
    for sgn in (-1, 1):
        bx, by = rot([(cx + sgn * rx, cy)], ang, cx, cy)[0]
        bow(c, bx, by, 3.2, RED, ang=ang, tails=True)


def zig_paper(c, x, y, ln=11, w=2.4, side=1, col=WHITE):
    pts, q = [], []
    n = 4
    for i in range(n + 1):
        yy = y + ln * i / n
        off = side * (1.6 if i % 2 else 0)
        pts.append((x + off, yy))
        q.append((x + off + side * w, yy + 1.2))
    c.poly(pts + q[::-1], col, sh=.5, ol=.35)


def knife(c, x, y, ang, L=9, col=(215, 222, 235)):
    blade = [(0, -1.0), (L * .75, -1.0), (L, 0), (L * .75, .6), (0, .6)]
    handle = [(-3.4, -.75), (0, -.75), (0, .65), (-3.4, .65)]
    c.poly(rot(tr(handle, x, y), ang, x, y), (70, 60, 80), sh=.3, ol=.35)
    c.poly(rot(tr(blade, x, y), ang, x, y), col, sh=.4, ol=.35)
    c.poly(rot(tr([(-.4, -1.5), (.4, -1.5), (.4, 1.3), (-.4, 1.3)], x, y), ang, x, y), GOLD, sh=0, ol=.3)


def fan(c, cx, cy, r, ang, col, col2=WHITE, spread=110, ribs=7):
    pts = [(cx, cy)] + ell_pts(cx, cy, r, r, ang - spread / 2, ang + spread / 2, n=24)
    c.poly(pts, col, sh=.9)
    inner = [(cx, cy)] + ell_pts(cx, cy, r * .45, r * .45, ang - spread / 2, ang + spread / 2, n=12)
    c.poly(inner, col2, sh=0, ol=.3)
    for i in range(ribs + 1):
        a = math.radians(ang - spread / 2 + spread * i / ribs)
        c.line([(cx, cy), (cx + r * .97 * math.cos(a), cy + r * .97 * math.sin(a))], line_of(col), .22)


def butterfly(c, x, y, s, col, a=220):
    for sg in (-1, 1):
        c.ell(x + sg * s * .55, y - s * .25, s * .6, s * .45, col, rot=sg * 25, sh=0, ol=.25, a=a)
        c.ell(x + sg * s * .45, y + s * .35, s * .38, s * .3, col, rot=-sg * 25, sh=0, ol=.25, a=a)
    c.line([(x, y - s * .4), (x, y + s * .5)], line_of(col), .2)


def yinyang(c, x, y, r):
    c.ell(x, y, r, r, WHITE, sh=0, ol=.35, lc=DARK)
    half = [(x, y - r)] + ell_pts(x, y, r, r, 270, 450, n=20)[1:] + \
        ell_pts(x, y + r / 2, r / 2, r / 2, 90, 270, n=10)[::-1][1:] + ell_pts(x, y - r / 2, r / 2, r / 2, 90, -90, n=10)[1:]
    c.poly(half, (35, 25, 40), sh=0, ol=0)
    c.ell(x, y - r / 2, r * .17, r * .17, (35, 25, 40), sh=0, ol=0)
    c.ell(x, y + r / 2, r * .17, r * .17, WHITE, sh=0, ol=0)


def trigram(c, x, y, ang, pat, col=DARK, s=1.0):
    for i, broken in enumerate(pat):
        yy = -s * (i - 1) * 1.0
        segs = [(-1.4, -.3), (.3, 1.4)] if broken else [(-1.4, 1.4)]
        for a0, a1 in segs:
            c.line(rot([(x + a0 * s, y + yy), (x + a1 * s, y + yy)], ang, x, y), col, .28 * s)


# ============================================================ characters ==
def draw_reimu(size=384):
    c = Cv(size)
    H, R = (70, 40, 36), (218, 36, 48)
    bow(c, 50, 17.5, 15, R, frill=WHITE, knot=(190, 25, 40))
    hair_back(c, H, bottom=64, spread=19.5)
    # gohei (behind right hand)
    c.line([(66, 74), (81, 38)], (170, 120, 70), .8)
    c.ell(81, 38, 1.0, 1.0, GOLD, sh=0, ol=.3)
    zig_paper(c, 81.5, 38.5, 13, side=1)
    zig_paper(c, 80.5, 38.5, 12, side=-1)
    legs(c, sock=WHITE, shoe=(120, 70, 50))
    sk = skirt(c, R, top=63, hem=83.5)
    c.clip(sk, lambda t: t.line([(28, 82), (72, 82)], WHITE, .9))
    torso(c, R)
    c.poly([(43, 54.5), (57, 54.5), (50, 61)], WHITE, sh=.4)
    wide_sleeve(c, 42, 56, 31, 70, WHITE, flare=6.5, cuff=R)
    wide_sleeve(c, 58, 56, 67, 71, WHITE, flare=6.0, cuff=R)
    hand(c, 66.6, 73.2)
    bow(c, 50, 57, 3.8, GOLD, tails=True)
    head(c)
    face(c, (175, 45, 45))
    bangs(c, H)
    for sd in (-1, 1):
        x = 30.2 if sd < 0 else 69.8
        c.strand([(x, 36), (x - sd * .4, 48), (x + sd * .2, 62)], 2.3, H, w1=.4, sh=.9)
        c.poly([(x - 2.2, 49), (x + 2.2, 49), (x + 2, 55), (x - 2, 55)], WHITE, sh=.5, ol=.4)
        c.line([(x - 2.1, 50.6), (x + 2.1, 50.6)], R, .45)
        c.line([(x - 2.0, 53.4), (x + 2.0, 53.4)], R, .45)
    return c.done()


def draw_marisa(size=384):
    c = Cv(size)
    Y, BK = (250, 222, 110), (48, 42, 58)
    # broom behind
    c.line([(83, 36), (67, 84)], (150, 100, 60), .9)
    c.poly([(66, 80), (70, 81), (77, 94), (70, 95.5), (61, 94)], (220, 180, 90), sm=1, sh=1.0)
    for x in (64, 67, 70, 73):
        c.line([(68.3, 82), (x, 94)], (170, 130, 60), .2)
    c.poly([(65.7, 79.5), (70.5, 81), (70, 83), (65.3, 81.5)], (150, 60, 40), sh=0, ol=.3)
    hair_back(c, Y, bottom=72, spread=22, waves=7, amp=2.2)
    legs(c, sock=WHITE, shoe=(60, 45, 40))
    skirt(c, BK, top=63, hem=84.5, wb=20)
    ap = [(43, 63), (57, 63), (62, 82), (38, 82)]
    c.poly(ap, WHITE, sm=1, sh=1.0)
    torso(c, BK)
    c.poly([(44.5, 53.5), (55.5, 53.5), (50, 59)], WHITE, sh=.3)
    bow(c, 50, 56.5, 3.2, WHITE)
    puff(c, 40, 57, WHITE)
    puff(c, 60, 57, WHITE)
    arm(c, 39, 58, 32, 70, WHITE, w=2.4)
    arm(c, 61, 58, 71, 69, WHITE, w=2.4)
    head(c)
    face(c, (225, 170, 40), mo='open')
    bangs(c, Y, tips=[(71, 50), (64.5, 37), (59, 40), (54, 36), (49.5, 42), (45, 36.5),
                      (40, 40), (35, 37), (29, 50)])
    # braid on the left
    for i in range(5):
        c.ell(29.5 + (i % 2) * .6, 40 + i * 3.6, 2.4, 2.2, Y, sh=.6)
    bow(c, 29.8, 57.5, 2.6, WHITE)
    # witch hat
    c.ell(50, 23.5, 31, 6.4, BK, sh=1.2)
    cone = [(35.5, 23), (38, 12), (47, 5.5), (56, 5.0), (63, 7.5), (58, 10), (62, 23)]
    c.poly(cone, BK, sm=2, sh=1.8)
    c.strand([(35.8, 20.5), (49, 22.2), (62, 20.5)], 1.7, WHITE, w1=1.7, sh=.5)
    bow(c, 60.5, 20.5, 6.5, WHITE, ang=-10)
    return c.done()


def draw_rumia(size=384):
    c = Cv(size)
    Y, BK, R = (248, 225, 120), (48, 42, 58), (210, 35, 45)
    c.ell(50, 56, 44, 40, (40, 20, 55), sh=0, ol=0, a=70)
    hair_back(c, Y, bottom=56, spread=21.5)
    legs(c, sock=(60, 50, 70), shoe=(110, 50, 45))
    skirt(c, BK, top=63, hem=85, wb=18)
    torso(c, WHITE)
    c.poly([(42, 54), (45.5, 54), (47, 66), (41, 66)], BK, sh=.8)
    c.poly([(58, 54), (54.5, 54), (53, 66), (59, 66)], BK, sh=.8)
    bow(c, 50, 56.5, 3.8, R)
    for sg in (-1, 1):
        c.strand([(50 + sg * 7, 56.5), (50 + sg * 20, 55), (50 + sg * 34, 54.5)], 2.3, WHITE, w1=2.0, sh=.9)
        hand(c, 50 + sg * 35.5, 54.8)
    head(c)
    face(c, (205, 40, 45), mo='open')
    bangs(c, Y, tips=[(70.5, 47), (64, 37), (58, 39), (53, 36), (49, 41), (44, 36),
                      (39.5, 39.5), (35, 37), (29.5, 47)])
    bow(c, 67, 23, 4.8, R, ang=15)
    return c.done()


def draw_cirno(size=384):
    c = Cv(size)
    B, BL, IC = (110, 185, 245), (55, 115, 225), (190, 240, 255)
    for sg in (-1, 1):
        for (ang, L) in ((-40, 30), (-12, 34), (18, 27)):
            a = math.radians(ang)
            bx, by = 50 + sg * 6, 58
            tx, ty = bx + sg * L * math.cos(a), by + L * math.sin(a) - 4
            mx, my = bx + (tx - bx) * .55, by + (ty - by) * .55
            nx, ny = -(ty - by) / L * 4.2, (tx - bx) / L * 4.2
            c.poly([(bx, by), (mx + nx, my + ny), (tx, ty), (mx - nx, my - ny)], IC, sh=1.6,
                   a=240, lc=(60, 120, 190), sc=(140, 205, 245))
            c.line([(bx + (tx - bx) * .2, by + (ty - by) * .2), (bx + (tx - bx) * .8, by + (ty - by) * .8)],
                   WHITE, .3)
    hair_back(c, B, bottom=57, spread=21)
    legs(c, sock=WHITE, shoe=(60, 90, 170))
    skirt(c, BL, top=63, hem=83, wb=19.5)
    torso(c, BL)
    c.poly([(44, 53.5), (56, 53.5), (54, 60), (46, 60)], WHITE, sh=.4)
    bow(c, 50, 57, 3.6, RED)
    puff(c, 40, 57, WHITE)
    puff(c, 60, 57, WHITE)
    arm(c, 39, 58, 31.5, 69, SKIN, w=2.1)
    arm(c, 61, 58, 69, 67, SKIN, w=2.1)
    head(c)
    face(c, (60, 130, 230), mo='open')
    bangs(c, B, tips=[(70.5, 48), (64, 36.5), (60, 40), (55, 36), (50, 42.5), (45, 36),
                      (40, 40), (36, 36.5), (29.5, 48)])
    bow(c, 50, 16.5, 12.5, BL, knot=(40, 90, 200))
    return c.done()


def draw_daiyousei(size=384):
    c = Cv(size)
    G, BL, YL = (110, 195, 110), (70, 110, 205), (250, 215, 70)
    for sg in (-1, 1):
        c.ell(50 + sg * 20, 50, 13, 6.5, (225, 255, 235), rot=sg * -30, sh=0, a=150, lc=(110, 170, 130))
        c.ell(50 + sg * 17, 64, 9, 4.5, (225, 255, 235), rot=sg * 25, sh=0, a=150, lc=(110, 170, 130))
    hair_back(c, G, bottom=57, spread=20.5)
    legs(c, sock=WHITE, shoe=(80, 60, 140))
    skirt(c, BL, top=63, hem=83, wb=18.5)
    torso(c, BL)
    c.poly([(44, 53.5), (56, 53.5), (54, 60), (46, 60)], WHITE, sh=.4)
    bow(c, 50, 57, 3.6, YL)
    puff(c, 40, 57, WHITE)
    puff(c, 60, 57, WHITE)
    arm(c, 39, 58, 33, 69, SKIN, w=2.1)
    arm(c, 61, 58, 67, 69, SKIN, w=2.1)
    head(c)
    face(c, (60, 150, 120))
    bangs(c, G)
    # side ponytail
    c.strand([(33, 24), (20, 34), (22, 56)], 3.2, G, w1=.3, bulge=1.5, sh=1.2)
    bow(c, 32, 24, 4.8, YL, ang=-30)
    return c.done()


def draw_meiling(size=384):
    c = Cv(size)
    R, GR = (220, 58, 55), (70, 165, 95)
    hair_back(c, R, bottom=86, spread=20, waves=7, amp=2.0)
    legs(c, sock=WHITE, shoe=(60, 50, 50), top=83)
    dress = [(40.5, 63), (59.5, 63), (65, 86), (35, 86)]
    c.poly(dress, GR, sm=1, sh=2.0)
    for sg in (-1, 1):
        c.poly([(50 + sg * 14, 76), (50 + sg * 15.5, 86.2), (50 + sg * 12.5, 86.2)], WHITE, sh=0, ol=.35)
    torso(c, GR)
    c.poly([(45, 53.5), (55, 53.5), (50, 58)], WHITE, sh=0)
    c.line([(47, 57), (53, 57)], (60, 50, 50), .4)
    puff(c, 40, 57, WHITE)
    puff(c, 60, 57, WHITE)
    arm(c, 39, 58, 33, 69, SKIN, w=2.2)
    arm(c, 61, 58, 67, 69, SKIN, w=2.2)
    c.ell(33, 69.8, 2.7, 2.6, SKIN, sh=.6)
    head(c)
    face(c, (90, 120, 170))
    bangs(c, R)
    for sg in (-1, 1):
        x = 50 + sg * 20
        for i in range(5):
            c.ell(x + sg * (i % 2) * .5, 40 + i * 3.5, 2.2, 2.1, R, sh=.6)
        bow(c, x, 57.5, 2.8, BLACK)
    # beret with star
    c.ell(50, 21.5, 20.5, 7.5, GR, sh=1.5)
    c.poly(chaikin([(30, 24), (70, 24), (69, 26.5), (31, 26.5)], 1), shade_of(GR), sh=0)
    c.poly(star_pts(50, 21, 4.6, 2.0), GOLD, sh=.4)
    return c.done()


def draw_patchouli(size=384):
    c = Cv(size)
    P, GW, ST = (165, 115, 200), (238, 215, 240), (205, 160, 220)
    hair_back(c, P, bottom=86, spread=20, waves=6, amp=1.6)
    for x, col in ((32, RED), (68, (70, 110, 220))):
        bow(c, x, 85, 3.0, col)
    legs(c, sock=GW, shoe=(150, 110, 160), top=84)
    gown = [(40.5, 58), (59.5, 58), (66, 88), (34, 88)]
    c.poly(gown, GW, sm=1, sh=2.2)
    c.clip(gown, lambda t: [t.line([(x, 55), (x + (x - 50) * .25, 92)], ST, .55) for x in range(30, 72, 4)])
    torso(c, GW)
    c.clip(torso(c, GW), lambda t: [t.line([(x, 50), (x, 70)], ST, .55) for x in range(30, 72, 4)])
    wide_sleeve(c, 42, 56, 45, 69, GW, flare=4.5, hand=False, cuff=ST)
    wide_sleeve(c, 58, 56, 55, 69, GW, flare=4.5, hand=False, cuff=ST)
    # book
    c.poly([(41, 64), (59, 64), (59, 76), (41, 76)], (150, 55, 60), sh=1.0)
    c.poly([(42, 64.8), (58, 64.8), (58, 66.5), (42, 66.5)], (245, 235, 210), sh=0, ol=.3)
    c.poly(star_pts(50, 71, 3.0, 1.3), GOLD, sh=0, ol=.3)
    hand(c, 42, 70)
    hand(c, 58, 70)
    head(c)
    face(c, (150, 80, 170), style='half', mo='flat')
    bangs(c, P)
    for sg in (-1, 1):
        c.strand([(50 + sg * 19.5, 36), (50 + sg * 21, 50), (50 + sg * 20, 64)], 2.3, P, w1=.5, sh=.9)
    mob_cap(c, (245, 225, 245), band=None, cy=21, frill=(250, 240, 250))
    bow(c, 36, 25, 3.8, RED, ang=-10)
    bow(c, 64, 25, 3.8, (70, 110, 220), ang=10)
    c.poly(ell_pts(50, 21, 4.0, 4.0, 60, 300, n=20) + ell_pts(52.2, 20.3, 3.2, 3.2, 290, 70, n=20)[::-1],
           GOLD, sh=0, ol=.35)
    return c.done()


def draw_sakuya(size=384):
    c = Cv(size)
    S, BL, GR = (215, 220, 235), (60, 85, 175), (70, 160, 100)
    hair_back(c, S, bottom=55, spread=20.5)
    legs(c, sock=WHITE, shoe=(60, 50, 60))
    skirt(c, BL, top=63, hem=83.5, wb=19)
    c.poly([(43, 63), (57, 63), (61, 81), (39, 81)], WHITE, sm=1, sh=1.0)
    for i in range(9):
        c.ell(39.5 + i * 2.6, 81.5, 1.5, 1.2, WHITE, sh=0, ol=.3)
    torso(c, BL)
    c.poly([(44.5, 53.5), (55.5, 53.5), (54, 60), (46, 60)], WHITE, sh=.3)
    bow(c, 50, 56.5, 3.2, GR)
    puff(c, 40, 57, BL)
    puff(c, 60, 57, BL)
    arm(c, 39, 58, 31, 67, WHITE, w=2.1, hand=False)
    arm(c, 61, 58, 69, 67, WHITE, w=2.1, hand=False)
    for sg, x in ((-1, 31), (1, 69)):
        for a in (-35, -70, -105):
            ang = a if sg > 0 else -180 - a
            knife(c, x, 67.5, ang, L=13)
        hand(c, x, 68)
    head(c)
    face(c, (70, 105, 205), mo='flat')
    bangs(c, S)
    for sg in (-1, 1):
        x = 50 + sg * 20.2
        for i in range(5):
            c.ell(x + sg * (i % 2) * .5, 40 + i * 3.3, 2.1, 2.0, S, sh=.6)
        bow(c, x, 56.5, 2.8, GR)
    # maid headdress
    c.strand([(31.5, 25), (50, 15.5), (68.5, 25)], 1.5, WHITE, w1=1.5, sh=.3)
    for i in range(9):
        a = math.radians(200 + 140 * i / 8)
        c.ell(50 + 20 * math.cos(a), 32 + 17 * math.sin(a), 2.3, 2.3, WHITE, sh=.5)
    return c.done()


def bat_wing(c, sg, col):
    pts = [(54, 58), (66, 44), (80, 36), (93, 32), (89, 44), (90, 52), (83, 50), (82, 60), (75, 56),
           (72, 65), (64, 61), (57, 66)]
    pts = pts if sg > 0 else mir(pts)
    c.poly(pts, col, sh=1.4)
    for p in ((80, 36), (89, 44), (82, 60)):
        q = p if sg > 0 else (100 - p[0], p[1])
        c.line([(50 + sg * 5, 57), q], line_of(col), .3)


def draw_remilia(size=384):
    c = Cv(size)
    H, PK = (160, 175, 220), (248, 208, 222)
    bat_wing(c, 1, (70, 45, 85))
    bat_wing(c, -1, (70, 45, 85))
    hair_back(c, H, bottom=56, spread=21)
    legs(c, sock=WHITE, shoe=(170, 40, 60))
    skirt(c, PK, top=63, hem=83, wb=19.5)
    for i in range(11):
        c.ell(31.5 + i * 3.7, 83.5, 2.0, 1.5, WHITE, sh=0, ol=.3)
    c.line([(40, 63.5), (60, 63.5)], RED, .6)
    torso(c, PK)
    bow(c, 50, 57, 3.8, RED)
    puff(c, 40, 57, PK)
    puff(c, 60, 57, PK)
    arm(c, 39, 58, 33, 68, SKIN, w=2.0)
    arm(c, 61, 58, 67, 68, SKIN, w=2.0)
    head(c)
    face(c, (205, 30, 45), mo='fang')
    bangs(c, H)
    mob_cap(c, (252, 232, 240), band=RED, cy=21, bowx=65)
    return c.done()


def draw_flandre(size=384):
    c = Cv(size)
    Y, R = (250, 225, 120), (215, 38, 50)
    cols = [(255, 95, 110), (255, 175, 80), (250, 235, 100), (130, 230, 130), (100, 200, 255),
            (130, 130, 250), (215, 120, 250)]
    for sg in (-1, 1):
        br = [(50 + sg * 4, 58), (50 + sg * 20, 34), (50 + sg * 41, 44)]
        c.line([bez(br, i / 20) for i in range(21)], (55, 40, 55), .75)
        for i, col in enumerate(cols):
            x, y = bez(br, .22 + .78 * i / 6)
            L = 5.5 + (i % 3) * 1.2
            c.poly([(x, y), (x + 1.8, y + L * .4), (x, y + L), (x - 1.8, y + L * .4)], col, sh=.8, a=235)
            c.ell(x, y, .6, .6, (55, 40, 55), sh=0, ol=0)
    hair_back(c, Y, bottom=55, spread=20.5)
    c.strand([(66, 26), (79, 34), (76, 55)], 3.0, Y, w1=.3, bulge=1.5, sh=1.1)
    legs(c, sock=WHITE, shoe=(160, 40, 50))
    skirt(c, R, top=63, hem=83, wb=19.5)
    for i in range(11):
        c.ell(31.5 + i * 3.7, 83.5, 2.0, 1.5, WHITE, sh=0, ol=.3)
    torso(c, R)
    c.poly([(44.5, 53.5), (55.5, 53.5), (54, 60), (46, 60)], WHITE, sh=.3)
    bow(c, 50, 57, 3.4, GOLD)
    puff(c, 40, 57, WHITE)
    puff(c, 60, 57, WHITE)
    arm(c, 39, 58, 33, 68, SKIN, w=2.0)
    arm(c, 61, 58, 67, 68, SKIN, w=2.0)
    head(c)
    face(c, (215, 30, 40), mo='fang')
    bangs(c, Y)
    mob_cap(c, (252, 238, 242), band=RED, cy=21, bowx=65)
    return c.done()


def draw_shanghai(c, x, y, s=1.0):
    Y, BL = (250, 222, 120), (70, 110, 210)
    def P(pts):
        return tr(pts, x, y, s)
    c.poly(P([(-3, 3), (3, 3), (5, 11), (-5, 11)]), BL, sm=1, sh=.8, ol=.35)
    c.poly(P([(-2, 4), (2, 4), (3, 10), (-3, 10)]), WHITE, sm=1, sh=.4, ol=.3)
    c.ell(x, y - 2 * s, 5.2 * s, 5 * s, Y, sh=.8, ol=.35)
    c.ell(x, y - .8 * s, 4 * s, 3.7 * s, SKIN, sh=.4, ol=.35)
    c.poly(P([(-4.7, -3), (-4, -6), (0, -7.5), (4, -6), (4.7, -3), (2, -4), (0, -2.8), (-2, -4)]), Y, sm=1, sh=.5, ol=.35)
    for sg in (-1, 1):
        c.ell(x + sg * 1.6 * s, y - .4 * s, .8 * s, 1.1 * s, (60, 90, 190), sh=0, ol=.2, lc=DARK)
    bow(c, x, y - 7 * s, 3.2 * s, RED, tails=False)
    c.line([P([(4.5, 5)])[0], P([(8, -8)])[0]], (190, 195, 210), .4)


def draw_alice(size=384):
    c = Cv(size)
    Y, BL = (248, 220, 130), (75, 115, 210)
    hair_back(c, Y, bottom=56, spread=21)
    legs(c, sock=WHITE, shoe=(90, 60, 50))
    skirt(c, BL, top=63, hem=84, wb=19.5)
    c.poly([(44, 63), (56, 63), (59, 80), (41, 80)], WHITE, sm=1, sh=.9)
    torso(c, BL)
    c.poly(chaikin([(38, 55), (62, 55), (65, 61), (50, 63.5), (35, 61)], 2), WHITE, sh=1.0)
    bow(c, 50, 56.5, 3.8, RED)
    arm(c, 38.5, 60, 32, 69, BL, w=2.1)
    arm(c, 61.5, 60, 69, 66, BL, w=2.1)
    head(c)
    face(c, (70, 110, 210))
    bangs(c, Y)
    for sg in (-1, 1):
        c.strand([(50 + sg * 19.5, 36), (50 + sg * 20.5, 46), (50 + sg * 19, 56)], 2.4, Y, w1=.5, sh=.9)
    c.strand([(30.5, 29), (50, 17), (69.5, 29)], 1.3, RED, w1=1.3, sh=.4)
    bow(c, 66, 22.5, 4.2, RED, ang=25)
    draw_shanghai(c, 82, 53, 1.4)
    return c.done()


def draw_youmu(size=384):
    c = Cv(size)
    S, GR = (228, 230, 238), (80, 155, 95)
    # ghost half
    tail = [(18, 32), (8, 40), (14, 52), (8, 60)]
    c.strand(tail, 7, (238, 245, 255), w1=.3, sh=1.3, a=230, lc=(140, 170, 210), sc=(200, 215, 240))
    # katana (Roukanken) held up in right hand
    blade = rot(tr([(0, -.8), (24, -.5), (27, .4), (0, .8)], 69, 68), -62, 69, 68)
    c.poly(blade, (220, 228, 240), sh=.5, ol=.4)
    c.poly(rot(tr([(-7, -.9), (0, -.9), (0, .9), (-7, .9)], 69, 68), -62, 69, 68), (40, 40, 55), sh=0, ol=.35)
    c.ell(69, 68, 1.2, 2.4, GOLD, rot=-62, sh=0, ol=.3)
    hair_back(c, S, bottom=55, spread=20.5)
    legs(c, sock=WHITE, shoe=(60, 50, 60))
    skirt(c, GR, top=63, hem=83, wb=19)
    torso(c, WHITE)
    c.poly([(42, 55), (47, 55), (46, 66), (41.2, 66)], GR, sh=.8)
    c.poly([(58, 55), (53, 55), (54, 66), (58.8, 66)], GR, sh=.8)
    bow(c, 50, 56.5, 3.0, BLACK)
    # short sword (Hakurouken) at the hip
    c.poly(rot(tr([(-1, -.6), (14, -.6), (14, .6), (-1, .6)], 36, 66), 25, 36, 66), (40, 40, 55), sh=0, ol=.35)
    puff(c, 40, 57, WHITE)
    puff(c, 60, 57, WHITE)
    arm(c, 39, 58, 33, 68, SKIN, w=2.0)
    arm(c, 61, 58, 67.5, 67, SKIN, w=2.0)
    c.ell(18, 32, 9, 8, (240, 246, 255), sh=1.4, a=235, lc=(140, 170, 210), sc=(200, 215, 240))
    head(c)
    face(c, (80, 140, 175), mo='flat')
    bangs(c, S, tips=[(70.5, 50), (64, 37), (59, 40), (54, 36), (50, 41), (45.5, 36),
                      (40.5, 40), (36, 37), (29.5, 50)])
    c.strand([(31, 27), (50, 18.2), (69, 27)], 1.2, BLACK, w1=1.2, sh=0)
    bow(c, 34, 25, 4.6, BLACK, ang=-35)
    return c.done()


def draw_yuyuko(size=384):
    c = Cv(size)
    P, KB, OB = (248, 165, 190), (170, 205, 240), (60, 80, 160)
    for (x, y, s) in ((13, 30, 3.5), (86, 44, 3.0), (15, 72, 2.6), (88, 78, 3.2)):
        butterfly(c, x, y, s, (255, 130, 190))
    hair_back(c, P, bottom=58, spread=22, waves=8, amp=2.2)
    legs(c, sock=WHITE, shoe=(80, 80, 120), top=84)
    kim = [(41, 60), (59, 60), (65, 87), (35, 87)]
    c.poly(kim, KB, sm=1, sh=2.2)
    c.clip(kim, lambda t: [t.poly(star_pts(x, y, 1.6, .8), (255, 200, 220), sh=0, ol=.2)
                           for x, y in ((40, 70), (58, 74), (46, 82), (60, 84), (38, 80), (53, 66))])
    torso(c, KB)
    c.poly([(44, 53.5), (50, 60), (56, 53.5), (54, 53.5), (50, 57.5), (46, 53.5)], WHITE, sh=0, ol=.3)
    c.poly([(40.5, 61), (59.5, 61), (60, 65), (40, 65)], OB, sh=.6)
    wide_sleeve(c, 42, 56, 32, 70, KB, flare=6.5, cuff=WHITE)
    wide_sleeve(c, 58, 56, 66, 71, KB, flare=6.0, cuff=WHITE)
    fan(c, 69, 71, 9, -70, (230, 120, 170), (250, 210, 230))
    hand(c, 68, 72.5)
    head(c)
    face(c, (225, 80, 120), mo='smile')
    bangs(c, P, tips=[(71, 48), (64.5, 36.5), (59, 39.5), (54, 36), (50, 41.5), (45.5, 36),
                      (41, 39.5), (35.5, 36.5), (29, 48)])
    for sg in (-1, 1):
        c.ell(50 + sg * 21, 48, 3.2, 4.0, P, sh=.8)
        c.ell(50 + sg * 21.5, 54, 2.6, 3.0, P, sh=.7)
    mob_cap(c, (225, 238, 252), band=None, cy=20.5, frill=(235, 245, 255))
    c.poly([(45, 25), (55, 25), (50, 33)], WHITE, sh=0, ol=.35)
    c.line(ell_pts(50, 27.5, 1.6, 1.6, 0, 300, n=12), RED, .35)
    return c.done()


def draw_chen(size=384):
    c = Cv(size)
    H, R, GR, BK = (175, 95, 55), (215, 42, 48), (70, 160, 95), (55, 38, 40)
    for sg in (-1, 1):
        c.strand([(50, 74), (50 + sg * 26, 88), (50 + sg * 34, 70), (50 + sg * 29, 56)], 1.8, BK, w1=1.2, sh=.7)
        c.ell(50 + sg * 29.2, 56.2, 1.3, 1.3, BK, sh=0)
    hair_back(c, H, bottom=55, spread=20.5)
    for sg in (-1, 1):
        ear = [(50 + sg * 10, 22), (50 + sg * 21, 8), (50 + sg * 20, 26)]
        c.poly(ear, BK, sh=.8)
        c.poly([(50 + sg * 12.5, 22), (50 + sg * 19.5, 12), (50 + sg * 18.5, 24)], (240, 170, 180), sh=0, ol=0)
    legs(c, sock=WHITE, shoe=(70, 45, 40))
    skirt(c, R, top=63, hem=83, wb=19.5)
    for i in range(11):
        c.ell(31.5 + i * 3.7, 83.5, 2.0, 1.5, WHITE, sh=0, ol=.3)
    torso(c, R)
    c.poly(chaikin([(41, 54.5), (59, 54.5), (57, 58.5), (50, 60), (43, 58.5)], 1), WHITE, sh=.4)
    bow(c, 50, 58, 3.4, GOLD)
    puff(c, 40, 57, R)
    puff(c, 60, 57, R)
    arm(c, 39, 58, 32, 66, WHITE, w=2.0)
    arm(c, 61, 58, 68, 66, WHITE, w=2.0)
    head(c)
    face(c, (195, 55, 40), mo='cat')
    bangs(c, H, tips=[(70, 47), (63.5, 37), (58.5, 39.5), (54, 36), (50, 41.5), (46, 36),
                      (41.5, 39.5), (36.5, 37), (30, 47)])
    # small green mob hat between the ears
    c.ell(50, 23, 14, 7.5, GR, sh=1.2)
    for i in range(7):
        t = -1 + 2 * i / 6
        c.ell(50 + t * 13, 28 + 1.5 * t * t, 2.1, 1.8, GR, sh=.4)
    c.ell(50, 22, 2.3, 2.3, GOLD, sh=0, ol=.35)
    c.ell(31, 32, 1.4, 1.4, GOLD, sh=0, ol=.3)
    return c.done()


def draw_ran(size=384):
    c = Cv(size)
    Y, BL, TL = (250, 215, 105), (65, 95, 185), (250, 200, 90)
    angs = [-122, -58, -144, -36, -166, -14, -188, 8, -210]
    for ang in angs:
        a = math.radians(ang)
        L = 41 + 10 * max(0, 1 - abs(ang + 90) / 70)
        tx, ty = 50 + L * math.cos(a), 64 + L * .95 * math.sin(a)
        bend = .35 if ang < -90 else -.35
        mx, my = 50 + L * .5 * math.cos(a + bend), 66 + L * .5 * math.sin(a + bend)
        ctrl = [(50, 72), (mx, my), (tx, ty)]
        c.fill(strand_pts(ctrl, 1.5, 0, 6.8, peak=.68, n=36), TL, sh=1.6)
        c.fill(strand_pts(ctrl, 1.5, 0, 6.8, t0=.84, peak=.68), WHITE, sh=.7)
        c.line([bez(ctrl, .45 + .3 * j / 6) for j in range(7)], shade_of(TL), .25)
    hair_back(c, Y, bottom=56, spread=20.5)
    legs(c, sock=WHITE, shoe=(80, 60, 50), top=84)
    dress = [(41, 60), (59, 60), (65, 87), (35, 87)]
    c.poly(dress, WHITE, sm=1, sh=2.0)
    torso(c, WHITE)
    tab = [(44, 55), (56, 55), (57.5, 83), (42.5, 83)]
    c.poly(tab, BL, sm=1, sh=1.4)
    for j, pat in enumerate(((0, 0, 0), (1, 0, 1), (0, 1, 1))):
        trigram(c, 50, 64 + j * 6, 0, pat, col=WHITE, s=1.1)
    wide_sleeve(c, 42, 56, 45.5, 69, WHITE, flare=4.8, hand=False, cuff=BL)
    wide_sleeve(c, 58, 56, 54.5, 69, WHITE, flare=4.8, hand=False, cuff=BL)
    head(c)
    face(c, (215, 160, 40))
    bangs(c, Y)
    for sg in (-1, 1):
        c.strand([(50 + sg * 19.5, 36), (50 + sg * 21, 46), (50 + sg * 19.5, 55)], 2.3, Y, w1=.4, sh=.9)
    # pillow hat with two points + ofuda
    hat = [(29, 29), (31, 17), (38, 7), (43, 15), (50, 14), (57, 15), (62, 7), (69, 17), (71, 29), (50, 27)]
    c.poly(hat, WHITE, sm=2, sh=1.8)
    for x in (37, 63):
        c.poly([(x - 1.6, 21), (x + 1.6, 21), (x + 1.6, 32), (x - 1.6, 32)], (250, 245, 225), sh=.3, ol=.3)
        c.line([(x, 23), (x, 30)], RED, .35)
        c.line([(x - .9, 25), (x + .9, 25)], RED, .25)
    return c.done()


def draw_yukari(size=384):
    c = Cv(size)
    Y, PU, TB, PK = (252, 222, 130), (130, 72, 175), (245, 240, 250), (250, 225, 238)
    gap(c, 50, 60, 42, 8, ang=-22, eyes=6)
    # parasol behind, over the right shoulder
    c.line([(67, 70), (78, 20)], (120, 80, 130), .6)
    cx, cy, r = 76, 21, 18
    canopy = ell_pts(cx, cy, r, r * .78, 180, 360, n=30, rot=16)
    rim = []
    base = ell_pts(cx, cy, r, r * .78, 0, 180, n=30, rot=16)
    ends = (canopy[-1], canopy[0])
    ax, ay = ends[0]
    bx, by = ends[1]
    for i in range(9):
        t0, t1 = i / 9, (i + .5) / 9
        rim.append((ax + (bx - ax) * t0, ay + (by - ay) * t0))
        rim.append((ax + (bx - ax) * t1 + 1.3, ay + (by - ay) * t1 + 2.6))
    c.poly(canopy + rim, PK, sh=2.4, sc=(235, 175, 210))
    for i in range(1, 6):
        c.line([canopy[i * 5], (cx + 1, cy + 2)], mix(PK, PU, .5), .3)
    for i in range(18):
        t = (i + .5) / 18
        c.ell(ax + (bx - ax) * t + .6, ay + (by - ay) * t + 1.6, 1.1, 1.1, WHITE, sh=0, ol=.25)
    c.ell(canopy[15][0], canopy[15][1] - .5, 1.1, 1.3, PU, sh=0, ol=.3)
    # hair: long and wavy
    hair_back(c, Y, bottom=88, spread=23, waves=9, amp=2.4)
    for x0 in (33, 40, 60, 67):
        c.line([bez([(x0, 50), (x0 + (x0 - 50) * .3 + 2, 68), (x0 + (x0 - 50) * .15, 84)], t / 10)
                for t in range(11)], shade_of(Y), .25)
    for x in (31, 69):
        bow(c, x, 86.5, 2.7, RED)
    legs(c, sock=WHITE, shoe=(110, 60, 140), top=85)
    dress = [(40.5, 60), (59.5, 60), (67, 88), (33, 88)]
    c.poly(dress, PU, sm=1, sh=2.3)
    for i in range(13):
        c.ell(33.5 + i * 2.75, 88.3, 1.7, 1.3, WHITE, sh=0, ol=.3)
    torso(c, PU)
    tab = [(43.5, 55), (56.5, 55), (60, 84), (40, 84)]
    c.poly(tab, TB, sm=1, sh=1.5)
    c.line([(43.5, 56), (40.2, 83.5)], RED, .3)
    c.line([(56.5, 56), (59.8, 83.5)], RED, .3)
    yinyang(c, 50, 69, 3.6)
    for k, pat in enumerate(((0, 0, 0), (1, 1, 1), (0, 1, 0), (1, 0, 1), (0, 0, 1), (1, 1, 0))):
        ang = -90 + k * 60
        tx, ty = 50 + 7.2 * math.cos(math.radians(ang)), 69 + 8.4 * math.sin(math.radians(ang))
        trigram(c, tx, ty, ang + 90, pat, col=PU, s=.75)
    c.poly(chaikin([(40, 55), (60, 55), (61.5, 58.8), (50, 60.2), (38.5, 58.8)], 1), WHITE, sh=.5)
    bow(c, 50, 57.5, 3.3, RED)
    puff(c, 40, 57.5, PU, r=4.3)
    puff(c, 60, 57.5, PU, r=4.3)
    arm(c, 38.5, 58.5, 30.5, 66, PU, w=2.2, hand=False)
    arm(c, 61.5, 58.5, 67.5, 68.5, PU, w=2.2, hand=False)
    for x, y in ((30.5, 66), (67.5, 68.5)):
        for i in range(4):
            c.ell(x - 1.8 + i * 1.2, y + .6, 1.1, 1.0, WHITE, sh=0, ol=.25)
    hand(c, 30.3, 67.5)
    fan(c, 30, 66, 10.5, -105, (120, 60, 160), (245, 230, 250), spread=115)
    c.ell(30, 66, .8, .8, GOLD, sh=0, ol=.2)
    hand(c, 67.6, 70.2)
    head(c)
    face(c, (165, 85, 185), mo='sly')
    bangs(c, Y, tips=[(71, 51), (65, 37), (60, 40.5), (55, 36), (50.5, 42), (46, 36),
                      (41, 40.5), (35.5, 37), (29, 51)])
    for sg in (-1, 1):
        c.strand([(50 + sg * 19.5, 36), (50 + sg * 22.5, 52), (50 + sg * 20, 67)], 2.6, Y, w1=.5, sh=1.0)
    mob_cap(c, (250, 240, 250), band=RED, cy=20.5, rx=24, bowx=36)
    return c.done()


CHARACTERS = {
    "yukari": {"name": "Yukari Yakumo", "color": (150, 80, 190),
               "spell": 'Boundary "Border of Prompt and Completion"', "draw": draw_yukari},
    "ran": {"name": "Ran Yakumo", "color": (250, 200, 80),
            "spell": 'Shikigami "Nine-Tailed Mixture of Experts"', "draw": draw_ran},
    "chen": {"name": "Chen", "color": (215, 45, 50),
             "spell": 'Oni Sign "Shikigami Subagent Dash"', "draw": draw_chen},
    "reimu": {"name": "Reimu Hakurei", "color": (230, 40, 50),
              "spell": 'Dream Sign "Fantasy Seal of 1M Tokens"', "draw": draw_reimu},
    "marisa": {"name": "Marisa Kirisame", "color": (245, 205, 70),
               "spell": 'Love Sign "Master Sparse Attention"', "draw": draw_marisa},
    "rumia": {"name": "Rumia", "color": (60, 40, 80),
              "spell": 'Darkness Sign "Dark Side of the Context"', "draw": draw_rumia},
    "cirno": {"name": "Cirno", "color": (110, 190, 250),
              "spell": 'Freeze Sign "Perfect Frozen Weights"', "draw": draw_cirno},
    "daiyousei": {"name": "Daiyousei", "color": (110, 200, 110),
                  "spell": 'Fairy Sign "Small Model, Big Heart"', "draw": draw_daiyousei},
    "meiling": {"name": "Hong Meiling", "color": (70, 170, 95),
                "spell": 'Gate Sign "Rainbow Rate-Limit Gatekeeper"', "draw": draw_meiling},
    "patchouli": {"name": "Patchouli Knowledge", "color": (170, 110, 210),
                  "spell": 'Earth & Water "Retrieval-Augmented Library"', "draw": draw_patchouli},
    "sakuya": {"name": "Sakuya Izayoi", "color": (170, 180, 210),
               "spell": 'Time Sign "Private Square: KV Cache Freeze"', "draw": draw_sakuya},
    "remilia": {"name": "Remilia Scarlet", "color": (200, 30, 60),
                "spell": 'Destiny "Sampled Fate at Temperature 0"', "draw": draw_remilia},
    "flandre": {"name": "Flandre Scarlet", "color": (240, 60, 70),
                "spell": 'Taboo "Four of a Kind: Parallel Subagents"', "draw": draw_flandre},
    "alice": {"name": "Alice Margatroid", "color": (80, 120, 215),
              "spell": 'Puppet Sign "Shanghai Tool-Use Doll"', "draw": draw_alice},
    "youmu": {"name": "Youmu Konpaku", "color": (100, 175, 110),
              "spell": 'Human Sign "Present Token Slash"', "draw": draw_youmu},
    "yuyuko": {"name": "Yuyuko Saigyouji", "color": (250, 150, 190),
               "spell": 'Deathly Sign "Checkpoint Resurrection"', "draw": draw_yuyuko},
}


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.makedirs(os.path.join(root, "build"), exist_ok=True)
    cell, cols = 320, 4
    keys = list(CHARACTERS)
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGBA", (cell * cols, cell * rows), (0, 0, 0, 255))
    for i, k in enumerate(keys):
        ch = CHARACTERS[k]
        assert len(ch["spell"]) < 48, (k, len(ch["spell"]))
        spr = ch["draw"](cell)
        assert spr.size == (cell, cell) and spr.mode == "RGBA"
        x, y = (i % cols) * cell, (i // cols) * cell
        bg = Image.new("RGBA", (cell, cell), mix(ch["color"], (235, 235, 240), .82) + (255,))
        bg.alpha_composite(spr)
        sheet.paste(bg, (x, y))
    sheet.save(os.path.join(root, "build", "sheet_a.png"))
    print("wrote build/sheet_a.png", len(keys), "characters")
