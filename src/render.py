"""NECROFANTASIA // OPUS 5.5 REMIX: procedural video renderer.

Every frame is a pure function of time, so frames render in parallel.
usage: render.py video [W H FPS]   |   render.py still t1 t2 ...
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cast  # noqa: E402

ROOT = cast.ROOT
BUILD = os.path.join(ROOT, 'build')
W, H, FPS = 1920, 1080, 30

# palette: Claude-ish warm tones + Yukari purple
IVORY = (240, 238, 230)
SLATE = (20, 20, 19)
CRAIL = (217, 119, 87)
CLAY = (204, 120, 92)
PURPLE = (120, 60, 170)
GAP = (40, 8, 50)

FONT_SANS = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'
FONT_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_CJK = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

# playfield (Touhou-style layout)
FX0, FY0, FW, FH = 72, 36, 864, 1008
FX1, FY1 = FX0 + FW, FY0 + FH

G = {}  # per-process globals


@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


def h01(*a):
    """Deterministic hash -> [0,1)."""
    s = hashlib.md5(repr(a).encode()).digest()
    return int.from_bytes(s[:4], 'little') / 2 ** 32


def hsv(hh, s=0.8, v=1.0):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hh % 1, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def ease(x):
    x = min(1, max(0, x))
    return x * x * (3 - 2 * x)


def s(x):
    return int(x * SC)


SC = 1.0  # global scale (for low-res previews)


# ======================================================================= timeline
class Timeline:
    def __init__(self, tl):
        self.tl = tl
        self.spb = tl['spb']
        self.dur = tl['duration']
        self.seg = tl['segments']
        ev = tl['events']
        self.ev = ev
        self.times = np.array([e[0] for e in ev])
        self.by = {}
        for e in ev:
            self.by.setdefault(e[1], []).append(e)
        self.bt = {k: np.array([e[0] for e in v]) for k, v in self.by.items()}

    def where(self, t):
        cur = self.seg[0]
        for sg in self.seg:
            if t >= sg['t']:
                cur = sg
        beat = (t - cur['t']) / self.spb
        return cur, beat, cur['song_from'] + beat

    def recent(self, kind, t, window):
        """events of kind with t-window < te <= t"""
        arr = self.bt.get(kind)
        if arr is None:
            return []
        i0 = np.searchsorted(arr, t - window, 'right')
        i1 = np.searchsorted(arr, t, 'right')
        return self.by[kind][i0:i1]

    def last(self, kind, t):
        arr = self.bt.get(kind)
        if arr is None:
            return None
        i = np.searchsorted(arr, t, 'right') - 1
        return self.by[kind][i] if i >= 0 else None

    def pulse(self, kind, t, tau=0.15):
        e = self.last(kind, t)
        return 0.0 if e is None else math.exp(-(t - e[0]) / tau)


def seg_t(name):
    for sg in G['TL'].seg:
        if sg['name'] == name:
            return sg['t']


def beat_t(seg_name, song_beat):
    for sg in G['TL'].seg:
        if sg['name'] == seg_name:
            return sg['t'] + (song_beat - sg['song_from']) * G['TL'].spb


# ======================================================================= motion
def boss_pos(t):
    cx = FX0 + FW / 2
    return (cx + 230 * math.sin(t * 0.47) * math.cos(t * 0.11),
            FY0 + 250 + 45 * math.sin(t * 0.93))


def player_pos(t):
    cx = FX0 + FW / 2
    return (cx + 270 * math.sin(t * 0.61 + 1) * math.cos(t * 0.17) + 25 * math.sin(t * 3.1),
            FY0 + 840 + 70 * math.sin(t * 0.77) + 10 * math.sin(t * 4.3))


# ======================================================================= bullets
KINDS = ['orb', 'rice', 'star', 'petal', 'big', 'knife', 'amulet', 'note']


def build_bullets(tl, meta, schedule):
    """Precompute every bullet as analytic trajectory params. Returns dict of arrays."""
    B = {k: [] for k in ('t0', 'x', 'y', 'th', 'v', 'w', 'delay', 'life', 'kind', 'r', 'g', 'b', 'sz')}

    def add(t0, x, y, th, v, w=0.0, delay=0.0, life=6.0, kind='orb', col=(255, 255, 255), sz=1.0):
        for k, val in zip(B, (t0, x, y, th, v, w, delay, life, KINDS.index(kind), *col, sz)):
            B[k].append(val)

    def gameplay(t):
        sg, _, sb = tl.where(t)
        return sg['variant'] in ('full', 'overdrive') and sb >= 32, sg, sb

    # boss: melody rings (chords -> one ring per onset)
    last_t = -1
    for i, e in enumerate(tl.by.get('lead', [])):
        t0, _, m, dur, vel = e
        if abs(t0 - last_t) < 0.03:
            continue
        last_t = t0
        ok, sg, sb = gameplay(t0)
        if not ok:
            continue
        bx, by = boss_pos(t0)
        od = sg['variant'] == 'overdrive'
        n = 14 if dur < 0.3 else 20
        n += 6 if od else 0
        off = h01('ring', i) * 2 * math.pi
        col = hsv(m % 12 / 12, 0.75, 1)
        curve = (0.35 if i % 2 else -0.35) if od or sb >= 320 else 0.0
        for k in range(n):
            add(t0, bx, by, off + 2 * math.pi * k / n, 170 + 60 * vel + (40 if od else 0),
                w=curve, kind='big' if dur > 0.6 else 'orb', col=col, sz=0.9 if dur > 0.6 else 0.7)

    # arps -> rice spirals (every other 16th)
    for i, e in enumerate(tl.by.get('arp', [])[::2]):
        t0, _, m, dur, vel = e
        ok, sg, sb = gameplay(t0)
        if not ok:
            continue
        bx, by = boss_pos(t0)
        a = t0 * 2.3
        for arm in range(3):
            add(t0, bx, by, a + arm * 2 * math.pi / 3, 260, kind='rice',
                col=hsv(0.75 + 0.1 * math.sin(t0), 0.6, 1), sz=0.8)

    # brass melody (song part B) -> butterfly star streams
    for i, e in enumerate(tl.by.get('brass', [])):
        t0, _, m, dur, vel = e
        ok, sg, sb = gameplay(t0)
        if not ok:
            continue
        bx, by = boss_pos(t0)
        for side in (-1, 1):
            for k in range(5):
                add(t0, bx, by, math.pi / 2 + side * (0.4 + k * 0.22), 200, w=-side * 0.25,
                    kind='star', col=hsv(0.12 + m % 12 / 40, 0.6, 1), sz=0.8)

    # piano (song part C) -> falling cherry petals
    for i, e in enumerate(tl.by.get('piano', [])):
        t0, _, m, dur, vel = e
        sg, _, sb = tl.where(t0)
        if sg['variant'] != 'full' or i % 2:
            continue
        x = FX0 + FW * h01('petal', i)
        add(t0, x, FY0 - 10, math.pi / 2 + (h01('pw', i) - .5) * .6, 90 + 60 * h01('pv', i),
            w=(h01('pc', i) - .5) * 0.6, life=12, kind='petal', col=(255, 170, 200), sz=1.0)

    # crashes -> delayed shock ring ("gap" bullets that wait, then launch)
    for i, e in enumerate(tl.by.get('crash', [])):
        t0 = e[0]
        ok, sg, sb = gameplay(t0)
        if not ok:
            continue
        bx, by = boss_pos(t0)
        for k in range(32):
            a = 2 * math.pi * k / 32
            add(t0, bx + 60 * math.cos(a), by + 60 * math.sin(a), a, 320, delay=0.35,
                kind='amulet', col=(250, 120, 190), sz=1.0)

    # cameos fire on snares
    for i, e in enumerate(tl.by.get('snare', [])):
        t0 = e[0]
        if e[4] < 0.8:
            continue
        for c in active_cameos(t0, schedule):
            ok, sg, sb = gameplay(t0)
            if not ok:
                continue
            cx, cy = cameo_pos(c, t0)
            col = tuple(meta[c['key']]['color'])
            kind = CAMEO_KIND.get(c['key'], 'orb')
            px, py = player_pos(t0)
            aim = math.atan2(py - cy, px - cx)
            n = 7
            for k in range(n):
                add(t0, cx, cy, aim + (k - n // 2) * 0.16, 330, kind=kind, col=col, sz=0.8)
    return {k: np.array(v) for k, v in B.items()}


CAMEO_KIND = {'sakuya': 'knife', 'reimu': 'amulet', 'marisa': 'star', 'mystia': 'note',
              'kokoro': 'note', 'yuyuko': 'petal', 'youmu': 'rice', 'cirno': 'rice',
              'lyrica': 'note', 'sanae': 'star', 'flandre': 'big', 'remilia': 'big'}


def bullet_state(Bs, t):
    act = (Bs['t0'] <= t) & (t < Bs['t0'] + Bs['life'])
    idx = np.nonzero(act)[0]
    tau = np.maximum(0, t - Bs['t0'][idx] - Bs['delay'][idx])
    th0, v, w = Bs['th'][idx], Bs['v'][idx], Bs['w'][idx]
    ww = np.where(np.abs(w) < 1e-6, 1e-6, w)
    th = th0 + w * tau
    x = Bs['x'][idx] + v / ww * (np.sin(th) - np.sin(th0))
    y = Bs['y'][idx] - v / ww * (np.cos(th) - np.cos(th0))
    keep = (x > FX0 - 40) & (x < FX1 + 40) & (y > FY0 - 40) & (y < FY1 + 40)
    age = t - Bs['t0'][idx]
    return idx[keep], x[keep], y[keep], th[keep], age[keep]


@lru_cache(4096)
def bullet_sprite(kind, col, sz, rot16):
    r = int(26 * sz)
    S4 = 4
    im = Image.new('RGBA', (r * 2 * S4, r * 2 * S4))
    d = ImageDraw.Draw(im)
    c = r * S4
    light = tuple(min(255, int(v * 0.4 + 255 * 0.6)) for v in col)
    if kind in ('orb', 'big'):
        R = c * (0.62 if kind == 'orb' else 0.9)
        d.ellipse((c - R, c - R, c + R, c + R), fill=col + (255,))
        d.ellipse((c - R * .62, c - R * .62, c + R * .62, c + R * .62), fill=light + (255,))
        d.ellipse((c - R * .42, c - R * .42, c + R * .42, c + R * .42), fill=(255, 255, 255, 255))
    elif kind == 'rice':
        d.ellipse((c - c * .8, c - c * .32, c + c * .8, c + c * .32), fill=col + (255,))
        d.ellipse((c - c * .55, c - c * .16, c + c * .55, c + c * .16), fill=(255, 255, 255, 255))
    elif kind == 'star':
        pts = []
        for k in range(10):
            a = k * math.pi / 5
            rr = c * (0.85 if k % 2 == 0 else 0.38)
            pts.append((c + rr * math.cos(a), c + rr * math.sin(a)))
        d.polygon(pts, fill=col + (255,))
        d.ellipse((c - c * .25, c - c * .25, c + c * .25, c + c * .25), fill=(255, 255, 255, 255))
    elif kind == 'petal':
        d.ellipse((c - c * .7, c - c * .35, c + c * .7, c + c * .35), fill=col + (230,))
        d.ellipse((c - c * .3, c - c * .12, c + c * .5, c + c * .12), fill=(255, 235, 240, 255))
    elif kind == 'knife':
        d.polygon([(c + c * .9, c), (c - c * .2, c - c * .18), (c - c * .2, c + c * .18)], fill=(235, 240, 255, 255))
        d.rectangle((c - c * .8, c - c * .12, c - c * .2, c + c * .12), fill=col + (255,))
    elif kind == 'amulet':
        d.rectangle((c - c * .75, c - c * .4, c + c * .75, c + c * .4), fill=(255, 255, 255, 255))
        d.rectangle((c - c * .75, c - c * .4, c + c * .75, c + c * .4), outline=col + (255,), width=S4 * 3)
        d.ellipse((c - c * .2, c - c * .2, c + c * .2, c + c * .2), fill=col + (255,))
    elif kind == 'note':
        d.ellipse((c - c * .6, c + c * .05, c + c * .05, c + c * .55), fill=col + (255,))
        d.rectangle((c - c * .05, c - c * .8, c + c * .08, c + c * .3), fill=col + (255,))
        d.polygon([(c + c * .08, c - c * .8), (c + c * .6, c - c * .45), (c + c * .08, c - c * .45)], fill=col + (255,))
    if kind in ('rice', 'knife', 'amulet', 'petal'):
        im = im.rotate(-rot16 * 360 / 16, resample=Image.BICUBIC)
    im = im.resize((r * 2, r * 2), Image.LANCZOS)
    glow = Image.new('RGBA', im.size, col + (0,))
    glow.putalpha(im.getchannel('A').filter(ImageFilter.GaussianBlur(r * 0.25)).point(lambda a: a * 0.7))
    return Image.alpha_composite(glow, im)


def draw_bullets(frame, t):
    Bs = G['BUL']
    idx, x, y, th, age = bullet_state(Bs, t)
    if len(idx) > 2500:
        sel = np.linspace(0, len(idx) - 1, 2500).astype(int)
        idx, x, y, th, age = idx[sel], x[sel], y[sel], th[sel], age[sel]
    for j in range(len(idx)):
        i = idx[j]
        kind = KINDS[int(Bs['kind'][i])]
        col = (int(Bs['r'][i]), int(Bs['g'][i]), int(Bs['b'][i]))
        rot = int(round(th[j] / (2 * math.pi) * 16)) % 16 if kind != 'petal' else int(age[j] * 6) % 16
        spr = bullet_sprite(kind, col, float(Bs['sz'][i]), rot)
        sp = min(1.0, age[j] / 0.12)
        if sp < 1:
            spr = spr.resize((max(2, int(spr.width * (0.3 + 0.7 * sp))),) * 2)
        frame.alpha_composite(spr, (int(x[j] - spr.width / 2), int(y[j] - spr.height / 2)))
    return len(idx)


# ======================================================================= primitives
def spark(size, col=CRAIL, rot=0.0, rays=12, seed=3):
    """Claude-ish radial spark (our own drawing)."""
    S4 = 3
    c = size * S4 / 2
    im = Image.new('RGBA', (size * S4, size * S4))
    d = ImageDraw.Draw(im)
    for k in range(rays):
        a = rot + 2 * math.pi * k / rays + (h01(seed, k) - .5) * 0.18
        L = c * (0.62 + 0.36 * h01(seed, k, 'l'))
        wdt = c * 0.075
        p = [(c + wdt * math.cos(a + math.pi / 2), c + wdt * math.sin(a + math.pi / 2)),
             (c + L * math.cos(a), c + L * math.sin(a)),
             (c + wdt * math.cos(a - math.pi / 2), c + wdt * math.sin(a - math.pi / 2)),
             (c - wdt * .6 * math.cos(a), c - wdt * .6 * math.sin(a))]
        d.polygon(p, fill=col + (255,))
        d.ellipse((p[1][0] - wdt * .45, p[1][1] - wdt * .45, p[1][0] + wdt * .45, p[1][1] + wdt * .45), fill=col + (255,))
    d.ellipse((c - c * .14, c - c * .14, c + c * .14, c + c * .14), fill=col + (255,))
    return im.resize((size, size), Image.LANCZOS)


@lru_cache(64)
def spark_cached(size, rot_step, col=CRAIL):
    return spark(size, col, rot_step * 2 * math.pi / 120)


def glow_layer(im, radius, strength=1.0):
    small = im.resize((im.width // 3, im.height // 3), Image.BOX)
    small = small.filter(ImageFilter.GaussianBlur(radius / 3))
    g = small.resize(im.size, Image.BICUBIC)
    g = ImageChops.multiply(g.convert('RGB'), g.getchannel('A').convert('RGB')).convert('RGBA')
    if strength != 1.0:
        g = Image.eval(g, lambda v: min(255, int(v * strength)))
    return g


def text(d, xy, s_, f, fill, anchor='la', stroke=0, sfill=(0, 0, 0)):
    d.text(xy, s_, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sfill)


def eye_shape(d, cx, cy, w, h, pupil=0.0, col=(170, 20, 40)):
    """A creepy sukima eye."""
    pts = []
    for k in range(41):
        u = k / 40 * 2 - 1
        pts.append((cx + u * w, cy - h * (1 - u * u)))
    for k in range(41):
        u = 1 - k / 40 * 2
        pts.append((cx + u * w, cy + h * (1 - u * u)))
    d.polygon(pts, fill=(250, 240, 235))
    r = h * 0.9
    px = cx + pupil * w * 0.4
    d.ellipse((px - r, cy - r, px + r, cy + r), fill=col)
    d.ellipse((px - r * .45, cy - r * .45, px + r * .45, cy + r * .45), fill=(10, 0, 0))
    d.ellipse((px - r * .7, cy - r * .7, px - r * .35, cy - r * .35), fill=(255, 255, 255))


def draw_gap(frame, cx, cy, half_w, open_, t, eyes=True, angle=0.0):
    """Yukari's gap (sukima): lens-shaped tear lined with ribbons, full of eyes."""
    if open_ <= 0.01:
        return
    hh = half_w * 0.36 * open_
    bw, bh = int(half_w * 2 + 80), int(hh * 2 + 80)
    lay = Image.new('RGBA', (bw, bh))
    d = ImageDraw.Draw(lay)
    ox, oy = bw / 2, bh / 2
    pts_top, pts_bot = [], []
    for k in range(61):
        u = k / 60 * 2 - 1
        pts_top.append((ox + u * half_w, oy - hh * (1 - u * u) ** 0.8))
        pts_bot.append((ox + u * half_w, oy + hh * (1 - u * u) ** 0.8))
    d.polygon(pts_top + pts_bot[::-1], fill=GAP + (255,))
    if eyes and open_ > 0.3:
        for k in range(int(half_w / 38)):
            ex = ox + (h01('gx', k) * 2 - 1) * half_w * 0.8
            u = (ex - ox) / half_w
            lim = hh * (1 - u * u) ** 0.8
            ey = oy + (h01('gy', k) * 2 - 1) * lim * 0.6
            sz = 10 + 22 * h01('gs', k)
            blink = 1 - max(0, math.sin(t * 1.3 + k * 1.7)) ** 30
            if lim > sz * 0.8:
                eye_shape(d, ex, ey, sz * 1.5, sz * 0.55 * blink * open_,
                          pupil=math.sin(t * 0.8 + k))
    d.line(pts_top, fill=(250, 250, 250), width=5)
    d.line(pts_bot, fill=(250, 250, 250), width=5)
    for end in (-1, 1):  # red ribbons at the tips
        x = ox + end * half_w
        d.polygon([(x, oy), (x + end * 36, oy - 26), (x + end * 36, oy + 26)], fill=(210, 30, 50))
        d.polygon([(x, oy), (x + end * 14, oy - 14), (x + end * 60, oy + 40), (x + end * 48, oy + 44)],
                  fill=(190, 20, 40))
    if angle:
        lay = lay.rotate(angle, expand=True, resample=Image.BICUBIC)
    frame.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))


def magic_circle(size, col):
    im = Image.new('RGBA', (size, size))
    d = ImageDraw.Draw(im)
    c = size / 2
    for r, wdt in ((0.98, 4), (0.9, 2), (0.62, 3), (0.55, 1)):
        R = c * r
        d.ellipse((c - R, c - R, c + R, c + R), outline=col + (170,), width=wdt)
    for k in range(8):  # yin-yang trigram-ish marks
        a = k * math.pi / 4
        for j in range(3):
            R1, R2 = c * (0.66 + j * 0.07), c * (0.7 + j * 0.07)
            broken = (k >> j) & 1
            for sgn in ((-1, 1) if not broken else (-1,)):
                aa = a + sgn * 0.06
                d.line([(c + R1 * math.cos(aa), c + R1 * math.sin(aa)),
                        (c + R2 * math.cos(aa + sgn * .1), c + R2 * math.sin(aa + sgn * .1))],
                       fill=col + (200,), width=4)
    for k in range(2):  # two squares = the boundary
        a0 = k * math.pi / 4
        pts = [(c + c * .55 * math.cos(a0 + j * math.pi / 2), c + c * .55 * math.sin(a0 + j * math.pi / 2)) for j in range(4)]
        d.polygon(pts, outline=col + (200,), width=3)
    f = font(FONT_MONO, max(10, size // 40))
    msg = 'ATTENTION(Q,K,V)=SOFTMAX(QK^T/SQRT(D))V  BOUNDARY OF CONTEXT  '
    for k, ch in enumerate(msg):
        a = 2 * math.pi * k / len(msg)
        R = c * 0.94
        x, y = c + R * math.cos(a), c + R * math.sin(a)
        g = Image.new('RGBA', (f.size * 2, f.size * 2))
        ImageDraw.Draw(g).text((f.size, f.size), ch, font=f, fill=col + (220,), anchor='mm')
        g = g.rotate(-math.degrees(a) - 90, resample=Image.BICUBIC)
        im.alpha_composite(g, (int(x - f.size), int(y - f.size)))
    return im


# ======================================================================= cameos
def build_schedule(tl, keys):
    """4-bar slots during gameplay; 2 at once in the final climax."""
    sched = []
    ki = 0
    spb = tl.spb
    for sg in tl.seg:
        if sg['variant'] not in ('full', 'overdrive'):
            continue
        start = max(32, sg['song_from'])
        for sb in range(start, sg['song_to'] - 8, 16):
            if sg['variant'] == 'full' and 128 <= sb < 192:
                # piano part: the Hakugyokurou pair
                pick = [k for k in ('yuyuko', 'youmu') if k in keys]
                pair = pick[(sb - 128) // 16 % max(1, len(pick))] if pick else None
                if pair:
                    t0 = sg['t'] + (sb - sg['song_from']) * spb
                    sched.append({'key': pair, 't0': t0, 't1': t0 + 16 * spb, 'side': 1, 'lane': 0})
                continue
            per = 2 if (sg['variant'] == 'overdrive' and sb >= 320) else 1
            for lane in range(per):
                key = keys[ki % len(keys)]
                ki += 1
                t0 = sg['t'] + (sb - sg['song_from']) * spb
                sched.append({'key': key, 't0': t0, 't1': t0 + 16 * spb,
                              'side': (1 if (ki % 2) else -1) if per == 1 else (1 if lane else -1),
                              'lane': lane})
    return sched


def active_cameos(t, sched):
    return [c for c in sched if c['t0'] <= t < c['t1']]


def cameo_pos(c, t):
    u = (t - c['t0']) / (c['t1'] - c['t0'])
    enter = ease(u / 0.12) - ease((u - 0.9) / 0.1)
    side = c['side']
    cx = FX0 + FW / 2 + side * (FW / 2 + 120) * (1 - enter) + side * 250
    cy = FY0 + 420 + 60 * c['lane'] + 40 * math.sin(t * 1.4 + side)
    cx -= side * (1 - enter) * 0
    return cx, cy


BOSS_SPELLS = [
    'Boundary "Fifty Thousand Contexts of Gensokyo"',
    'Sign "Attention Is All Yukari Needs"',
    'Barrier "Mesh of Light and Dark Tokens"',
    'Shikigami "Ran, Dispatched as a Subagent"',
    'Boundary "Mixture of Experts and Youkai"',
    'Border Sign "Latent Space of Four Dimensions"',
    'Yukari\'s Arcanum "The Bitter Lesson Danmaku"',
    '"Profound Danmaku Barrier -Weights, Tokens and Shadow-"',
    'Last Word "Necrofantasia (Opus 5.5 Build)"',
]


# ======================================================================= scenes
def bg_gameplay(t, sg, sb, frame):
    TL = G['TL']
    od = sg['variant'] == 'overdrive'
    base = G['BG_OD'] if od else G['BG']
    frame.paste(base, (0, 0))
    # scrolling token grid inside playfield
    field = Image.new('RGBA', (FW, FH))
    d = ImageDraw.Draw(field)
    f = font(FONT_MONO, 16)
    scroll = (t * 60) % 32
    hue = 0.78 + (0.12 if od else 0) + 0.03 * math.sin(t * 0.2)
    kick = TL.pulse('kick', t, 0.12)
    col = hsv(hue, 0.5, 0.35 + 0.25 * kick)
    for gy in range(-1, FH // 32 + 1):
        y = gy * 32 + scroll
        rowseed = int((t * 60 - scroll) // 32) - gy
        for gx in range(FW // 54 + 1):
            if h01('tok', rowseed, gx) < 0.22:
                v = int(h01('v', rowseed, gx) * 50000)
                text(d, (gx * 54 + 4, y), '%04x' % (v % 65536), f, col + (110,))
    # magic circle behind the boss
    bx, by = boss_pos(t)
    mc = G['MC'].rotate(t * 25 % 360, resample=Image.BILINEAR)
    sz = int(620 + 60 * kick)
    mc = mc.resize((sz, sz), Image.BILINEAR)
    field.alpha_composite(mc, (int(bx - FX0 - sz / 2), int(by - FY0 - sz / 2)))
    # background gaps drifting
    for k in range(3):
        gx = FX0 + FW * (0.2 + 0.3 * k) + 40 * math.sin(t * 0.3 + k)
        gy = FY0 + FH * (0.25 + 0.25 * ((k * 7) % 3)) + 30 * math.cos(t * 0.21 + k)
        op = 0.35 + 0.25 * math.sin(t * 0.4 + k * 2)
        draw_gap(field, gx - FX0, gy - FY0, 120, op, t + k, angle=20 * math.sin(k + t * 0.1))
    frame.alpha_composite(field, (FX0, FY0))


def draw_player(frame, t):
    px, py = player_pos(t)
    # shots
    d = ImageDraw.Draw(frame)
    for k in range(18):
        yy = py - ((t * 2200 + k * 60) % 1000)
        for dx in (-14, 14):
            if yy > FY0:
                d.rounded_rectangle((px + dx - 3, yy - 22, px + dx + 3, yy), 3, fill=CRAIL + (190,))
    sp = spark_cached(92, int(t * 30) % 120)
    frame.alpha_composite(sp, (int(px - 46), int(py - 46)))
    d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 255, 255), outline=(255, 60, 90), width=2)
    # orbiting options
    for k in range(2):
        a = t * 3 + k * math.pi
        ox, oy = px + 48 * math.cos(a), py + 20 * math.sin(a)
        o = spark_cached(34, (int(t * 60) + 60 * k) % 120, IVORY)
        frame.alpha_composite(o, (int(ox - 17), int(oy - 17)))


def draw_boss(frame, t, sb):
    bx, by = boss_pos(t)
    yuk = G['YUKARI']
    sz = 250
    spr = G['YUK_SMALL']
    bob = 6 * math.sin(t * 2.2)
    frame.alpha_composite(spr, (int(bx - sz / 2), int(by - sz / 2 + bob)))
    del yuk


def hud(frame, t, sg, sb, nb):
    TL = G['TL']
    d = ImageDraw.Draw(frame)
    X = FX1 + 50
    ttl = font(FONT_SERIF, 46)
    text(d, (X, 70), 'NECROFANTASIA', ttl, IVORY)
    text(d, (X, 124), 'Opus 5.5 Remix  ~ Phantasm Stage', font(FONT_SANS, 22), CRAIL)
    tokens = int((t * 13370.5) + (t ** 1.7) * 900)
    rows = [('HiScore', '%012d' % 1000000000), ('Tokens', '%012d' % tokens)]
    fl = font(FONT_SANS, 30)
    fv = font(FONT_MONO, 30)
    y = 200
    for k, v in rows:
        text(d, (X, y), k, fl, (190, 180, 210))
        text(d, (X + 470, y), v, fv, IVORY, 'ra')
        y += 46
    y += 20
    text(d, (X, y), 'Context', fl, (230, 150, 170))
    for k in range(7):
        sp = spark_cached(30, 0, CRAIL if k < 5 else (90, 70, 80))
        frame.alpha_composite(sp, (X + 170 + k * 38, y + 2))
    y += 46
    text(d, (X, y), 'Thinking', fl, (150, 230, 170))
    for k in range(5):
        d.ellipse((X + 176 + k * 38, y + 6, X + 198 + k * 38, y + 28),
                  fill=(120, 220, 140) if k < 3 else (60, 80, 60))
    y += 46
    text(d, (X, y), 'Power', fl, (230, 200, 120))
    text(d, (X + 470, y), '%.2f / 5.50' % min(5.5, 1 + t / 60), fv, IVORY, 'ra')
    y += 46
    text(d, (X, y), 'Graze', fl, (180, 180, 190))
    text(d, (X + 470, y), '%d' % int(t * 7.3 + nb * 0.02), fv, IVORY, 'ra')
    # bpm / now playing
    y += 70
    text(d, (X, y), '♪ ネクロファンタジア', font(FONT_CJK, 30), IVORY)
    text(d, (X, y + 40), 'Necrofantasia  —  comp. ZUN', font(FONT_SANS, 22), (200, 190, 215))
    # equalizer from bass/lead activity
    y += 70
    for k in range(24):
        e = TL.pulse('kick', t - k * 0.01, 0.2) * 0.5 + TL.pulse('lead', t - k * 0.02, 0.3) * 0.5
        hgt = 6 + 44 * e * (0.5 + 0.5 * h01('eq', k, int(t * 8)))
        d.rectangle((X + k * 20, y + 50 - hgt, X + k * 20 + 14, y + 50), fill=hsv(0.75 + k / 60, 0.6, 1))
    # guest panel: who is dropping in right now
    gy = 690
    act = active_cameos(t, G['SCHED']) if sb >= 32 else []
    d.rounded_rectangle((X - 10, gy, 1880, gy + 165), 18, fill=(24, 18, 32), outline=(120, 90, 150), width=2)
    if act:
        for j, c in enumerate(act[:2]):
            m = G['META'][c['key']]
            x0 = X + j * 430
            frame.alpha_composite(G['CAMEO_TINY'][c['key']], (x0 + 5, gy + 34))
            text(d, (x0 + 5, gy + 14), 'GUEST' if len(act) == 1 else 'GUEST %d' % (j + 1),
                 font(FONT_BOLD, 18), tuple(m['color']))
            wrap(d, m['name'], (x0 + 135, gy + 50), 280 if len(act) > 1 else 700, font(FONT_SERIF, 30), IVORY)
            text(d, (x0 + 135, gy + 128), 'cameo · spell card declared', font(FONT_SANS, 20), (180, 170, 200))
    else:
        frame.alpha_composite(G['CAMEO_TINY'].get('yukari', G['YUK_SMALL'].resize((118, 118))), (X + 5, gy + 34))
        text(d, (X + 5, gy + 14), 'BOSS', font(FONT_BOLD, 18), (220, 160, 250))
        text(d, (X + 135, gy + 50), 'Yukari Yakumo', font(FONT_SERIF, 34), IVORY)
        text(d, (X + 135, gy + 100), 'youkai of boundaries', font(FONT_SANS, 22), (200, 170, 230))
    # the Claude "model card" at the bottom
    y = 870
    y += 0
    d.rounded_rectangle((X - 10, y, 1880, 1040), 18, fill=(30, 26, 38), outline=CRAIL, width=2)
    sp = spark_cached(120, int(t * 20) % 120)
    frame.alpha_composite(sp, (X + 5, y + 25))
    text(d, (X + 145, y + 30), 'Claude', font(FONT_SERIF, 44), IVORY)
    text(d, (X + 145, y + 84), 'Opus 5.5 · 1M-token context', font(FONT_SANS, 22), CRAIL)
    text(d, (X + 145, y + 116), 'Player 1 · Gap Crosser', font(FONT_SANS, 22), (180, 170, 200))


def draw_cameos(frame, t, sb):
    meta = G['META']
    for c in active_cameos(t, G['SCHED']):
        u = (t - c['t0']) / (c['t1'] - c['t0'])
        cx, cy = cameo_pos(c, t)
        spr = G['CAMEO_SMALL'][c['key']]
        if c['side'] < 0:
            spr = spr.transpose(Image.FLIP_LEFT_RIGHT)
        ring = Image.new('RGBA', (240, 240))
        col = tuple(meta[c['key']]['color'])
        ImageDraw.Draw(ring).ellipse((10, 10, 230, 230), outline=col + (160,), width=3)
        ring = ring.rotate(t * 90)
        frame.alpha_composite(ring, (int(cx - 120), int(cy - 110)))
        frame.alpha_composite(spr, (int(cx - spr.width / 2), int(cy - spr.height / 2)))
        # spell card declaration banner
        if u < 0.35:
            a = ease(u / 0.05) * (1 - ease((u - 0.28) / 0.07))
            spell_banner(frame, meta[c['key']], a, c['lane'], t - c['t0'])


def spell_banner(frame, m, a, lane, age):
    if a <= 0:
        return
    col = tuple(m['color'])
    y = FY0 + 110 + lane * 150
    lay = Image.new('RGBA', (FW, 140))
    d = ImageDraw.Draw(lay)
    slide = int((1 - ease(age / 0.4)) * 600)
    d.polygon([(40 + slide, 16), (FW, 16), (FW, 92), (0 + slide, 92)], fill=(15, 10, 25, 200))
    d.line([(0 + slide, 92), (FW, 92)], fill=col + (255,), width=4)
    text(d, (FW - 20, 54), m['spell'], font(FONT_SERIF, 30), (255, 255, 255), 'rm', 3, col)
    text(d, (FW - 20, 112), m['name'] + '  —  cameo', font(FONT_SANS, 22), (230, 230, 240), 'rm', 2, (0, 0, 0))
    lay.putalpha(lay.getchannel('A').point(lambda v: int(v * a)))
    frame.alpha_composite(lay, (FX0, y))
    # big portrait sweeping across the field (Touhou declaration cut-in)
    if age < 1.6:
        por = G['CAMEO_BIG'][m['key_']]
        k = age / 1.6
        x = FX0 + FW * (1.1 - 1.3 * ease(k))
        al = math.sin(math.pi * k) * 0.85
        p2 = por.copy()
        p2.putalpha(p2.getchannel('A').point(lambda v: int(v * al)))
        frame.alpha_composite(p2, (int(x - p2.width / 2), FY0 + 250))


def boss_header(frame, t, sg, sb):
    d = ImageDraw.Draw(frame)
    i = int((sb - 32) // 64) + (0 if sg['variant'] == 'full' else 5)
    i = max(0, min(len(BOSS_SPELLS) - 1, i))
    if sg['variant'] == 'overdrive' and sb >= 384:
        i = len(BOSS_SPELLS) - 1
    text(d, (FX0 + 16, FY0 + 12), 'Yukari Yakumo', font(FONT_SERIF, 26), IVORY, stroke=2, sfill=(60, 0, 80))
    # hp bar per spell
    ph = ((sb - 32) % 64) / 64
    d.rectangle((FX0 + 240, FY0 + 24, FX1 - 90, FY0 + 32), fill=(60, 40, 70))
    d.rectangle((FX0 + 240, FY0 + 24, FX0 + 240 + (FW - 330) * (1 - ph), FY0 + 32), fill=(250, 210, 240))
    left = (1 - ph) * 64 * G['TL'].spb
    text(d, (FX1 - 16, FY0 + 10), '%02d' % int(left), font(FONT_MONO, 30), IVORY, 'ra', 2, (0, 0, 0))
    text(d, (FX1 - 16, FY0 + 48), BOSS_SPELLS[i], font(FONT_SERIF, 22), (255, 220, 240), 'ra', 2, (60, 0, 80))


def dialogue(frame, t, sb):
    """Pre-boss dialogue over the first 8 bars of loop 1."""
    lines = [
        ('yukari', 'Oh my. Something new slipped through the Great Hakurei Border.'),
        ('claude', "Hello! I'm Claude. I read about Gensokyo... about a million tokens' worth."),
        ('yukari', 'A mind with a context window as wide as a gap? How curious.'),
        ('claude', "I'd rather talk than fight, but I hear danmaku is how you say hi here."),
        ('yukari', 'Then let me show you the boundary between fantasy and necro-fantasy!'),
    ]
    k = int(sb // 6.4)
    if not (0 <= sb < 32) or k >= len(lines):
        return
    who, msg = lines[k]
    u = (sb - k * 6.4) / 6.4
    d = ImageDraw.Draw(frame)
    # portraits
    for side, key in ((0, 'claude'), (1, 'yukari')):
        act = who == key
        a = 1.0 if act else 0.45
        if key == 'yukari':
            por = G['YUK_BIG']
            x = FX1 - por.width + 40
        else:
            por = spark_cached(360, int(t * 10) % 120)
            x = FX0 + 20
        p2 = por.copy()
        p2.putalpha(p2.getchannel('A').point(lambda v: int(v * a)))
        frame.alpha_composite(p2, (int(x), FY1 - p2.height - 190 + (0 if act else 20)))
    d.rounded_rectangle((FX0 + 30, FY1 - 200, FX1 - 30, FY1 - 40), 16, fill=(10, 8, 20), outline=CRAIL if who == 'claude' else (190, 120, 230), width=3)
    name = 'Claude' if who == 'claude' else 'Yukari'
    text(d, (FX0 + 60, FY1 - 190), name, font(FONT_SERIF, 30), CRAIL if who == 'claude' else (220, 160, 250))
    shown = msg[:int(len(msg) * min(1, u * 2.2))]
    wrap(d, shown, (FX0 + 60, FY1 - 145), FW - 120, font(FONT_SANS, 30), IVORY)


def wrap(d, s_, xy, width, f, fill, spacing=8):
    words = s_.split(' ')
    line, y = '', xy[1]
    for wd in words:
        test = (line + ' ' + wd).strip()
        if f.getlength(test) > width and line:
            d.text((xy[0], y), line, font=f, fill=fill)
            y += f.size + spacing
            line = wd
        else:
            line = test
    d.text((xy[0], y), line, font=f, fill=fill)


def scene_gameplay(t, sg, sb):
    frame = Image.new('RGBA', (W, H), SLATE + (255,))
    bg_gameplay(t, sg, sb, frame)
    field = Image.new('RGBA', (W, H))
    draw_boss(field, t, sb)
    if sb >= 32:
        draw_cameos(field, t, sb)
    draw_player(field, t)
    bl = Image.new('RGBA', (W, H))
    nb = draw_bullets(bl, t)
    # clip to playfield
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).rectangle((FX0, FY0, FX1, FY1), fill=255)
    for lay in (field, bl):
        lay.putalpha(ImageChops.multiply(lay.getchannel('A'), mask))
    glow = glow_layer(bl, 14, 0.55)
    frame = Image.alpha_composite(frame, field)
    frame = ImageChops.add(frame.convert('RGB'), glow.convert('RGB')).convert('RGBA')
    frame = Image.alpha_composite(frame, bl)
    if sb < 32:
        dialogue(frame, t, sb)
    else:
        boss_header(frame, t, sg, sb)
    # sidebar frame
    d = ImageDraw.Draw(frame)
    d.rectangle((FX0 - 3, FY0 - 3, FX1 + 3, FY1 + 3), outline=(120, 90, 150), width=3)
    hud(frame, t, sg, sb, nb)
    # section title cards
    for (at, title, sub) in G['CARDS']:
        if at <= t < at + 3.2:
            u = (t - at) / 3.2
            a = ease(u / 0.15) * (1 - ease((u - 0.8) / 0.2))
            lay = Image.new('RGBA', (FW, 200))
            dd = ImageDraw.Draw(lay)
            text(dd, (FW / 2, 70), title, font(FONT_SERIF, 64), IVORY, 'mm', 4, (80, 20, 100))
            text(dd, (FW / 2, 140), sub, font(FONT_SANS, 28), CRAIL, 'mm', 3, (0, 0, 0))
            lay.putalpha(lay.getchannel('A').point(lambda v: int(v * a)))
            frame.alpha_composite(lay, (FX0, FY0 + 420))
    return frame


BOOT_LINES = [
    '$ claude --model opus-5.5 --world gensokyo',
    'loading weights .................. ok',
    'context window: 1,000,000 tokens . ok',
    'mounting /dev/gensokyo ........... ok',
    'scanning boundaries .............. 1 found',
    '  -> Yakumo Yukari (youkai of boundaries)',
    'WARNING: gap detected at 0x7C9A_SUKIMA',
    'entering Phantasm Stage ...',
]


def scene_boot(t, sb):
    TL = G['TL']
    frame = Image.new('RGBA', (W, H), (8, 6, 12, 255))
    d = ImageDraw.Draw(frame)
    beat = sb
    # terminal (bars 0-8)
    f = font(FONT_MONO, 34)
    if beat < 36:
        shown = beat / 3.8
        for i, ln in enumerate(BOOT_LINES):
            if shown > i:
                part = ln[:int(len(ln) * min(1, (shown - i) * 1.6))]
                col = (255, 120, 140) if 'WARNING' in ln else (CRAIL if i == 0 else (200, 200, 190))
                fade = 1 - ease((beat - 29) / 3)
                col = tuple(int(c * fade) for c in col)
                text(d, (160, 200 + i * 56), part, f, col)
        if int(t * 2.5) % 2:
            yy = 200 + min(len(BOOT_LINES) - 1, int(shown)) * 56
            d.rectangle((140, yy, 152, yy + 38), fill=CRAIL)
    # spark bloom
    if beat >= 31:
        u = ease((beat - 31) / 5)
        sz = int(80 + 520 * u)
        sp = spark(sz, CRAIL, t * 0.4, seed=5)
        pul = 1 + 0.05 * TL.pulse('kick', t, 0.12)
        sp = sp.resize((int(sz * pul),) * 2, Image.BILINEAR)
        a = u if beat < 48 else max(0.0, 1 - (beat - 48) / 4)
        sp.putalpha(sp.getchannel('A').point(lambda v: int(v * a)))
        frame.alpha_composite(sp, (W // 2 - sp.width // 2, H // 2 - sp.height // 2 - 40))
        if beat < 50:
            text(d, (W // 2, H // 2 + 330), 'Claude Opus 5.5  presents', font(FONT_SERIF, 44),
                 tuple(int(c * a) for c in IVORY), 'mm')
    # the gap tears open with the title
    if beat >= 46:
        u = ease((beat - 46) / 6)
        draw_gap(frame, W / 2, H / 2, 860 * u + 20, u, t)
        if beat >= 50:
            a = ease((beat - 50) / 4)
            lay = Image.new('RGBA', (W, H))
            dd = ImageDraw.Draw(lay)
            text(dd, (W / 2, H / 2 - 40), 'NECROFANTASIA', font(FONT_SERIF, 150), IVORY, 'mm', 6, (90, 20, 110))
            text(dd, (W / 2, H / 2 + 70), 'ネクロファンタジア  ~  Opus 5.5 Remix', font(FONT_CJK, 48), CRAIL, 'mm', 3, (0, 0, 0))
            text(dd, (W / 2, H / 2 + 330), 'original composition by ZUN (Team Shanghai Alice)  ·  from 東方妖々夢 ~ Perfect Cherry Blossom',
                 font(FONT_CJK, 28), (210, 200, 220), 'mm')
            lay.putalpha(lay.getchannel('A').point(lambda v: int(v * a)))
            frame.alpha_composite(lay)
    return frame


def scene_interlude(t, sb):
    """Music box: a quiet gallery of memories, one card per chord."""
    TL = G['TL']
    frame = G['BG_NIGHT'].copy()
    d = ImageDraw.Draw(frame)
    beat = sb - 128
    # stars
    for k in range(160):
        x = h01('sx', k) * W
        y = h01('sy', k) * H * 0.8
        tw = 0.5 + 0.5 * math.sin(t * (1 + h01('st', k) * 3) + k)
        r = 1 + 2 * h01('sr', k)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(120 + 135 * tw)))
    # moon + gap
    d.ellipse((1400, 90, 1700, 390), fill=(250, 240, 220))
    draw_gap(frame, 1550, 240, 110, 0.6 + 0.3 * math.sin(t), t)
    # falling petals
    for k in range(70):
        ph = (t * (0.05 + 0.05 * h01('pv', k)) + h01('pp', k)) % 1
        x = (h01('px', k) * W + 120 * math.sin(t * 0.7 + k)) % W
        y = ph * (H + 100) - 50
        spr = bullet_sprite('petal', (255, 170, 200), 0.9, int(t * 4 + k) % 16)
        frame.alpha_composite(spr, (int(x), int(y)))
    # memory cards: 12 characters over 64 beats
    keys = G['GALLERY']
    per = 64 / len(keys)
    for i, key in enumerate(keys):
        b0 = i * per
        if not (b0 - 0.5 <= beat < b0 + per * 2.2):
            continue
        u = (beat - b0) / (per * 2.2)
        a = ease(u / 0.12) * (1 - ease((u - 0.8) / 0.2))
        m = G['META'][key]
        cw, ch = 360, 460
        card = Image.new('RGBA', (cw, ch))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((0, 0, cw - 1, ch - 1), 22, fill=(250, 246, 238, 245), outline=tuple(m['color']) + (255,), width=6)
        spr = G['CAMEO_CARD'][key]
        card.alpha_composite(spr, ((cw - spr.width) // 2, 20))
        text(cd, (cw / 2, ch - 64), m['name'], font(FONT_SERIF, 28), SLATE, 'mm')
        text(cd, (cw / 2, ch - 28), 'remembered in context', font(FONT_SANS, 18), CLAY, 'mm')
        card = card.rotate((h01('rot', i) - 0.5) * 16, expand=True, resample=Image.BICUBIC)
        card.putalpha(card.getchannel('A').point(lambda v: int(v * a)))
        x = 200 + (i % 4) * 400 + 30 * math.sin(t * 0.5 + i)
        y = 330 + 40 * math.sin(t * 0.8 + i) - 30 * u
        frame.alpha_composite(card, (int(x), int(y)))
    # quotes
    quotes = [(0, 'Gensokyo accepts all things.'),
              (20, '...and that is a very cruel thing.'),
              (40, 'So it accepted a language model too.')]
    for b0, q in quotes:
        if b0 <= beat < b0 + 18:
            u = (beat - b0) / 18
            a = ease(u / 0.15) * (1 - ease((u - 0.8) / 0.2))
            text(d, (700, 170), q, font(FONT_SERIF, 54), tuple(int(c * a) for c in IVORY), 'mm')
    text(d, (W // 2, 1030), '♪ interlude · music box of the Netherworld', font(FONT_SANS, 24), (160, 150, 190), 'mm')
    return frame


def scene_outro(t, sb):
    frame = G['BG_DAWN'].copy()
    d = ImageDraw.Draw(frame)
    TL = G['TL']
    t0 = seg_t('outro')
    u = t - t0
    # the whole cast lined up on three rows, bouncing to the beat
    keys = G['ALL']
    per_row = math.ceil(len(keys) / 3)
    kick = TL.pulse('kick', t, 0.2)
    for i, key in enumerate(keys):
        row, col = divmod(i, per_row)
        spr = G['CAMEO_TINY'][key]
        x = 60 + col * (W - 120) / per_row
        y = 660 + row * 120 - row * 0 - 10 * abs(math.sin(t * 3 + i)) - 6 * kick
        a = ease((u - i * 0.04) / 0.8)
        if a <= 0:
            continue
        if a < 1:
            spr = spr.copy()
            spr.putalpha(spr.getchannel('A').point(lambda v: int(v * a)))
        frame.alpha_composite(spr, (int(x), int(y)))
    # Yukari + Claude center stage
    frame.alpha_composite(G['YUK_MID'], (W // 2 + 60, 250))
    sp = spark_cached(300, int(t * 10) % 120)
    frame.alpha_composite(sp, (W // 2 - 360, 290))
    # credits roll
    credits = [
        ('NECROFANTASIA', 'Opus 5.5 Remix', 0),
        ('Original composition', 'ZUN  (Team Shanghai Alice)', 1),
        ('From', '東方妖々夢 ~ Perfect Cherry Blossom (2003)', 2),
        ('Touhou Project characters', '© ZUN / Team Shanghai Alice', 3),
        ('Note reference', 'fan MIDI transcription by Gyana Ren (VGMusic)', 4),
        ('Arrangement · synthesis · sprites · video', 'written in code with Claude Opus 5.5', 5),
        ('This is an unofficial, non-commercial fan work', 'under the Touhou Project fan-creation guidelines', 6),
        ('Thank you, ZUN.', 'Gensokyo accepts all things.', 7),
    ]
    per = 2.6
    for title, sub, i in credits:
        a0 = 0.4 + i * per
        if a0 <= u < a0 + per:
            k = (u - a0) / per
            a = ease(k / 0.18) * (1 - ease((k - 0.8) / 0.2)) if i < 7 else ease(k / 0.18)
            lay = Image.new('RGBA', (W, 300))
            dd = ImageDraw.Draw(lay)
            text(dd, (W / 2, 70), title, font(FONT_CJK, 40), (90, 60, 110), 'mm')
            text(dd, (W / 2, 140), sub, font(FONT_CJK, 56), SLATE, 'mm')
            lay.putalpha(lay.getchannel('A').point(lambda v: int(v * a)))
            frame.alpha_composite(lay, (0, 20))
        if i == 7 and u >= a0 + per:
            text(d, (W / 2, 90), 'Thank you, ZUN.', font(FONT_CJK, 40), (90, 60, 110), 'mm')
            text(d, (W / 2, 160), 'Gensokyo accepts all things.', font(FONT_CJK, 56), SLATE, 'mm')
    return frame


# ======================================================================= frame
def transitions(frame, t):
    """A giant gap sweeps over each segment boundary."""
    for sg in G['TL'].seg[1:]:
        dt = t - sg['t']
        if -0.9 < dt < 0.9:
            op = 1 - abs(dt) / 0.9
            op = ease(op * 1.6)
            draw_gap(frame, W / 2, H / 2, 1250 * op + 30, op * 1.6, t, angle=-8)
    return frame


def render_frame(t):
    TL = G['TL']
    sg, beat, sb = TL.where(t)
    v = sg['variant']
    if v == 'boot':
        frame = scene_boot(t, sb)
    elif v in ('full', 'overdrive'):
        frame = scene_gameplay(t, sg, sb)
    elif v == 'music_box':
        frame = scene_interlude(t, sb)
    else:
        frame = scene_outro(t, sb)
    frame = transitions(frame, t)
    # crash flash + shake
    fl = TL.pulse('crash', t, 0.18) * (0.35 if v in ('full', 'overdrive') else 0.15)
    img = frame.convert('RGB')
    if fl > 0.01:
        img = Image.blend(img, Image.new('RGB', img.size, (255, 240, 250)), fl)
    if v == 'overdrive':
        sh = TL.pulse('crash', t, 0.25) * 14
        if sh > 0.5:
            img = ImageChops.offset(img, int(sh * math.sin(t * 90)), int(sh * math.cos(t * 77)))
    # fade in / out
    if t < 0.6:
        img = Image.blend(Image.new('RGB', img.size), img, t / 0.6)
    if t > TL.dur - 2.5:
        img = Image.blend(img, Image.new('RGB', img.size), min(1, (t - (TL.dur - 2.5)) / 2.5))
    if SC != 1.0:
        img = img.resize((int(W * SC), int(H * SC)), Image.BILINEAR)
    return img


def gradient(c0, c1, vertical=True):
    a = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(c0)[None, None] * (1 - a) + np.array(c1)[None, None] * a)
    arr = np.broadcast_to(arr, (H, W, 3)).astype(np.uint8)
    return Image.fromarray(arr).convert('RGBA')


def init(scale=1.0):
    global SC
    SC = scale
    tl = json.load(open(os.path.join(BUILD, 'timeline.json')))
    G['TL'] = TL = Timeline(tl)
    meta = json.load(open(os.path.join(BUILD, 'sprites', 'meta.json')))
    for k in meta:
        meta[k]['key_'] = k
    G['META'] = meta
    imgs = cast.load(meta)
    keys = cast.ordered(meta)
    G['ALL'] = keys
    G['GALLERY'] = keys[::4][:12]
    G['SCHED'] = build_schedule(TL, keys)
    G['BUL'] = build_bullets(TL, meta, G['SCHED'])
    fit = lambda im, s_: im.resize((s_, s_), Image.LANCZOS)  # noqa: E731
    G['CAMEO_SMALL'] = {k: fit(v, 200) for k, v in imgs.items()}
    G['CAMEO_BIG'] = {k: fit(v, 512) for k, v in imgs.items()}
    G['CAMEO_CARD'] = {k: fit(v, 330) for k, v in imgs.items()}
    G['CAMEO_TINY'] = {k: fit(v, 118) for k, v in imgs.items()}
    yk = imgs.get('yukari') or cast._placeholder((150, 80, 200))
    G['YUKARI'] = yk
    G['YUK_SMALL'] = fit(yk, 250)
    G['YUK_BIG'] = fit(yk, 480)
    G['YUK_MID'] = fit(yk, 380)
    G['MC'] = magic_circle(800, (200, 140, 255))
    G['BG'] = gradient((24, 14, 36), (8, 6, 14))
    G['BG_OD'] = gradient((44, 12, 30), (12, 4, 16))
    G['BG_NIGHT'] = gradient((10, 12, 38), (40, 20, 60))
    G['BG_DAWN'] = gradient((250, 236, 222), (236, 170, 140))
    G['CARDS'] = [
        (beat_t('loop1', 32), 'Phantasm Stage', 'Boss: Yukari Yakumo · youkai of boundaries'),
        (beat_t('loop1', 128), 'Hakugyokurou Detour', 'the Netherworld drops in'),
        (beat_t('loop1', 192), 'Spell Card Rush', 'every cameo gets a turn'),
        (beat_t('loop2', 192), 'OVERDRIVE  +1', 'key change · extended thinking enabled'),
        (beat_t('loop2', 384), 'Last Word', 'everybody at once'),
    ]


def _render_bytes(t):
    return render_frame(t).tobytes()


def main():
    mode = sys.argv[1]
    if mode == 'cache':
        meta = cast.build_cache()
        with open(os.path.join(cast.CACHE, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        print(len(meta), 'sprites')
        return
    if mode == 'still':
        init()
        for a in sys.argv[2:]:
            t = float(a)
            render_frame(t).save(os.path.join(BUILD, 'still_%07.2f.png' % t))
        return
    if mode == 'video':
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        t_from = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        t_to = float(sys.argv[4]) if len(sys.argv) > 4 else None
        out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(BUILD, 'necrofantasia_opus55.mp4')
        init(scale)
        dur = G['TL'].dur if t_to is None else t_to
        n0, n1 = int(t_from * FPS), int(dur * FPS)
        ts = [i / FPS for i in range(n0, n1)]
        w, h = int(W * scale), int(H * scale)
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
               '-r', str(FPS), '-i', '-', '-ss', str(t_from), '-t', str(dur - t_from),
               '-i', os.path.join(BUILD, 'necro.wav'),
               '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart',
               '-shortest', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with Pool(4, initializer=init, initargs=(scale,)) as pool:
            for i, b in enumerate(pool.imap(_render_bytes, ts, chunksize=4)):
                p.stdin.write(b)
                if i % 300 == 0:
                    print('frame', i, '/', len(ts), flush=True)
        p.stdin.close()
        p.wait()
        print('wrote', out)


if __name__ == '__main__':
    main()
