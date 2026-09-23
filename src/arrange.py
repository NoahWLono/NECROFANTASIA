"""NECROFANTASIA // OPUS 5.5 REMIX: arrangement + mixdown.

Writes build/necro.wav and build/timeline.json (section + note events for the visuals).
"""
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(__file__))
import synth as S  # noqa: E402
from midiref import load  # noqa: E402

BPM = 164
SPB = 60 / BPM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build')

# song sections, in song beats (beat 0 = bar 0, song ends at beat 452)
SONG_END = 452

# (name, song_from, song_to, out_start, transpose, variant)
SEGMENTS = [
    ('boot',      0,   64,   0, 0, 'boot'),
    ('loop1',    -1, SONG_END, 64, 0, 'full'),
    ('interlude', 128, 192, 64 + SONG_END, 0, 'music_box'),
    ('loop2',    192, SONG_END, 64 + SONG_END + 64, 1, 'overdrive'),
    ('outro',     0,   32, 64 + SONG_END + 64 + 260, 1, 'outro'),
]
TOTAL_BEATS = SEGMENTS[-1][3] + 32 + 24  # + held chord / reverb tail


def song_part(b):
    """Which part of the original song a song-beat belongs to."""
    if b < 64:
        return 'A'
    if b < 128:
        return 'B'
    if b < 192:
        return 'C'
    if b < 320:
        return 'D'
    return 'E'


# instrument choice per reference track -> (voice, gain, pan)
def voice_for(track, dur, variant):
    if track == 1:
        return ('lead', 0.7, -0.15)
    if track == 2:
        return ('lead2', 0.4, 0.3)
    if track == 3:
        return ('arp', 0.7, 0.0)
    if track in (4, 5):
        return ('piano', 0.3 if track == 4 else 0.22, -0.2 if track == 4 else 0.2)
    if track == 6:
        return ('bassarp', 0.6, 0.0)
    if track == 7:
        return ('pad', 0.35, 0.0) if dur >= 2 else ('brass', 0.45, 0.1)
    if track == 8:
        return ('sub', 0.38, 0.0)
    return None


class Mixer:
    def __init__(self, seconds):
        self.n = int(seconds * S.SR)
        self.bus = {}

    def add(self, busname, mono, t0, p=0.0):
        if busname not in self.bus:
            self.bus[busname] = np.zeros((self.n, 2))
        i = int(t0 * S.SR)
        if i >= self.n or i + len(mono) <= 0:
            return
        if i < 0:
            mono, i = mono[-i:], 0
        mono = mono[: self.n - i]
        self.bus[busname][i:i + len(mono)] += S.pan(mono, p)


def build():
    os.makedirs(BUILD, exist_ok=True)
    ref = load()
    secs = TOTAL_BEATS * SPB
    mix = Mixer(secs)
    events = []   # for visuals: (t, voice, midi, dur, vel)
    kicks = []
    cache = {}

    def note(voice, m, dur_s, vel, t, p, busname=None):
        key = (voice, m, round(dur_s, 3), round(vel, 2))
        if key not in cache:
            cache[key] = S.INSTRUMENTS[voice](m, dur_s, vel)
        mix.add(busname or voice, cache[key], t, p)
        events.append((round(t, 4), voice, int(m), round(dur_s, 3), round(vel, 2)))

    for name, a, b, out0, tr, var in SEGMENTS:
        for track, notes in ref.items():
            if track in (9, 10):
                continue  # drums are rewritten from scratch below
            for (sb, db, m, v) in notes:
                if not (a <= sb < b):
                    continue
                vc = voice_for(track, db, var)
                if vc is None:
                    continue
                voice, g, p = vc
                ob = out0 + (sb - a)
                t, d = ob * SPB, db * SPB
                m2 = m + tr
                if var == 'boot':
                    # intro: only harmony + filtered arps, melody held back
                    if voice in ('lead', 'lead2'):
                        continue
                    if voice == 'arp':
                        voice, g = 'bell', 0.3 * min(1, (sb + 8) / 48)
                    if voice == 'sub' and sb < 32:
                        continue
                    if voice == 'bassarp':
                        g *= min(1, sb / 48)
                elif var == 'music_box':
                    if voice in ('pad',):
                        g *= 0.8
                    elif voice == 'piano':
                        g *= 0.55
                        note('bell', m2 + 12, d, v * 0.1, t, -p)
                    else:
                        continue
                elif var == 'outro':
                    if voice not in ('pad', 'lead', 'arp'):
                        continue
                    if voice == 'arp':
                        voice = 'bell'
                    g *= max(0.0, 1 - sb / 40)
                elif var == 'overdrive' and voice == 'lead':
                    note('lead', m2 + 12, d, v * g * 0.45, t, 0.35, 'lead')
                note(voice, m2, d, v * g, t, p)

    # ------------------------------------------------ drums (new, not from ref)
    kit = {'kick': S.drum_kick(), 'snare': S.drum_snare(), 'hat': S.drum_hat(),
           'ohat': S.drum_hat(open_=True), 'crash': S.drum_crash()}

    def hit(kind, beat, vel=1.0, p=0.0):
        t = beat * SPB
        mix.add('drums', kit[kind] * vel, t, p)
        events.append((round(t, 4), kind, 0, 0, round(vel, 2)))
        if kind == 'kick':
            kicks.append(t)

    def fill(beat, bars=1):
        for k in range(bars * 16):
            hit('snare', beat + k / 4, 0.3 + 0.7 * k / (bars * 16), (k % 4 - 1.5) / 3)

    def pattern(beat, style, bar_in_phrase):
        for q in range(4):
            bt = beat + q
            if style == 'boot':
                if q in (0,):
                    hit('kick', bt, 0.7)
                hit('hat', bt + 0.5, 0.3)
            elif style == 'halftime':
                if q == 0:
                    hit('kick', bt)
                if q == 2:
                    hit('snare', bt, 0.8)
                for e in (0, 0.5):
                    hit('hat', bt + e, 0.3)
            elif style in ('full', 'overdrive'):
                hit('kick', bt)
                if q in (1, 3):
                    hit('snare', bt, 0.85)
                for s in range(4):
                    hit('hat', bt + s / 4, 0.25 + (0.15 if s == 2 else 0), 0.3)
                if style == 'overdrive':
                    hit('ohat', bt + 0.5, 0.35, -0.3)
                    if q == 3 and bar_in_phrase % 2:
                        hit('kick', bt + 0.75, 0.7)
        if bar_in_phrase == 0:
            hit('crash', beat, 0.9)

    for name, a, b, out0, tr, var in SEGMENTS:
        nbars = (b - max(a, 0)) // 4
        base = out0 + (max(a, 0) - a)
        for i in range(nbars):
            sb = max(a, 0) + 4 * i
            beat = base + 4 * i
            part = song_part(sb)
            phrase_bar = (sb // 4) % 8
            if var == 'boot':
                if i >= 8:
                    pattern(beat, 'boot', i % 8 or 1)
                if i == 15:
                    fill(beat, 1)
            elif var == 'music_box':
                if i >= 12:
                    pattern(beat, 'halftime', 1)
                if i == 15:
                    fill(beat)
            elif var == 'outro':
                if i == 0:
                    hit('crash', beat)
            else:
                style = var
                if part == 'A':
                    style = 'halftime' if sb < 32 else var
                if part == 'C':
                    style = 'halftime' if sb >= 160 else None
                if style:
                    last = phrase_bar == 7 and part != 'C'
                    if last and style != 'halftime':
                        pattern(beat, style, 1)
                        fill(beat + 2, 0)
                        for k in range(8):
                            hit('snare', beat + 2 + k / 4, 0.4 + k / 12)
                    else:
                        pattern(beat, style, phrase_bar)
                if part == 'C' and sb == 188:
                    fill(beat)

    boot_end = SEGMENTS[1][3] * SPB
    r = S.riser(8 * SPB)
    mix.add('fx', r, boot_end - 8 * SPB)
    inter_end = SEGMENTS[3][3] * SPB
    mix.add('fx', r, inter_end - 8 * SPB)

    # final held chord (C# minor +1 -> D minor with a Picardy D major on top)
    end_beat = SEGMENTS[-1][3] + 32
    for m in (38, 50, 57, 62, 66, 69, 74):
        note('pad', m, 10.0, 0.35, end_beat * SPB, (m % 5 - 2) / 3)
    for m in (74, 78, 81, 86):
        note('bell', m, 3.0, 0.3, end_beat * SPB, (m % 3 - 1) / 2)
    hit('crash', end_beat, 0.8)
    hit('kick', end_beat, 1.0)

    # ------------------------------------------------ mixdown
    n = mix.n
    t = np.arange(n) / S.SR
    duck = np.ones(n)
    for k in kicks:
        i = int(k * S.SR)
        j = min(n, i + int(0.25 * S.SR))
        duck[i:j] = np.minimum(duck[i:j], 1 - 0.55 * np.exp(-(t[i:j] - k) / 0.09))
    duck = S.onepole_lp(duck, 200)

    def lp(x, fc):
        sos = signal.butter(2, fc / (S.SR / 2), output='sos')
        return signal.sosfilt(sos, x, axis=0)

    def hp(x, fc):
        sos = signal.butter(2, fc / (S.SR / 2), 'high', output='sos')
        return signal.sosfilt(sos, x, axis=0)

    B = mix.bus
    z = lambda k: B.get(k, np.zeros((n, 2)))  # noqa: E731
    # boot filter sweep on bassarp/bell: crossfade between lp copies
    sweep_end = boot_end
    k = np.clip(t / sweep_end, 0, 1)[:, None]
    for name in ('bassarp', 'bell', 'pad'):
        x = z(name)
        B[name] = np.where(t[:, None] < sweep_end, (1 - k) * lp(x, 600) + k * x, x)

    dry = (z('lead') * 1.0 + z('lead2') * 0.9 + z('arp') * 0.8 + z('brass') * 0.8
           + z('piano') + z('bell') * 0.9
           + (z('pad') * 0.9 + z('bassarp') * 0.9 + z('sub')) * duck[:, None]
           + z('drums') * 0.3 + z('fx') * 0.6)
    send = (z('lead') * 0.35 + z('lead2') * 0.3 + z('piano') * 0.5 + z('bell') * 0.6
            + z('pad') * 0.4 + z('brass') * 0.3 + hp(z('drums'), 3000) * 0.08 + z('fx') * 0.5)
    delay = S.pingpong(z('lead') + z('bell') * 0.6, SPB * 0.75, 0.35, 5)
    wet = S.reverb(hp(send + delay * 0.5, 180), S.make_ir(2.8))
    for kname, x in sorted(B.items()):
        i0, i1 = int(540 * SPB * S.SR), int(560 * SPB * S.SR)
        print('  bus %-8s rms %.3f  interlude %.3f' % (kname, np.sqrt((x ** 2).mean()),
                                                     np.sqrt((x[i0:i1] ** 2).mean())))
    out = dry + delay * 0.25 + wet * 0.55
    # section gain: interlude + loop1 piano part sit back, loop2 pushes
    g = np.ones(n)
    for s in SEGMENTS:
        t0 = s[3] * SPB
        if s[5] == 'music_box':
            t1 = t0 + 64 * SPB
            ramp = np.clip((t - (t1 - 4 * SPB)) / (4 * SPB), 0, 1)
            g = np.where((t >= t0) & (t < t1), 0.55 + 0.45 * ramp, g)
    out *= S.onepole_lp(g, 5)[:, None]
    out = hp(out, 28)
    out = out / np.percentile(np.abs(out), 99.9)
    out = np.tanh(out * 1.1) / np.tanh(1.1)
    fade = np.clip((secs - t) / 6.0, 0, 1)[:, None]
    out *= fade * 0.93
    wavfile.write(os.path.join(BUILD, 'necro.wav'), S.SR, (out * 32767).astype(np.int16))

    tl = {
        'bpm': BPM, 'spb': SPB, 'duration': secs,
        'segments': [{'name': s[0], 'song_from': s[1], 'song_to': s[2], 'out_beat': s[3],
                      't': s[3] * SPB, 'transpose': s[4], 'variant': s[5]} for s in SEGMENTS],
        'end_chord_t': end_beat * SPB,
        'events': sorted(events),
    }
    with open(os.path.join(BUILD, 'timeline.json'), 'w') as f:
        json.dump(tl, f)
    print('wrote %.1fs, %d events' % (secs, len(events)))


if __name__ == '__main__':
    build()
