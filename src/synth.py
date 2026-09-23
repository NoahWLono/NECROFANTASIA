"""Tiny numpy synth: oscillators, envelopes, instruments, drums, effects."""
import numpy as np
from scipy import signal

SR = 44100
RNG = np.random.default_rng(7)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- oscillators
def _blep(ph, dt):
    dt = np.broadcast_to(dt, ph.shape)
    out = np.zeros_like(ph)
    a = ph < dt
    t = ph[a] / dt[a]
    out[a] = t + t - t * t - 1
    b = ph > 1 - dt
    t = (ph[b] - 1) / dt[b]
    out[b] = t * t + t + t + 1
    return out


def phase_of(freq, n, phi0=0.0):
    """freq: scalar or array (Hz). Returns phase in [0,1) and dt."""
    if np.ndim(freq):
        ph = (phi0 + np.cumsum(freq) / SR) % 1.0
        return ph, freq / SR
    t = np.arange(n) / SR
    return (phi0 + freq * t) % 1.0, freq / SR


def saw(freq, n, phi0=0.0):
    ph, dt = phase_of(freq, n, phi0)
    return 2 * ph - 1 - _blep(ph, dt)


def square(freq, n, phi0=0.0, pw=0.5):
    ph, dt = phase_of(freq, n, phi0)
    s1 = 2 * ph - 1 - _blep(ph, dt)
    ph2 = (ph + pw) % 1.0
    s2 = 2 * ph2 - 1 - _blep(ph2, dt)
    return (s1 - s2) * 0.5


def sine(freq, n, phi0=0.0):
    ph, _ = phase_of(freq, n, phi0)
    return np.sin(2 * np.pi * ph)


def adsr(n, a, d, s, r, gate):
    """gate: seconds held. Returns envelope of length n."""
    t = np.arange(n) / SR
    env = np.where(t < a, t / max(a, 1e-4),
                   s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    g = min(gate, n / SR)
    ga = np.interp(g, t, env) if n else 0
    rel = t > g
    env[rel] = ga * np.exp(-(t[rel] - g) / max(r, 1e-4))
    return env


def onepole_lp(x, fc):
    b, a = signal.butter(1, min(fc, SR * 0.45) / (SR / 2))
    return signal.lfilter(b, a, x)


def vib(freq, n, depth=0.12, rate=5.5, delay=0.18):
    t = np.arange(n) / SR
    amt = np.clip((t - delay) / 0.3, 0, 1) * depth
    return freq * 2 ** (amt * np.sin(2 * np.pi * rate * t) / 12)


# ---------------------------------------------------------------- instruments
# each returns (mono ndarray) for a note; caller places + pans
def inst_lead(m, dur, vel):
    n = int((dur + 0.35) * SR)
    f = vib(mtof(m), n, depth=0.18 if dur > 0.5 else 0)
    x = sum(saw(f * 2 ** (d / 1200), n, RNG.random()) for d in (-18, -7, 0, 7, 18)) / 5
    x += 0.35 * square(f / 2, n)
    x = onepole_lp(x, 5200)
    return x * adsr(n, 0.008, 0.25, 0.75, 0.18, dur) * vel


def inst_lead2(m, dur, vel):
    n = int((dur + 0.3) * SR)
    f = vib(mtof(m), n, depth=0.1 if dur > 0.5 else 0, rate=6)
    x = square(f, n, pw=0.3) + 0.4 * saw(f * 1.003, n)
    x = onepole_lp(x, 4200)
    return x * adsr(n, 0.005, 0.2, 0.65, 0.15, dur) * vel * 0.8


def inst_arp(m, dur, vel):
    n = int((dur + 0.25) * SR)
    f = mtof(m)
    t = np.arange(n) / SR
    bright = saw(f, n) * np.exp(-t / 0.07)
    body = square(f, n, pw=0.25) * 0.5 + sine(f * 2, n) * 0.3
    return (bright + body) * adsr(n, 0.002, 0.12, 0.25, 0.08, dur) * vel * 0.55


def inst_piano(m, dur, vel):
    n = int((dur + 1.4) * SR)
    f = mtof(m)
    t = np.arange(n) / SR
    idx = 2.2 * np.exp(-t / 0.35) * vel + 0.2
    mod = np.sin(2 * np.pi * f * 14 / 4 * t) * 0.15 * np.exp(-t / 0.05)
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t) + mod)
    car += 0.3 * np.sin(2 * np.pi * f * 2.001 * t) * np.exp(-t / 0.6)
    env = adsr(n, 0.002, 0.9, 0.0, 0.4, dur + 0.1) * 0.8 + np.exp(-t / 2.5) * 0.2
    env *= np.clip((dur + 0.6 - t) / 0.4, 0, 1)
    return car * env * vel * 0.7


def inst_bell(m, dur, vel):
    n = int((dur + 1.8) * SR)
    f = mtof(m)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t + 1.6 * np.exp(-t / 0.4) * np.sin(2 * np.pi * f * 3.5 * t))
    return x * np.exp(-t / 0.9) * vel * 0.5


def inst_brass(m, dur, vel):
    n = int((dur + 0.25) * SR)
    f = vib(mtof(m), n, depth=0.12)
    x = sum(saw(f * 2 ** (d / 1200), n, RNG.random()) for d in (-9, 0, 9)) / 3
    env = adsr(n, 0.04, 0.3, 0.8, 0.15, dur)
    x = onepole_lp(x, 2600)
    return x * env * vel * 0.8


def inst_pad(m, dur, vel):
    n = int((dur + 0.9) * SR)
    f = mtof(m)
    x = sum(saw(f * 2 ** (d / 1200), n, RNG.random()) for d in (-22, -11, 0, 11, 22)) / 5
    x = onepole_lp(onepole_lp(x, 1800), 2400)
    return x * adsr(n, 0.35, 1.0, 0.85, 0.7, dur) * vel * 0.6


def inst_bassarp(m, dur, vel):
    n = int((dur + 0.12) * SR)
    f = mtof(m)
    t = np.arange(n) / SR
    x = saw(f, n) * (0.4 + 0.6 * np.exp(-t / 0.05)) + 0.6 * sine(f, n)
    x = onepole_lp(x, 2200)
    return x * adsr(n, 0.002, 0.08, 0.5, 0.04, dur) * vel * 0.7


def inst_sub(m, dur, vel):
    n = int((dur + 0.05) * SR)
    f = mtof(m)
    x = sine(f, n) + 0.35 * onepole_lp(saw(f, n), 700)
    return x * adsr(n, 0.003, 0.08, 0.7, 0.03, dur) * vel * 0.9


INSTRUMENTS = {
    'lead': inst_lead, 'lead2': inst_lead2, 'arp': inst_arp, 'piano': inst_piano,
    'bell': inst_bell, 'brass': inst_brass, 'pad': inst_pad, 'bassarp': inst_bassarp,
    'sub': inst_sub,
}


# ---------------------------------------------------------------- drums
def drum_kick(vel=1.0):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 45 + 140 * np.exp(-t / 0.03)
    ph = np.cumsum(f) / SR
    x = np.sin(2 * np.pi * ph) * np.exp(-t / 0.22)
    click = RNG.standard_normal(n) * np.exp(-t / 0.003) * 0.3
    return np.tanh(2.2 * (x + click)) * vel


def drum_snare(vel=1.0):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    nz = signal.lfilter(*signal.butter(2, 1500 / (SR / 2), 'high'), RNG.standard_normal(n))
    return (0.6 * tone + 0.8 * nz * np.exp(-t / 0.12)) * vel


def drum_hat(vel=1.0, open_=False):
    n = int((0.35 if open_ else 0.07) * SR)
    t = np.arange(n) / SR
    nz = signal.lfilter(*signal.butter(2, 7000 / (SR / 2), 'high'), RNG.standard_normal(n))
    return nz * np.exp(-t / (0.12 if open_ else 0.018)) * vel * 0.5


def drum_crash(vel=1.0):
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    nz = signal.lfilter(*signal.butter(2, 4000 / (SR / 2), 'high'), RNG.standard_normal(n))
    return nz * np.exp(-t / 0.7) * vel * 0.45


def drum_tom(pitch=110, vel=1.0):
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    f = pitch * (1 + 0.6 * np.exp(-t / 0.04))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) * vel


def riser(seconds):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    nz = RNG.standard_normal(n)
    # crude sweep: blend lp copies
    lo = onepole_lp(nz, 500)
    hi = signal.lfilter(*signal.butter(2, 3000 / (SR / 2), 'high'), nz)
    k = t / seconds
    return ((1 - k) * lo + k * hi) * k ** 2 * 0.5


# ---------------------------------------------------------------- effects
def make_ir(seconds=2.6, predelay=0.02):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for c in range(2):
        nz = RNG.standard_normal(n)
        nz = onepole_lp(nz, 6000)
        ir[:, c] = nz * np.exp(-t / (seconds / 6.9)) * (t > predelay)
    return ir / np.sqrt((ir ** 2).sum(0))


def reverb(x, ir):
    out = np.zeros((len(x) + len(ir) - 1, 2))
    for c in range(2):
        out[:, c] = signal.oaconvolve(x[:, c], ir[:, c])
    return out[:len(x)]


def pingpong(x, delay_s, fb=0.4, taps=4):
    d = int(delay_s * SR)
    out = np.zeros_like(x)
    for k in range(1, taps + 1):
        g = fb ** k
        side = (k % 2)  # alternate L/R
        out[k * d:, side] += x[:-k * d or None, 0] * g if k * d < len(x) else 0
    return out


def pan(mono, p):
    """p in [-1,1] -> stereo (equal power)."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], 1)
