"""Shared landscape pieces: mountain ridges, pine shorelines, lake reflections, stars, the moon."""
import math
import time

import numpy as np

import gfx
from gfx import H, W, XX, YY

XS = np.arange(W)


def ridge(seed, base, amp, cells=6, octaves=4):
    return gfx.ridge(XS, seed, base, amp, cells, octaves)


def pines(heights, rng, density=0.35, size=(5, 12)):
    """Return new heights with little conifer spikes added along a ridge line (silhouette treeline)."""
    out = heights.copy()
    x = 0
    while x < W:
        if rng.random() < density:
            h = rng.integers(size[0], size[1])
            half = max(2, h // 3)
            for dx in range(-half, half + 1):
                xx = x + dx
                if 0 <= xx < W:
                    out[xx] = min(out[xx], heights[x] - h + abs(dx) * h / half)
            x += half + int(rng.integers(1, 4))
        else:
            x += 2
    return out


def reflect(img, waterline, t, strength=0.5, ripple=1.0, sky_only=None):
    """Mirror everything above the waterline into the water, with a horizontal ripple and darker wave lines."""
    wl = int(waterline)
    rows = np.arange(wl, H)
    d = rows - wl
    src = np.clip(wl - 1 - d, 0, H - 1)
    shift = (np.sin(d * 0.55 + t * 2.1) * (0.6 + d * 0.06) * ripple).astype(np.int32)
    cols = (XS[None, :] + shift[:, None]) % W
    refl = img[src[:, None], cols] * strength
    wave = 0.75 + 0.25 * np.sin(d * 1.7 - t * 3.0)
    img[wl:, :] = refl * wave[:, None]
    return img


def sky_gradient(img, top, bottom, y0=0, y1=H):
    g = np.linspace(top, bottom, y1 - y0, dtype=np.float32)
    img[y0:y1, :] = np.maximum(img[y0:y1, :], g[:, None])


class Stars:
    def __init__(self, rng, n=120, ymax=150, bright=(0.4, 1.0)):
        self.x = rng.integers(0, W, n)
        self.y = rng.integers(0, ymax, n)
        self.b = rng.uniform(*bright, n).astype(np.float32)
        self.p = rng.uniform(0, 2 * math.pi, n)
        self.f = rng.uniform(0.6, 2.4, n)

    def draw(self, img, t, gain=1.0):
        v = self.b * (0.65 + 0.35 * np.sin(self.p + self.f * t)) * gain
        gfx.plot(img, self.x, self.y, v)


def moon_phase(ts=None):
    """0 = new, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter."""
    ts = time.time() if ts is None else ts
    days = (ts - 947182440) / 86400.0          # 2000-01-06 18:14 UTC new moon
    return (days / 29.530588853) % 1.0


def phase_name(phase=None):
    p = moon_phase() if phase is None else phase
    names = ("NEW MOON", "WAXING CRESCENT", "FIRST QUARTER", "WAXING GIBBOUS", "FULL MOON", "WANING GIBBOUS",
             "LAST QUARTER", "WANING CRESCENT")
    return names[int((p * 8 + 0.5) % 8)]


def moon(img, cx, cy, r, phase=None, val=0.95, halo=True, craters=True):
    phase = moon_phase() if phase is None else phase
    nx = (XX - cx) / r
    ny = (YY - cy) / r
    inside = nx * nx + ny * ny <= 1.0
    k = math.cos(2 * math.pi * phase)
    s = np.sqrt(np.clip(1 - ny * ny, 0, 1))
    lit = (nx > k * s) if phase < 0.5 else (nx < -k * s)
    body = np.where(lit, val, 0.07)
    if craters:
        for (u, v, rr) in ((-0.3, -0.25, 0.22), (0.25, 0.3, 0.18), (0.35, -0.35, 0.12), (-0.1, 0.45, 0.1)):
            m = (nx - u) ** 2 + (ny - v) ** 2 <= rr * rr
            body = np.where(m & lit, val * 0.72, body)
    img[inside] = np.maximum(img[inside], body[inside])
    if halo:
        d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
        illum = 0.5 - 0.5 * k
        hv = np.clip(1 - (d - r) / (r * 1.6), 0, 1) * 0.16 * illum
        hv[inside] = 0
        np.maximum(img, hv, out=img)
