"""Tiny numpy raster toolkit for a 320x240 single-channel phosphor canvas (values 0..1 = beam intensity).

Everything here works on float32 arrays shaped (H, W). No colour anywhere: the tube supplies the green.
Drawing ops are vectorised per primitive so a Pi 3B+ (Cortex-A53) keeps ~30 fps.
"""
import math

import numpy as np

import fontdata

W, H = 320, 240

# ---------------------------------------------------------------- ordered dithering (Bayer)
_B2 = np.array([[0, 2], [3, 1]])


def _bayer(n):
    m = _B2
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


BAYER4 = _bayer(4).astype(np.float32)
BAYER8 = _bayer(8).astype(np.float32)
BAYER4_FULL = np.tile(BAYER4, (H // 4, W // 4))
BAYER8_FULL = np.tile(BAYER8, (H // 8, W // 8))


def dither(img, levels=5, matrix=None):
    """Quantise 0..1 intensity to `levels` steps with an ordered (Bayer) dither: the 1980s way to fake greys."""
    m = BAYER4_FULL if matrix is None else matrix
    n = levels - 1
    v = np.clip(img, 0.0, 1.0) * n
    return np.floor(v + m).clip(0, n) / n


def onebit(img, matrix=None):
    return dither(img, 2, matrix)


# ---------------------------------------------------------------- coordinates
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def canvas(val=0.0):
    return np.full((H, W), val, np.float32)


# ---------------------------------------------------------------- primitives
def plot(img, xs, ys, val=1.0, mode="max"):
    """Set many points. xs/ys float arrays (any shape). Out-of-range points are dropped."""
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi, yi = xi[ok], yi[ok]
    if np.ndim(val):
        v = np.asarray(val, np.float32).ravel()[ok]
    else:
        v = np.float32(val)
    if mode == "max":
        if np.ndim(v):                       # duplicates: sort so the largest value is written last
            o = np.argsort(v, kind="stable")
            yi, xi, v = yi[o], xi[o], v[o]
        img[yi, xi] = np.maximum(img[yi, xi], v)
    elif mode == "add":
        np.add.at(img, (yi, xi), v)
    else:
        img[yi, xi] = v


def line_points(x0, y0, x1, y1):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


def line(img, x0, y0, x1, y1, val=1.0, mode="max"):
    xs, ys = line_points(x0, y0, x1, y1)
    plot(img, xs, ys, val, mode)


def lines(img, segs, val=1.0, mode="max"):
    """segs: iterable of (x0, y0, x1, y1). Vectorised: every segment is sampled at the longest one's density."""
    a = np.asarray(list(segs), np.float32).reshape(-1, 4)
    if not len(a):
        return
    if mode == "add":
        return _lines_loop(img, a, val, mode)
    dx, dy = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    n = int(min(1024, np.abs(np.concatenate([dx, dy])).max())) + 2
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    xs = a[:, :1] + dx[:, None] * t
    ys = a[:, 1:2] + dy[:, None] * t
    v = np.broadcast_to(np.asarray(val, np.float32).reshape(-1, 1), xs.shape) if np.ndim(val) else val
    plot(img, xs, ys, v, mode)


def _lines_loop(img, segs, val, mode):
    xs, ys, vs = [], [], []
    for i, (x0, y0, x1, y1) in enumerate(segs):
        a, b = line_points(x0, y0, x1, y1)
        xs.append(a)
        ys.append(b)
        if np.ndim(val):
            vs.append(np.full(a.shape, val[i], np.float32))
    if not xs:
        return
    plot(img, np.concatenate(xs), np.concatenate(ys), np.concatenate(vs) if vs else val, mode)


def polyline(img, xs, ys, val=1.0, closed=False):
    xs = list(xs)
    ys = list(ys)
    if closed:
        xs.append(xs[0])
        ys.append(ys[0])
    lines(img, [(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(xs) - 1)], val)


def disc(img, cx, cy, r, val=1.0, aspect=1.0):
    m = ((XX - cx) / aspect) ** 2 + (YY - cy) ** 2 <= r * r
    img[m] = np.maximum(img[m], val)


def ring(img, cx, cy, r, val=1.0, steps=None):
    steps = steps or max(16, int(2 * math.pi * r))
    a = np.linspace(0, 2 * math.pi, steps, endpoint=False)
    plot(img, cx + r * np.cos(a), cy + r * np.sin(a), val)


def rect(img, x0, y0, x1, y1, val=1.0, fill=True):
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0c, x1c, y0c, y1c = max(x0, 0), min(x1, W - 1), max(y0, 0), min(y1, H - 1)
    if x0c > x1c or y0c > y1c:
        return
    if fill:
        img[y0c:y1c + 1, x0c:x1c + 1] = val
    else:
        lines(img, [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)], val)


# ---------------------------------------------------------------- blur / glow
def blur(img, r=1):
    """Cheap separable box blur (radius r) via cumulative sums."""
    if r <= 0:
        return img.copy()
    k = 2 * r + 1
    p = np.pad(img, ((0, 0), (r + 1, r)), mode="edge")
    c = np.cumsum(p, axis=1)
    out = (c[:, k:] - c[:, :-k]) / k
    p = np.pad(out, ((r + 1, r), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0)
    return ((c[k:, :] - c[:-k, :]) / k).astype(np.float32)


def glow(img, r=2, amount=0.6):
    return np.clip(np.maximum(img, blur(img, r) * (1 + amount)), 0, 1)


# ---------------------------------------------------------------- text
_FONTS = {"vga8": fontdata.VGA8, "small": fontdata.SMALL}
_GLYPH_CACHE = {}


def glyph(ch, font="vga8"):
    key = (ch, font)
    g = _GLYPH_CACHE.get(key)
    if g is None:
        w, h, table = _FONTS[font]
        rows = table.get(ch) or table.get("?")
        g = np.array([[(r >> (w - 1 - x)) & 1 for x in range(w)] for r in rows], np.float32)
        _GLYPH_CACHE[key] = g
    return g


def text_size(s, font="vga8", scale=1, spacing=0):
    w, h, _ = _FONTS[font]
    return len(s) * (w + spacing) * scale, h * scale


def text(img, x, y, s, val=1.0, font="vga8", scale=1, spacing=0, mode="max"):
    """Blit bitmap text at (x, y) top-left. Returns the x after the last glyph."""
    w, h, _ = _FONTS[font]
    x = int(x)
    y = int(y)
    for ch in s:
        g = glyph(ch, font)
        if scale > 1:
            g = np.kron(g, np.ones((scale, scale), np.float32))
        gh, gw = g.shape
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + gw, W), min(y + gh, H)
        if x0 < x1 and y0 < y1:
            sub = g[y0 - y:y1 - y, x0 - x:x1 - x] * val
            if mode == "max":
                np.maximum(img[y0:y1, x0:x1], sub, out=img[y0:y1, x0:x1])
            else:                                   # "set": paint glyph cell (inverse video friendly)
                img[y0:y1, x0:x1] = sub
        x += (w + spacing) * scale
    return x


def text_halo(img, x, y, s, val=1.0, font="vga8", scale=1, spacing=0, pad=1):
    """Text with a black 1-px outline so it stays legible over clouds, aurora or bright sky."""
    m = canvas()
    text(m, x, y, s, 1.0, font, scale, spacing, "set")
    k = m > 0
    d = k.copy()
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            d |= np.roll(np.roll(k, dy, 0), dx, 1)
    img[d] = 0.0
    img[k] = np.maximum(img[k], val)


def text_center(img, y, s, val=1.0, font="vga8", scale=1, spacing=0, cx=W // 2):
    tw, _ = text_size(s, font, scale, spacing)
    return text(img, cx - tw // 2, y, s, val, font, scale, spacing)


# ---------------------------------------------------------------- noise
def value_noise(shape, cells, rng, octaves=1, persistence=0.5):
    """Tileable smooth value noise (cosine-interpolated random lattice), sum of octaves, normalised to 0..1."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    cy, cx = cells
    for _ in range(octaves):
        g = rng.random((cy, cx)).astype(np.float32)
        ys = np.arange(h, dtype=np.float32) * cy / h
        xs = np.arange(w, dtype=np.float32) * cx / w
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        fy = (1 - np.cos((ys - y0) * math.pi)) / 2
        fx = (1 - np.cos((xs - x0) * math.pi)) / 2
        y1 = (y0 + 1) % cy
        x1 = (x0 + 1) % cx
        y0 %= cy
        x0 %= cx
        a = g[np.ix_(y0, x0)]
        b = g[np.ix_(y0, x1)]
        c = g[np.ix_(y1, x0)]
        d = g[np.ix_(y1, x1)]
        top = a + (b - a) * fx[None, :]
        bot = c + (d - c) * fx[None, :]
        out += amp * (top + (bot - top) * fy[:, None])
        total += amp
        amp *= persistence
        cy *= 2
        cx *= 2
    return out / total


def noise1d(n, cells, rng, octaves=3):
    return value_noise((1, n), (1, cells), rng, octaves)[0]


def ridge(xs, rng_seed, base, amp, cells=6, octaves=4, width=W):
    """A 1-D mountain silhouette height (pixels from top) for x positions."""
    rng = np.random.default_rng(rng_seed)
    n = noise1d(width, cells, rng, octaves)
    return base - amp * n[np.clip(np.asarray(xs, int), 0, width - 1)]


def fill_below(img, heights, val=0.0, mode="set"):
    """Fill every column from height[x] down to the bottom (silhouettes, occluding what is behind)."""
    m = YY >= np.asarray(heights, np.float32)[None, :]
    if mode == "set":
        img[m] = val
    else:
        img[m] = np.maximum(img[m], val)
    return m
