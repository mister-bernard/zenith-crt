"""1980s-style pieces: terminal, vector flight, lake at night, circuit board, demo effects, wireframes, radar."""
import math
import time

import numpy as np

import feeds
import gfx
import land
from gfx import H, W, XX, YY
from scene import Scene, TextMode, TEXT_COLS, TEXT_ROWS

TM = None


def textmode():
    global TM
    if TM is None:
        TM = TextMode()
    return TM


# =====================================================================================================
class Terminal(Scene):
    """A green-screen session: memory test, DIR, RUN. 40x30 text mode with the IBM 8x8 font."""
    name, kind, hold = "terminal", "classic", (13.0, 16.0)
    levels = 0

    def start(self, rng):
        super().start(rng)
        d = time.strftime("%m-%d-83")
        files = [("STARS", "BAS", 2048), ("AURORA", "BAS", 4096), ("WEATHER", "BAS", 3072),
                 ("LIFE", "BAS", 1536), ("TUBE", "DOC", 812)]
        rng.shuffle(files)
        ev = [(0.2, "out", "ZENITH ZVM-123   P31 GREEN"), (0.5, "out", "ZENITH BIOS V1.83 (C) 1983"),
              (0.9, "mem", ""), (2.4, "out", ""), (2.6, "cmd", "A>DIR"), (3.5, "out", " VOLUME IN DRIVE A IS GREENGLOW")]
        tt = 3.7
        for n, e, s in files[:4]:
            ev.append((tt, "out", f" {n:<8} {e}  {s:>6}  {d}"))
            tt += 0.12
        ev += [(tt + 0.1, "out", f"        4 FILE(S)  {301056 - rng.integers(0, 40) * 512:>7} BYTES FREE"),
               (tt + 0.4, "out", ""), (tt + 0.7, "cmd", "A>RUN SLIDES"), (tt + 2.4, "dots", "LOADING")]
        self.events = ev

    def frame(self, t, dt):
        lines, cur = [], None
        for t0, kind, s in self.events:
            if t < t0:
                break
            if kind == "cmd":
                n = min(len(s), 2 + int((t - t0 - 0.3) * 9)) if t - t0 > 0.3 else 2
                s = s[:n]
            elif kind == "mem":
                k = min(640, int((t - t0) / 1.3 * 640) // 16 * 16)
                s = f"MEMORY TEST: {k:>3}K" + (" OK" if k >= 640 else "")
            elif kind == "dots":
                s = s + "." * min(12, int((t - t0) * 3))
            lines.append(s)
            cur = (len(lines) - 1, len(s))
        mg = 2
        idx = textmode().from_lines([""] * mg + [" " * mg + s for s in lines])
        img = textmode().render(idx) * 0.85
        if cur and int(t * 2.2) % 2 == 0:
            y, x = (cur[0] + mg) * 8, (cur[1] + mg) * 8
            if x < W - 8:
                img[y:y + 7, x:x + 8] = 0.85
        return img


# =====================================================================================================
class VectorFlight(Scene):
    """Perspective grid flight toward a slit sun behind wireframe mountains."""
    name, kind, hold = "vector-flight", "classic", (10.0, 16.0)
    tao = "top"

    def start(self, rng):
        super().start(rng)
        self.hz = 128
        self.stars = land.Stars(rng, 90, self.hz - 10, (0.3, 0.9))
        xs = np.arange(0, W + 16, 16)
        self.mx = xs
        self.my = self.hz - np.abs(rng.normal(0, 1, len(xs))) * 16 - rng.random(len(xs)) * 8
        self.my[(xs > 120) & (xs < 200)] = self.hz - 4 - rng.random(((xs > 120) & (xs < 200)).sum()) * 6
        self.speed = rng.uniform(2.5, 4.0)
        self.sun_r = 46
        sun = gfx.canvas()
        cx, cy = 160, self.hz - 6
        d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
        m = (d <= self.sun_r) & (YY < self.hz)
        grad = np.clip(1.05 - (YY - (cy - self.sun_r)) / (self.sun_r * 1.25), 0.35, 1.0)
        sun[m] = grad[m]
        for k in range(6):                                # the slits get wider toward the horizon
            y = cy - 6 - k * 7
            sun[max(0, y - k // 2 - 1):y + 1, :] = 0
        hz = self.hz
        ridge_y = np.interp(np.arange(W), self.mx, self.my)
        sky = YY < ridge_y[None, :]                          # mountains occlude the sun and the stars
        sun[~sky] = 0
        segs = [(self.mx[i], self.my[i], self.mx[i + 1], self.my[i + 1]) for i in range(len(self.mx) - 1)]
        segs += [(self.mx[i], self.my[i], self.mx[i] + 8, hz) for i in range(len(self.mx) - 1)]
        gfx.lines(sun, segs, 0.55)
        sun[hz:, :] = 0
        self.static = sun
        self.sky = sky

    def frame(self, t, dt):
        img = gfx.canvas()
        self.stars.draw(img, t)
        img[~self.sky] = 0
        np.maximum(img, self.static, out=img)
        hz = self.hz
        # grid floor
        f, ch, cx = 150.0, 0.85, 160.0
        sway = 1.6 * math.sin(t * 0.35)
        zs = (np.arange(1, 40) - (t * self.speed) % 1.0)
        zs = zs[zs > 0.6]
        ys = hz + ch * f / zs
        hv = np.clip(1.2 - zs / 30.0, 0.15, 1.0)
        gfx.lines(img, [(0, y, W - 1, y) for y in ys if y < H], hv[ys < H])
        zn, zf = 0.75, 45.0
        segs = []
        for xw in np.arange(-24, 25, 1.5):
            x = xw + sway
            segs.append((cx + x * f / zf, hz + ch * f / zf, cx + x * f / zn, hz + ch * f / zn))
        gfx.lines(img, segs, 0.7)
        gfx.line(img, 0, hz, W - 1, hz, 1.0)
        return img


# =====================================================================================================
BIPLANE = [
    "....##..........",
    "..######........",
    "....##....#.....",
    "#############...",
    "################",
    "#############...",
    "....##....#.....",
    "..######........",
]


def sprite(rows):
    return np.array([[1.0 if c == "#" else 0.0 for c in r] for r in rows], np.float32)


class LakeNight(Scene):
    """Mountains over a still lake, a phase-correct moon, and a biplane droning across."""
    name, kind, hold = "lake-night", "classic", (14.0, 20.0)
    tao = "top"

    def start(self, rng):
        super().start(rng)
        seed = int(rng.integers(1 << 30))
        self.wl = 168
        img = gfx.canvas()
        land.sky_gradient(img, 0.0, 0.22, 0, self.wl)
        self.moonxy = (int(rng.integers(200, 280)), int(rng.integers(34, 60)))
        land.moon(img, *self.moonxy, 13)
        far = land.ridge(seed, 150, 70, 5, 4)
        mid = land.ridge(seed + 1, 160, 44, 7, 4)
        near = land.pines(land.ridge(seed + 2, 172, 18, 9, 3), rng, 0.45)
        self.layers = []
        for hgt, val in ((far, 0.30), (mid, 0.15), (near, 0.0)):
            gfx.fill_below(img, hgt, val)
            if val:
                gfx.plot(img, np.arange(W), hgt, min(1.0, val + 0.25))
        self.base = img
        self.stars = land.Stars(rng, 110, 120)
        self.star_mask = None
        self.plane = sprite(BIPLANE)
        self.plane_y = int(rng.integers(64, 96))
        self.plane_v = rng.uniform(16, 22)
        self.plane_x0 = -20 + rng.uniform(0, 60)
        self.meteor_t = rng.uniform(3, 10)
        self.meteor = (rng.uniform(40, 200), rng.uniform(10, 50), rng.uniform(0.4, 0.9))

    def frame(self, t, dt):
        img = gfx.canvas()
        self.stars.draw(img, t)
        img[self.base > 0.02] = 0                    # stars only in clear sky
        np.maximum(img, self.base, out=img)
        mt = t - self.meteor_t
        if 0 <= mt < 0.7:
            x0, y0, s = self.meteor
            gfx.line(img, x0 + mt * 140, y0 + mt * 140 * s, x0 + mt * 140 - 18, y0 + (mt * 140 - 18) * s, 0.9)
        x = int(self.plane_x0 + t * self.plane_v)
        y = self.plane_y + int(round(2 * math.sin(t * 0.8)))
        sp = self.plane
        h, w = sp.shape
        if -w < x < W:
            x0, x1 = max(x, 0), min(x + w, W)
            img[y:y + h, x0:x1] = np.where(sp[:, x0 - x:x1 - x] > 0, 0.85, img[y:y + h, x0:x1])
            if int(t * 1.6) % 2 == 0 and 0 <= x + w + 1 < W:
                img[y + 4, x + w + 1] = 1.0
        land.reflect(img, self.wl, t, 0.55)
        return img


# =====================================================================================================
class Circuit(Scene):
    """A printed-circuit board drawing itself, then signals running the traces."""
    name, kind, hold = "circuit", "classic", (12.0, 18.0)
    LABELS = ["8088", "Z80A", "ROM", "PIO", "6845", "UART", "DRAM", "CTC"]

    def start(self, rng):
        super().start(rng)
        g = 8
        cols, rows = W // g, H // g
        occ = np.zeros((rows, cols), bool)
        chips, starts = [], []
        for _ in range(int(rng.integers(3, 6))):
            for _try in range(30):
                cw, chh = int(rng.integers(5, 9)), int(rng.integers(2, 4))
                cx, cy = int(rng.integers(2, cols - cw - 2)), int(rng.integers(2, rows - chh - 2))
                if not occ[cy - 2:cy + chh + 3, cx - 2:cx + cw + 3].any():
                    occ[cy:cy + chh + 1, cx:cx + cw + 1] = True
                    chips.append((cx, cy, cw, chh, self.LABELS[int(rng.integers(len(self.LABELS)))]))
                    for px in range(cx, cx + cw + 1):
                        starts.append((px, cy - 1, 0, -1))
                        starts.append((px, cy + chh + 1, 0, 1))
                    break
        rng.shuffle(starts)
        for _ in range(14):                                   # a few free-running traces from the board edges
            e = int(rng.integers(4))
            starts.append([(int(rng.integers(cols)), 0, 0, 1), (int(rng.integers(cols)), rows - 1, 0, -1),
                           (0, int(rng.integers(rows)), 1, 0), (cols - 1, int(rng.integers(rows)), -1, 0)][e])
        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        traces = []
        for (x, y, dx, dy) in starts[:60]:
            if not (0 <= x < cols and 0 <= y < rows) or occ[y, x]:
                continue
            pts = [(x, y)]
            occ[y, x] = True
            d = dirs.index((dx, dy))
            for _step in range(int(rng.integers(5, 28))):
                if rng.random() < 0.22:
                    d = (d + (1 if rng.random() < 0.5 else -1)) % 8
                nx, ny = x + dirs[d][0], y + dirs[d][1]
                if not (0 <= nx < cols and 0 <= ny < rows) or occ[ny, nx]:
                    break
                x, y = nx, ny
                occ[y, x] = True
                pts.append((x, y))
            if len(pts) >= 4:
                traces.append(pts)
        self.traces = []
        for pts in traces:
            xs, ys = [], []
            for (a, b), (c, d) in zip(pts, pts[1:]):
                lx, ly = gfx.line_points(a * g + 4, b * g + 4, c * g + 4, d * g + 4)
                xs.append(lx[:-1])
                ys.append(ly[:-1])
            self.traces.append((np.concatenate(xs), np.concatenate(ys), pts[-1]))
        chipimg = gfx.canvas()
        for cx, cy, cw, chh, lab in chips:
            x0, y0, x1, y1 = cx * g, cy * g, (cx + cw + 1) * g - 1, (cy + chh + 1) * g - 1
            gfx.rect(chipimg, x0, y0, x1, y1, 0.12)
            gfx.rect(chipimg, x0, y0, x1, y1, 0.7, fill=False)
            for px in range(cx, cx + cw + 1):
                gfx.rect(chipimg, px * g + 3, y0 - 3, px * g + 4, y0 - 1, 0.6)
                gfx.rect(chipimg, px * g + 3, y1 + 1, px * g + 4, y1 + 3, 0.6)
            gfx.disc(chipimg, x0 + 4, y0 + 4, 1.5, 0.5)
            tw, th = gfx.text_size(lab, "small")
            gfx.text(chipimg, (x0 + x1) // 2 - tw // 2, (y0 + y1) // 2 - th // 2 + 1, lab, 0.8, "small")
        self.chips = chipimg
        self.pulses = []
        for _ in range(28):
            k = int(rng.integers(len(self.traces))) if self.traces else 0
            self.pulses.append([k, rng.uniform(0, 400), rng.uniform(45, 110)])

    def frame(self, t, dt):
        img = self.chips.copy()
        grow = min(1.0, t / 3.0)
        for xs, ys, end in self.traces:
            n = int(len(xs) * grow)
            gfx.plot(img, xs[:n], ys[:n], 0.38)
            if grow >= 1.0:
                gfx.ring(img, end[0] * 8 + 4, end[1] * 8 + 4, 2.2, 0.6, 14)
        if t > 2.5 and self.traces:
            for p in self.pulses:
                xs, ys, _ = self.traces[p[0]]
                n = len(xs)
                pos = int((p[1] + (t - 2.5) * p[2]) % (n + 30))
                if pos < n:
                    lo = max(0, pos - 10)
                    v = np.linspace(0.4, 1.0, pos - lo + 1)
                    gfx.plot(img, xs[lo:pos + 1], ys[lo:pos + 1], v)
        return img


# =====================================================================================================
SCROLL_TEXT = ("      *** ZENITH PRESENTS ...  A LITTLE GREEN GLOW FOR A DIM ROOM ...  "
               "P31 PHOSPHOR, 525 LINES, ONE BEAM ...  ALL GRAPHICS COMPUTED LIVE ON A RASPBERRY PI ...  "
               "GREETINGS TO EVERY TUBE STILL WARM IN THE TWENTY-TWENTIES ...  BLUE SKIES AND TAILWINDS, G ...  "
               "ZVM-123 FOREVER ...      ")


class CopperScroll(Scene):
    """Demoscene: copper bars, a raster-wobbling logo, a sine scroller over parallax stars."""
    name, kind, hold = "copper-scroll", "classic", (14.0, 20.0)

    def start(self, rng):
        super().start(rng)
        self.layers = [(rng.integers(0, W, 40), rng.integers(0, H, 40), sp, v)
                       for sp, v in ((12, 0.25), (28, 0.45), (60, 0.8))]
        logo = gfx.canvas()
        word = "ZENITH" if rng.random() < 0.6 else "GREEN"
        sc = 4 if len(word) <= 6 else 3
        tw, th = gfx.text_size(word, scale=sc, spacing=1)
        gfx.text(logo, 160 - tw // 2, 22, word, 1.0, scale=sc, spacing=1)
        shade = np.clip(1.0 - (YY - 22) / (th * 1.6), 0.45, 1.0)
        self.logo = logo * shade
        self.logo_rows = (22, 22 + th)
        self.phase = rng.uniform(0, 6.28)

    def frame(self, t, dt):
        img = gfx.canvas()
        for xs, ys, sp, v in self.layers:
            gfx.plot(img, (xs - t * sp) % W, ys, v)
        ycol = np.arange(H, dtype=np.float32)
        bars = np.zeros(H, np.float32)
        for k in range(4):
            c = 128 + 62 * math.sin(t * 1.25 + k * 0.62 + self.phase)
            bars = np.maximum(bars, np.clip(1 - np.abs(ycol - c) / 7.0, 0, 1) * (0.55 + 0.1 * k))
        img = np.maximum(img, bars[:, None] * 0.85)
        y0, y1 = self.logo_rows
        rows = np.arange(y0 - 2, y1 + 2)
        shift = (3.5 * np.sin(t * 2.2 + rows * 0.07)).astype(int)
        img[y0 - 2:y1 + 2] = np.maximum(img[y0 - 2:y1 + 2],
                                         self.logo[rows[:, None], (np.arange(W)[None, :] - shift[:, None]) % W])
        cw = 16
        off = t * 72
        first = int(off // cw)
        for i in range(first, first + W // cw + 2):
            ch = SCROLL_TEXT[i % len(SCROLL_TEXT)]
            x = i * cw - off
            y = 196 + 14 * math.sin(x * 0.028 + t * 2.4)
            gfx.text(img, x, y, ch, 1.0, scale=2)
        return img


# =====================================================================================================
class Tunnel(Scene):
    """The classic table-driven texture tunnel, wandering centre."""
    name, kind, hold = "tunnel", "classic", (10.0, 14.0)

    def start(self, rng):
        super().start(rng)
        yy, xx = np.mgrid[-H:H, -W:W].astype(np.float32)
        r = np.sqrt(xx * xx + (yy * 1.0) ** 2) + 1e-3
        self.u = (2400.0 / r).astype(np.float32)                 # depth
        self.v = ((np.arctan2(yy, xx) / (2 * math.pi) + 0.5) * 64).astype(np.float32)
        self.shade = np.clip(r / 150.0, 0, 1).astype(np.float32)
        ty, tx = np.mgrid[0:64, 0:64]
        style = int(rng.integers(3))
        if style == 0:
            tex = ((tx // 8 + ty // 8) % 2).astype(np.float32)
        elif style == 1:
            tex = (((ty % 16) < 2) | (((tx + (ty // 16) * 8) % 16) < 2)).astype(np.float32)       # bricks
        else:
            tex = (0.5 + 0.5 * np.sin(tx / 64 * 2 * math.pi * 4) * np.sin(ty / 64 * 2 * math.pi * 3)).astype(np.float32)
        self.tex = tex * 0.9 + 0.1
        self.spd = rng.uniform(28, 44)
        self.rot = rng.uniform(-8, 8)

    def frame(self, t, dt):
        ox = int(W / 2 + 60 * math.sin(t * 0.7))
        oy = int(H / 2 + 40 * math.sin(t * 0.53 + 1))
        sl = (slice(H - oy, 2 * H - oy), slice(W - ox, 2 * W - ox))
        u = (self.u[sl] + t * self.spd).astype(np.int32) & 63
        v = (self.v[sl] + t * self.rot).astype(np.int32) & 63
        return self.tex[u, v] * self.shade[sl]


# =====================================================================================================
def _tesseract():
    v = np.array([[(i >> k & 1) * 2 - 1 for k in range(4)] for i in range(16)], np.float32)
    e = [(i, j) for i in range(16) for j in range(i + 1, 16) if bin(i ^ j).count("1") == 1]
    return v, e


def _torus(n=18, m=9, R=1.0, r=0.42):
    v, e = [], []
    for i in range(n):
        for j in range(m):
            a, b = 2 * math.pi * i / n, 2 * math.pi * j / m
            v.append([(R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b), 0])
            e.append((i * m + j, ((i + 1) % n) * m + j))
            e.append((i * m + j, i * m + (j + 1) % m))
    return np.array(v, np.float32) * 1.3, e


def _icosa():
    p = (1 + 5 ** 0.5) / 2
    v = []
    for a in (-1, 1):
        for b in (-p, p):
            v += [[0, a, b], [a, b, 0], [b, 0, a]]
    v = np.array([x + [0] for x in v], np.float32) / 1.2
    e = [(i, j) for i in range(12) for j in range(i + 1, 12) if abs(np.linalg.norm(v[i] - v[j]) - 2 / 1.2) < 0.05]
    return v, e


class Wireframe(Scene):
    """Rotating vector objects with depth cueing: a 4-D tesseract, a torus or an icosahedron."""
    name, kind, hold = "wireframe", "classic", (12.0, 16.0)
    OBJS = {"TESSERACT": _tesseract, "TORUS": _torus, "ICOSAHEDRON": _icosa}

    def start(self, rng):
        super().start(rng)
        self.oname = list(self.OBJS)[int(rng.integers(3))] if rng.random() > 0.4 else "TESSERACT"
        self.v, self.e = self.OBJS[self.oname]()
        self.e = np.array(self.e)
        self.rates = rng.uniform(0.25, 0.7, 4) * rng.choice([-1, 1], 4)
        self.stars = land.Stars(rng, 70, H, (0.2, 0.5))

    def frame(self, t, dt):
        v = self.v.copy()
        a = self.rates * t

        def rot(i, j, ang):
            c, s = math.cos(ang), math.sin(ang)
            vi, vj = v[:, i].copy(), v[:, j].copy()
            v[:, i], v[:, j] = c * vi - s * vj, s * vi + c * vj
        if self.oname == "TESSERACT":
            rot(0, 3, a[0])
            rot(1, 3, a[1] * 0.7)
            w = 1.0 / (2.6 - v[:, 3])
            v[:, :3] *= w[:, None] * 2.0
        rot(0, 2, a[2])
        rot(1, 2, a[3])
        rot(0, 1, a[0] * 0.3)
        z = v[:, 2] + 4.2
        f = 330.0
        sx = 160 + v[:, 0] * f / z * 0.5
        sy = 112 + v[:, 1] * f / z * 0.5
        img = gfx.canvas()
        self.stars.draw(img, t)
        e = self.e
        depth = (z[e[:, 0]] + z[e[:, 1]]) / 2
        dv = np.clip(1.25 - (depth - depth.min()) / (np.ptp(depth) + 1e-6) * 0.8, 0.35, 1.0)
        gfx.lines(img, zip(sx[e[:, 0]], sy[e[:, 0]], sx[e[:, 1]], sy[e[:, 1]]), dv)
        gfx.plot(img, sx, sy, 1.0)
        gfx.text(img, 16, 212, f"OBJ {self.oname}  V{len(v)} E{len(e)}", 0.45, "small")
        return img


# =====================================================================================================
class AsciiDonut(Scene):
    """A spinning torus rendered in 40x30 text mode with an ASCII luminance ramp."""
    name, kind, hold = "ascii-torus", "classic", (10.0, 14.0)
    levels = 0

    def start(self, rng):
        super().start(rng)
        th, ph = np.meshgrid(np.linspace(0, 2 * math.pi, 70, endpoint=False),
                             np.linspace(0, 2 * math.pi, 160, endpoint=False))
        self.th, self.ph = th.ravel(), ph.ravel()
        self.k = rng.uniform(0.8, 1.2, 2)

    def frame(self, t, dt):
        A, B = t * 0.9 * self.k[0], t * 0.45 * self.k[1]
        ct, st, cp, sp = np.cos(self.th), np.sin(self.th), np.cos(self.ph), np.sin(self.ph)
        cA, sA, cB, sB = math.cos(A), math.sin(A), math.cos(B), math.sin(B)
        R1, R2, K2 = 1.0, 2.0, 5.0
        ch = R2 + R1 * ct
        x = ch * (cB * cp + sA * sB * sp) - R1 * st * cA * sB
        y = ch * (sB * cp - sA * cB * sp) + R1 * st * cA * cB
        z = K2 + cA * ch * sp + R1 * st * sA
        ooz = 1 / z
        K1 = TEXT_ROWS * K2 * 3 / (8 * (R1 + R2))
        xp = (TEXT_COLS / 2 + K1 * ooz * x).astype(int)
        yp = (TEXT_ROWS / 2 - K1 * ooz * y).astype(int)
        L = cp * ct * sB - cA * ct * sp - sA * st + cB * (cA * st - ct * sA * sp)
        ok = (xp >= 0) & (xp < TEXT_COLS) & (yp >= 0) & (yp < TEXT_ROWS)
        xp, yp, ooz, L = xp[ok], yp[ok], ooz[ok], L[ok]
        o = np.argsort(ooz)
        lum = np.zeros((TEXT_ROWS, TEXT_COLS), np.float32)
        hit = np.zeros((TEXT_ROWS, TEXT_COLS), bool)
        lum[yp[o], xp[o]] = np.clip(L[o] / 1.45, 0, 1)
        hit[yp[o], xp[o]] = True
        tm = textmode()
        idx = tm.luminance(lum)
        idx[~hit] = tm.index[" "]
        img = tm.render(idx, np.where(hit, 0.55 + 0.45 * lum, 0))
        return gfx.dither(img, 4)


# =====================================================================================================
class Planet(Scene):
    """A banded gas giant with rings, a small moon on its orbit, deep space."""
    name, kind, hold = "planet", "classic", (12.0, 18.0)
    tao = "bottom"

    def start(self, rng):
        super().start(rng)
        self.cx, self.cy, self.r = 166 + int(rng.integers(-12, 12)), 120, int(rng.integers(56, 72))
        nx = (XX - self.cx) / self.r
        ny = (YY - self.cy) / self.r
        d2 = nx * nx + ny * ny
        self.inside = d2 <= 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        tilt = 0.35
        ny2 = ny * math.cos(tilt) - nx * math.sin(tilt)
        nx2 = nx * math.cos(tilt) + ny * math.sin(tilt)
        self.lat = np.arcsin(np.clip(ny2, -1, 1))
        self.lon = np.arctan2(nx2, nz)
        light = np.array([-0.55, -0.45, 0.70])
        light /= np.linalg.norm(light)
        self.lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1) ** 0.8
        bands = gfx.value_noise((64, 256), (4, 8), rng, 3)
        lat_band = 0.5 + 0.5 * np.sin(np.linspace(0, 9 * math.pi, 64))
        self.tex = (0.35 + 0.45 * lat_band[:, None] + 0.35 * (bands - 0.5)).astype(np.float32)
        # rings in the planet's tilted equatorial plane, seen at an angle
        rx = (XX - self.cx)
        ry = (YY - self.cy)
        u = rx * math.cos(tilt) + ry * math.sin(tilt)
        w = (-rx * math.sin(tilt) + ry * math.cos(tilt)) / 0.28
        rr = np.sqrt(u * u + w * w) / self.r
        ring = ((rr > 1.35) & (rr < 2.05)).astype(np.float32)
        ring *= 0.55 + 0.35 * np.sin(rr * 38) ** 2
        ring[(rr > 1.72) & (rr < 1.78)] = 0                  # a Cassini-style gap
        self.ring = ring
        self.ring_front = w > 0
        self.stars = land.Stars(rng, 140, H, (0.25, 0.9))
        self.moon_ph = rng.uniform(0, 6.28)

    def frame(self, t, dt):
        img = gfx.canvas()
        self.stars.draw(img, t)
        back = self.ring * (~self.ring_front)
        np.maximum(img, back, out=img)
        lon = ((self.lon + t * 0.18) / (2 * math.pi) * 256).astype(np.int32) & 255
        lat = ((self.lat / math.pi + 0.5) * 63).astype(np.int32).clip(0, 63)
        surf = self.tex[lat, lon] * (0.08 + 0.95 * self.lambert)
        img[self.inside] = surf[self.inside]
        front = self.ring * self.ring_front
        np.maximum(img, front, out=img)
        a = t * 0.45 + self.moon_ph
        mx, my = self.cx + math.cos(a) * self.r * 2.45, self.cy + math.sin(a) * self.r * 0.62 + math.cos(a) * 12
        if math.sin(a) > 0 or (mx - self.cx) ** 2 + (my - self.cy) ** 2 > self.r ** 2:
            gfx.disc(img, mx, my, 5, 0.85)
        return img


# =====================================================================================================
class Radar(Scene):
    """A PPI radar scope: rotating sweep, long-persistence echoes, data blocks, ground clutter.
    Live: real traffic from ADS-B (feeds.AdsbPoller) around [radar] lat/lon (default San Francisco Intl),
    range_nm wide; [radar] marks = [["ID", lat, lon], ...] and label override the airport diamonds and scope tag. Offline/tests: synthetic N-number traffic."""
    name, kind, hold = "radar", "classic", (16.0, 22.0)
    levels = 6
    MARKS = (("SFO", 37.6213, -122.3790), ("OAK", 37.7213, -122.2208), ("SJC", 37.3626, -121.9291))   # default; [radar] marks overrides

    def start(self, rng):
        super().start(rng)
        rc = (self.ctx.get("cfg") or {}).get("radar") or {}
        self.lat0 = float(rc.get("lat", feeds.DEFAULT_RADAR[0]))
        self.lon0 = float(rc.get("lon", feeds.DEFAULT_RADAR[1]))
        self.range = float(rc.get("range_nm", 40))
        self.marks = [(str(m[0]), float(m[1]), float(m[2])) for m in rc.get("marks", self.MARKS)]
        self.label = str(rc.get("label", "KSFO"))
        fd = self.ctx.get("feeds")
        self.poller = None
        if rc.get("live", True) and fd is not None and getattr(fd, "offline", True) is False:
            self.poller = feeds.adsb(self.lat0, self.lon0, self.range)
            self.poller.get()
        self.cx, self.cy, self.R = 160, 120, 104
        self.k = self.R / self.range                                  # px per NM
        dx, dy = XX - self.cx, YY - self.cy
        self.rad = np.sqrt(dx * dx + dy * dy)
        self.ang = (np.arctan2(dx, -dy)) % (2 * math.pi)        # 0 = north, clockwise
        self.inside = self.rad <= self.R
        clutter = gfx.value_noise((H, W), (12, 16), rng, 3)
        self.clutter = np.where(self.inside & (self.rad < 20), np.clip((clutter - 0.6) * 4, 0, 0.8), 0).astype(np.float32)
        scope = gfx.canvas()
        for q in (1, 2, 3, 4):
            gfx.ring(scope, self.cx, self.cy, self.R * q / 4, 0.28)
        for k in range(72):
            a = k * math.pi / 36
            r0 = self.R - (6 if k % 6 == 0 else 3)
            gfx.line(scope, self.cx + r0 * math.sin(a), self.cy - r0 * math.cos(a),
                     self.cx + self.R * math.sin(a), self.cy - self.R * math.cos(a), 0.45)
        kx = math.cos(math.radians(self.lat0)) * 60
        for lab, la, lo in self.marks:                                # airports as small diamonds
            x = self.cx + (lo - self.lon0) * kx * self.k
            y = self.cy - (la - self.lat0) * 60 * self.k
            if (x - self.cx) ** 2 + (y - self.cy) ** 2 < (self.R - 6) ** 2:
                gfx.polyline(scope, [x, x + 3, x, x - 3, x], [y - 3, y, y + 3, y, y - 3], 0.5)
                gfx.text(scope, x - 8, y + 5, lab, 0.4, "small")
        gfx.text(scope, self.cx - 2, self.cy - self.R + 8, "N", 0.6, "small")
        gfx.text(scope, 16, 16, "RNG %dNM" % self.range, 0.5, "small")
        gfx.text(scope, 16, 28, self.label[:8], 0.5, "small")
        self.scope = scope
        self.period = 4.0
        self.tgt = []
        if self.poller is None:                                       # synthetic traffic
            for _ in range(int(rng.integers(6, 10))):
                r = rng.uniform(8, 37)
                a = rng.uniform(0, 2 * math.pi)
                call = "N%d%s" % (rng.integers(100, 9999), "".join(chr(65 + rng.integers(26)) for _ in range(rng.integers(0, 3))))
                self.tgt.append({"id": call, "call": call, "x_nm": r * math.sin(a), "y_nm": r * math.cos(a),
                                 "alt": int(rng.integers(20, 120)) * 100, "gs": float(rng.uniform(90, 240)),
                                 "trk": float(rng.uniform(0, 360)), "age": 0.0})
        self.ts = time.time()
        self.t0 = None
        self.glow = gfx.canvas()                         # sweep afterglow: fast
        self.buf = gfx.canvas()                          # echoes + data blocks: fade over ~one revolution
        self.prev = 0.0

    def _targets(self, t):
        """Current targets dead-reckoned to now, as (x_px, y_px, trk, gs, call, alt)."""
        if self.poller is not None:
            data, ts, err = self.poller.get()
            self.err = err
            if data is not None:
                self.tgt, self.ts = data, ts
        now = time.time() if self.poller is not None else self.ts + t
        out = []
        for a in self.tgt:
            el = min(60.0, now - self.ts + a.get("age", 0.0)) / 3600.0
            r = math.radians(a["trk"])
            x = a["x_nm"] + a["gs"] * el * math.sin(r)
            y = a["y_nm"] + a["gs"] * el * math.cos(r)
            out.append((self.cx + x * self.k, self.cy - y * self.k, r, a["gs"], a["call"], a["alt"], a.get("route") or ""))
        return out

    def frame(self, t, dt):
        self.glow *= 0.955 ** (dt * 30)
        self.buf *= math.exp(-0.42 * dt)                 # ~20% left after a 4 s revolution, then repainted at full
        a1 = (t / self.period) * 2 * math.pi
        a0 = self.prev
        self.prev = a1
        lo, hi = a0 % (2 * math.pi), a1 % (2 * math.pi)
        if hi >= lo:
            wedge = (self.ang >= lo) & (self.ang <= hi + 0.02)
        else:
            wedge = (self.ang >= lo) | (self.ang <= hi + 0.02)
        wedge &= self.inside
        self.glow[wedge] = np.maximum(self.glow[wedge], np.maximum(self.clutter[wedge], 0.16))
        tg = self._targets(t)
        for x, y, trk, gs, call, alt, route in tg:
            ta = math.atan2(x - self.cx, -(y - self.cy)) % (2 * math.pi)
            inwedge = (lo <= ta <= hi + 0.02) if hi >= lo else (ta >= lo or ta <= hi + 0.02)
            if inwedge and (x - self.cx) ** 2 + (y - self.cy) ** 2 < (self.R - 4) ** 2:
                gfx.rect(self.buf, x - 1, y - 1, x + 1, y + 1, 1.0)
                lead = gs / 60.0 * self.k                             # 1-minute velocity leader
                gfx.line(self.buf, x, y, x + lead * math.sin(trk), y - lead * math.cos(trk), 0.7)
                bx = x + 8 if x < self.cx + 60 else x - 8 - 6 * max(len(call), len(route[:11]))
                gfx.text(self.buf, bx, y - 15, call, 0.85, "small")
                gfx.text(self.buf, bx, y - 5, "GND" if alt is None else "%03d" % (int(alt) // 100), 0.75, "small")
                if route:
                    gfx.text(self.buf, bx, y + 5, route[:11], 0.75, "small")
                gfx.ring(self.buf, x, y, 4, 0.8)                  # the hit blooms, then fades with the echo
        img = np.maximum(np.maximum(self.buf, self.glow), self.scope)
        ex, ey = self.cx + self.R * math.sin(a1), self.cy - self.R * math.cos(a1)
        gfx.line(img, self.cx, self.cy, ex, ey, 0.95)
        if self.poller is not None:
            n = sum(1 for x, y, *_ in tg if (x - self.cx) ** 2 + (y - self.cy) ** 2 < self.R ** 2)
            msg = ("ADS-B LIVE  %d TGT" % n) if not getattr(self, "err", None) else "ADS-B: NO DATA"
            gfx.text(img, 16, 212, msg, 0.55, "small")
        return img
