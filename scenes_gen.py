"""Generative pieces, different every time: automata, ridgelines, flow fields, harmonographs, fractals, plasma."""
import math
import time

import numpy as np

import gfx
import land
from gfx import H, W, XX, YY
from scene import Scene


# =====================================================================================================
class Life(Scene):
    """Conway's Life on a 160x120 torus with phosphor trails; re-seeds itself when it settles."""
    name, kind, hold = "life", "generative", (14.0, 20.0)

    def start(self, rng):
        super().start(rng)
        self.g = rng.random((120, 160)) < rng.uniform(0.18, 0.32)
        self.trail = self.g.astype(np.float32)
        self.acc = 0.0
        self.pop = []

    def step(self):
        g = self.g
        n = sum(np.roll(np.roll(g, dy, 0), dx, 1).astype(np.uint8)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        self.g = (n == 3) | (g & (n == 2))
        self.pop.append(int(self.g.sum()))
        if len(self.pop) > 50:
            self.pop.pop(0)
            if max(self.pop) - min(self.pop) < 6:              # settled into still lifes / blinkers: stir it
                y, x = self.rng.integers(10, 110), self.rng.integers(10, 150)
                self.g[y - 8:y + 8, x - 8:x + 8] |= self.rng.random((16, 16)) < 0.4
                self.pop.clear()

    def frame(self, t, dt):
        self.acc += dt
        while self.acc >= 1 / 14:
            self.acc -= 1 / 14
            self.step()
            self.trail = np.maximum(self.trail * 0.82, self.g)
        v = np.where(self.g, 1.0, self.trail * 0.55)
        return np.repeat(np.repeat(v, 2, 0), 2, 1).astype(np.float32)


# =====================================================================================================
class Elementary(Scene):
    """A 1-D cellular automaton (Wolfram rule) printing line after line like a teletype."""
    name, kind, hold = "rule-ca", "generative", (10.0, 15.0)
    RULES = [30, 45, 54, 60, 73, 90, 105, 110, 122, 126, 137, 150, 182, 225]
    levels = 0

    def start(self, rng):
        super().start(rng)
        self.rule = int(rng.choice(self.RULES))
        self.tab = np.array([(self.rule >> i) & 1 for i in range(8)], bool)
        row = np.zeros(160, bool)
        if self.rule in (30, 45, 73, 90, 105, 126, 150, 182) and rng.random() < 0.5:
            row[80] = True
        else:
            row = rng.random(160) < 0.5
        self.rows = [row]
        self.acc = 0.0

    def frame(self, t, dt):
        self.acc += dt
        while self.acc >= 1 / 24:
            self.acc -= 1 / 24
            r = self.rows[-1]
            idx = (np.roll(r, 1).astype(int) << 2) | (r.astype(int) << 1) | np.roll(r, -1).astype(int)
            self.rows.append(self.tab[idx])
            if len(self.rows) > 116:
                self.rows.pop(0)
        img = gfx.canvas()
        a = np.array(self.rows, np.float32)
        n = len(self.rows)
        age = np.linspace(0.45, 1.0, n, dtype=np.float32)[:, None]
        block = np.repeat(np.repeat(a * age, 2, 0), 2, 1)
        img[:2 * n] = block
        img = gfx.dither(img, 4)
        gfx.rect(img, 236, 226, 319, 239, 0.0)
        gfx.text(img, 244, 212, "RULE %d" % self.rule, 0.75, "small")
        return img


# =====================================================================================================
class Ridgelines(Scene):
    """Stacked noise ridgelines with hidden-line removal (floating horizon), drifting through the terrain."""
    name, kind, hold = "ridgelines", "generative", (12.0, 18.0)
    rate = 15

    def start(self, rng):
        super().start(rng)
        self.nl = int(rng.integers(26, 33))
        self.field = gfx.value_noise((512, W), (16, 10), rng, 4)
        self.spd = rng.uniform(5, 9)
        xs = np.arange(W, dtype=np.float32)
        c1, c2 = rng.uniform(110, 210), rng.uniform(0.6, 1.2)
        env = np.exp(-((xs - c1) / 62.0) ** 2) * c2 + np.exp(-((xs - 320 + c1) / 45.0) ** 2) * 0.5
        self.env = (0.08 + env).astype(np.float32)
        self.amp = rng.uniform(55, 80)

    def frame(self, t, dt):
        img = gfx.canvas()
        horizon = np.full(W, H + 10, np.float32)
        top, bot = 40.0, 222.0
        gap = (bot - top) / (self.nl - 1)
        off = t * self.spd
        xs = np.arange(W)
        for i in range(self.nl - 1, -1, -1):                     # front (bottom) to back (top)
            row = int(i * 7 + off) % 512
            fr = (i * 7 + off) % 1
            h = self.field[row] * (1 - fr) + self.field[(row + 1) % 512] * fr
            y = top + i * gap - self.amp * self.env * np.maximum(h - 0.35, 0) * 1.6
            vis = y < horizon
            ya, yb = y[:-1], y[1:]
            seg_vis = vis[:-1] | vis[1:]
            k = np.nonzero(seg_vis)[0]
            if len(k):
                tt = np.linspace(0, 1, 4, dtype=np.float32)[None, :]
                px = xs[k][:, None] + tt
                py = ya[k][:, None] + (yb[k] - ya[k])[:, None] * tt
                lim = np.minimum(horizon[k], horizon[np.minimum(k + 1, W - 1)])[:, None]
                ok = py < lim
                val = 0.45 + 0.55 * (i / self.nl)
                gfx.plot(img, px[ok], py[ok], val)
            horizon = np.minimum(horizon, y)
        return img


# =====================================================================================================
class FlowField(Scene):
    """Particles combing through a slowly turning noise vector field; long phosphor trails."""
    name, kind, hold = "flow-field", "generative", (14.0, 20.0)

    def start(self, rng):
        super().start(rng)
        self.a1 = (gfx.value_noise((H, W), (4, 5), rng, 2) * 4 * math.pi).astype(np.float32)
        self.a2 = (gfx.value_noise((H, W), (3, 4), rng, 2) * 4 * math.pi).astype(np.float32)
        n = 1400
        self.px = rng.uniform(0, W, n).astype(np.float32)
        self.py = rng.uniform(0, H, n).astype(np.float32)
        self.life = rng.uniform(0, 6, n).astype(np.float32)
        self.buf = gfx.canvas()
        self.spd = rng.uniform(32, 48)

    def frame(self, t, dt):
        mix = 0.5 + 0.5 * math.sin(t * 0.12)
        xi = self.px.astype(np.int32).clip(0, W - 1)
        yi = self.py.astype(np.int32).clip(0, H - 1)
        a = self.a1[yi, xi] * mix + self.a2[yi, xi] * (1 - mix) + t * 0.05
        self.px += np.cos(a) * self.spd * dt
        self.py += np.sin(a) * self.spd * dt
        self.life -= dt
        dead = (self.px < 0) | (self.px >= W) | (self.py < 0) | (self.py >= H) | (self.life < 0)
        k = int(dead.sum())
        if k:
            self.px[dead] = self.rng.uniform(0, W, k)
            self.py[dead] = self.rng.uniform(0, H, k)
            self.life[dead] = self.rng.uniform(3, 8, k)
        self.buf *= 0.94 ** (dt * 30)
        gfx.plot(self.buf, self.px, self.py, 0.85)
        return self.buf.copy()


# =====================================================================================================
class Harmonograph(Scene):
    """A damped two-pendulum harmonograph drawing itself like a pen plotter on the tube."""
    name, kind, hold = "harmonograph", "generative", (15.0, 20.0)
    tao = "top"
    RATIOS = [(2, 3), (3, 4), (1, 2), (3, 2), (4, 5), (2, 5), (5, 6), (1, 3), (3, 5)]

    def start(self, rng):
        super().start(rng)
        a, b = self.RATIOS[int(rng.integers(len(self.RATIOS)))]
        d = rng.uniform(-0.012, 0.012, 4)
        self.f = np.array([a + d[0], b + d[1], a + d[2], b + d[3]]) * 2.0
        self.p = rng.uniform(0, 2 * math.pi, 4)
        self.amp = np.array([62, 40, 62, 40]) * rng.uniform(0.8, 1.15, 4)
        self.damp = rng.uniform(0.010, 0.022, 4)
        self.T = 0.0
        self.buf = gfx.canvas()
        self.last = self.pen(0.0)

    def pen(self, T):
        e = np.exp(-self.damp * T)
        s = np.sin(self.f * T + self.p) * self.amp * e
        return 160 + (s[0] + s[1]) * 1.15, 120 + (s[2] + s[3]) * 0.98

    def frame(self, t, dt):
        steps = 360                                       # dense enough that consecutive pen samples touch
        Ts = self.T + np.arange(1, steps + 1) * (dt * 5.0 / steps)
        e = np.exp(-self.damp[:, None] * Ts[None, :])
        s = np.sin(self.f[:, None] * Ts[None, :] + self.p[:, None]) * self.amp[:, None] * e
        xs = 160 + (s[0] + s[1]) * 1.15
        ys = 120 + (s[2] + s[3]) * 0.98
        xi, yi = np.rint(xs).astype(np.int32), np.rint(ys).astype(np.int32)
        keep = np.ones(len(xi), bool)
        keep[1:] = (xi[1:] != xi[:-1]) | (yi[1:] != yi[:-1])   # one ink dab per pixel visited
        gfx.plot(self.buf, xi[keep], yi[keep], 0.18, mode="add")
        np.clip(self.buf, 0, 1, out=self.buf)
        self.T = Ts[-1]
        self.last = (xs[-1], ys[-1])
        img = np.minimum(self.buf * 1.6, 1.0)
        gfx.disc(img, xs[-1], ys[-1], 1.5, 1.0)
        return img


# =====================================================================================================
class Mandelbrot(Scene):
    """Computed line by line like a 1985 micro, then brought to life with palette cycling."""
    name, kind, hold = "mandelbrot", "generative", (14.0, 20.0)
    SPOTS = [(-0.743643887, 0.131825904, 0.010), (-0.7453, 0.1127, 0.0065), (0.2549, 0.0005, 0.010),
             (-1.25066, 0.02012, 0.0040), (-0.1592, 1.0317, 0.034), (-0.60, 0.0, 3.0), (-0.7746, 0.1242, 0.012),
             (-1.7687, 0.0018, 0.012)]

    def start(self, rng):
        super().start(rng)
        cx, cy, w = self.SPOTS[int(rng.integers(len(self.SPOTS)))]
        w *= rng.uniform(0.8, 1.25)
        h = w * H / W
        xs = np.linspace(cx - w / 2, cx + w / 2, W)
        ys = np.linspace(cy - h / 2, cy + h / 2, H)
        self.C = xs[None, :] + 1j * ys[:, None]
        self.n = np.zeros((H, W), np.float32)
        self.maxit = 128 if w < 0.1 else 72
        self.row = 0
        self.label = "%+.5f %+.5fi" % (cx, cy)

    def compute_rows(self, k):
        r0, r1 = self.row, min(H, self.row + k)
        c = self.C[r0:r1]
        z = np.zeros_like(c)
        n = np.zeros(c.shape, np.float32)
        alive = np.ones(c.shape, bool)
        for i in range(self.maxit):
            z[alive] = z[alive] * z[alive] + c[alive]
            esc = alive & (np.abs(z) > 4)
            if esc.any():
                n[esc] = i + 1 - np.log2(np.log2(np.abs(z[esc])))
                alive &= ~esc
            if not alive.any():
                break
        n[alive] = -1
        self.n[r0:r1] = n
        self.row = r1

    def frame(self, t, dt):
        t0 = time.perf_counter()
        while self.row < H and time.perf_counter() - t0 < 0.012:       # ~12 ms of arithmetic per frame
            self.compute_rows(1)
        img = gfx.canvas()
        done = self.n[:self.row]
        cyc = 0.5 + 0.5 * np.sin(done * 0.42 - (t * 2.2 if self.row >= H else 0.0))
        img[:self.row] = np.where(done < 0, 0.0, 0.12 + 0.88 * cyc)
        if self.row < H:
            img[self.row:self.row + 1, :] = 0.9
            gfx.text(img, 16, 212, "COMPUTING %3d%%" % (self.row * 100 // H), 0.7, "small")
        else:
            gfx.rect(img, 6, 224, 6 + gfx.text_size(self.label, "small")[0] + 6, 239, 0.0)
            gfx.text(img, 16, 212, self.label, 0.6, "small")
        return img


# =====================================================================================================
class Tree(Scene):
    """A recursive tree grows level by level and sways in a gentle wind, under a few stars."""
    name, kind, hold = "fractal-tree", "generative", (12.0, 18.0)

    def start(self, rng):
        super().start(rng)
        self.depth = 11
        self.spread = rng.uniform(0.32, 0.5)
        self.shrink = rng.uniform(0.70, 0.78)
        self.asym = rng.uniform(-0.08, 0.08)
        self.len0 = rng.uniform(44, 54)
        self.stars = land.Stars(rng, 60, 130, (0.2, 0.6))
        self.ground = land.ridge(int(rng.integers(1 << 30)), 236, 10, 3, 2)

    def frame(self, t, dt):
        img = gfx.canvas()
        self.stars.draw(img, t)
        gfx.fill_below(img, self.ground, 0.0)
        gfx.plot(img, np.arange(W), self.ground, 0.4)
        grow = min(float(self.depth), t / 0.45)
        x = np.array([160.0])
        y = np.array([self.ground[160]])
        a = np.array([-math.pi / 2])
        L = self.len0
        wind = 0.06 * math.sin(t * 0.9) + 0.03 * math.sin(t * 2.3)
        for lev in range(self.depth):
            frac = min(1.0, max(0.0, grow - lev))
            if frac <= 0:
                break
            aa = a + wind * (lev + 1) * 0.35
            x2 = x + np.cos(aa) * L * frac
            y2 = y + np.sin(aa) * L * frac
            n = int(L * frac) + 2
            tt = np.linspace(0, 1, n, dtype=np.float32)[None, :]
            val = 0.45 + 0.55 * lev / (self.depth - 1)
            gfx.plot(img, x[:, None] + (x2 - x)[:, None] * tt, y[:, None] + (y2 - y)[:, None] * tt, val)
            if lev < 3:                                          # thicker trunk
                gfx.plot(img, x[:, None] + (x2 - x)[:, None] * tt + 1, y[:, None] + (y2 - y)[:, None] * tt, val)
            if frac < 1:
                break
            sp = self.spread * (1 + 0.15 * math.sin(t * 0.6 + lev))
            x = np.repeat(x2, 2)
            y = np.repeat(y2, 2)
            a = np.stack([aa - sp + self.asym, aa + sp + self.asym], 1).ravel()
            L *= self.shrink
        if grow >= self.depth:
            gfx.plot(img, x, y, 1.0)
        return img


# =====================================================================================================
class Plasma(Scene):
    """The old-school sine plasma, computed at quarter resolution and ordered-dithered to three levels."""
    name, kind, hold = "plasma", "generative", (9.0, 13.0)
    levels = 0

    def start(self, rng):
        super().start(rng)
        yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.k = rng.uniform(0.04, 0.09, 4)
        self.levels_out = int(rng.choice([2, 3, 3, 4]))

    def frame(self, t, dt):
        xx, yy, k = self.xx, self.yy, self.k
        cx = 80 + 50 * math.sin(t * 0.33)
        cy = 60 + 40 * math.cos(t * 0.27)
        v = (np.sin(xx * k[0] + t) + np.sin(yy * k[1] - t * 1.3) + np.sin((xx + yy) * k[2] + t * 0.7)
             + np.sin(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * k[3] * 2 - t * 1.1))
        v = (v + 4) / 8
        v = 0.5 + 0.5 * np.sin(v * 2 * math.pi * 2 + t * 0.4)
        big = np.repeat(np.repeat(v, 2, 0), 2, 1)
        return gfx.dither(big.astype(np.float32), self.levels_out, gfx.BAYER8_FULL)
