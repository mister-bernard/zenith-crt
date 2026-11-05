"""Live-data pieces: today's weather as a landscape, and the NOAA aurora forecast (curtains + polar oval)."""
import math
import time

import numpy as np

import almanac
import feeds
import gfx
import land
from gfx import H, W, XX, YY
from scene import Scene


def _fmt_t(v):
    return "--" if v is None else "%d" % round(v)


def _day(ts):
    """Local calendar day, e.g. 'NOV 24'."""
    return time.strftime("%b %d", time.localtime(ts)).upper().replace(" 0", " ")


# =====================================================================================================
class Weather(Scene):
    """Today's conditions painted as a scene: sun or phase-correct moon, cloud banks moved by the real wind,
    rain/snow/fog/lightning, a small readout. Not a dashboard."""
    name, kind, hold = "weather", "weather", (18.0, 22.0)
    SHORT = {"LIGHT DRIZZLE": "DRIZZLE", "DENSE DRIZZLE": "DRIZZLE", "MAINLY CLEAR": "CLEAR", "PARTLY CLOUDY": "PT CLOUDY",
             "FREEZING DRIZZLE": "FRZ DRIZZLE", "FREEZING RAIN": "FRZ RAIN", "THUNDERSTORM": "T-STORM"}

    def start(self, rng):
        super().start(rng)
        w, _ = self.ctx["feeds"].get()
        self.w = w
        kind = (w or {}).get("kind", "partly")
        self.kind = kind
        self.day = (w or {}).get("is_day", True)
        cloud = (w or {}).get("cloud")
        self.cloud = (cloud if cloud is not None else {"clear": 10, "partly": 45, "overcast": 95}.get(kind, 85)) / 100.0
        wind = (w or {}).get("wind") or 4.0
        deg = (w or {}).get("wind_deg")
        east = -math.sin(math.radians(deg)) if deg is not None else 1.0       # wind FROM west blows east (+x)
        self.drift = (3 + wind * 0.9) * (1 if east >= 0 else -1)
        self.slant = max(-0.5, min(0.5, east * wind / 40.0))
        seed = int(rng.integers(1 << 30))
        self.wl = 196
        far = land.ridge(seed, 168, 52, 5, 4)
        near = land.pines(land.ridge(seed + 1, 200, 16, 8, 3), rng, 0.4)
        self.far, self.near = far, near
        self.clouds = [gfx.value_noise((110, W * 2), (3, 10), rng, 4), gfx.value_noise((90, W * 2), (2, 7), rng, 4)]
        self.stars = land.Stars(rng, 90, 150, (0.3, 0.9))
        n = {"drizzle": 140, "rain": 300, "storm": 380, "snow": 240}.get(kind, 0)
        if w and kind == "rain" and w.get("code") in (65, 82):
            n = 460
        self.n = n
        self.px = rng.uniform(0, W, n)
        self.py = rng.uniform(-H, H, n)
        self.pv = rng.uniform(0.8, 1.2, n)
        self.flash_at = rng.uniform(2, 5)
        self.bolt = None
        self.fog = gfx.value_noise((H, W * 2), (4, 8), rng, 3) if kind == "fog" else None
        self.sunxy = (70, 64)

    def _sky(self, img, t):
        if self.day:
            land.sky_gradient(img, 0.10, 0.30, 0, self.wl)
            if self.cloud < 0.85 and self.kind not in ("rain", "storm", "snow", "fog"):
                cx, cy = self.sunxy
                gfx.disc(img, cx, cy, 13, 1.0)
                for k in range(12):
                    a = k * math.pi / 6 + t * 0.15
                    r0, r1 = 18, 25 + (4 if k % 2 else 0) + 2 * math.sin(t * 2 + k)
                    gfx.line(img, cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r1 * math.cos(a), cy + r1 * math.sin(a), 0.8)
        else:
            land.sky_gradient(img, 0.0, 0.14, 0, self.wl)
            if self.cloud < 0.9:
                self.stars.draw(img, t, gain=1 - self.cloud)
            m = almanac.moon()
            land.moon(img, 40, 40, 11, phase=m["phase"])   # top left: the readout owns the top right
            x = 60
            gfx.text_halo(img, x, 24, m["name"], 0.7, "small")
            gfx.text_halo(img, x, 36, "DAY %d  %d%% LIT" % (int(m["age"]) + 1, round(m["illum"] * 100)), 0.5, "small")
            nf, nn = m["next_full"], m["next_new"]
            if nf < nn:
                ev = "FULL %s  %s MOON" % (_day(nf), m["full_name"])
            else:
                ev = "NEW MOON %s" % _day(nn)
            gfx.text_halo(img, 22, 56, ev, 0.5, "small")

    def _clouds(self, img, t):
        if self.cloud < 0.06:
            return
        thr = 1.0 - self.cloud * 0.62
        for li, (tex, y0, sp) in enumerate(((self.clouds[0], 8, 1.0), (self.clouds[1], 30, 0.6))):
            h = tex.shape[0]
            off = int(t * self.drift * sp) % (W * 2)
            cols = (np.arange(W) + off) % (W * 2)
            c = tex[:, cols]
            dens = np.clip((c - thr) / 0.18, 0, 1)
            if dens.max() == 0:
                continue
            top = np.vstack([dens[:1].repeat(3, 0), dens[:-3]])
            lit = np.clip(dens - top * 0.7, 0, 1)                # sunward edge is brighter
            ramp = np.clip(np.minimum(np.arange(h), np.arange(h)[::-1]) / 14.0, 0, 1)[:, None]
            dens, lit = dens * ramp, lit * ramp                  # feather the layer's top/bottom: no hard seams
            base = 0.55 if self.day else 0.22
            dens = np.floor(dens * 3.999) / 3.0                       # banded shading, like a pixel artist would
            lit = np.floor(lit * 2.999) / 2.0
            val = dens * base * (0.7 + 0.3 * li) + lit * (0.35 if self.day else 0.2)
            if self.kind in ("rain", "storm"):
                val *= 0.75
            seg = img[y0:y0 + h]
            mask = dens > 0.02
            seg[mask] = np.maximum(val[mask], seg[mask] * (1 - dens[mask]))

    def _precip(self, img, t, dt):
        if not self.n:
            return
        if self.kind == "snow":
            self.py += dt * 28 * self.pv
            self.px += dt * (self.drift * 0.6 + 8 * np.sin(t * 1.3 + self.pv * 9))
            self.px %= W
            wrap = self.py > H
            self.py[wrap] -= H + 10
            gfx.plot(img, self.px, self.py, 0.9)
            big = self.pv > 1.1
            gfx.plot(img, self.px[big] + 1, self.py[big], 0.9)
        else:
            sp = 260 if self.kind != "drizzle" else 150
            self.py += dt * sp * self.pv
            self.px += dt * sp * self.pv * self.slant
            self.px %= W
            wrap = self.py > self.wl + 6
            self.py[wrap] -= self.wl + 20
            ln = 7 if self.kind != "drizzle" else 3
            for k in range(ln):
                gfx.plot(img, self.px - self.slant * k, self.py - k, 0.25 + 0.5 * k / ln)
            hit = wrap.nonzero()[0]
            gfx.plot(img, self.px[hit], self.wl + 4 + (hit % 30), 0.8)

    def _storm(self, img, t):
        if self.kind != "storm":
            return
        if t >= self.flash_at:
            rng = self.rng
            x = rng.uniform(40, 280)
            pts = [(x, 30)]
            y = 30
            while y < self.wl - 6:
                y += rng.uniform(6, 14)
                x += rng.uniform(-9, 9)
                pts.append((x, y))
            self.bolt = (t, pts)
            self.flash_at = t + rng.uniform(2.5, 7)
        if self.bolt and t - self.bolt[0] < 0.35:
            age = t - self.bolt[0]
            k = 1 - age / 0.35
            img += 0.25 * k * (YY < self.wl)
            xs, ys = zip(*self.bolt[1])
            gfx.polyline(img, xs, ys, 1.0 * k)

    def _readout(self, img):
        w = self.w
        if not w:
            gfx.text_halo(img, 16, 212, "WX DATA OFFLINE", 0.55, "small")
            return
        temp = _fmt_t(w.get("temp")) + "°"
        x1 = W - 18
        tw, th = gfx.text_size(temp, scale=3)
        gfx.text_halo(img, x1 - tw, 18, temp, 1.0, scale=3)
        cond = w.get("text", "")
        gfx.text_halo(img, x1 - gfx.text_size(cond, "small")[0], 18 + th + 4, cond, 0.85, "small")
        lab = (w.get("label") or "").upper()
        if lab:
            gfx.text_halo(img, x1 - gfx.text_size(lab, "small")[0], 18 + th + 17, lab, 0.6, "small")
        y = 18 + th + 34
        for d in (w.get("days") or [])[:2]:                     # the next two days, right-aligned under the readout
            txt = self.SHORT.get(d["text"], d["text"])
            f = "%s %s/%s %s" % (d["dow"], _fmt_t(d["hi"]), _fmt_t(d["lo"]), txt)
            if d.get("pop") and d["pop"] >= 20:
                f += " %d%%" % d["pop"]
            gfx.text_halo(img, x1 - gfx.text_size(f, "small")[0], y, f, 0.55, "small")
            y += 12
        gfx.rect(img, 0, 209, W - 1, H - 1, 0.0)
        page = int(self.t // 6) % 3
        if page == 2:
            m = almanac.moon()
            line = "MOON %s %d%%" % (m["name"].replace(" MOON", ""), round(m["illum"] * 100))
            nf, nn = m["next_full"], m["next_new"]
            right = ("FULL %s %s" % (_day(nf), m["full_name"])) if nf < nn else ("NEW %s" % _day(nn))
        elif page == 0:
            line = "HI %s°  LO %s°  WIND %s %s" % (_fmt_t(w.get("hi")), _fmt_t(w.get("lo")),
                                                 w.get("wind_dir", ""), _fmt_t(w.get("wind")))
            if w.get("wind_unit"):
                line += " " + w["wind_unit"]
            right = "SUN %s-%s" % (w.get("sunrise", ""), w.get("sunset", "")) if w.get("sunrise") else ""
        else:
            dl, dd = w.get("daylen"), w.get("daylen_delta")
            line = "DAYLIGHT %dH%02dM" % divmod(dl, 60) if dl else "DAYLIGHT --"
            if dd:
                line += " (%+dM TOMORROW)" % dd
            sh = almanac.next_shower()
            right = "%s %s" % (sh["name"], _day(sh["ts"])) if sh else ""
        gfx.text_halo(img, 16, 212, line, 0.7, "small")
        if right:
            gfx.text_halo(img, W - 16 - gfx.text_size(right, "small")[0], 212, right, 0.5, "small")

    def frame(self, t, dt):
        self.t = t
        img = gfx.canvas()
        self._sky(img, t)
        self._clouds(img, t)
        gfx.fill_below(img, self.far, 0.18 if self.day else 0.06)
        gfx.plot(img, np.arange(W), self.far, 0.45 if self.day else 0.35)
        gfx.fill_below(img, self.near, 0.0)
        land.reflect(img, self.wl, t, 0.45)
        if self.fog is not None:
            off = int(t * 6) % (W * 2)
            f = self.fog[:, (np.arange(W) + off) % (W * 2)]
            band = np.clip(1 - np.abs(YY - 170) / 70, 0, 1)
            np.maximum(img, f * band * 0.55, out=img)
        self._storm(img, t)
        self._precip(img, t, dt)
        self._readout(img)
        return np.clip(img, 0, 1)


# =====================================================================================================
class AuroraSky(Scene):
    """Aurora curtains over the mountains and lake, driven by NOAA SWPC OVATION + Kp.
    Quiet = a faint low glow on the northern horizon; active = tall, bright, fast folding curtains."""
    name, kind, hold = "aurora-sky", "aurora", (18.0, 20.0)

    def start(self, rng):
        super().start(rng)
        _, a = self.ctx["feeds"].get()
        self.a = a
        s = feeds.aurora_strength(a)
        self.s = s
        seed = int(rng.integers(1 << 30))
        self.wl = 168
        self.mtn = land.pines(land.ridge(seed, 169, 20, 6, 4), rng, 0.3, (3, 7))
        self.far = land.ridge(seed + 5, 166, 26, 4, 3)
        self.stars = land.Stars(rng, 170, 175, (0.25, 1.0))
        ncur = 1 + int(s > 0.35) + int(s > 0.65)
        self.topfade = np.clip((YY[:self.wl] - 4) / 70.0, 0, 1)
        self.cur = []
        for k in range(ncur):
            self.cur.append({
                "base": 150 - 40 * s - 20 * k - 30 * max(0, s - 0.5) * rng.uniform(0.6, 1.0),
                "bend": gfx.noise1d(2048, 22, rng, 3) * (22 + 40 * s),
                "rays": gfx.noise1d(4096, 600, rng, 2),
                "bright": gfx.noise1d(2048, 10, rng, 3),
                "h": 14 + 75 * s * rng.uniform(0.8, 1.1),
                "spd": rng.uniform(8, 18) * (0.5 + s),
                "ray_spd": rng.uniform(20, 50) * (0.6 + s),
                "gain": (0.35 + 0.65 * s) * (1.0 - 0.25 * k),
            })

    def frame(self, t, dt):
        img = gfx.canvas()
        land.sky_gradient(img, 0.0, 0.06, 0, self.wl)
        self.stars.draw(img, t, 0.9)
        xs = np.arange(W)
        ycol = YY[:self.wl]
        for c in self.cur:
            o = int(t * c["spd"])
            base = c["base"] + c["bend"][(xs + o) % 2048] - c["bend"].mean()
            ray = c["rays"][(xs * 3 + int(t * c["ray_spd"])) % 4096]
            ray = 0.45 + 0.55 * np.clip((ray - 0.3) * 2.0, 0, 1)
            br = 0.25 + 0.75 * c["bright"][(xs + int(t * 7) + o // 2) % 2048] ** 1.5
            fold = 1.0 + np.clip(np.abs(np.gradient(base)) * 0.9, 0, 1.2)       # sheets seen edge-on glow brighter
            up = np.clip(base[None, :] - ycol, 0, None)
            down = np.clip(ycol - base[None, :], 0, None)
            hcol = c["h"] * (0.6 + 0.8 * br)[None, :]
            prof = np.where(up > 0, np.exp(-up / hcol) * (0.55 + 0.45 * np.exp(-up / 6.0)), np.exp(-down / 1.6))
            v = prof * np.clip(ray * br * fold * c["gain"], 0, 1)[None, :] * self.topfade
            np.maximum(img[:self.wl], v, out=img[:self.wl])
        gfx.fill_below(img, self.far, 0.0)
        gfx.fill_below(img, self.mtn, 0.0)
        land.reflect(img, self.wl, t, 0.5)
        self._status(img, t)
        return np.clip(img, 0, 1)

    def _status(self, img, t):
        a = self.a
        gfx.rect(img, 0, 208, W - 1, H - 1, 0.0)
        if not a or a.get("kp") is None and a.get("visible") is None:
            gfx.text(img, 16, 211, "NOAA SWPC: NO SIGNAL  QUIET SKY", 0.5, "small")
            return
        kp = a.get("kp")
        left = "AURORA  KP %.1f  %s" % (kp, a.get("activity", "")) if kp is not None else "AURORA"
        gfx.text(img, 16, 211, left, 0.8, "small")
        bits = []
        if a.get("visible") is not None:
            bits.append("IN VIEW %d%%" % round(a["visible"]))
        if a.get("kp_max24") is not None:
            bits.append("24H KP %.1f" % a["kp_max24"])
        r = "  ".join(bits)
        gfx.text(img, W - 16 - gfx.text_size(r, "small")[0], 211, r, 0.55, "small")


# =====================================================================================================
class AuroraOval(Scene):
    """The OVATION 30-minute forecast as a polar map: oval brightness, graticule, home blinking, the sun's side."""
    name, kind, hold = "aurora-oval", "aurora", (14.0, 18.0)
    LAT_MIN = 30.0

    def start(self, rng):
        super().start(rng)
        _, a = self.ctx["feeds"].get()
        self.a = a
        loc = self.ctx["loc"]
        self.hlat, self.hlon = float(loc["lat"]), float(loc["lon"])
        self.cx, self.cy, self.R = 116, 120, 106
        dx, dy = XX - self.cx, YY - self.cy
        rr = np.sqrt(dx * dx + dy * dy)
        self.inside = rr <= self.R
        lat = 90 - rr / self.R * (90 - self.LAT_MIN)
        # home longitude points straight down (6 o'clock); east is counter-clockwise looking down on the pole
        lon = (self.hlon + np.degrees(np.arctan2(dx, dy))) % 360
        self.lat, self.lon = lat, lon
        grid = a.get("grid") if a else None
        if grid:
            g = np.array(grid, np.float32)
            v = g[np.clip(lat.round().astype(int), 0, 90), lon.round().astype(int) % 360]
            self.offline = False
        else:                                                  # model offline: a modest synthetic oval
            ring = np.exp(-((lat - 66 + 3 * np.cos(np.radians(lon - self.hlon))) / 3.5) ** 2) * 18
            v = ring.astype(np.float32)
            self.offline = True
        self.val = np.where(self.inside & (lat >= self.LAT_MIN), np.clip(v / 35.0, 0, 1) ** 0.7, 0).astype(np.float32)
        self.shimmer = gfx.value_noise((H, W), (8, 10), rng, 2)
        base = gfx.canvas()
        for la in (40, 50, 60, 70, 80):
            gfx.ring(base, self.cx, self.cy, (90 - la) / (90 - self.LAT_MIN) * self.R, 0.2, 180)
        for k in range(12):
            ang = k * math.pi / 6
            r0 = 0.12 * self.R
            gfx.lines(base, [(self.cx + r0 * math.sin(ang), self.cy + r0 * math.cos(ang),
                              self.cx + self.R * math.sin(ang), self.cy + self.R * math.cos(ang))], 0.12)
        gfx.ring(base, self.cx, self.cy, self.R, 0.45, 400)
        self.base = base
        hr = (90 - self.hlat) / (90 - self.LAT_MIN) * self.R
        self.home = (self.cx, self.cy + hr)
        utc = time.gmtime()
        sub_lon = (12 - (utc.tm_hour + utc.tm_min / 60)) * 15                  # longitude where it is noon
        ang = math.radians(sub_lon - self.hlon)
        self.sun = (self.cx + (self.R + 7) * math.sin(ang), self.cy + (self.R + 7) * math.cos(ang))

    def frame(self, t, dt):
        img = self.base.copy()
        sh = np.roll(self.shimmer, int(t * 9), axis=1)
        v = self.val * (0.75 + 0.35 * sh) * min(1.0, t / 2.0)
        np.maximum(img, v, out=img)
        hx, hy = self.home
        if int(t * 2) % 2 == 0:
            gfx.lines(img, [(hx - 4, hy, hx + 4, hy), (hx, hy - 4, hx, hy + 4)], 1.0)
        else:
            gfx.plot(img, [hx], [hy], 1.0)
        sx, sy = self.sun
        gfx.disc(img, sx, sy, 3, 0.9)
        gfx.ring(img, sx, sy, 5.5, 0.5)
        self._panel(img)
        return np.clip(img, 0, 1)

    def _panel(self, img):
        a = self.a or {}
        x = 228
        gfx.text(img, x, 16, "AURORA", 1.0)
        gfx.text(img, x, 28, "OVATION", 0.6, "small")
        y = 46
        rows = []
        kp = a.get("kp")
        rows.append(("KP NOW", "%.1f" % kp if kp is not None else "--"))
        rows.append(("STATE", a.get("activity", "NO DATA") if a else "NO DATA"))
        if a.get("kp_max24") is not None:
            rows.append(("24H MAX", "%.1f" % a["kp_max24"]))
        if a.get("visible") is not None:
            rows.append(("IN VIEW", "%d%%" % round(a["visible"])))
        if a.get("overhead") is not None:
            rows.append(("OVERHEAD", "%d%%" % a["overhead"]))
        for k, v in rows:
            gfx.text(img, x, y, k, 0.5, "small")
            gfx.text(img, x, y + 11, v, 0.95, "small")
            y += 26
        ft = (a.get("forecast") or "")[11:16]
        label = "MODEL OFFLINE" if self.offline else ("FCST %sZ" % ft if ft else "")
        if label:
            gfx.text(img, x, 200, label, 0.45, "small")
        gfx.text(img, x, 212, "+ HOME  o SUN", 0.45, "small")
