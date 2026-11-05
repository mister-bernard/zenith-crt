#!/usr/bin/env python3
"""ZENITH: a green-phosphor pixel-art slideshow for a Zenith ZVM-123 on Raspberry Pi composite video.

README
======
Run (on the Pi, from the console or as the systemd service):
    python3 zenith.py                         # full screen on the composite output (direct KMS, no X/SDL needed)
    python3 zenith.py --list                  # list the scenes
    python3 zenith.py --only aurora-sky       # loop one scene (handy while tuning)
    python3 zenith.py --bench                 # time every scene's render on this CPU, no display
    python3 zenith.py --preview out.mp4 --seconds 120   # headless: render with a simulated tube (for screens)
    python3 zenith.py --snap DIR              # headless: one PNG per scene (simulated tube)

Packages (Debian/Raspberry Pi OS):  apt install python3-numpy   (that is all the default KMS backend needs)
    python3-pygame is an optional second backend (SDL KMSDRM, or a window on a desktop); /dev/fb0 is the last resort.
    Headless --preview/--snap also want ffmpeg (video) or Pillow (PNG); the display path needs neither.

Config: ./config.toml next to this file, or ~/.config/zenith/config.toml (see config.example.toml):
    [location] label/lat/lon/units   -- weather + aurora location; if absent, IP geolocation, then San Francisco
    [display]  fps, levels, mode (auto|240p|480i), output (grey|green), crt_sim, tao (faint Tao line on quiet scenes), margin_x/margin_y (margin_top/margin_bottom override), black/white/gamma (kms output curve),
               backend (auto|kms|pygame|fbdev)
    [slides]   min_seconds, max_seconds, transition_min, transition_max, exclude = [names]

Why the output is GREY, not green: a monochrome composite monitor displays luma only. Sending #00FF00 gives
~59% brightness (Y = 0.587 G) and puts a colour subcarrier on the signal that a mono tube shows as dot crawl.
Equal R=G=B carries no chroma, so we send grey and the P31 phosphor makes it green. Use output = "green"
only on a colour screen.

Resolution: everything is drawn on a 320x240 canvas (40x30 text cells of the IBM 8x8 font). On 480i every
canvas line is drawn twice (once per field), so one-pixel details never flicker at 30 Hz. If the kernel offers
a 240-line progressive mode it is used instead and the tube draws its own real scanlines.

Data: weather from Open-Meteo (no key), aurora from NOAA SWPC (OVATION 30-min forecast + planetary Kp).
Fetched in a background thread, cached in ~/.cache/zenith, every scene copes with no data.

Logs go to stderr (journald under systemd). Ctrl+C / SIGTERM exit cleanly and blank the screen.
"""
import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import feeds  # noqa: E402
import gfx  # noqa: E402
import scenes_classic as sc  # noqa: E402
import scenes_gen as sg  # noqa: E402
import scenes_live as sl  # noqa: E402
import tao  # noqa: E402
from gfx import H, W  # noqa: E402

log = logging.getLogger("zenith")

SCENES = [sc.Terminal, sc.VectorFlight, sc.LakeNight, sc.Circuit, sc.CopperScroll, sc.Tunnel, sc.Wireframe,
          sc.AsciiDonut, sc.Planet, sc.Radar,
          sg.Life, sg.Elementary, sg.Ridgelines, sg.FlowField, sg.Harmonograph, sg.Mandelbrot, sg.Tree, sg.Plasma,
          sl.Weather, sl.AuroraSky, sl.AuroraOval]

DEFAULTS = {
    "location": {},
    "display": {"fps": 30, "levels": 6, "mode": "auto", "output": "grey", "crt_sim": False,
                "margin_x": 0, "margin_y": 0, "backend": "auto",
                "black": 0.0, "white": 1.0, "gamma": 1.0},
    "slides": {"min_seconds": 8, "max_seconds": 20, "transition_min": 1.0, "transition_max": 2.5, "exclude": []},
}


def load_config(path=None):
    import tomllib
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    cands = [Path(path)] if path else [HERE / "config.toml", Path.home() / ".config/zenith/config.toml"]
    for p in cands:
        if p.is_file():
            with open(p, "rb") as f:
                user = tomllib.load(f)
            for k, v in user.items():
                if isinstance(v, dict):
                    cfg.setdefault(k, {}).update(v)
            log.info("config: %s", p)
            break
    return cfg


# ===================================================================================== transitions
_RANK = None


def _rank():
    """A fixed random order of 2x2 cells (fizzle dissolve), like an LFSR walk."""
    global _RANK
    if _RANK is None:
        r = np.random.default_rng(1983).permutation((H // 2) * (W // 2)).reshape(H // 2, W // 2) / ((H // 2) * (W // 2))
        _RANK = np.repeat(np.repeat(r, 2, 0), 2, 1).astype(np.float32)
    return _RANK


def _smooth(p):
    return p * p * (3 - 2 * p)


def tr_fade(a, b, p):
    """Fade through black."""
    return a * max(0.0, 1 - 2 * p) if p < 0.5 else b * (2 * p - 1)


def tr_dither(a, b, p):
    """Ordered-dither dissolve (8x8 Bayer threshold)."""
    return np.where(gfx.BAYER8_FULL < _smooth(p), b, a)


def tr_fizzle(a, b, p):
    return np.where(_rank() < p, b, a)


def tr_sweep(a, b, p):
    """A bright beam sweeps down and leaves the new picture behind it."""
    y = int(_smooth(p) * (H + 6))
    out = a.copy()
    out[:max(0, min(H, y))] = b[:max(0, min(H, y))]
    for k, v in ((0, 1.0), (1, 0.6), (2, 0.3)):
        if 0 <= y - k < H:
            out[y - k] = np.maximum(out[y - k], v)
    return out


def tr_pixelate(a, b, p):
    """Soft pixelate out, switch, de-pixelate in."""
    sizes = [1, 2, 4, 5, 8, 10, 16, 20]
    q = 1 - abs(2 * p - 1)
    s = sizes[min(len(sizes) - 1, int(q * len(sizes)))]
    img = a if p < 0.5 else b
    if s == 1:
        return img
    m = img.reshape(H // s, s, W // s, s).mean(axis=(1, 3))
    return np.repeat(np.repeat(m, s, 0), s, 1)


def tr_roll(a, b, p):
    """Vertical-hold slip: the old frame rolls up past a black blanking bar, the new one rolls in."""
    bar = 18
    off = int(_smooth(p) * (H + bar))
    tall = np.vstack([a, np.zeros((bar, W), np.float32), b])
    return tall[off:off + H]


def tr_interlace(a, b, p):
    """Odd lines first, then even (a nod to interlaced fields)."""
    out = a.copy()
    q = _smooth(p) * 2
    rows = np.arange(H)
    first = (rows % 2 == 0) & (rows < q * H)
    second = (rows % 2 == 1) & (rows < (q - 1) * H)
    out[first | second] = b[first | second]
    return out


TRANSITIONS = [tr_fade, tr_dither, tr_fizzle, tr_sweep, tr_pixelate, tr_roll, tr_interlace]


# ===================================================================================== deck
class Deck:
    """A shuffled programme per cycle: classic and generative interleaved, weather and aurora once each."""

    def __init__(self, ctx, cfg, rng, only=None):
        ex = set(cfg["slides"].get("exclude") or [])
        self.pool = [cls for cls in SCENES if cls.name not in ex and (not only or cls.name in only)]
        if not self.pool:
            raise SystemExit("no scenes selected")
        self.ctx, self.cfg, self.rng = ctx, cfg, rng
        self.queue = []
        self.last = None

    def _cycle(self):
        by = {}
        for c in self.pool:
            by.setdefault(c.kind, []).append(c)
        cl = list(by.get("classic", []))
        ge = list(by.get("generative", []))
        self.rng.shuffle(cl)
        self.rng.shuffle(ge)
        seq = []
        while cl or ge:
            if cl:
                seq.append(cl.pop())
            if ge:
                seq.append(ge.pop())
        live = by.get("weather", []) + by.get("aurora", [])
        n = len(seq)
        for i, c in enumerate(live):                                   # spread the live pieces through the cycle
            seq.insert(min(len(seq), int((i + 1) * n / (len(live) + 1)) + i), c)
        if seq and self.last is not None and seq[0] is self.last and len(seq) > 1:
            seq.append(seq.pop(0))
        return seq

    def next(self, force=None):
        """force: a scene name to play now (remote control); the shuffled programme resumes after it."""
        cls = next((c for c in SCENES if c.name == force), None) if force else None
        if cls is None:
            if not self.queue:
                self.queue = self._cycle()
            cls = self.queue.pop(0)
        self.last = cls
        s = cls(self.ctx)
        lo, hi = cls.hold
        mn, mx = float(self.cfg["slides"]["min_seconds"]), float(self.cfg["slides"]["max_seconds"])
        lo, hi = max(lo, mn), min(hi, mx)
        hold = self.rng.uniform(lo, hi) if hi > lo else max(mn, min(mx, lo))
        return s, hold


# ===================================================================================== output
def to_levels(img, levels):
    return gfx.dither(img, levels) if levels else np.clip(img, 0, 1)


def crt_sim(img, out_w=640, out_h=480, tint="green"):
    """For screens that are not a tube: P31-ish colour, horizontal beam blur, bloom, scanlines, vignette.
    Returns uint8 RGB (out_h, out_w, 3)."""
    sy, sx = out_h // H, out_w // W
    big = np.repeat(np.repeat(img, sy, 0), sx, 1)
    k = np.array([0.15, 0.7, 0.15], np.float32)                          # composite bandwidth smear
    big = k[0] * np.roll(big, 1, 1) + k[1] * big + k[2] * np.roll(big, -1, 1)
    bloom = gfx_blur_any(big, 4)
    v = np.clip(big * 0.95 + bloom * 0.45, 0, 1.2)
    scan = np.ones(out_h, np.float32)
    scan[sy - 1::sy] = 0.55
    v = v * scan[:, None]
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    vig = 1 - 0.35 * (((xx - out_w / 2) / (out_w / 2)) ** 2 + ((yy - out_h / 2) / (out_h / 2)) ** 2) ** 1.5
    v = np.clip(v * vig, 0, 1.2)
    if tint == "grey":
        rgb = np.stack([v, v, v], -1)
    else:                                                               # P31: yellow-green, whiter when saturated
        rgb = np.stack([0.22 * v + 0.55 * np.clip(v - 0.9, 0, 1), 1.0 * v, 0.30 * v + 0.4 * np.clip(v - 0.9, 0, 1)], -1)
        rgb += 0.012                                                   # faint glass glow of a lit tube
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def gfx_blur_any(img, r):
    k = 2 * r + 1
    p = np.pad(img, ((0, 0), (r + 1, r)), mode="edge")
    c = np.cumsum(p, axis=1)
    o = (c[:, k:] - c[:, :-k]) / k
    p = np.pad(o, ((r + 1, r), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0)
    return (c[k:, :] - c[:-k, :]) / k


class PygameOut:
    """Full screen through SDL. On the Pi console this is KMSDRM straight to the composite connector."""

    def __init__(self, cfg):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.display.init()
        modes = pygame.display.list_modes() or []
        info = pygame.display.Info()
        want = cfg["display"].get("mode", "auto")
        size = (info.current_w, info.current_h)
        p240 = [m for m in modes if m[1] in (240, 288) and m[0] >= 640]
        if want in ("auto", "240p") and p240:
            size = p240[0]
        elif want == "480i" and (720, 480) in modes:
            size = (720, 480)
        windowed = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if windowed and (size[0] <= 0 or size[0] > 1600):
            size = (720, 480)
        flags = 0 if windowed else pygame.FULLSCREEN
        self.screen = pygame.display.set_mode(size, flags)
        pygame.mouse.set_visible(False)
        pygame.display.set_caption("zenith")
        self.size = self.screen.get_size()
        mx, my = int(cfg["display"].get("margin_x", 0)), int(cfg["display"].get("margin_y", 0))
        self.dest = pygame.Rect(mx, my, self.size[0] - 2 * mx, self.size[1] - 2 * my)
        self.surf = pygame.Surface((W, H), depth=8)
        tint = cfg["display"].get("output", "grey")
        pal = [(0, v, 0) if tint == "green" else (v, v, v) for v in range(256)]
        self.surf.set_palette(pal)
        log.info("display: SDL %s, modes %s, using %s (dest %s)", pygame.display.get_driver(), modes[:8], self.size,
                 tuple(self.dest))

    def show(self, img):
        a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        self.pg.surfarray.blit_array(self.surf, a.T)
        if self.dest.size == (W, H):
            self.screen.blit(self.surf, self.dest)
        else:
            self.screen.blit(self.pg.transform.scale(self.surf, self.dest.size), self.dest)
        self.pg.display.flip()
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key in (self.pg.K_ESCAPE, self.pg.K_q)):
                raise KeyboardInterrupt

    def close(self):
        try:
            self.screen.fill((0, 0, 0))
            self.pg.display.flip()
        finally:
            self.pg.quit()


class FbOut:
    """Fallback: write straight to /dev/fb0 (fbdev emulation of the KMS driver)."""

    def __init__(self, cfg, dev="/dev/fb0"):
        base = Path("/sys/class/graphics") / Path(dev).name
        self.w, self.h = map(int, (base / "virtual_size").read_text().strip().split(","))
        self.bpp = int((base / "bits_per_pixel").read_text())
        self.stride = int((base / "stride").read_text()) if (base / "stride").exists() else self.w * self.bpp // 8
        self.f = open(dev, "r+b", buffering=0)
        import mmap
        self.mm = mmap.mmap(self.f.fileno(), self.stride * self.h)
        self.yi = (np.arange(self.h) * H // self.h).clip(0, H - 1)
        self.xi = (np.arange(self.w) * W // self.w).clip(0, W - 1)
        self.green = cfg["display"].get("output", "grey") == "green"
        log.info("display: fbdev %dx%d %dbpp", self.w, self.h, self.bpp)

    def show(self, img):
        v = (np.clip(img, 0, 1) * 255).astype(np.uint16)[self.yi][:, self.xi]
        if self.bpp == 16:
            g6 = (v >> 2) << 5
            px = g6 if self.green else ((v >> 3) << 11) | g6 | (v >> 3)
            buf = np.zeros((self.h, self.stride // 2), np.uint16)
            buf[:, :self.w] = px
        else:
            v32 = v.astype(np.uint32)
            px = (v32 << 8) if self.green else (v32 << 16) | (v32 << 8) | v32
            buf = np.zeros((self.h, self.stride // 4), np.uint32)
            buf[:, :self.w] = px
        self.mm[:] = buf.tobytes()

    def close(self):
        self.mm[:] = bytes(len(self.mm))
        self.mm.close()
        self.f.close()


# ===================================================================================== the show
class Show:
    def __init__(self, cfg, offline=False, seed=None, only=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.feeds = feeds.Feeds(cfg, offline=offline)
        self.ctx = {"feeds": self.feeds, "cfg": cfg, "loc": self.feeds.loc}
        self.deck = Deck(self.ctx, cfg, self.rng, only)
        self.levels = int(cfg["display"].get("levels", 6))
        s = cfg["slides"]
        self.tmin, self.tmax = float(s["transition_min"]), float(s["transition_max"])
        self.tao = bool(cfg["display"].get("tao", True))
        self.tao_lines = None
        self.control = None if offline else feeds.CACHE / "control"
        self.ctl_checked = 0.0
        self.pinned = False

    # remote control: write a scene name, "next", "pin <name>" or "unpin" into ~/.cache/zenith/control
    def _control(self):
        try:
            cmd = self.control.read_text().strip().lower()
            self.control.unlink()
        except OSError:
            return None
        if not cmd:
            return None
        log.info("control: %s", cmd)
        names = {c.name for c in SCENES}
        if cmd == "unpin":
            self.pinned = False
            return "next"
        if cmd.startswith("pin "):
            cmd = cmd[4:].strip()
            if cmd in names:
                self.pinned = True
                return cmd
        if cmd == "next" or cmd in names:
            self.pinned = False if cmd == "next" else self.pinned
            return cmd
        log.warning("control: unknown command %r", cmd)
        return None

    def _status(self, scene, hold):
        if not self.control:
            return
        try:
            (self.control.parent / "now.json").write_text(json.dumps(
                {"scene": scene.name, "since": time.time(), "hold": hold if hold < 1e8 else None, "pinned": self.pinned}))
        except OSError:
            pass

    def frames(self, fps, clock=None):
        """Yield finished 0..1 frames forever. clock() gives seconds (real time live, frame time offline)."""
        clock = clock or time.monotonic
        scene, hold = self.deck.next()
        scene.start(np.random.default_rng(self.rng.integers(1 << 62)))
        t0 = clock()
        last = t0
        trans = None
        cached = None
        log.info("scene %s (%.1fs)", scene.name, hold)
        self._status(scene, hold)
        while True:
            now = clock()
            dt = min(0.2, max(1e-3, now - last))
            last = now
            t = now - t0
            if scene.rate and cached is not None and now - cached[0] < 1.0 / scene.rate:
                img = cached[1]
            else:
                try:
                    img = scene.frame(t, now - cached[0] if (scene.rate and cached is not None) else dt)
                except Exception:                                    # a broken scene must never stop the show
                    log.exception("scene %s failed; skipping", scene.name)
                    img, hold = gfx.canvas(), 0
                lv = self.levels if scene.levels is None else scene.levels
                if self.tao and getattr(scene, "tao", None) and lv:
                    if self.tao_lines is None:
                        self.tao_lines = tao.pick(self.rng)[0]
                    img = np.array(img, np.float32, copy=True)
                    tao.draw(img, self.tao_lines, scene.tao)
                img = to_levels(img, lv)
                cached = (now, img)
            if trans:
                tf, a_img, t_start, t_len = trans
                p = (now - t_start) / t_len
                if p >= 1:
                    trans = None
                else:
                    img = tf(a_img, img, max(0.0, p))
            yield img
            force = None
            if self.control and now - self.ctl_checked > 0.5:
                self.ctl_checked = now
                force = self._control()
                if force == "next":
                    force, hold = None, 0
                elif force:
                    hold = 0
            if t >= hold and not trans:
                prev = img
                scene, hold = self.deck.next(force)
                self.tao_lines = None
                if self.pinned:
                    hold = 1e9
                scene.start(np.random.default_rng(self.rng.integers(1 << 62)))
                t0 = last = clock()
                cached = None
                tf = TRANSITIONS[int(self.rng.integers(len(TRANSITIONS)))]
                tl = float(self.rng.uniform(self.tmin, self.tmax))
                trans = (tf, prev, t0, tl)
                log.info("scene %s (%.1fs) via %s %.1fs", scene.name, hold, tf.__name__[3:], tl)
                self._status(scene, hold)


def open_output(cfg):
    """backend: auto = KMS dumb buffers (no SDL/EGL needed) -> pygame/SDL -> /dev/fb0."""
    import kms
    want = cfg["display"].get("backend", "auto")
    order = {"auto": ["kms", "pygame", "fbdev"], "kms": ["kms"], "pygame": ["pygame"], "fbdev": ["fbdev"]}[want]
    if want == "auto" and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        order = ["pygame"]                                             # a desktop session: just open a window
    last = None
    for b in order:
        try:
            return {"kms": kms.KmsOut, "pygame": PygameOut, "fbdev": FbOut}[b](cfg)
        except Exception as e:                                         # noqa: BLE001 - try the next one
            log.warning("display backend %s unavailable: %s", b, e)
            last = e
    raise SystemExit("no usable display: %s" % last)


STOP = False


def _stop(*_):
    global STOP
    STOP = True


def run_live(cfg, args):
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    show = Show(cfg, offline=args.offline, only=args.only)
    show.feeds.start()
    out = None
    try:
        out = open_output(cfg)
        fps = float(cfg["display"].get("fps", 30))
        period = 1.0 / fps
        slow, n, t_stat = 0, 0, time.monotonic()
        nxt = time.monotonic()
        for img in show.frames(fps):
            if cfg["display"].get("crt_sim"):
                img = crt_sim(img, W, H, "grey")[:, :, 1] / 255.0
            out.show(img)
            n += 1
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                slow += 1
                nxt = time.monotonic()
            if time.monotonic() - t_stat > 300:
                log.info("fps %.1f (late frames %d)", n / (time.monotonic() - t_stat), slow)
                n, slow, t_stat = 0, 0, time.monotonic()
            if STOP:
                break
    except KeyboardInterrupt:
        pass
    finally:
        show.feeds.stop()
        if out:
            out.close()
        log.info("bye")


def run_offline(cfg, args):
    """Render with a frame clock (not real time): --preview video, --snap PNGs, --bench timings."""
    fps = float(args.fps or cfg["display"].get("fps", 30))
    if args.bench:
        return bench(cfg, args)
    if args.snap:
        from PIL import Image
        d = Path(args.snap)
        d.mkdir(parents=True, exist_ok=True)
        f = feeds.Feeds(cfg, offline=args.offline)
        if not args.offline:
            f.fetch_weather()
            f.fetch_aurora()
        ctx = {"feeds": f, "cfg": cfg, "loc": f.loc}
        for cls in SCENES:
            if args.only and cls.name not in args.only:
                continue
            s = cls(ctx)
            s.start(np.random.default_rng(args.seed))
            img = None
            t = 0.0
            while t < args.at:
                img = s.frame(t, 1 / fps)
                t += 1 / fps
            lv = int(cfg["display"]["levels"]) if s.levels is None else s.levels
            img = to_levels(img, lv)
            Image.fromarray(crt_sim(img)).save(d / f"{cls.name}.png")
            Image.fromarray((img * 255).astype(np.uint8)).save(d / f"{cls.name}-raw.png")
            print(d / f"{cls.name}.png")
        return
    import subprocess
    show = Show(cfg, offline=args.offline, seed=args.seed, only=args.only)
    if not args.offline:
        show.feeds.fetch_weather()
        show.feeds.fetch_aurora()
    clk = [0.0]
    n = int(args.seconds * fps)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-preset", "medium", args.preview], stdin=subprocess.PIPE)
    for i, img in enumerate(show.frames(fps, clock=lambda: clk[0])):
        ff.stdin.write(crt_sim(img).tobytes())
        clk[0] += 1 / fps
        if i + 1 >= n:
            break
    ff.stdin.close()
    ff.wait()
    print(args.preview)


def bench(cfg, args):
    f = feeds.Feeds(cfg, offline=True)
    ctx = {"feeds": f, "cfg": cfg, "loc": f.loc}
    rows = []
    for cls in SCENES:
        if args.only and cls.name not in args.only:
            continue
        s = cls(ctx)
        t0 = time.perf_counter()
        s.start(np.random.default_rng(1))
        st = time.perf_counter() - t0
        times = []
        t = 0.0
        for _ in range(60):
            a = time.perf_counter()
            img = s.frame(t, 1 / 30)
            lv = 6 if s.levels is None else s.levels
            to_levels(img, lv)
            times.append(time.perf_counter() - a)
            t += 1 / 30
        times.sort()
        rows.append((cls.name, st * 1000, np.mean(times) * 1000, times[int(len(times) * 0.95)] * 1000))
    a, b = np.zeros((H, W), np.float32), np.ones((H, W), np.float32)
    for tf in TRANSITIONS:
        t0 = time.perf_counter()
        for i in range(30):
            to_levels(tf(a, b, i / 30), 6)
        rows.append(("tr:" + tf.__name__[3:], 0.0, (time.perf_counter() - t0) / 30 * 1000, 0.0))
    print("%-16s %8s %8s %8s" % ("scene", "start_ms", "mean_ms", "p95_ms"))
    for r in rows:
        print("%-16s %8.1f %8.1f %8.1f" % r)
    worst = max(r[2] for r in rows)
    print("worst mean %.1f ms -> sustainable ~%.0f fps (plus display blit)" % (worst, 1000 / max(worst, 1e-3)))


def main():
    ap = argparse.ArgumentParser(description="Green-phosphor pixel-art slideshow for the Zenith ZVM-123")
    ap.add_argument("--config")
    ap.add_argument("--only", nargs="*", help="scene names to include")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--offline", action="store_true", help="no network (data scenes show their fallbacks)")
    ap.add_argument("--preview", metavar="OUT.mp4")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--snap", metavar="DIR")
    ap.add_argument("--at", type=float, default=6.0, help="--snap: seconds into each scene")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    if args.list:
        for c in SCENES:
            print("%-14s %-10s %s" % (c.name, c.kind, (c.__doc__ or "").strip().splitlines()[0]))
        return
    if args.preview or args.snap or args.bench:
        return run_offline(cfg, args)
    run_live(cfg, args)


if __name__ == "__main__":
    main()
