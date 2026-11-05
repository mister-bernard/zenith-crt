#!/usr/bin/env python3
"""Offline tests for the Zenith slideshow: every scene, every transition, the deck, the data parsers and
every weather/aurora state render without error and stay inside the 320x240, 0..1 contract.
Run: python3 -m unittest test_zenith -v   (needs numpy only; no display, no network)."""
import math
import time
import unittest

import numpy as np

import feeds
import gfx
import zenith
from gfx import H, W


class FakeFeeds:
    def __init__(self, weather=None, aurora=None):
        self.w, self.a = weather, aurora
        self.loc = {"lat": 37.62, "lon": -122.38, "label": "TEST"}

    def get(self):
        return self.w, self.a


def ctx(weather=None, aurora=None):
    f = FakeFeeds(weather, aurora)
    return {"feeds": f, "cfg": zenith.DEFAULTS, "loc": f.loc}


def run_scene(cls, c, seconds=4.0, fps=15, seed=1):
    s = cls(c)
    s.start(np.random.default_rng(seed))
    t, img = 0.0, None
    while t < seconds:
        img = s.frame(t, 1 / fps)
        t += 1 / fps
    lv = 6 if s.levels is None else s.levels
    return zenith.to_levels(img, lv)


def check(tc, img, name):
    tc.assertEqual(img.shape, (H, W), name)
    tc.assertTrue(np.isfinite(img).all(), name)
    tc.assertGreaterEqual(float(img.min()), 0.0, name)
    tc.assertLessEqual(float(img.max()), 1.0 + 1e-6, name)


WEATHER = {"label": "TEST", "code": 63, "kind": "rain", "text": "RAIN", "temp": 41.2, "hi": 46, "lo": 35,
           "humidity": 90, "cloud": 100, "precip": 1.2, "pop": 90, "wind": 12, "gust": 20, "wind_dir": "SW",
           "wind_deg": 225, "is_day": True, "sunrise": "06:38", "sunset": "17:07", "units": "°F", "wind_unit": "MPH"}


def aurora(kp, vis):
    grid = [[0] * 360 for _ in range(91)]
    for lon in range(360):
        for lat in range(60, 72):
            grid[lat][lon] = int(vis)
    return {"grid": grid, "overhead": 0, "visible": vis, "kp": kp, "kp_max24": kp + 1, "activity": feeds.activity(kp),
            "forecast": "2026-11-02T05:27:00Z"}


class Scenes(unittest.TestCase):
    def test_every_scene_renders(self):
        for cls in zenith.SCENES:
            with self.subTest(scene=cls.name):
                img = run_scene(cls, ctx(WEATHER, aurora(3.3, 20)))
                check(self, img, cls.name)
                self.assertGreater(float(img.mean()), 0.003, cls.name + " is blank at t=4s")

    def test_live_scenes_without_data(self):
        for cls in (zenith.sl.Weather, zenith.sl.AuroraSky, zenith.sl.AuroraOval):
            with self.subTest(scene=cls.name):
                check(self, run_scene(cls, ctx(None, None)), cls.name)

    def test_every_weather_kind(self):
        for code in sorted(feeds.WMO):
            for day in (True, False):
                w = dict(WEATHER, code=code, kind=feeds.WMO[code][0], text=feeds.WMO[code][1], is_day=day,
                         cloud={"clear": 5, "partly": 50}.get(feeds.WMO[code][0], 100))
                with self.subTest(code=code, day=day):
                    check(self, run_scene(zenith.sl.Weather, ctx(w), seconds=7.0, fps=10), "weather")

    def test_aurora_strength_changes_the_sky(self):
        quiet = run_scene(zenith.sl.AuroraSky, ctx(None, aurora(1.0, 2)), seconds=3)
        storm = run_scene(zenith.sl.AuroraSky, ctx(None, aurora(7.0, 90)), seconds=3)
        sky = slice(0, 150)
        self.assertGreater(float(storm[sky].mean()), float(quiet[sky].mean()) * 1.5)

    def test_scene_text_glyphs_exist(self):
        for s in ("°", "%", "'", "·", "+", "-", ">", "▄", "█"):
            self.assertIn(s, gfx.fontdata.VGA8[2])


class Transitions(unittest.TestCase):
    def test_endpoints_and_shape(self):
        a, b = np.zeros((H, W), np.float32), np.full((H, W), 0.8, np.float32)
        for tf in zenith.TRANSITIONS:
            with self.subTest(tr=tf.__name__):
                for p in np.linspace(0, 0.999, 13):
                    out = tf(a, b, float(p))
                    check(self, np.asarray(out, np.float32), tf.__name__)
                end = np.asarray(tf(a, b, 0.999), np.float32)
                self.assertGreater(float(end.mean()), 0.6, tf.__name__ + " should be ~all B at the end")


class Deck(unittest.TestCase):
    def test_cycle_has_everything_once_and_no_repeats(self):
        d = zenith.Deck(ctx(), zenith.DEFAULTS, np.random.default_rng(5))
        names = [d.next()[0].name for _ in range(len(zenith.SCENES) * 3)]
        first = names[:len(zenith.SCENES)]
        self.assertEqual(sorted(first), sorted(c.name for c in zenith.SCENES))
        self.assertTrue(all(a != b for a, b in zip(names, names[1:])))

    def test_hold_inside_config(self):
        cfg = {**zenith.DEFAULTS, "slides": dict(zenith.DEFAULTS["slides"], min_seconds=8, max_seconds=20)}
        d = zenith.Deck(ctx(), cfg, np.random.default_rng(1))
        for _ in range(50):
            _, hold = d.next()
            self.assertTrue(8 <= hold <= 20, hold)

    def test_show_runs_through_scene_changes(self):
        cfg = {k: dict(v) for k, v in zenith.DEFAULTS.items()}
        cfg["slides"].update(min_seconds=1, max_seconds=1.5, transition_min=0.3, transition_max=0.5)
        show = zenith.Show(cfg, offline=True, seed=3)
        clk = [0.0]
        for i, img in enumerate(show.frames(10, clock=lambda: clk[0])):
            check(self, img, "show")
            clk[0] += 0.1
            if i > 400:                                     # ~40 s of show time, dozens of scene changes
                break


    def test_remote_control(self):
        import json, pathlib, tempfile
        d = pathlib.Path(tempfile.mkdtemp())
        cfg = {k: dict(v) for k, v in zenith.DEFAULTS.items()}
        show = zenith.Show(cfg, offline=True, seed=1)
        show.control = d / "control"
        clk, seen = [0.0], []
        g = show.frames(10, clock=lambda: clk[0])
        for i in range(500):
            next(g)
            clk[0] += 0.1
            if i == 20:
                (d / "control").write_text("radar")
            if i == 60:
                (d / "control").write_text("pin tunnel")
            if i == 400:
                (d / "control").write_text("unpin")
            st = json.loads((d / "now.json").read_text())
            if not seen or seen[-1][1] != st["scene"]:
                seen.append((i, st["scene"], st["pinned"]))
        names = [x[1] for x in seen]
        self.assertEqual(names[1:3], ["radar", "tunnel"])
        self.assertTrue(seen[2][2])
        self.assertGreater(seen[3][0], 400)                 # pinned past every normal hold, released by unpin


class Words(unittest.TestCase):
    def test_textdata_fits_the_fonts(self):
        import textdata
        self.assertGreater(len(textdata.TAO), 150)
        for _, s in textdata.TAO:
            self.assertTrue(30 <= len(s) <= 150, s)
            for c in s.upper():
                self.assertIn(c, gfx.fontdata.SMALL[2], s)

    def test_tao_is_still_and_faint(self):
        import tao
        cfg = {k: dict(v) for k, v in zenith.DEFAULTS.items()}
        show = zenith.Show(cfg, offline=True, seed=2, only=["planet"])
        clk = [0.0]
        g = show.frames(10, clock=lambda: clk[0])
        first = None
        for i in range(30):
            img = next(g)
            clk[0] += 0.1
            first = first if first is not None else show.tao_lines
            self.assertEqual(show.tao_lines, first)                  # same line for the whole scene
        self.assertTrue(first and all(len(x) <= tao.COLS for x in first))
        self.assertLess(tao.VAL, 0.5)

    def test_almanac_against_published_phases(self):
        import calendar
        import almanac
        u = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M"))
        for full in ("2024-11-15 21:28", "2025-01-13 22:27", "2026-01-03 10:03"):     # USNO
            f = almanac.phases_around(u(full) - 5 * 86400)[2]
            self.assertLess(abs(f - u(full)), 300, full)
        for new in ("2024-11-01 12:47", "2025-01-29 12:36"):
            n = almanac.phases_around(u(new) - 5 * 86400)[1]
            self.assertLess(abs(n - u(new)), 300, new)
        self.assertEqual(almanac.full_name(u("2025-10-07 03:47")), "HARVEST")
        self.assertEqual(almanac.full_name(u("2024-11-15 21:28")), "BEAVER")
        self.assertEqual(almanac.next_shower(u("2026-11-02 12:00"))["name"], "LEONIDS")


class Parsers(unittest.TestCase):
    def test_weather(self):
        j = {"current": {"temperature_2m": 40.6, "weather_code": 71, "is_day": 0, "wind_speed_10m": 8,
                         "wind_direction_10m": 200, "cloud_cover": 100, "time": "2026-11-02T20:00"},
             "current_units": {"temperature_2m": "°F"},
             "daily": {"temperature_2m_max": [44], "temperature_2m_min": [30], "sunrise": ["2026-11-02T06:38"],
                       "sunset": ["2026-11-02T17:07"], "weather_code": [71]}}
        j["daily"].update({"time": ["2026-11-02", "2026-11-03", "2026-11-04"], "temperature_2m_max": [44, 50, 48],
                           "temperature_2m_min": [30, 33, 31], "weather_code": [71, 3, 61],
                           "sunrise": ["2026-11-02T06:38", "2026-11-03T06:39"], "sunset": ["2026-11-02T17:07", "2026-11-03T17:06"],
                           "precipitation_probability_max": [10, 5, 70]})
        w = feeds.parse_weather(j, "X")
        self.assertEqual([(d["dow"], d["hi"], d["kind"], d["pop"]) for d in w["days"]],
                         [("TUE", 50, "overcast", 5), ("WED", 48, "rain", 70)])
        self.assertEqual((w["daylen"], w["daylen_delta"]), (629, -2))
        self.assertEqual((w["kind"], w["text"], w["wind_dir"], w["is_day"]), ("snow", "LIGHT SNOW", "S", False))
        self.assertEqual((w["sunrise"], w["sunset"]), ("06:38", "17:07"))

    def test_ovation(self):
        coords = [[lon, lat, 0] for lon in range(360) for lat in range(-90, 91)]
        for c in coords:
            if c[1] == 48 and 225 <= c[0] <= 255:
                c[2] = 40
        a = feeds.parse_ovation({"coordinates": coords, "Observation Time": "x", "Forecast Time": "y"}, 37.62, -122.38)
        self.assertEqual(a["overhead"], 0)
        self.assertGreater(a["visible"], 10)            # 10 degrees poleward at our longitude is in view
        self.assertEqual(len(a["grid"]), 91)

    def test_adsb(self):
        j = {"ac": [{"hex": "abc", "flight": "SWA1 ", "lat": 37.6213 + 0.5, "lon": -122.379, "alt_baro": 9000,
                     "gs": 300, "track": 90, "seen_pos": 1},
                    {"hex": "def", "r": "N12345", "lat": 37.6213, "lon": -122.379 + 1, "alt_baro": "ground"},
                    {"hex": "nopos"}]}
        a = feeds.parse_adsb(j, 37.6213, -122.379)
        self.assertEqual(len(a), 2)
        self.assertEqual((a[0]["call"], round(a[0]["y_nm"]), a[0]["alt"]), ("SWA1", 30, 9000))
        self.assertIsNone(a[1]["alt"])
        self.assertAlmostEqual(a[1]["x_nm"], 60 * math.cos(math.radians(37.6213)), 3)

    def test_route_leg_validation(self):
        ORD = {"iata": "ORD", "lat": 41.9786, "lon": -87.9048}
        DEN = {"iata": "DEN", "lat": 39.8617, "lon": -104.673}
        SEA = {"iata": "SEA", "lat": 47.449, "lon": -122.309}
        MSP = {"iata": "MSP", "lat": 44.8848, "lon": -93.2223}
        SNA = {"iata": "SNA", "lat": 33.6757, "lon": -117.868}
        SMF = {"iata": "SMF", "lat": 38.6954, "lon": -121.591}
        mid = (46.29, -106.02)                                          # on the ORD-SEA great circle, over Montana
        # database routes that don't match the aircraft must be dropped
        self.assertEqual(feeds.route_leg([ORD, DEN, ORD], *mid, 279.5), "")          # nowhere near ORD-DEN
        self.assertEqual(feeds.route_leg([SNA, SMF], 39.9, -104.6, 156.9), "")       # descending into DEN, not on SNA-SMF
        self.assertEqual(feeds.route_leg([DEN, ORD], 41.97, -88.10, 90), "DEN-ORD")   # on final at ORD
        # a westbound ORD-SEA flight fits; the same position flying east does not
        self.assertEqual(feeds.route_leg([ORD, SEA], *mid, 285), "ORD-SEA")
        self.assertEqual(feeds.route_leg([ORD, SEA], *mid, 95), "")
        # multi-stop: pick the leg actually being flown
        self.assertEqual(feeds.route_leg([SEA, MSP, ORD], 44.6, -91.0, 120), "MSP-ORD")
        self.assertEqual(feeds.route_leg([SEA, MSP, ORD], 47.5, -120.0, 85), "SEA-MSP")

    def test_kp_both_formats(self):
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:00:00", time.gmtime(now + 3 * 3600))
        dicts = [{"time_tag": iso, "kp": 4.67, "observed": "predicted"}]
        lists = [["time_tag", "kp", "observed", "noaa_scale"], [iso, "4.67", "predicted", None]]
        for fc in (dicts, lists):
            kp, mx = feeds.parse_kp([{"estimated_kp": 2.33}], fc, now)
            self.assertEqual((kp, mx), (2.33, 4.67))

    def test_activity_and_strength(self):
        self.assertEqual([feeds.activity(k) for k in (1, 3.2, 4.5, 5.3, 8.7)],
                         ["QUIET", "UNSETTLED", "ACTIVE", "STORM G1", "STORM G4"])
        self.assertTrue(0.1 <= feeds.aurora_strength(None) <= 0.2)
        self.assertLess(feeds.aurora_strength(aurora(1, 2)), feeds.aurora_strength(aurora(6, 80)))
        self.assertLessEqual(feeds.aurora_strength(aurora(9, 100)), 1.0)

    def test_moon_phase_known_dates(self):
        import land
        full = 1731706080                                 # full moon 2024-11-15 21:28 UTC
        self.assertAlmostEqual(land.moon_phase(full), 0.5, delta=0.04)
        self.assertAlmostEqual(math.cos(2 * math.pi * land.moon_phase(947182440)), 1.0, places=5)


class Kms(unittest.TestCase):
    def test_struct_sizes_match_the_kernel_abi(self):
        import ctypes as C
        import kms
        sizes = {kms.ModeInfo: 68, kms.CardRes: 64, kms.Crtc: 104, kms.Encoder: 20, kms.Connector: 80,
                 kms.CreateDumb: 32, kms.MapDumb: 16, kms.FbCmd: 28, kms.FbDirty: 24, kms.PageFlip: 24}
        for st, n in sizes.items():
            self.assertEqual(C.sizeof(st), n, st.__name__)
        self.assertEqual(kms.GETRESOURCES, 0xC04064A0)            # DRM_IOCTL_MODE_GETRESOURCES from drm.h
        self.assertEqual(kms.CREATE_DUMB, 0xC02064B2)

    def test_mode_choice(self):
        import kms
        m480 = kms.ModeInfo(hdisplay=720, vdisplay=480, flags=kms.FLAG_INTERLACE, type=kms.TYPE_PREFERRED, vrefresh=60)
        m240 = kms.ModeInfo(hdisplay=720, vdisplay=240, flags=0, vrefresh=60)
        hd = kms.ModeInfo(hdisplay=1920, vdisplay=1080, type=kms.TYPE_PREFERRED)
        self.assertEqual(kms.pick_mode([m480, m240]).vdisplay, 240)
        self.assertEqual(kms.pick_mode([m480, m240], "480i").vdisplay, 480)
        self.assertEqual(kms.pick_mode([m480]).vdisplay, 480)
        self.assertEqual(kms.pick_mode([hd]).hdisplay, 1920)
        m288 = kms.ModeInfo(hdisplay=720, vdisplay=288, flags=0, vrefresh=50)
        self.assertEqual(kms.pick_mode([m480, m288, m240]).vdisplay, 240)     # NTSC tube: 60 Hz, not PAL 288p
        self.assertEqual(kms.pick_mode([m480, m288]).vdisplay, 480)
        self.assertTrue(kms.is_composite_raster(720, 480))
        self.assertFalse(kms.is_composite_raster(1280, 800))


class Output(unittest.TestCase):
    def test_dither_levels(self):
        g = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
        for lv in (2, 3, 6):
            q = gfx.dither(g, lv)
            self.assertLessEqual(len(np.unique(q)), lv)
            self.assertAlmostEqual(float(q.mean()), float(g.mean()), delta=0.03)

    def test_crt_sim_shape(self):
        rgb = zenith.crt_sim(np.random.default_rng(0).random((H, W)).astype(np.float32))
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
