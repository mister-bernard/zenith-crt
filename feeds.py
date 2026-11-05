"""Live data for the slideshow: today's weather (Open-Meteo, no key) and the NOAA SWPC aurora forecast.

Both are fetched by one background thread, cached on disk (so a reboot without network still has
something to show) and exposed as plain dicts. Every consumer must cope with None (= no data yet).
"""
import calendar
import json
import logging
import math
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("zenith.feeds")

UA = "zenith-crt-slideshow/1.0 (private home display)"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "zenith"

OPEN_METEO = ("https://api.open-meteo.com/v1/forecast?latitude={lat:.3f}&longitude={lon:.3f}"
              "&current=temperature_2m,relative_humidity_2m,is_day,precipitation,weather_code,cloud_cover,"
              "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
              "&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_probability_max"
              "&temperature_unit={tu}&wind_speed_unit={wu}&timezone=auto&forecast_days=4")
OVATION = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
KP_NOW = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
KP_FCST = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
IPGEO = "https://ipapi.co/json/"


def http_json(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _cache_put(name, obj):
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / (name + ".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(CACHE / (name + ".json"))
    except OSError as e:
        log.warning("cache write %s failed: %s", name, e)


def _cache_get(name, max_age):
    p = CACHE / (name + ".json")
    try:
        obj = json.loads(p.read_text())
        if time.time() - obj.get("fetched", 0) <= max_age:
            return obj
    except (OSError, ValueError):
        pass
    return None


# ------------------------------------------------------------------ weather
WMO = {  # code -> (kind, label)
    0: ("clear", "CLEAR"), 1: ("clear", "MAINLY CLEAR"), 2: ("partly", "PARTLY CLOUDY"), 3: ("overcast", "OVERCAST"),
    45: ("fog", "FOG"), 48: ("fog", "RIME FOG"),
    51: ("drizzle", "LIGHT DRIZZLE"), 53: ("drizzle", "DRIZZLE"), 55: ("drizzle", "HEAVY DRIZZLE"),
    56: ("drizzle", "FREEZING DRIZZLE"), 57: ("drizzle", "FREEZING DRIZZLE"),
    61: ("rain", "LIGHT RAIN"), 63: ("rain", "RAIN"), 65: ("rain", "HEAVY RAIN"),
    66: ("rain", "FREEZING RAIN"), 67: ("rain", "FREEZING RAIN"),
    71: ("snow", "LIGHT SNOW"), 73: ("snow", "SNOW"), 75: ("snow", "HEAVY SNOW"), 77: ("snow", "SNOW GRAINS"),
    80: ("rain", "SHOWERS"), 81: ("rain", "SHOWERS"), 82: ("rain", "VIOLENT SHOWERS"),
    85: ("snow", "SNOW SHOWERS"), 86: ("snow", "SNOW SHOWERS"),
    95: ("storm", "THUNDERSTORM"), 96: ("storm", "T-STORM, HAIL"), 99: ("storm", "T-STORM, HAIL"),
}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def parse_weather(j, label=""):
    cur, day = j["current"], j.get("daily") or {}

    def d0(k):
        v = day.get(k)
        return v[0] if v else None

    code = int(cur.get("weather_code") or 0)
    kind, text = WMO.get(code, ("partly", "WX %d" % code))
    wd = cur.get("wind_direction_10m")
    return {
        "label": label,
        "code": code, "kind": kind, "text": text,
        "temp": cur.get("temperature_2m"), "hi": d0("temperature_2m_max"), "lo": d0("temperature_2m_min"),
        "humidity": cur.get("relative_humidity_2m"), "cloud": cur.get("cloud_cover"),
        "precip": cur.get("precipitation"), "pop": d0("precipitation_probability_max"),
        "wind": cur.get("wind_speed_10m"), "gust": cur.get("wind_gusts_10m"),
        "wind_dir": COMPASS[int(((wd or 0) + 22.5) // 45) % 8] if wd is not None else "",
        "wind_deg": wd,
        "is_day": bool(cur.get("is_day", 1)),
        "sunrise": (d0("sunrise") or "")[-5:], "sunset": (d0("sunset") or "")[-5:],
        "units": j.get("current_units", {}).get("temperature_2m", "°F"),
        "time": cur.get("time", ""),
        "days": _days(day),
        "daylen": _daylen(day, 0), "daylen_delta": (_daylen(day, 1) - _daylen(day, 0)) if _daylen(day, 1) else None,
    }


def _daylen(day, i):
    """Minutes of daylight on forecast day i (from the local sunrise/sunset strings), or None."""
    try:
        a, b = day["sunrise"][i], day["sunset"][i]
        return (int(b[11:13]) * 60 + int(b[14:16])) - (int(a[11:13]) * 60 + int(a[14:16]))
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _days(day):
    """The next three days after today: date, weekday, hi, lo, kind, text, pop."""
    out = []
    for i in range(1, min(4, len(day.get("time") or []))):
        try:
            d = day["time"][i]
            code = int(day["weather_code"][i] or 0)
            kind, text = WMO.get(code, ("partly", "WX %d" % code))
            dow = time.strftime("%a", time.strptime(d, "%Y-%m-%d")).upper()
            pops = day.get("precipitation_probability_max") or []
            out.append({"date": d, "dow": dow, "hi": day["temperature_2m_max"][i], "lo": day["temperature_2m_min"][i],
                        "code": code, "kind": kind, "text": text, "pop": pops[i] if i < len(pops) else None})
        except (KeyError, IndexError, TypeError, ValueError):
            break
    return out


def ascii_text(s):
    """Typographic punctuation -> the 7-bit set our fonts carry; drop anything else."""
    tr = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...",
          "\u00a0": " ", "\u00e9": "e", "\u00e8": "e", "\u00e1": "a", "\u00f3": "o", "\u00ed": "i", "\u00fc": "u",
          "\u00f6": "o", "\u00e4": "a", "\u00f1": "n", "\u00e7": "c", "\u00b0": "\u00b0"}
    s = "".join(tr.get(c, c) for c in (s or ""))
    return " ".join("".join(c for c in s if 32 <= ord(c) < 127 or c == "\u00b0").split())


# ------------------------------------------------------------------ aurora
def activity(kp):
    if kp is None:
        return "NO DATA"
    if kp >= 5:
        return "STORM G%d" % min(5, int(kp) - 4)
    if kp >= 4:
        return "ACTIVE"
    if kp >= 3:
        return "UNSETTLED"
    return "QUIET"


def parse_ovation(j, lat, lon):
    """OVATION grid -> compact 2-D list [lat 0..90][lon 0..359] of probability (northern hemisphere only),
    plus the chance overhead and the best chance within the sky visible from (lat, lon)."""
    grid = [[0] * 360 for _ in range(91)]
    for lo, la, v in j["coordinates"]:
        if 0 <= la <= 90:
            grid[int(la)][int(lo) % 360] = int(v)
    lon360 = int(round(lon)) % 360
    la0 = int(round(lat))
    overhead = grid[min(max(la0, 0), 90)][lon360]
    # Low on the northern horizon you can see aurora up to roughly 10-12 degrees of latitude poleward.
    best = 0
    for dla in range(0, 13):
        la = la0 + dla
        if la > 90:
            break
        for dlo in range(-15, 16):
            w = 1.0 - dla / 16.0                     # farther north = lower on the horizon = less impressive
            best = max(best, grid[la][(lon360 + dlo) % 360] * w)
    return {"grid": grid, "overhead": overhead, "visible": round(best, 1),
            "observed": j.get("Observation Time", ""), "forecast": j.get("Forecast Time", "")}


def parse_kp(now_rows, fc_rows, now_ts=None):
    now_ts = now_ts or time.time()
    kp_now = None
    if now_rows:
        r = now_rows[-1]
        kp_now = r.get("estimated_kp", r.get("kp_index"))
    kp_max24 = None
    if fc_rows:
        rows = fc_rows
        if isinstance(rows[0], list):                      # older table format: header row + lists
            hdr = rows[0]
            rows = [dict(zip(hdr, r)) for r in rows[1:]]
        for r in rows:
            try:
                ts = calendar.timegm(time.strptime(r["time_tag"][:19], "%Y-%m-%dT%H:%M:%S"))
                kp = float(r["kp"])
            except (KeyError, ValueError, TypeError):
                continue
            if r.get("observed") in ("predicted", "estimated") and now_ts - 3 * 3600 <= ts <= now_ts + 24 * 3600:
                kp_max24 = kp if kp_max24 is None else max(kp_max24, kp)
    return kp_now, kp_max24


# ------------------------------------------------------------------ location
def resolve_location(cfg):
    loc = dict(cfg.get("location") or {})
    if "lat" in loc and "lon" in loc:
        return loc
    cached = _cache_get("location", 30 * 86400)
    if cached:
        return {**cached["loc"], **loc}
    try:
        j = http_json(IPGEO, timeout=8)
        found = {"lat": float(j["latitude"]), "lon": float(j["longitude"]), "label": (j.get("city") or "").upper()}
        _cache_put("location", {"fetched": time.time(), "loc": found})
        log.info("location from IP geolocation")
        return {**found, **loc}
    except Exception as e:                                  # noqa: BLE001 - any failure means use the default
        log.warning("IP geolocation failed (%s); using San Francisco", e)
        return {"lat": 37.77, "lon": -122.42, "label": "SAN FRANCISCO", **loc}


# ------------------------------------------------------------------ the fetcher
class Feeds:
    WEATHER_EVERY = 30 * 60
    AURORA_EVERY = 15 * 60

    def __init__(self, cfg, offline=False):
        self.cfg = cfg
        self.loc = resolve_location(cfg) if not offline else {"lat": 37.77, "lon": -122.42, "label": "OFFLINE", **(cfg.get("location") or {})}
        self.units = (self.loc.get("units") or "imperial").lower()
        self.weather = None
        self.aurora = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._next = {"weather": 0.0, "aurora": 0.0}
        self.offline = offline
        if not offline:
            c = _cache_get("weather", 6 * 3600)
            if c:
                self.weather = c["data"]
            c = _cache_get("aurora", 6 * 3600)
            if c:
                self.aurora = c["data"]

    # public snapshot accessors
    def get(self):
        with self.lock:
            return self.weather, self.aurora

    def start(self):
        if self.offline:
            return
        threading.Thread(target=self._run, name="feeds", daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            if now >= self._next["weather"]:
                self._next["weather"] = now + (self.WEATHER_EVERY if self.fetch_weather() else 300)
            if now >= self._next["aurora"]:
                self._next["aurora"] = now + (self.AURORA_EVERY if self.fetch_aurora() else 300)
            self._stop.wait(20)

    def fetch_weather(self):
        imperial = self.units != "metric"
        url = OPEN_METEO.format(lat=float(self.loc["lat"]), lon=float(self.loc["lon"]),
                                tu="fahrenheit" if imperial else "celsius", wu="mph" if imperial else "kmh")
        try:
            w = parse_weather(http_json(url), self.loc.get("label", ""))
            w["wind_unit"] = "MPH" if imperial else "KMH"
            w["fetched"] = time.time()
        except Exception as e:                              # noqa: BLE001
            log.warning("weather fetch failed: %s", e)
            return False
        with self.lock:
            self.weather = w
        _cache_put("weather", {"fetched": time.time(), "data": w})
        log.info("weather: %s %s", w["text"], w["temp"])
        return True

    def fetch_aurora(self):
        lat, lon = float(self.loc["lat"]), float(self.loc["lon"])
        out = {"fetched": time.time()}
        ok = False
        try:
            out.update(parse_ovation(http_json(OVATION, timeout=20), lat, lon))
            ok = True
        except Exception as e:                              # noqa: BLE001
            log.warning("ovation fetch failed: %s", e)
        try:
            out["kp"], out["kp_max24"] = parse_kp(http_json(KP_NOW), http_json(KP_FCST))
            ok = True
        except Exception as e:                              # noqa: BLE001
            log.warning("kp fetch failed: %s", e)
        if not ok:
            return False
        out["activity"] = activity(out.get("kp"))
        out["home_lat"], out["home_lon"] = lat, lon
        with self.lock:
            prev = self.aurora or {}
            if "grid" not in out and "grid" in prev:        # partial failure: keep the last oval
                for k in ("grid", "overhead", "visible", "observed", "forecast"):
                    out[k] = prev.get(k)
            self.aurora = out
        _cache_put("aurora", {"fetched": time.time(), "data": out})
        log.info("aurora: kp=%s max24=%s visible=%s", out.get("kp"), out.get("kp_max24"), out.get("visible"))
        return True


def aurora_strength(a):
    """0..1 drive for the animation: blends the OVATION chance in view with Kp."""
    if not a:
        return 0.15
    kp = a.get("kp") or 0.0
    vis = (a.get("visible") or 0) / 100.0
    k = max(0.0, min(1.0, (kp - 1.0) / 6.0))
    return max(0.12, min(1.0, 0.12 + 0.45 * math.sqrt(vis) + 0.5 * k))


# ---------------------------------------------------------------- live ADS-B (radar scene)
# adsb.lol: community ADS-B aggregator with the ADS-B Exchange v2 JSON schema, free, no key (ADSBx's own API is paid).
ADSB_URL = "https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/{nm:d}"
ROUTE_URL = "https://api.adsb.lol/api/0/route/{call}/{lat:.4f}/{lon:.4f}"   # often 500 (no route known)
ADSBDB_URL = "https://api.adsbdb.com/v0/callsign/{call}"                     # fallback; free, no key
DEFAULT_RADAR = (37.6213, -122.3790)            # San Francisco Intl (KSFO); set [radar] lat/lon in config.toml


def parse_adsb(j, lat0, lon0, now=None):
    """-> list of dicts: id, call, x_nm (east), y_nm (north), alt (ft or None=ground), gs (kt), trk (deg), age (s)."""
    out = []
    now = now or time.time()
    k = math.cos(math.radians(lat0)) * 60.0
    for a in (j or {}).get("ac") or []:
        if a.get("lat") is None or a.get("lon") is None:
            continue
        alt = a.get("alt_baro")
        alt = None if alt == "ground" else (alt if isinstance(alt, (int, float)) else a.get("alt_geom"))
        call = (a.get("flight") or "").strip() or a.get("r") or a.get("hex", "?").upper()
        out.append({"id": a.get("hex"), "call": call[:8], "x_nm": (a["lon"] - lon0) * k, "y_nm": (a["lat"] - lat0) * 60.0,
                    "alt": alt, "gs": float(a.get("gs") or 0.0),
                    "trk": float(a.get("track") if a.get("track") is not None else a.get("true_heading") or 0.0),
                    "age": float(a.get("seen_pos") or a.get("seen") or 0.0), "type": a.get("t") or ""})
    return out


class AdsbPoller:
    """Polls adsb.lol in a background thread only while someone asked for data in the last 30 s (the radar scene)."""
    EVERY = 8.0                                      # dead reckoning fills the gaps between polls

    def __init__(self, lat, lon, nm):
        self.lat, self.lon, self.nm = lat, lon, int(nm)
        self.data, self.ts, self.err = None, 0.0, None
        self.want = 0.0
        self.routes = {}                                 # callsign -> (route "SEA-GEG" or "", fetched at)
        self.lock = threading.Lock()
        self.thread = None

    def get(self):
        self.want = time.time()
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, name="adsb", daemon=True)
            self.thread.start()
        with self.lock:
            return self.data, self.ts, self.err

    def _run(self):
        while time.time() - self.want < 30:
            t0 = time.time()
            try:
                j = http_json(ADSB_URL.format(lat=self.lat, lon=self.lon, nm=self.nm), timeout=8)
                tg = parse_adsb(j, self.lat, self.lon)
                self._routes(tg, j)
                with self.lock:
                    self.data, self.ts, self.err = tg, time.time(), None
                wait = self.EVERY
            except Exception as e:                       # network, HTTP, JSON: keep the last picture, say so
                with self.lock:
                    self.err = str(e)[:60] if time.time() - self.ts > 60 else None   # brief gaps: keep showing LIVE
                log.warning("adsb: %s", e)
                wait = 30.0 if "429" in str(e) else self.EVERY * 2               # rate-limited: back off
            time.sleep(max(0.5, wait - (time.time() - t0)))


    def _routes(self, tg, j):
        """Origin-destination for airline callsigns (ICAO prefix + number), max 3 lookups per poll, cached 1 h."""
        pos = {a.get("hex"): (a.get("lat"), a.get("lon")) for a in (j or {}).get("ac") or []}
        n = 0
        for a in tg:
            c = a["call"]
            hit = self.routes.get(c)
            if hit is None or time.time() - hit[1] > 3600:
                if n >= 3 or not re.match(r"^[A-Z]{3}[0-9][0-9A-Z]{0,4}$", c):
                    continue
                n += 1
                la, lo = pos.get(a["id"], (self.lat, self.lon))
                cands = []                                # every candidate route; route_leg() picks the one that fits
                try:
                    r = http_json(ROUTE_URL.format(call=c, lat=la, lon=lo), timeout=6)
                    if r.get("_airports"):
                        cands.append(r["_airports"])
                except Exception as e:
                    log.info("adsb.lol route %s: %s", c, e)
                try:
                    f = (http_json(ADSBDB_URL.format(call=c), timeout=6).get("response") or {})
                    f = f.get("flightroute") if isinstance(f, dict) else None
                    if f:
                        cands.append([{"iata": p.get("iata_code") or p.get("icao_code"), "lat": p.get("latitude"),
                                       "lon": p.get("longitude")}
                                      for p in (f.get("origin"), f.get("midpoint"), f.get("destination")) if p])
                except Exception as e:
                    log.info("adsbdb route %s: %s", c, e)
                # nothing known (both services often have gaps): ask again in 5 min instead of an hour
                self.routes[c] = hit = (cands, time.time() - (0 if cands else 3300))
            la, lo = pos.get(a["id"], (self.lat, self.lon))
            a["route"] = next((r for r in (route_leg(c, la, lo, a["trk"]) for c in hit[0]) if r), "")



def _gc(la1, lo1, la2, lo2):
    """-> (distance NM, initial bearing rad) on a sphere."""
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    d = math.acos(max(-1.0, min(1.0, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl))))
    b = math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return d * 3440.065, b


def route_leg(airports, lat, lon, track):
    """Pick the leg of a (possibly multi-stop, possibly stale) database route that this aircraft is actually flying.
    A leg A->B counts only if the aircraft is near the A-B great circle (cross-track <= max(50 NM, 10% of the leg)),
    between A and B along it, and (unless within 20 NM of B, i.e. on approach) heading within 60 deg of the bearing
    to B. -> "AAA-BBB" or "" when no leg fits (then we show no route rather than a wrong one)."""
    best = None
    for a, b in zip(airports, airports[1:]):
        try:
            la_a, lo_a, la_b, lo_b = float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        leg, b_ab = _gc(la_a, lo_a, la_b, lo_b)
        d_ap, b_ap = _gc(la_a, lo_a, lat, lon)
        if leg < 20:
            continue
        R = 3440.065
        xt = abs(math.asin(max(-1.0, min(1.0, math.sin(d_ap / R) * math.sin(b_ap - b_ab))))) * R
        at = math.acos(max(-1.0, min(1.0, math.cos(d_ap / R) / max(1e-9, math.cos(xt / R))))) * R
        if math.cos(b_ap - b_ab) < 0:
            at = -at
        if xt > max(50.0, 0.10 * leg) or not (-30 <= at <= leg + 30):
            continue
        d_pb, b_pb = _gc(lat, lon, la_b, lo_b)
        if d_pb > 20 and track is not None:
            diff = abs((math.degrees(b_pb) - track + 180) % 360 - 180)
            if diff > 60:
                continue
        code = "%s-%s" % (a.get("iata") or a.get("icao"), b.get("iata") or b.get("icao"))
        if best is None or xt < best[0]:
            best = (xt, code)
    return best[1] if best else ""


_adsb = {}


def adsb(lat, lon, nm):
    key = (round(lat, 4), round(lon, 4), int(nm))
    if key not in _adsb:
        _adsb[key] = AdsbPoller(lat, lon, nm)
    return _adsb[key]
