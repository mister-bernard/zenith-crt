"""Deterministic sky almanac: true times of new/full moon (Meeus, Astronomical Algorithms ch. 49, main periodic
terms: accurate to a few minutes), moon age and illumination, traditional full-moon names, meteor-shower peaks."""
import calendar
import math
import time

SYNODIC = 29.530588861


def _jd_to_unix(jd):
    return (jd - 2440587.5) * 86400.0


def _phase_jde(k, full):
    T = k / 1236.85
    jde = (2451550.09766 + SYNODIC * k + 0.00015437 * T * T - 0.000000150 * T ** 3 + 0.00000000073 * T ** 4)
    E = 1 - 0.002516 * T - 0.0000074 * T * T
    r = math.radians
    M = r(2.5534 + 29.10535670 * k - 0.0000014 * T * T - 0.00000011 * T ** 3)
    Mp = r(201.5643 + 385.81693528 * k + 0.0107582 * T * T + 0.00001238 * T ** 3 - 0.000000058 * T ** 4)
    F = r(160.7108 + 390.67050284 * k - 0.0016118 * T * T - 0.00000227 * T ** 3 + 0.000000011 * T ** 4)
    Om = r(124.7746 - 1.56375588 * k + 0.0020672 * T * T + 0.00000215 * T ** 3)
    s = math.sin
    a, b = (-0.40614, 0.17302) if full else (-0.40720, 0.17241)
    c, d, e = (0.01614, 0.01043, 0.00734) if full else (0.01608, 0.01039, 0.00739)
    f, g = (-0.00515, 0.00209) if full else (-0.00514, 0.00208)
    jde += (a * s(Mp) + b * E * s(M) + c * s(2 * Mp) + d * s(2 * F) + e * E * s(Mp - M) + f * E * s(Mp + M)
            + g * E * E * s(2 * M) - 0.00111 * s(Mp - 2 * F) - 0.00057 * s(Mp + 2 * F) + 0.00056 * E * s(2 * Mp + M)
            - 0.00042 * s(3 * Mp) + 0.00042 * E * s(M + 2 * F) + 0.00038 * E * s(M - 2 * F)
            - 0.00024 * E * s(2 * Mp - M) - 0.00017 * s(Om))
    return jde - 69.0 / 86400.0                     # TT -> UT (delta T ~69 s)


def phases_around(ts=None):
    """-> (last_new, next_new, next_full) unix times bracketing ts."""
    ts = time.time() if ts is None else ts
    k0 = math.floor((ts / 86400.0 + 2440587.5 - 2451550.09766) / SYNODIC) - 1
    news = [_jd_to_unix(_phase_jde(k, False)) for k in range(k0, k0 + 4)]
    fulls = [_jd_to_unix(_phase_jde(k + 0.5, True)) for k in range(k0, k0 + 4)]
    last_new = max(t for t in news if t <= ts)
    next_new = min(t for t in news if t > ts)
    next_full = min(t for t in fulls if t > ts)
    return last_new, next_new, next_full


FULL_NAMES = {1: "WOLF", 2: "SNOW", 3: "WORM", 4: "PINK", 5: "FLOWER", 6: "STRAWBERRY", 7: "BUCK", 8: "STURGEON",
              9: "CORN", 10: "HUNTER'S", 11: "BEAVER", 12: "COLD"}


def full_name(ts):
    """Traditional (Algonquin / Farmer's Almanac) name; the full moon nearest the September equinox is the Harvest
    Moon, and the one after it the Hunter's Moon."""
    t = time.gmtime(ts)
    eq = calendar.timegm((t.tm_year, 9, 22, 19, 0, 0))
    harvest = None
    for k in range(-2, 3):                               # full moons around the equinox
        f = phases_around(eq + k * SYNODIC * 86400)[2]
        if harvest is None or abs(f - eq) < abs(harvest - eq):
            harvest = f
    if abs(ts - harvest) < 86400 * 2:
        return "HARVEST"
    if 0 < ts - harvest < SYNODIC * 86400 * 1.1:
        return "HUNTER'S"
    m = t.tm_mon
    if m in (9, 10):
        return "CORN" if ts < harvest else FULL_NAMES[11 if m == 10 else 10]
    return FULL_NAMES[m]


def moon(ts=None):
    ts = time.time() if ts is None else ts
    last_new, next_new, next_full = phases_around(ts)
    age = (ts - last_new) / 86400.0
    p = age / ((next_new - last_new) / 86400.0)
    illum = (1 - math.cos(2 * math.pi * p)) / 2
    names = ("NEW MOON", "WAXING CRESCENT", "FIRST QUARTER", "WAXING GIBBOUS", "FULL MOON", "WANING GIBBOUS",
             "LAST QUARTER", "WANING CRESCENT")
    name = names[int((p * 8 + 0.5) % 8)]
    return {"phase": p, "age": age, "illum": illum, "name": name, "next_full": next_full, "next_new": next_new,
            "full_name": full_name(next_full)}


# peak dates (UTC, typical; vary by a day year to year) and best zenithal hourly rate
SHOWERS = [(1, 3, "QUADRANTIDS", 110), (4, 22, "LYRIDS", 18), (5, 6, "ETA AQUARIIDS", 50), (7, 30, "S DELTA AQUARIIDS", 25),
           (8, 12, "PERSEIDS", 100), (10, 8, "DRACONIDS", 10), (10, 21, "ORIONIDS", 20), (11, 17, "LEONIDS", 15),
           (12, 14, "GEMINIDS", 150), (12, 22, "URSIDS", 10)]


def next_shower(ts=None):
    ts = time.time() if ts is None else ts
    y = time.gmtime(ts).tm_year
    for yy in (y, y + 1):
        for m, d, name, zhr in SHOWERS:
            t = calendar.timegm((yy, m, d, 12, 0, 0))
            if t > ts - 86400:
                return {"name": name, "ts": t, "zhr": zhr, "days": (t - ts) / 86400.0}
