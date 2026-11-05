#!/usr/bin/env python3
"""Calibration card for the Zenith: border at the picture edge, 10% steps inward, crosshair, grey ramp, big text.
Stop the service first (sudo systemctl stop zenith-crt). Usage: python3 testcard.py [seconds] [mode] [white]"""
import logging
import sys
import time

import numpy as np

import gfx
import kms
import zenith
from gfx import H, W

logging.basicConfig(level=logging.INFO, format="%(message)s")
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 120
mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
img = np.zeros((H, W), np.float32)
kind = sys.argv[3] if len(sys.argv) > 3 else "card"
if kind == "white":
    img[:] = 1.0
elif kind == "edges":                                     # nested 1-px frames every 3 px: count what's visible per side
    for i in range(9):
        o = 3 * i
        img[o, o:W - o] = img[H - 1 - o, o:W - o] = 1.0
        img[o:H - o, o] = img[o:H - o, W - 1 - o] = 1.0
    gfx.text_center(img, 100, "EDGES 0 MARGIN", scale=2)
    gfx.text_center(img, 130, "FRAMES EVERY 3 PX", scale=1)
elif kind == "ramp":                                       # raw DAC levels: which row shows 8 distinct steps?
    for i, top in enumerate(tuple(float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else "1,.6,.4,.25,.15,.08".split(",")))):
        y0 = 12 + i * 37
        gfx.text(img, 4, y0 + 10, "%3s" % ("%g" % round(top * 255)), scale=1)
        for k in range(8):
            img[y0:y0 + 28, 36 + k * 35:68 + k * 35] = top * k / 7
else:
    for i, f in enumerate((0.0, 0.05, 0.10)):            # nested frames: outer = edge of the signal, then 5%, 10%
        x, y = int(W * f), int(H * f)
        v = 1.0 if i == 0 else 0.6
        img[y, x:W - x] = img[H - 1 - y, x:W - x] = v
        img[y:H - y, x] = img[y:H - y, W - 1 - x] = v
    img[H // 2, :] = img[:, W // 2] = 0.5
    for k in range(8):                                    # grey steps
        img[150:180, 40 + k * 30:70 + k * 30] = k / 7
    gfx.text_center(img, 50, "ZENITH TEST", scale=3)
    gfx.text_center(img, 90, "NTSC %s" % mode.upper(), scale=2)
cfg = zenith.load_config()
cfg["display"]["mode"] = mode
if kind == "edges" and "cfg" not in sys.argv[4:]:
    cfg["display"]["margin_x"] = cfg["display"]["margin_y"] = 0
out = kms.KmsOut(cfg)
if kind == "ramp":
    out.lut = np.arange(256, dtype=np.uint32) * 0x010101       # bypass the configured curve
q = (np.clip(img, 0, 1) * 255).astype(np.uint8)
t0 = time.time()
try:
    while time.time() - t0 < secs:
        out.show(q)
        time.sleep(0.1)
finally:
    out.close()
