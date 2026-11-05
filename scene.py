"""Scene base class and the shared text-mode renderer."""
import numpy as np

import gfx
from gfx import H, W


class Scene:
    name = "scene"
    kind = "generative"          # classic | generative | weather | aurora
    hold = (8.0, 20.0)           # seconds on screen (the deck picks within this, clipped to the config range)
    levels = None                # None = deck default dither levels; 0 = scene output is final (already quantised)
    rate = None                  # max frames per second this scene needs (None = every frame); saves CPU on the Pi
    tao = None                   # "top" | "bottom": a faint, still Tao Te Ching line in that quiet strip

    def __init__(self, ctx):
        self.ctx = ctx           # dict: feeds, cfg, loc

    def start(self, rng):
        self.rng = rng

    def frame(self, t, dt):      # -> float32 (H, W) in 0..1
        raise NotImplementedError


# ---------------------------------------------------------------- text mode: 40x30 cells of 8x8 VGA glyphs
TEXT_COLS, TEXT_ROWS = W // 8, H // 8
_RAMP = " .,-~:;=!*#$@"


class TextMode:
    """Vectorised character-cell renderer: grid of glyph indices -> one 240x320 image (no per-glyph python loop)."""

    def __init__(self, chars=None):
        chars = chars or "".join(chr(c) for c in range(32, 127)) + "░▒▓█▀▄▌▐■°·"
        self.chars = chars
        self.index = {c: i for i, c in enumerate(chars)}
        self.atlas = np.stack([gfx.glyph(c) for c in chars])          # (n, 8, 8)
        self.ramp = np.array([self.index[c] for c in _RAMP])

    def render(self, idx, val=None):
        """idx: (rows, cols) int glyph indices; val: optional (rows, cols) brightness."""
        r, c = idx.shape
        g = self.atlas[idx]                                            # (r, c, 8, 8)
        if val is not None:
            g = g * val[:, :, None, None]
        return g.transpose(0, 2, 1, 3).reshape(r * 8, c * 8)

    def from_lines(self, lines, rows=TEXT_ROWS, cols=TEXT_COLS):
        idx = np.zeros((rows, cols), np.int32)
        for y, s in enumerate(lines[:rows]):
            for x, ch in enumerate(s[:cols]):
                idx[y, x] = self.index.get(ch, self.index["?"])
        return idx

    def luminance(self, lum):
        """(rows, cols) luminance 0..1 -> glyph indices along the ASCII-art ramp."""
        k = np.clip((lum * (len(self.ramp) - 1)).round().astype(int), 0, len(self.ramp) - 1)
        return self.ramp[k]
