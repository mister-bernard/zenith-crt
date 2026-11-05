"""A line of the Tao Te Ching (Legge, 1891, public domain), set faint and still into a quiet part of some scenes.
Scenes opt in with tao = "top" | "bottom"; the line is chosen once per scene and never moves."""
import textwrap

import gfx
import textdata

VAL = 0.3                 # faint: after dithering only part of each glyph lights, like text seen through haze
COLS = 48


def pick(rng):
    ch, s = textdata.TAO[int(rng.integers(len(textdata.TAO)))]
    return textwrap.wrap(s.upper(), COLS)[:3], ch


def draw(img, lines, where):
    y = 16 if where == "top" else 214 - 11 * (len(lines) - 1)
    for line in lines:
        gfx.text(img, 16, y, line, VAL, "small")
        y += 11
