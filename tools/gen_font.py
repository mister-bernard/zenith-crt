#!/usr/bin/env python3
"""Build crt/fontdata.py from Linux console PSF fonts (kbd package) so the Pi needs no font files.
VGA8 = the 8x8 IBM-PC text-mode font (40 columns at 320 px, period-correct); Terminus12x6 = small status text."""
import gzip, struct, sys
from pathlib import Path

WANT = [chr(c) for c in range(32, 127)] + list("°·░▒▓█▀▄▌▐■←→↑↓")


def load_psf(path):
    d = gzip.open(path).read()
    if d[:2] == b"\x36\x04":                                  # PSF1
        mode, h = d[2], d[3]
        n = 512 if mode & 1 else 256
        w, off, size, has_tab = 8, 4, h, bool(mode & 2)
    elif d[:4] == b"\x72\xb5\x4a\x86":                        # PSF2
        _, off, flags, n, size, h, w = struct.unpack("<7I", d[4:32])
        has_tab = bool(flags & 1)
    else:
        raise SystemExit(f"not PSF: {path}")
    glyphs = [d[off + i * size: off + (i + 1) * size] for i in range(n)]
    cmap, pos = {}, off + n * size
    if has_tab:
        tab, g = d[pos:], 0
        if d[:2] == b"\x36\x04":
            i = 0
            while g < n and i + 1 < len(tab):
                u = struct.unpack("<H", tab[i:i + 2])[0]; i += 2
                if u == 0xFFFF: g += 1
                elif u != 0xFFFE: cmap.setdefault(chr(u), g)
        else:
            i = 0
            while g < n and i < len(tab):
                b = tab[i]
                if b == 0xFF: g += 1; i += 1; continue
                if b == 0xFE: i += 1; continue
                j = i + 1
                while j < len(tab) and tab[j] & 0xC0 == 0x80: j += 1
                cmap.setdefault(tab[i:j].decode("utf-8", "replace"), g); i = j
    bpr = (w + 7) // 8
    out = {}
    for ch in WANT:
        if ch not in cmap: continue
        raw = glyphs[cmap[ch]]
        rows = [int.from_bytes(raw[r * bpr:(r + 1) * bpr], "big") >> (bpr * 8 - w) for r in range(h)]
        out[ch] = rows
    return w, h, out


def main():
    root = Path("/usr/share/consolefonts")
    fonts = {"vga8": "Uni2-VGA8.psf.gz", "small": "Uni2-Terminus12x6.psf.gz"}
    lines = ['"""Bitmap fonts extracted from Linux console fonts (kbd) by tools/gen_font.py. Do not edit."""', ""]
    for name, fn in fonts.items():
        w, h, g = load_psf(root / fn)
        missing = [c for c in WANT if c not in g]
        print(name, fn, f"{w}x{h}", len(g), "glyphs; missing:", "".join(missing), file=sys.stderr)
        lines.append(f"{name.upper()} = ({w}, {h}, {{")
        for ch, rows in g.items():
            lines.append(f"    {ch!r}: {rows!r},")
        lines.append("})")
    Path(__file__).resolve().parent.parent.joinpath("fontdata.py").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
