"""Direct KMS output: dumb buffers + page flips through DRM ioctls (ctypes only, no SDL, no EGL, no X).

Why: the Pi console has no compositor, and this path needs nothing but /dev/dri/card0 and the 'video' group.
It is the same mechanism every KMS driver supports (vc4 on the Pi, bochs/virtio in a VM), so it can be proven
off-device. We pick the connected connector (composite preferred), choose a mode (240p if offered, else the
preferred one), draw into two XRGB8888 dumb buffers and flip between them. The original CRTC state is restored
on close, so the text console comes back.
"""
import ctypes as C
import errno
import fcntl
import logging
import mmap
import os
import select

import numpy as np

log = logging.getLogger("zenith.kms")


def _IOWR(nr, size):
    return (3 << 30) | (size << 16) | (ord("d") << 8) | nr


def _IO(nr):
    return (ord("d") << 8) | nr


class ModeInfo(C.Structure):
    _fields_ = [("clock", C.c_uint32), ("hdisplay", C.c_uint16), ("hsync_start", C.c_uint16),
                ("hsync_end", C.c_uint16), ("htotal", C.c_uint16), ("hskew", C.c_uint16),
                ("vdisplay", C.c_uint16), ("vsync_start", C.c_uint16), ("vsync_end", C.c_uint16),
                ("vtotal", C.c_uint16), ("vscan", C.c_uint16), ("vrefresh", C.c_uint32), ("flags", C.c_uint32),
                ("type", C.c_uint32), ("name", C.c_char * 32)]


class CardRes(C.Structure):
    _fields_ = [("fb_id_ptr", C.c_uint64), ("crtc_id_ptr", C.c_uint64), ("connector_id_ptr", C.c_uint64),
                ("encoder_id_ptr", C.c_uint64), ("count_fbs", C.c_uint32), ("count_crtcs", C.c_uint32),
                ("count_connectors", C.c_uint32), ("count_encoders", C.c_uint32), ("min_width", C.c_uint32),
                ("max_width", C.c_uint32), ("min_height", C.c_uint32), ("max_height", C.c_uint32)]


class Crtc(C.Structure):
    _fields_ = [("set_connectors_ptr", C.c_uint64), ("count_connectors", C.c_uint32), ("crtc_id", C.c_uint32),
                ("fb_id", C.c_uint32), ("x", C.c_uint32), ("y", C.c_uint32), ("gamma_size", C.c_uint32),
                ("mode_valid", C.c_uint32), ("mode", ModeInfo)]


class Encoder(C.Structure):
    _fields_ = [("encoder_id", C.c_uint32), ("encoder_type", C.c_uint32), ("crtc_id", C.c_uint32),
                ("possible_crtcs", C.c_uint32), ("possible_clones", C.c_uint32)]


class Connector(C.Structure):
    _fields_ = [("encoders_ptr", C.c_uint64), ("modes_ptr", C.c_uint64), ("props_ptr", C.c_uint64),
                ("prop_values_ptr", C.c_uint64), ("count_modes", C.c_uint32), ("count_props", C.c_uint32),
                ("count_encoders", C.c_uint32), ("encoder_id", C.c_uint32), ("connector_id", C.c_uint32),
                ("connector_type", C.c_uint32), ("connector_type_id", C.c_uint32), ("connection", C.c_uint32),
                ("mm_width", C.c_uint32), ("mm_height", C.c_uint32), ("subpixel", C.c_uint32), ("pad", C.c_uint32)]


class CreateDumb(C.Structure):
    _fields_ = [("height", C.c_uint32), ("width", C.c_uint32), ("bpp", C.c_uint32), ("flags", C.c_uint32),
                ("handle", C.c_uint32), ("pitch", C.c_uint32), ("size", C.c_uint64)]


class MapDumb(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("pad", C.c_uint32), ("offset", C.c_uint64)]


class DestroyDumb(C.Structure):
    _fields_ = [("handle", C.c_uint32)]


class FbCmd(C.Structure):
    _fields_ = [("fb_id", C.c_uint32), ("width", C.c_uint32), ("height", C.c_uint32), ("pitch", C.c_uint32),
                ("bpp", C.c_uint32), ("depth", C.c_uint32), ("handle", C.c_uint32)]


class FbDirty(C.Structure):
    _fields_ = [("fb_id", C.c_uint32), ("flags", C.c_uint32), ("color", C.c_uint32), ("num_clips", C.c_uint32),
                ("clips_ptr", C.c_uint64)]


class PageFlip(C.Structure):
    _fields_ = [("crtc_id", C.c_uint32), ("fb_id", C.c_uint32), ("flags", C.c_uint32), ("reserved", C.c_uint32),
                ("user_data", C.c_uint64)]


SET_MASTER = _IO(0x1E)
DROP_MASTER = _IO(0x1F)
GETRESOURCES = _IOWR(0xA0, C.sizeof(CardRes))
GETCRTC = _IOWR(0xA1, C.sizeof(Crtc))
SETCRTC = _IOWR(0xA2, C.sizeof(Crtc))
GETENCODER = _IOWR(0xA6, C.sizeof(Encoder))
GETCONNECTOR = _IOWR(0xA7, C.sizeof(Connector))
ADDFB = _IOWR(0xAE, C.sizeof(FbCmd))
RMFB = _IOWR(0xAF, C.sizeof(C.c_uint32))
PAGE_FLIP = _IOWR(0xB0, C.sizeof(PageFlip))
DIRTYFB = _IOWR(0xB1, C.sizeof(FbDirty))
CREATE_DUMB = _IOWR(0xB2, C.sizeof(CreateDumb))
MAP_DUMB = _IOWR(0xB3, C.sizeof(MapDumb))
DESTROY_DUMB = _IOWR(0xB4, C.sizeof(DestroyDumb))

CONNECTED = 1
FLAG_INTERLACE = 1 << 4
TYPE_PREFERRED = 1 << 3
PAGE_FLIP_EVENT = 1
CONN_TYPES = {1: "VGA", 5: "Composite", 6: "SVIDEO", 9: "DVI-D", 10: "DVI-A", 11: "HDMI-A", 13: "TV",
              14: "eDP", 15: "Virtual", 16: "DSI", 17: "DPI", 18: "Writeback", 7: "LVDS", 8: "Component"}


def _ioctl(fd, req, arg):
    fcntl.ioctl(fd, req, arg, True)


def _arr(ctype, n):
    a = (ctype * max(n, 1))()
    return a, C.addressof(a)


def mode_label(m):
    return "%dx%d%s@%d" % (m.hdisplay, m.vdisplay, "i" if m.flags & FLAG_INTERLACE else "p", m.vrefresh)


def pick_mode(modes, want="auto"):
    """240-line progressive if offered (the tube draws real scanlines); else the preferred mode; else the first."""
    if want in ("auto", "240p"):
        pref = next((m for m in modes if m.type & TYPE_PREFERRED), modes[0])
        for m in modes:                                  # same field rate as the preferred mode: never PAL 288p on NTSC
            if m.vdisplay in (240, 288) and not m.flags & FLAG_INTERLACE and abs(m.vrefresh - pref.vrefresh) <= 1:
                return m
    if want == "480i":
        for m in modes:
            if m.vdisplay == 480 and m.flags & FLAG_INTERLACE:
                return m
    for m in modes:
        if m.type & TYPE_PREFERRED:
            return m
    return modes[0]


def is_composite_raster(w, h):
    """Composite/SDTV rasters have non-square pixels and fill the whole 4:3 tube; everything else is letterboxed."""
    return h in (240, 288, 480, 576) and w in (640, 704, 720)


class KmsOut:
    def __init__(self, cfg, card="/dev/dri/card0", src_w=320, src_h=240):
        self.fd = os.open(card, os.O_RDWR | os.O_CLOEXEC)
        try:
            _ioctl(self.fd, SET_MASTER, 0)
        except OSError as e:
            if e.errno not in (errno.EINVAL, errno.EBUSY, errno.EACCES):
                raise
            # first opener is master already, or a compositor holds it (then SETCRTC below fails loudly)
        res = CardRes()
        _ioctl(self.fd, GETRESOURCES, res)
        crtcs, res.crtc_id_ptr = _arr(C.c_uint32, res.count_crtcs)
        conns, res.connector_id_ptr = _arr(C.c_uint32, res.count_connectors)
        encs, res.encoder_id_ptr = _arr(C.c_uint32, res.count_encoders)
        fbs, res.fb_id_ptr = _arr(C.c_uint32, res.count_fbs)
        _ioctl(self.fd, GETRESOURCES, res)
        self.crtc_ids = list(crtcs[:res.count_crtcs])
        found = []
        for cid in conns[:res.count_connectors]:
            con, modes, encids = self._connector(cid)
            name = "%s-%d" % (CONN_TYPES.get(con.connector_type, "T%d" % con.connector_type), con.connector_type_id)
            log.info("kms: connector %s %s, modes %s", name, "connected" if con.connection == CONNECTED else "off",
                     [mode_label(m) for m in modes[:8]])
            if con.connection == CONNECTED and modes:
                found.append((con.connector_type in (5, 6, 13), con, modes, encids, name))
        if not found:
            raise RuntimeError("no connected display connector")
        found.sort(key=lambda f: not f[0])                       # composite / TV first
        _, con, modes, encids, self.name = found[0]
        self.conn_id = con.connector_id
        self.mode = pick_mode(modes, cfg["display"].get("mode", "auto"))
        self.crtc_id = self._crtc_for(con, encids)
        self.saved = Crtc(crtc_id=self.crtc_id)
        _ioctl(self.fd, GETCRTC, self.saved)
        self.w, self.h = self.mode.hdisplay, self.mode.vdisplay
        self.bufs = [self._dumb(self.w, self.h) for _ in range(2)]
        self.front = 0
        self.green = cfg["display"].get("output", "grey") == "green"
        self._layout(cfg, src_w, src_h)
        try:
            self._setcrtc(self.bufs[0]["fb"])
        except OSError as e:                                     # vc4 VEC lists 240p but legacy SETCRTC rejects it
            alt = pick_mode(modes, "480i")
            if e.errno != errno.EINVAL or alt.vdisplay == self.h:
                raise
            log.warning("kms: %s rejected (%s); using %s", mode_label(self.mode), e, mode_label(alt))
            self.close_bufs()
            self.mode, self.w, self.h = alt, alt.hdisplay, alt.vdisplay
            self.bufs = [self._dumb(self.w, self.h) for _ in range(2)]
            self._layout(cfg, src_w, src_h)
            self._setcrtc(self.bufs[0]["fb"])
        self.flip_ok = True
        self.pending = False
        log.info("display: KMS %s %s on crtc %d, picture %s", self.name, mode_label(self.mode), self.crtc_id, self.dst)

    # -------------------------------------------------------------- setup helpers
    def _layout(self, cfg, src_w, src_h):
        mx, my = int(cfg["display"].get("margin_x", 0)), int(cfg["display"].get("margin_y", 0))
        mt, mb = int(cfg["display"].get("margin_top", my)), int(cfg["display"].get("margin_bottom", my))
        if is_composite_raster(self.w, self.h):                 # margin_top/bottom shift the picture vertically
            x0, y0, dw, dh = mx, mt, self.w - 2 * mx, self.h - mt - mb
        else:                                                    # square pixels: centre a 4:3 picture
            dh = self.h - 2 * my
            dw = min(self.w - 2 * mx, dh * 4 // 3)
            dh = dw * 3 // 4
            x0, y0 = (self.w - dw) // 2, (self.h - dh) // 2
        self.dst = (x0, y0, dw, dh)
        self.yi = (np.arange(dh) * src_h // dh).clip(0, src_h - 1)
        self.xi = (np.arange(dw) * src_w // dw).clip(0, src_w - 1)
        # transfer curve: the Zenith's video input clips early (any grey above ~15% reads as full white), so the
        # picture is squeezed into [black, white] of the DAC range with a gamma; tune in [display] black/white/gamma
        d = cfg["display"]
        lo, hi, gam = float(d.get("black", 0.0)), float(d.get("white", 1.0)), float(d.get("gamma", 1.0))
        y = np.round(255 * (lo + (hi - lo) * (np.arange(256) / 255.0) ** gam)).clip(0, 255).astype(np.uint32)
        self.lut = (y << 8) if self.green else (y * 0x010101)
        for b in self.bufs:
            b["px"][:] = 0

    def close_bufs(self):
        for b in self.bufs:
            try:
                b["px"] = None
                b["mm"].close()
                _ioctl(self.fd, RMFB, C.c_uint32(b["fb"]))
                _ioctl(self.fd, DESTROY_DUMB, DestroyDumb(handle=b["handle"]))
            except (OSError, BufferError):
                pass
        self.bufs = []

    def _connector(self, cid):
        con = Connector(connector_id=cid)
        _ioctl(self.fd, GETCONNECTOR, con)
        n_modes, n_enc = con.count_modes, con.count_encoders
        modes, con.modes_ptr = _arr(ModeInfo, n_modes)
        encs, con.encoders_ptr = _arr(C.c_uint32, n_enc)
        props, con.props_ptr = _arr(C.c_uint32, con.count_props)
        vals, con.prop_values_ptr = _arr(C.c_uint64, con.count_props)
        con.count_modes, con.count_encoders = n_modes, n_enc
        _ioctl(self.fd, GETCONNECTOR, con)
        k = min(n_modes, con.count_modes)
        return con, [ModeInfo.from_buffer_copy(modes[i]) for i in range(k)], list(encs[:min(n_enc, con.count_encoders)])

    def _crtc_for(self, con, encids):
        order = ([con.encoder_id] if con.encoder_id else []) + [e for e in encids if e != con.encoder_id]
        for eid in order:
            enc = Encoder(encoder_id=eid)
            _ioctl(self.fd, GETENCODER, enc)
            if enc.crtc_id:
                return enc.crtc_id
            for i, c in enumerate(self.crtc_ids):
                if enc.possible_crtcs & (1 << i):
                    return c
        raise RuntimeError("no CRTC for connector")

    def _dumb(self, w, h):
        cd = CreateDumb(height=h, width=w, bpp=32)
        _ioctl(self.fd, CREATE_DUMB, cd)
        fb = FbCmd(width=w, height=h, pitch=cd.pitch, bpp=32, depth=24, handle=cd.handle)
        _ioctl(self.fd, ADDFB, fb)
        md = MapDumb(handle=cd.handle)
        _ioctl(self.fd, MAP_DUMB, md)
        mm = mmap.mmap(self.fd, cd.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=md.offset)
        px = np.frombuffer(mm, np.uint32, count=(cd.pitch // 4) * h).reshape(h, cd.pitch // 4)
        return {"handle": cd.handle, "fb": fb.fb_id, "mm": mm, "px": px}

    def _setcrtc(self, fb_id):
        conn = (C.c_uint32 * 1)(self.conn_id)
        c = Crtc(set_connectors_ptr=C.addressof(conn), count_connectors=1, crtc_id=self.crtc_id, fb_id=fb_id,
                 mode_valid=1, mode=self.mode)
        _ioctl(self.fd, SETCRTC, c)

    # -------------------------------------------------------------- per frame
    def _wait_flip(self, timeout):
        if not self.pending:
            return
        r, _, _ = select.select([self.fd], [], [], timeout)
        if r:
            os.read(self.fd, 1024)                              # drm_event_vblank; we only need "it happened"
        self.pending = False

    def show(self, img):
        self._wait_flip(0.1)
        back = self.bufs[1 - self.front]
        v = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        x0, y0, dw, dh = self.dst
        k = dh // v.shape[0]
        if k * v.shape[0] == dh:                                 # integer line repeat (480i: each line twice)
            row = np.take(self.lut[v], self.xi, axis=1)
            for r in range(k):
                back["px"][y0 + r:y0 + dh:k, x0:x0 + dw] = row
        else:
            back["px"][y0:y0 + dh, x0:x0 + dw] = self.lut[v[self.yi][:, self.xi]]
        if self.flip_ok:
            try:
                _ioctl(self.fd, PAGE_FLIP, PageFlip(crtc_id=self.crtc_id, fb_id=back["fb"], flags=PAGE_FLIP_EVENT))
                self.pending = True
            except OSError as e:
                log.warning("page flip unsupported (%s); drawing single-buffered", e)
                self.flip_ok = False
        if not self.flip_ok:                                     # single buffer: copy into the visible one
            self.bufs[self.front]["px"][:] = back["px"]
            back = self.bufs[self.front]
            try:
                _ioctl(self.fd, DIRTYFB, FbDirty(fb_id=back["fb"]))
            except OSError:
                pass
            return
        self.front = 1 - self.front

    def close(self):
        try:
            self._wait_flip(0.2)
            s = self.saved
            if s.mode_valid and s.fb_id:
                conn = (C.c_uint32 * 1)(self.conn_id)
                s.set_connectors_ptr, s.count_connectors = C.addressof(conn), 1
                _ioctl(self.fd, SETCRTC, s)                     # give the console its framebuffer back
            else:
                self.bufs[self.front]["px"][:] = 0
        except OSError as e:
            log.warning("restore crtc: %s", e)
        self.close_bufs()
        try:
            _ioctl(self.fd, DROP_MASTER, 0)
        except OSError:
            pass
        os.close(self.fd)
