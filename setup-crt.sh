#!/bin/bash
# setup-crt.sh: turn on composite video and install the Zenith green-CRT slideshow on a Raspberry Pi.
# Mr. Bernard for G, 2026-11-02 (TG 70355/70360/70406). Run as root from a checkout of this directory:
#     sudo bash setup-crt.sh              # install, then reboot
#     sudo bash setup-crt.sh --no-reboot  # install, reboot later yourself
#     sudo bash setup-crt.sh --rollback   # put boot config back, disable the service
# Standalone by design (G TG 70406): the installed copy lives in /opt/zenith, runs as its own system user
# "zenith", fetches only public no-key feeds (Open-Meteo, NOAA SWPC, adsb.lol) and needs nobody's account,
# SSH access or any AI service. Re-running this script updates /opt/zenith from the checkout it is run from.
#
# What it changes (and nothing else):
#   1. /boot/firmware/config.txt   dtoverlay=vc4-kms-v3d  ->  dtoverlay=vc4-kms-v3d,composite
#      /boot/firmware/cmdline.txt  + video=Composite-1:720x480@60ie,tv_mode=NTSC
#      (both backed up next to themselves as *.bak-zenith-<time>; skipped on a non-Pi test machine)
#   2. apt: python3-numpy python3-pygame + the Mesa EGL/GBM bits SDL's KMSDRM driver needs (Debian packages, no pip)
#   3. system user zenith (groups video, render: may draw on /dev/dri/card0); code copied to /opt/zenith (root-owned)
#   4. /etc/systemd/system/zenith-crt.service (runs as zenith, starts at boot, restarts on failure; cache in
#      /var/cache/zenith) + /usr/local/bin/zenith-ctl (sudo zenith-ctl radar | next | pin <scene> | unpin | now)
# Idempotent: safe to run twice.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
SRC=$(cd "$(dirname "$0")" && pwd)
APP=/opt/zenith
BOOT=/boot/firmware
STAMP=$(date +%Y%m%d-%H%M%S)
MODE=${1:-install}

if [ "$MODE" = "--rollback" ]; then
    systemctl disable --now zenith-crt 2>/dev/null || true
    for f in config.txt cmdline.txt; do
        b=$(ls -1t $BOOT/$f.bak-zenith-* 2>/dev/null | tail -1 || true)     # the oldest backup = the original
        if [ -n "$b" ]; then cp -a "$b" "$BOOT/$f"; echo "restored $BOOT/$f from $b"; fi
    done
    echo "rolled back; reboot to drop composite: sudo systemctl reboot"
    exit 0
fi

[ -f "$SRC/zenith.py" ] || { echo "missing $SRC/zenith.py"; exit 1; }

echo "== 1/5 composite video (NTSC)"
if [ -f "$BOOT/config.txt" ] && [ -f "$BOOT/cmdline.txt" ]; then
    cp -a "$BOOT/config.txt" "$BOOT/config.txt.bak-zenith-$STAMP"
    cp -a "$BOOT/cmdline.txt" "$BOOT/cmdline.txt.bak-zenith-$STAMP"
    if grep -qE '^dtoverlay=vc4-kms-v3d,.*composite' "$BOOT/config.txt"; then
        echo "   config.txt already has composite"
    elif grep -qE '^dtoverlay=vc4-kms-v3d$' "$BOOT/config.txt"; then
        sed -i 's/^dtoverlay=vc4-kms-v3d$/dtoverlay=vc4-kms-v3d,composite/' "$BOOT/config.txt"
        echo "   config.txt: dtoverlay=vc4-kms-v3d,composite"
    else
        echo "   WARNING: no plain 'dtoverlay=vc4-kms-v3d' line; add ',composite' to it by hand"; grep -n vc4 "$BOOT/config.txt" || true
    fi
    if grep -q 'video=Composite-1' "$BOOT/cmdline.txt"; then
        echo "   cmdline.txt already sets Composite-1"
    else
        sed -i '1 s/$/ video=Composite-1:720x480@60ie,tv_mode=NTSC/' "$BOOT/cmdline.txt"
        echo "   cmdline.txt: + video=Composite-1:720x480@60ie,tv_mode=NTSC"
    fi
    [ "$(wc -l < "$BOOT/cmdline.txt")" -le 1 ] || { echo "cmdline.txt is not one line; restoring"; cp -a "$BOOT/cmdline.txt.bak-zenith-$STAMP" "$BOOT/cmdline.txt"; exit 1; }
else
    echo "   not a Raspberry Pi boot partition: skipping (test machine)"
fi

echo "== 2/5 packages"
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-numpy python3-pygame \
    libegl1 libegl-mesa0 libgles2 libgbm1 libgl1-mesa-dri

echo "== 3/5 system user zenith + /opt/zenith"
id zenith >/dev/null 2>&1 || useradd --system --home-dir /var/lib/zenith --no-create-home --shell /usr/sbin/nologin zenith
usermod -aG video,render zenith
install -d -o root -g root -m 755 "$APP"
if [ "$SRC" != "$APP" ]; then                  # full source incl. tests + this script, so the Pi can rebuild itself
    for f in "$SRC"/*.py "$SRC"/*.toml "$SRC"/*.sh "$SRC"/*.md; do
        [ -f "$f" ] && install -o root -g root -m 644 "$f" "$APP/"
    done
fi
chmod 755 "$APP"/*.sh
[ -f "$APP/config.toml" ] || cp "$APP/config.example.toml" "$APP/config.toml"
python3 -m compileall -q "$APP" || true
cat > /usr/local/bin/zenith-ctl <<'CTL'
#!/bin/sh
# zenith-ctl <scene|next|pin <scene>|unpin|now|list>: steer the running slideshow
D=/var/cache/zenith
case "$1" in
  now)  cat "$D/now.json"; echo ;;
  list) python3 /opt/zenith/zenith.py --list ;;
  "")   echo "usage: zenith-ctl <scene>|next|pin <scene>|unpin|now|list"; exit 2 ;;
  *)    printf '%s\n' "$*" > "$D/control.tmp" && chown zenith: "$D/control.tmp" && mv "$D/control.tmp" "$D/control" ;;
esac
CTL
chmod 755 /usr/local/bin/zenith-ctl

echo "== 4/5 service"
cat > /etc/systemd/system/zenith-crt.service <<'UNIT'
[Unit]
Description=Zenith ZVM-123 green-phosphor slideshow (composite)
After=network-online.target systemd-user-sessions.service
Wants=network-online.target

[Service]
User=zenith
Group=zenith
SupplementaryGroups=video render
WorkingDirectory=/opt/zenith
CacheDirectory=zenith
Environment=XDG_CACHE_HOME=/var/cache HOME=/var/lib/zenith SDL_VIDEODRIVER=kmsdrm PYGAME_HIDE_SUPPORT_PROMPT=1
ExecStart=/usr/bin/python3 /opt/zenith/zenith.py
Restart=on-failure
RestartSec=10
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
MemoryMax=300M

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable zenith-crt

echo "== 5/5 start"
systemctl restart zenith-crt

echo
echo "done. The slideshow starts by itself after the reboot."
echo "control: sudo zenith-ctl list | <scene> | pin <scene> | unpin | next | now"
echo "rollback any time:  sudo bash $SRC/setup-crt.sh --rollback"
if [ "$MODE" != "--no-reboot" ] && [ -f "$BOOT/config.txt" ]; then
    echo "rebooting in 5 s (Ctrl+C to skip)"; sleep 5; systemctl reboot
fi
