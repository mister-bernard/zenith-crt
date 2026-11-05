#!/bin/sh
# tcrun.sh <testcard.py args>: stop the show, put up a card, restart the show when the card ends.
# Run as root. Runs the card in its own session so a later tcrun (or `tcrun.sh stop`) kills the whole group.
sudo systemctl stop zenith-crt
if [ -f /tmp/tc.pid ]; then kill -- -"$(cat /tmp/tc.pid)" 2>/dev/null; sleep 1; fi
[ "$1" = stop ] && { rm -f /tmp/tc.pid; sudo systemctl start zenith-crt; exit 0; }
cd /opt/zenith
setsid sh -c 'python3 testcard.py "$@" >/tmp/tc.log 2>&1; sudo systemctl start zenith-crt' sh "$@" >/dev/null 2>&1 </dev/null &
echo $! > /tmp/tc.pid
