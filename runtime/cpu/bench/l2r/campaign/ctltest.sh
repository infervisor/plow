#!/usr/bin/env bash
# perf stat --control fifo test: enable after 1 s, disable after 2 s
cd /tmp/g4c/l2r
rm -f t.fifo; mkfifo t.fifo
perf --version
child() { exec 9>t.fifo; sleep 1; echo enable >&9; sleep 1; echo disable >&9; sleep 0.5; }
export -f child
sudo perf stat -a -x, -D -1 --control fifo:t.fifo -e r1f25 -- sudo -u "$(id -un)" bash /tmp/g4c/l2r/child.sh 2>&1 | tail -4
rm -f t.fifo
