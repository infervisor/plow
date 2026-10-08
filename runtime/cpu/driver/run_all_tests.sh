#!/bin/bash
# Build, (re)load and exercise the pseudo-lock driver from this directory.
set -euo pipefail
cd "$(dirname "$0")"
make
if lsmod | grep -q '^pseudo_lock_sram '; then sudo rmmod pseudo_lock_sram; fi
sudo insmod ./pseudo_lock_sram.ko "$@"
sudo chgrp "$(id -g)" /dev/pseudo_lock
cat /sys/class/misc/pseudo_lock/caps
cc -O2 -pthread sram_bench.c -o sram_bench
./sram_bench 10
READ_LOCKED=1 WORKER_CLOS=1 ./sram_bench 10
