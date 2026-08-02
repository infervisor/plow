#!/usr/bin/env bash
# Block until the hsaco batch build finishes.
while pgrep -f build_hsaco_batch.sh >/dev/null; do sleep 10; done
for B in 2 4 8; do
  echo "hsaco-b$B: $(ls /home/lava/plow/build-amd/hsaco-b$B/*.elf 2>/dev/null | wc -l) elfs"
done
