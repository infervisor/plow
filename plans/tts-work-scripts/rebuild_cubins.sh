#!/usr/bin/env bash
# rebuild_cubins.sh <assets> — rebuild the packet-paired objects after a runtime/ source edit.
source /root/tts-work/cuda-env.sh
cmake --build "$1/.cubin-build" --target sm120_cubins 2>&1 | grep -v "warning #177\|Remark\|rowoff\|^\s*$" | tail -n 12
