#!/bin/sh
# Second A/B round: candidate only (baseline A already measured in round 1, same session/lease day).
# Round 1 numbers: A(dispatch fix only) 32.004 / 31.993 ms/tok at ctx 1k / 4k, tokens 24/24.
set -e
D=/home/lava/models/glm52_tp
PKT=$D/glm52_tp4_64k.pkt
W=/home/lava/models/GLM-5.2-plow
cd "$D"
LD_LIBRARY_PATH=/opt/rocm/lib
export LD_LIBRARY_PATH
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

for E in "$D/interp_mfold.elf" "$D/interp_decode_fix.elf"; do
    echo "############ $E"
    echo "---- token identity (--gen 24) ----"
    PLOW_INTERP=$E ./glm52_decode "$PKT" "$W" --tp 4 --gen 24 2>&1 | tail -4
    echo "---- sweep 1024,4096 (median of 65) ----"
    PLOW_INTERP=$E ./glm52_decode "$PKT" "$W" --tp 4 --sweep 1024,4096 --steps 65 2>&1 | tail -5
done
