#!/bin/sh
# Clean timing A/B, interleaved B/A/B/A so drift cannot masquerade as a win. Sweep only —
# token identity is deterministic and was settled in rounds 1-2.
set -e
D=/home/lava/models/glm52_tp
PKT=$D/glm52_tp4_64k.pkt
W=/home/lava/models/GLM-5.2-plow
cd "$D"
LD_LIBRARY_PATH=/opt/rocm/lib
export LD_LIBRARY_PATH
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

for E in "$D/interp_mfold.elf" "$D/interp_decode_fix.elf" "$D/interp_mfold.elf" "$D/interp_decode_fix.elf"; do
    echo "############ $E"
    PLOW_INTERP=$E ./glm52_decode "$PKT" "$W" --tp 4 --sweep 1024,4096 --steps 65 2>&1 | tail -4
done
