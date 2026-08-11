#!/bin/sh
# A/B the MLA_MERGE_FOLD rewrite against the current best decode object, end to end.
# Run under: perf-data/tools/gpulease -n 4 mfab sg render -c '.... this script ...'
# NOTE: no `env -i` anywhere — that strips ROCR_VISIBLE_DEVICES and destroys the lease (§0a/§6e).
set -e
D=/home/lava/models/glm52_tp
PKT=$D/glm52_tp4_64k.pkt
W=/home/lava/models/GLM-5.2-plow
A="${A:-$D/interp_decode_fix.elf}"   # baseline: dispatch fix only
B="${B:-$D/interp_mfold.elf}"        # candidate: dispatch fix + fold/merge rewrite
cd "$D"
LD_LIBRARY_PATH=/opt/rocm/lib
export LD_LIBRARY_PATH
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

for tag in A B; do
    case $tag in A) E=$A ;; B) E=$B ;; esac
    echo "############ $tag = $E"
    echo "---- sweep 1024,4096 (median of 65) ----"
    PLOW_INTERP=$E ./glm52_decode "$PKT" "$W" --tp 4 --sweep 1024,4096 --steps 65 2>&1 |
        grep -Ev "^(binding|loading)" | tail -12
    echo "---- token identity (--gen 24) ----"
    PLOW_INTERP=$E ./glm52_decode "$PKT" "$W" --tp 4 --gen 24 2>&1 | tail -6
done
