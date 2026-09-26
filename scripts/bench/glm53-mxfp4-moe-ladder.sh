#!/usr/bin/env bash
# GLM-5.3 MXFP4 TP8 grouped A4W4 MoE (op85 GLU+bridge, op86 DOWN+scatter) over the prefill and
# decode rungs, on ONE leased GPU. Run through the queue, never bare:
#   scripts/bench/gpuq.py submit glm53-moe-ladder 1 scripts/bench/glm53-mxfp4-moe-ladder.sh <bin> <out>
# <bin> is runtime/tests/moe_prefill_a4w4_bench_gfx950.hip built for gfx950 (see its header).
set -euo pipefail
bin=$(realpath -- "${1:?usage: glm53-mxfp4-moe-ladder.sh BIN OUT}")
out=$(realpath -m -- "${2:?usage: glm53-mxfp4-moe-ladder.sh BIN OUT}")
mkdir -p "$out"
grid=${GRID:-256}
iters=${ITERS:-31}
routed_rungs=${ROUTED_RUNGS:-"1 2 4 8 16 32 64 128 1024 2048 4096 8192 16384"}
dense_rungs=${DENSE_RUNGS:-"1024 2048 4096 8192 16384"}
sha256sum "$bin" > "$out/bin.sha256"
# argv: T grid iters act beta lbeta I H experts topk [varied]
for t in $routed_rungs; do
    "$bin" "$t" "$grid" "$iters" 1 0 0 256 6144 256 8 > "$out/routed-T$t.log" 2>&1
done
for t in $dense_rungs; do
    "$bin" "$t" "$grid" "$iters" 1 0 0 1536 6144 1 1 > "$out/dense-T$t.log" 2>&1
done
grep -H "op85\|op86\|total  \|oracle" "$out"/*.log
