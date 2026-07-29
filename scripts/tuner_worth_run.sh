#!/usr/bin/env bash
# WHAT IS THE PREFILL-GEMM TUNER WORTH, END TO END? Nobody had measured it.
#
# The `tunedb` is GEMM-ONLY (`tunedb::gemm_op_case` is its sole lookup;
# `crates/devgen/src/lib.rs:324,:454`) and the DECODE program contains ZERO `Gemm` ops —
# every decode matmul is `Gemv`/`GemvQkv`/`GemvGlu`/`GemvFp8Blk`, confirmed in `build.json`'s
# per-program `arms` list. So the tuner cannot move ms/token AT ALL. It moves PREFILL, i.e.
# TTFT, and nothing else. This script prices exactly that.
#
# THE ARMS. One tree, one plowrt, one object dir, one checkpoint, one ladder — the ONLY
# difference is the packet:
#   tuned    plowc reads `tuning/` and picks tiles from measurements
#   notuned  plowc --no-tuning, whose documented contract is "identical to the pre-tuner
#            compiler" (`crates/plowc/src/main.rs:836`); it sets `PLOW_TUNEDB=""`, and an
#            empty store is the analytical model's fallback tier
#
# INTERLEAVED, NOT SEQUENTIAL. `A B A B`, four separate weight loads. A one-shot `A then B`
# cannot separate the arm from drift, and every cell here costs a 167 s TP4 load anyway.
#
# BEFORE RUNNING: the store must not be stale against THIS tree. `tuning/` records are keyed
# by the PREPROCESSED interp digest, so any `interp.hip`/`op_gemm.h`/`op_moe.h` edit stales
# the whole campaign and the "tuned" arm silently becomes the "notuned" arm — the A/B then
# measures nothing and reports a confident zero. Verify with `PLOW_TUNE_DUMP=1` on the emit:
# it must read HIT, not MISS. (Measured 2026-07-29 on this tree BEFORE re-running the
# campaign: 0 HIT / 2472 MISS.)
#
#   $1 out dir (default /home/lava/models/glm52_tunerab)   $2 port (default 8188)
set -u
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB="${1:-/home/lava/models/glm52_tunerab}"
PORT="${2:-8188}"
REPS="${REPS:-2}"

for r in $(seq 1 "$REPS"); do
  for arm in tuned notuned; do
    echo "############ rep $r arm $arm ############"
    IN_LENS="${IN_LENS:-4096 16384}" CONCS=1 OUTLEN="${OUTLEN:-128}" \
    NPROMPT="${NPROMPT:-8}" BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:---num-warmups 2}" \
    LOG="/tmp/tunerab_server_${arm}_r${r}.log" \
      bash "$WT/scripts/bench_plowrt_serve.sh" "$AB/$arm" "$PORT" glm-5.2 zai-org/GLM-5.2-FP8 1200
    echo
  done
done
