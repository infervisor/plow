#!/usr/bin/env bash
# One PLOW_TRACE_PHASE=1 decode trace of the Gemma-4 31B T=4 program, for the
# residue attribution. Runs inside nix (the CPU reference backend is silent).
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-dec-trace \
#     nix develop /app/plow --command scripts/gemma31_dec_trace.sh
set -euo pipefail
R=/app/plow/build-gemma31/residue
BIN="${BIN:-/app/plow/build-gemma31/residue/target/release}"
CKPT=/app/plow/build-gemma31/checkpoint
BLOB="${BLOB:-$R/assets-ctl/model.pkt}"
OBJ="${OBJ:-$R/obj-trace-merged}"
OUT="${OUT:-$R/dec_t4.bin}"
CTX="${CTX:-1024}"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PLOW_L2_PLACE_DISPATCH=1
export PLOW_TRACE_RAW="$OUT"
"$BIN/plowrt" amd-bench --blob "$BLOB" --hsaco "$OBJ" --checkpoint "$CKPT" \
    --batched --steps "${STEPS:-24}" --ctx "$CTX" 2>&1 | tail -40
ls -l "$OUT"
