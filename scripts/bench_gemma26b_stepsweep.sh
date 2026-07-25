#!/usr/bin/env bash
# bench_gemma26b_stepsweep.sh — single-stream (and B=4) kernel-only TPOT sweep
# for gemma-4-26B-A4B on sm_120, bf16 vs fp8, via crates/plowrt/examples/step_bench.
#
#   scripts/bench_gemma26b_stepsweep.sh <assets_root> <out_log>
#
# <assets_root> holds bf16-b1/ fp8-b1/ bf16-b4/ fp8-b4/ serve-asset dirs.
# Each config runs under its OWN gpulease so parallel agents serialize; bf16 and
# fp8 are interleaved to control for GPU thermal/clock drift. PLOW_STEP_TIME=1
# arms the engine's device-event breakdown (dev_interp_ms/upload/download),
# emitted once at step 128 as a tracing INFO line.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?usage: bench_gemma26b_stepsweep.sh <assets_root> <out_log>}"
OUT="${2:?usage: bench_gemma26b_stepsweep.sh <assets_root> <out_log>}"
SB="$ROOT/target/release/examples/step_bench"
STEPS="${STEPS:-128}"
export GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-7200}"
export GPU_LEASE_IDLE_MIB="${GPU_LEASE_IDLE_MIB:-40000}"

# config: dir slots ctx
CONFIGS=(
  "bf16-b1 1 128"
  "fp8-b1  1 128"
  "bf16-b1 1 1024"
  "fp8-b1  1 1024"
  "bf16-b1 1 4096"
  "fp8-b1  1 4096"
  "bf16-b4 4 2048"
  "fp8-b4  4 2048"
)

: > "$OUT"
for cfg in "${CONFIGS[@]}"; do
  read -r dir slots ctx <<< "$cfg"
  tag="sw-${dir}-c${ctx}"
  echo "===== CONFIG dir=$dir slots=$slots ctx=$ctx steps=$STEPS =====" | tee -a "$OUT"
  for attempt in 1 2 3; do
    timeout 1800 /usr/local/bin/gpulease "$tag" \
      env PLOW_STEP_TIME=1 "$SB" "$ASSETS/$dir" "$slots" "$ctx" "$STEPS" \
      >> "$OUT" 2>&1
    rc=$?
    [ "$rc" = 75 ] && { echo "  (gpulease busy rc=75, retry $attempt)" | tee -a "$OUT"; sleep 15; continue; }
    echo "  (rc=$rc)" | tee -a "$OUT"
    break
  done
done
echo "===== SWEEP DONE =====" | tee -a "$OUT"
