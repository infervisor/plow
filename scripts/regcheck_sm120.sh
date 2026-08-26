#!/usr/bin/env bash
# regcheck_sm120.sh — ptxas -v register/stack/spill report for the sm_120 cubins.
#
#   scripts/regcheck_sm120.sh [extra cmake -D flags...]
#
# No CUDA/ptxas -v wrapper existed for the NVIDIA side (only the AMD/HIP
# equivalent, scripts/regcheck_prefill.sh). Builds via
# scripts/build_sm120_cubin.sh with -Xptxas -v,--warn-on-spills appended,
# then greps registers/stack/spills/smem per compiled object out of the log.
# Report-only: never deletes, never touches served assets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT
LOG="$OUT_DIR/build.log"

EXTRA="${PLOW_EXTRA_DEFINES:-}"
EXTRA="${EXTRA:+$EXTRA }-Xptxas -v,--warn-on-spills"

PLOW_EXTRA_DEFINES="$EXTRA" bash "$HERE/build_sm120_cubin.sh" "$OUT_DIR/interp_sm120.cubin" "$@" \
    > "$LOG" 2>&1 || { echo "BUILD FAILED — see $LOG"; cat "$LOG"; exit 1; }

echo "=== regcheck_sm120: $* ==="
printf "%-22s %-45s %8s %6s %6s %6s\n" "object" "symbol" "regs" "stack" "spillS" "spillL"
awk '
  /^\[.*%\] nvcc/ { obj = $0; sub(/^\[[^]]*\] nvcc /, "", obj); sub(/ \(.*/, "", obj) }
  /ptxas info *: Function properties for / { sym = $0; sub(/.*Function properties for /, "", sym) }
  /bytes stack frame, / { stack = $1; spillS = $5; spillL = $9 }
  /ptxas info *: Used [0-9]+ registers/ {
    regs = $5
    short = sym
    if (length(sym) > 43) short = substr(sym, 1, 40) "..."
    printf "%-22s %-45s %8s %6s %6s %6s\n", obj, short, regs, stack, spillS, spillL
  }
' "$LOG"

echo
NZ="$(grep -E '[1-9][0-9]* bytes spill (stores|loads)' "$LOG" || true)"
if [ -n "$NZ" ]; then
    echo "-- NONZERO SPILLS --"
    echo "$NZ"
else
    echo "-- 0 spills across every compiled symbol --"
fi

echo
echo "full log: $LOG (not deleted — copy out before it's cleaned up on exit if needed)"
trap - EXIT
