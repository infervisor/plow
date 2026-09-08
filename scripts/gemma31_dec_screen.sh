#!/usr/bin/env bash
# Fast A/B screen of decode blobs/objects through `amd-bench --batched`, one lease,
# interleaved control/candidate/control so the bracket prices its own drift.
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-dec-screen \
#     nix develop /app/plow --command scripts/gemma31_dec_screen.sh <label>=<blob>[:<obj>] ...
set -euo pipefail
R=/app/plow/build-gemma31/residue
BIN="${BIN:-$R/target/release}"
CKPT=/app/plow/build-gemma31/checkpoint
DEFOBJ="${DEFOBJ:-/app/plow/build-gemma31/hsaco-tiered}"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PLOW_L2_PLACE_DISPATCH=1
STEPS="${STEPS:-48}"
CTX="${CTX:-1024}"
REPS="${REPS:-2}"
for rep in $(seq 1 "$REPS"); do
  for spec in "$@"; do
    label="${spec%%=*}"; rest="${spec#*=}"
    blob="${rest%%:*}"; obj="${rest#*:}"
    [ "$obj" = "$rest" ] && obj="$DEFOBJ"
    line=$("$BIN/plowrt" amd-bench --blob "$blob" --hsaco "$obj" --checkpoint "$CKPT" \
        --batched --steps "$STEPS" --ctx "$CTX" 2>/dev/null | grep -E '^ +tpot ')
    echo "rep$rep  $label  $line"
  done
done
