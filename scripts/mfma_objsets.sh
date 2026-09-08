#!/usr/bin/env bash
# Build the A/B object sets for the MFMA batched-decode candidate, both from THIS tree so the
# only difference between them is GV_MFMA4. Run inside `nix develop /app/plow`.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CTL="${1:-$ROOT/build-mfma/hsaco-ctl}"
CAND="${2:-$ROOT/build-mfma/hsaco-mfma}"

echo "=== control (GV_MFMA4 off) -> $CTL"
PLOW_DECODE_BATCH=4 PLOW_DECODE_TIERS=1,2 scripts/build_gfx942.sh "$CTL" || exit 1
echo "=== candidate (PLOW_GEMV_MFMA4=1) -> $CAND"
PLOW_GEMV_MFMA4=1 PLOW_DECODE_BATCH=4 PLOW_DECODE_TIERS=1,2 scripts/build_gfx942.sh "$CAND" || exit 1
echo "=== done"
