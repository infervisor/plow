#!/usr/bin/env bash
# Build BOTH gfx950 object sets for the L2-placement A/B: control (no placement dispatch) and
# treatment (-DPLOW_L2_PLACE_DISPATCH). Run OUTSIDE nix — nix's glibc shadows the system one and
# hipcc dies with GLIBC_2.38 not found (knob-contract §0a).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/l2place}"

echo "=== control objects (no L2 dispatch) ==="
PLOW_L2_PLACE=0 "$REPO/scripts/build_gfx950.sh" "$OUT/objs_off" 2>&1 | tail -6

echo
echo "=== treatment objects (-DPLOW_L2_PLACE_DISPATCH) ==="
PLOW_L2_PLACE=1 "$REPO/scripts/build_gfx950.sh" "$OUT/objs_on" 2>&1 | tail -6
