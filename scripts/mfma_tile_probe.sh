#!/usr/bin/env bash
# TILE SENSITIVITY PROBE. The MFMA arm's best STANDALONE tile at MM=4 is UN=6/YT=2 at 194 VGPR;
# this builds the low-register UN=2/YT=2 tile (86 VGPR standalone, 17% slower standalone) so the
# served A/B can say whether the megakernel's 256-VGPR allocation is what caps the win.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$ROOT/build-mfma/hsaco-mfma-u2y2}"
PLOW_GEMV_MFMA4=1 PLOW_GEMV_MFMA4_UN_M4=2 PLOW_GEMV_MFMA4_YT_M4=2 \
  PLOW_DECODE_BATCH=4 PLOW_DECODE_TIERS=1,2 scripts/build_gfx942.sh "$OUT"
echo "=== tile probe objects done: $OUT"
