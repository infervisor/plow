#!/usr/bin/env bash
# More MM=4 MFMA tiles for the tile probe. Each argument is one "<UN>x<YT>" pair.
#   nix develop /app/plow --command scripts/mfma_tile_build_more.sh 11x1 4x3
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for t in "$@"; do
  un="${t%x*}"; yt="${t#*x}"
  OUT="$ROOT/build-mfma/hsaco-mfma-u${un}y${yt}"
  echo "=== building UN=$un YT=$yt -> $OUT"
  PLOW_GEMV_MFMA4=1 PLOW_GEMV_MFMA4_UN_M4="$un" PLOW_GEMV_MFMA4_YT_M4="$yt" \
    PLOW_DECODE_BATCH=4 PLOW_DECODE_TIERS=1,2 scripts/build_gfx942.sh "$OUT" || exit 1
done
echo "=== tile builds done"
