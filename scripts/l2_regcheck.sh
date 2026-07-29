#!/usr/bin/env bash
# Register cost of -DPLOW_L2_PLACE_DISPATCH against the plain GQ decode object.
#
# The decode object sits at 248 VGPR / occupancy 2, and the window selection is in the hottest
# loop in the interpreter — so "does it cost occupancy" is a build-time question with a
# build-time answer. Run OUTSIDE nix (knob-contract §0a).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
INC="-I$R/amd -I$R/common"
BASE="-DPLOW_BUCKET_DECODE=1 -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=1"

for D in "" "-DPLOW_L2_PLACE_DISPATCH"; do
  U=$(/opt/rocm/bin/hipcc --offload-arch=gfx950 -O3 -w $BASE $D --genco \
        -Rpass-analysis=kernel-resource-usage "$R/amd/interp.hip" -o /dev/null $INC 2>&1)
  V=$(echo "$U" | grep -oP 'VGPRs: \K\d+' | head -1)
  A=$(echo "$U" | grep -oP 'AGPRs: \K\d+' | head -1)
  O=$(echo "$U" | grep -oP 'Occupancy \[waves/SIMD\]: \K\d+' | head -1)
  S=$(echo "$U" | grep -oP 'VGPRs Spill: \K\d+' | head -1)
  printf "%-28s VGPR=%-4s AGPR=%-4s total=%-4s occ=%-3s spill=%s\n" \
    "${D:-baseline (no place)}" "$V" "$A" "$((V + A))" "$O" "$S"
done
