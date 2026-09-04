#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-mla-merge-fold-v128}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w -Rpass-analysis=kernel-resource-usage --genco \
    "$ROOT/runtime/bench/amd/mla_merge_fold_v128/kernel.hip" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/kernel.co" 2>"$OUT/resources"
"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/mla_merge_fold_v128/bench.cpp" \
    -o "$OUT/bench" -lamdhip64

resource() {
    sed -n "/Function Name: $1/,/Function Name:/p" "$OUT/resources" |
        grep -F "$2:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}
for name in k_fold_v128_tb1 k_fold_v128_tb2 k_fold_v128_tb4 k_fold_v128_tb8 \
    k_fold_v128_tb2_serial_merge k_fold_v128_tb4_serial_merge \
    k_fold_v128_tb8_serial_merge; do
    vgpr=$(resource "$name" VGPRs); sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill'); sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]'); lds=$(resource "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a -n "$sspill" -a \
        -n "$occ" -a -n "$lds"
    printf '%-20s VGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s\n' \
        "$name" "$vgpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
    if ((scratch != 0 || vspill != 0 || sspill != 0 || occ < 2 || lds > 49152)); then
        echo "FAIL: $name crossed the gfx950 isolated resource gate" >&2
        exit 1
    fi
done

if [ "${BUILD_ONLY:-0}" = 1 ]; then
    exit 0
fi
"$ROOT/perf-data/tools/gpulease" -n 1 mla-merge-fold-v128 \
    "$OUT/bench" "$OUT/kernel.co" "${SAMPLES:-31}"
