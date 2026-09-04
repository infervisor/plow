#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-kda-intra-wave-items}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w -Rpass-analysis=kernel-resource-usage \
    "$ROOT/runtime/bench/amd/kda_intra_wave_items/kernel.hip" \
    "$ROOT/runtime/bench/amd/kda_intra_wave_items/bench.cpp" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/bench" 2>"$OUT/resources"

resource() {
    sed -n "/Function Name: $1/,/Function Name:/p" "$OUT/resources" |
        grep -F "$2:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}
for name in k_intra_control k_intra_wave_items k_wu_qpre k_carry_qpre; do
    vgpr=$(resource "$name" VGPRs); sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill'); sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]'); lds=$(resource "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a -n "$sspill" -a -n "$occ" -a -n "$lds"
    printf '%-20s VGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s\n' \
        "$name" "$vgpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
    if ((scratch != 0 || vspill != 0 || sspill != 0 || occ < 2 || lds > 147456)); then
        echo "FAIL: $name crossed the gfx950 resource gate" >&2
        exit 1
    fi
    if [ "$name" = k_intra_wave_items ] && ((vgpr > 96 || lds > 131072)); then
        echo "FAIL: wave-item Intra exceeded its specialist budget" >&2
        exit 1
    fi
done

"$ROOT/perf-data/tools/gpulease" -n 1 kda-intra-wave-items \
    "$OUT/bench" "${T:-8192}" "${H:-12}" "${SAMPLES:-21}"
