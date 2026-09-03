#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-kda-carry-key-precompute}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w -Rpass-analysis=kernel-resource-usage \
    "$ROOT/runtime/bench/amd/kda_carry_key_precompute/kernel.hip" \
    "$ROOT/runtime/bench/amd/kda_carry_key_precompute/bench.cpp" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/bench" 2>"$OUT/resources"

resource() {
    sed -n "/Function Name: $1/,/Function Name:/p" "$OUT/resources" |
        grep -F "$2:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}
for name in k_wu_control k_wu_key_factors k_carry_control k_carry_precomputed; do
    vgpr=$(resource "$name" VGPRs); sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill'); sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]'); lds=$(resource "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a -n "$sspill" -a -n "$occ" -a -n "$lds"
    printf '%-22s VGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s\n' \
        "$name" "$vgpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
    if ((scratch != 0 || vspill != 0 || sspill != 0 || occ < 2 || lds > 163840)); then
        echo "FAIL: $name crossed the gfx950 lean-object resource gate" >&2; exit 1
    fi
done

"$ROOT/perf-data/tools/gpulease" -n 1 kda-carry-key-precompute \
    "$OUT/bench" "${T:-8192}" "${H:-12}" 128 128 "${SAMPLES:-21}"
