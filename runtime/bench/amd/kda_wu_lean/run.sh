#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-kda-wu-lean}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w -Rpass-analysis=kernel-resource-usage \
    "$ROOT/runtime/bench/amd/kda_wu_lean/kernel.hip" \
    "$ROOT/runtime/bench/amd/kda_wu_lean/kernel_wg256.hip" \
    "$ROOT/runtime/bench/amd/kda_wu_lean/bench.cpp" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/bench" \
    2>"$OUT/resources"

resource() {
    sed -n "/Function Name: $1/,/Function Name:/p" "$OUT/resources" |
        grep -F "$2:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

for name in k_wu_control k_wu_key_factor_object k_wu_lean k_wu_lean_keys; do
    vgpr=$(resource "$name" VGPRs); agpr=$(resource "$name" AGPRs)
    sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill'); sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]'); lds=$(resource "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a -n "$sspill" -a -n "$occ"
    printf '%-24s VGPR=%s AGPR=%s SGPR=%s occ=%s lds=%s scratch=%s spills=%s/%s\n' \
        "$name" "$vgpr" "${agpr:-0}" "$sgpr" "$occ" "$lds" "$scratch" "$vspill" "$sspill"
    if ((scratch != 0 || vspill != 0 || sspill != 0)); then
        echo "FAIL: $name crossed the zero-private/spill gate" >&2
        exit 1
    fi
done

[ "${COMPILE_ONLY:-0}" = 1 ] && exit 0

"$ROOT/perf-data/tools/gpulease" -n 1 kda-wu-lean \
    "$OUT/bench" "${T:-8192}" "${H:-12}" "${SAMPLES:-21}" "${MODE:-0}" |
    tee "$OUT/results.txt"
