#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-kda-carry-regstate}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w -Rpass-analysis=kernel-resource-usage \
    "$ROOT/runtime/bench/amd/kda_carry_regstate/kernel.hip" \
    "$ROOT/runtime/bench/amd/kda_carry_regstate/bench.cpp" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/bench" \
    2>"$OUT/resources"

resource() {
    sed -n "/Function Name: $1/,/Function Name:/p" "$OUT/resources" |
        grep -F "$2:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

for name in k_carry_control k_carry_regstate k_carry_regstate_hwcvt k_carry_regstate_keyfeed; do
    vgpr=$(resource "$name" VGPRs); agpr=$(resource "$name" AGPRs)
    sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill'); sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]'); lds=$(resource "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a -n "$sspill" -a -n "$occ"
    printf '%-20s VGPR=%s AGPR=%s SGPR=%s occ=%s lds=%s scratch=%s spills=%s/%s\n' \
        "$name" "$vgpr" "${agpr:-0}" "$sgpr" "$occ" "$lds" "$scratch" "$vspill" "$sspill"
    if ((scratch != 0 || vspill != 0 || sspill != 0)); then
        echo "FAIL: $name crossed the zero-private/spill gate" >&2
        exit 1
    fi
done

[ "${COMPILE_ONLY:-0}" = 1 ] && exit 0

"$ROOT/perf-data/tools/gpulease" -n 1 kda-carry-regstate \
    "$OUT/bench" "${T:-8192}" "${H:-12}" "${SAMPLES:-21}" "${TIMERS:-1}" "${MODE:-0}" |
    tee "$OUT/results.txt"
