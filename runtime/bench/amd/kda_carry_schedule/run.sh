#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-kda-carry-schedule}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w -Rpass-analysis=kernel-resource-usage \
    "$ROOT/runtime/bench/amd/kda_carry_schedule/kernel.hip" \
    "$ROOT/runtime/bench/amd/kda_carry_schedule/bench.cpp" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/bench" \
    2>"$OUT/resources"

resource() {
    sed -n "/Function Name: $1/,/Function Name:/p" "$OUT/resources" |
        grep -F "$2:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

for name in k_carry_control k_carry_v16_wg256 \
            k_carry_v32_wg512 k_carry_v32_staged k_carry_tail_overlap \
            k_carry_key_stage k_carry_key_stage_tail_overlap; do
    vgpr=$(resource "$name" VGPRs); sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill'); sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a -n "$sspill" -a -n "$occ"
    printf '%-24s VGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s\n' \
        "$name" "$vgpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill"
    if ((scratch != 0 || vspill != 0 || sspill != 0)); then
        echo "FAIL: $name crossed the zero-private/spill gate" >&2
        exit 1
    fi
done

v8_scratch=$(resource k_carry_v8_wg256 'ScratchSize [bytes/lane]')
v8_vspill=$(resource k_carry_v8_wg256 'VGPRs Spill')
v8_sspill=$(resource k_carry_v8_wg256 'SGPRs Spill')
test -n "$v8_scratch" -a -n "$v8_vspill" -a -n "$v8_sspill"
run_v8=1
if ((v8_scratch != 0 || v8_vspill != 0 || v8_sspill != 0)); then
    run_v8=0
    echo "k_carry_v8_wg256 rejected statically: scratch=$v8_scratch spills=$v8_vspill/$v8_sspill"
fi

wg256_spill=$(resource k_carry_v32_wg256 'SGPRs Spill')
test -n "$wg256_spill" -a "$wg256_spill" -gt 0
echo "k_carry_v32_wg256 rejected statically: SGPR spills=$wg256_spill"

"$ROOT/perf-data/tools/gpulease" -n 1 kda-carry-schedule \
    "$OUT/bench" "${T:-8192}" "${H:-12}" "${SAMPLES:-21}" "$run_v8" |
    tee "$OUT/results.txt"
