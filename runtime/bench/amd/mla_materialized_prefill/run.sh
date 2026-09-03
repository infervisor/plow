#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-mla-materialized-prefill}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"

"$HIPCC" --offload-arch=gfx950 -O3 -w \
    -DPLOW_BUCKET_FLASH=1 -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 \
    -Rpass-analysis=kernel-resource-usage --genco "$ROOT/runtime/bench/amd/mla_materialized_prefill/kernel.hip" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/kernel.co" 2>"$OUT/resources"

resource() {
    local name=$1 field=$2
    sed -n "/Function Name: $name/,/Function Name:/p" "$OUT/resources" |
        grep -F "$field:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

for name in k_absorbed k_absorbed_fold k_materialized k_materialized_lds; do
    vgpr=$(resource "$name" VGPRs)
    agpr=$(resource "$name" AGPRs)
    sgpr=$(resource "$name" TotalSGPRs)
    scratch=$(resource "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$name" 'VGPRs Spill')
    sspill=$(resource "$name" 'SGPRs Spill')
    occ=$(resource "$name" 'Occupancy [waves/SIMD]')
    lds=$(resource "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$agpr" -a -n "$sgpr" -a -n "$scratch" -a \
        -n "$vspill" -a -n "$sspill" -a -n "$occ" -a -n "$lds"
    printf '%-18s VGPR=%s AGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s\n' \
        "$name" "$vgpr" "$agpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
    if ((scratch != 0 || vspill != 0 || sspill != 0 || occ < 1 || lds > 163840)); then
        echo "FAIL: $name crossed the lean-object resource gate" >&2
        exit 1
    fi
done

"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/mla_materialized_prefill/bench.cpp" \
    -o "$OUT/bench" -lamdhip64
"$ROOT/perf-data/tools/gpulease" -n 1 mla-materialized-prefill \
    "$OUT/bench" "$OUT/kernel.co" 9
