#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${1:-/tmp/plow-mla-prefill-8k}
HIPCC=${HIPCC:-$(command -v hipcc)}
SRC="$ROOT/runtime/bench/amd/mla_prefill_8k_sweep.hip"
HOST="$ROOT/runtime/bench/amd/mla_prefill_8k_sweep.cpp"
mkdir -p "$OUT"

build() {
    local name=$1 define=$2
    "$HIPCC" --offload-arch=gfx950 -O3 -w $define \
        -Rpass-analysis=kernel-resource-usage --genco "$SRC" \
        -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" \
        -o "$OUT/$name.co" 2>"$OUT/$name.resources"
}

field() {
    local file=$1 name=$2
    sed -n '/Function Name: k_mla_prefill_v2/,/Function Name:/p' "$file" |
        grep -F "$name:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

build control ""
build sv "-DPLOW_MLA_PF_SV=1"

for name in control sv; do
    report="$OUT/$name.resources"
    vgpr=$(field "$report" VGPRs)
    agpr=$(field "$report" AGPRs)
    scratch=$(field "$report" 'ScratchSize [bytes/lane]')
    spill=$(field "$report" 'VGPRs Spill')
    occ=$(field "$report" 'Occupancy [waves/SIMD]')
    lds=$(field "$report" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$agpr" -a -n "$scratch" -a -n "$spill" -a -n "$occ" -a -n "$lds"
    total=$((vgpr + agpr))
    printf '%-8s VGPR=%s AGPR=%s total=%s occ=%s scratch=%s spill=%s LDS=%s\n' \
        "$name" "$vgpr" "$agpr" "$total" "$occ" "$scratch" "$spill" "$lds"
    if ((total > 384 || occ < 1 || scratch != 0 || spill != 0 || lds > 163840)); then
        echo "FAIL: $name V2 crossed the lean-object resource gate" >&2
        exit 1
    fi
done

"$HIPCC" -O2 -w "$HOST" -o "$OUT/bench" -lamdhip64
"$ROOT/perf-data/tools/gpulease" -n 1 mla-prefill-8k-sweep \
    "$OUT/bench" "$OUT/control.co" "$OUT/sv.co" 9
