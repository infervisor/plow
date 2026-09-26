#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${1:-/tmp/plow-mla-prefill-8k}
MODE=${2:-v2}
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
    local file=$1 kernel=$2 name=$3
    sed -n "/Function Name: $kernel/,/Function Name:/p" "$file" |
        grep -F "$name:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

case "$MODE" in
    v2)
        build current "-DPLOW_MLA_PF_SV=1"
        build tr16 "-DPLOW_MLA_PF_SV=1 -DPLOW_MLA_PF_TR16=1"
        candidate=tr16
        kernel=k_mla_prefill_v2
        ;;
    v2-prod)
        build current ""
        build tr16 "-DPLOW_MLA_PF_SV=1 -DPLOW_MLA_PF_TR16=1"
        candidate=tr16
        kernel=k_mla_prefill_v2
        ;;
    mfma)
        build current "-DPLOW_MLA_PF_SV=1"
        build split "-DPLOW_MLA_PF_SV=1 -DPLOW_MLA_PF_SMX=1"
        candidate=split
        kernel=k_mla_prefill_mfma
        ;;
    *) echo "usage: $0 [output-dir] [v2|v2-prod|mfma]" >&2; exit 2 ;;
esac

for name in current "$candidate"; do
    report="$OUT/$name.resources"
    vgpr=$(field "$report" "$kernel" VGPRs)
    agpr=$(field "$report" "$kernel" AGPRs)
    sgpr=$(field "$report" "$kernel" TotalSGPRs)
    scratch=$(field "$report" "$kernel" 'ScratchSize [bytes/lane]')
    sgpr_spill=$(field "$report" "$kernel" 'SGPRs Spill')
    vgpr_spill=$(field "$report" "$kernel" 'VGPRs Spill')
    occ=$(field "$report" "$kernel" 'Occupancy [waves/SIMD]')
    lds=$(field "$report" "$kernel" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$agpr" -a -n "$sgpr" -a -n "$scratch" \
        -a -n "$sgpr_spill" -a -n "$vgpr_spill" -a -n "$occ" -a -n "$lds"
    total=$((vgpr + agpr))
    wg=256
    [ "$MODE" = mfma ] && wg=512
    printf '%-8s wave=64 WG=%s VGPR=%s AGPR=%s total=%s SGPR=%s occ=%s scratch=%s sgpr_spill=%s vgpr_spill=%s LDS=%s\n' \
        "$name" "$wg" "$vgpr" "$agpr" "$total" "$sgpr" "$occ" "$scratch" \
        "$sgpr_spill" "$vgpr_spill" "$lds"
    if ((total > 384 || occ < 1 || sgpr_spill != 0 || lds > 163840)) ||
       { [ "$MODE" = v2 -o "$name" != current ] &&
         ((scratch != 0 || vgpr_spill != 0)); }; then
        echo "FAIL: $name $kernel crossed the lean-object resource gate" >&2
        exit 1
    fi
done

"$HIPCC" -O2 -w "$HOST" -o "$OUT/bench" -lamdhip64
"$OUT/bench" "$OUT/current.co" "$OUT/$candidate.co" 9 "$MODE"
