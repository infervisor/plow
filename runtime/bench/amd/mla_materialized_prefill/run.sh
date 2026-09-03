#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-mla-materialized-prefill}
HIPCC=${HIPCC:-$(command -v hipcc)}
AITER_ROOT=${AITER_ROOT:-/tmp/aiter-main}
mkdir -p "$OUT"
test -f "$AITER_ROOT/csrc/include/fmha_fwd_hd192_v128_bf16_opus_kernel.hpp"

"$HIPCC" --offload-arch=gfx950 -O3 -w \
    -DPLOW_BUCKET_FLASH=1 -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 \
    -Rpass-analysis=kernel-resource-usage --genco "$ROOT/runtime/bench/amd/mla_materialized_prefill/kernel.hip" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/kernel.co" 2>"$OUT/resources"

"$HIPCC" --offload-arch=gfx950 -std=c++20 -O3 -w \
    -Rpass-analysis=kernel-resource-usage --genco \
    "$ROOT/runtime/bench/amd/mla_materialized_prefill/opus_probe.hip" \
    -I"$AITER_ROOT/csrc/include" -o "$OUT/opus.co" 2>"$OUT/opus-resources"

ROCM_ROOT=$(dirname "$(dirname "$(readlink -f "$HIPCC")")")
"$ROCM_ROOT/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input="$OUT/opus.co" \
    --output="$OUT/opus.elf"
wave=$("$ROCM_ROOT/lib/llvm/bin/llvm-readobj" --notes "$OUT/opus.elf" |
    sed -n 's/^ *\.wavefront_size: *//p' | head -1)
test "$wave" = 64 || { echo "FAIL: Opus comparator wavefront=$wave, expected 64" >&2; exit 1; }

resource() {
    local file=$1 name=$2 field=$3
    sed -n "/Function Name: $name/,/Function Name:/p" "$file" |
        grep -F "$field:" | head -1 | sed 's/^.*: *//; s/ .*$//'
}

for name in k_absorbed k_absorbed_fold k_materialized k_materialized_lds; do
    vgpr=$(resource "$OUT/resources" "$name" VGPRs)
    agpr=$(resource "$OUT/resources" "$name" AGPRs)
    sgpr=$(resource "$OUT/resources" "$name" TotalSGPRs)
    scratch=$(resource "$OUT/resources" "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$OUT/resources" "$name" 'VGPRs Spill')
    sspill=$(resource "$OUT/resources" "$name" 'SGPRs Spill')
    occ=$(resource "$OUT/resources" "$name" 'Occupancy [waves/SIMD]')
    lds=$(resource "$OUT/resources" "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$agpr" -a -n "$sgpr" -a -n "$scratch" -a \
        -n "$vspill" -a -n "$sspill" -a -n "$occ" -a -n "$lds"
    printf '%-18s VGPR=%s AGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s\n' \
        "$name" "$vgpr" "$agpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
    if ((scratch != 0 || vspill != 0 || sspill != 0 || occ < 1 || lds > 163840)); then
        echo "FAIL: $name crossed the lean-object resource gate" >&2
        exit 1
    fi
done

opus_name=_Z20gqa_d192_v128_kernel
vgpr=$(resource "$OUT/opus-resources" "$opus_name" VGPRs)
agpr=$(resource "$OUT/opus-resources" "$opus_name" AGPRs)
sgpr=$(resource "$OUT/opus-resources" "$opus_name" TotalSGPRs)
scratch=$(resource "$OUT/opus-resources" "$opus_name" 'ScratchSize [bytes/lane]')
vspill=$(resource "$OUT/opus-resources" "$opus_name" 'VGPRs Spill')
sspill=$(resource "$OUT/opus-resources" "$opus_name" 'SGPRs Spill')
occ=$(resource "$OUT/opus-resources" "$opus_name" 'Occupancy [waves/SIMD]')
lds=$(resource "$OUT/opus-resources" "$opus_name" 'LDS Size [bytes/block]')
printf '%-18s VGPR=%s AGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s\n' \
    opus_hd192_v128 "$vgpr" "$agpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
if ((vgpr > 256 || sgpr > 128 || scratch != 0 || vspill != 0 || sspill != 0 || occ < 2 || lds > 163840)); then
    echo "FAIL: Opus comparator crossed the lean-object resource gate" >&2
    exit 1
fi
"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/mla_materialized_prefill/bench.cpp" \
    -o "$OUT/bench" -lamdhip64
"$ROOT/perf-data/tools/gpulease" -n 1 mla-materialized-prefill \
    "$OUT/bench" "$OUT/kernel.co" "$OUT/opus.co" "${SAMPLES:-9}"
