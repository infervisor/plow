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

ROCM_ROOT=$(dirname "$(dirname "$(readlink -f "$HIPCC")")")
bash "$ROOT/runtime/cmake/hipcc_hsaco.sh" "$HIPCC" \
    "$ROCM_ROOT/lib/llvm/bin/clang-offload-bundler" gfx950 \
    "$OUT/opus.elf" plow_mla_materialized_hd192_v128_gfx950 256 2 \
    -std=c++20 -I"$ROOT/runtime/amd/third_party/aiter_opus" \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_materialized_opus_abi_1 \
    "$ROOT/runtime/amd/mla_materialized_opus.hip"
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

for name in k_materialize_q k_materialize_kv k_absorb_q k_absorb_qrope; do
    vgpr=$(resource "$OUT/resources" "$name" VGPRs)
    sgpr=$(resource "$OUT/resources" "$name" TotalSGPRs)
    scratch=$(resource "$OUT/resources" "$name" 'ScratchSize [bytes/lane]')
    vspill=$(resource "$OUT/resources" "$name" 'VGPRs Spill')
    sspill=$(resource "$OUT/resources" "$name" 'SGPRs Spill')
    occ=$(resource "$OUT/resources" "$name" 'Occupancy [waves/SIMD]')
    lds=$(resource "$OUT/resources" "$name" 'LDS Size [bytes/block]')
    test -n "$vgpr" -a -n "$sgpr" -a -n "$scratch" -a -n "$vspill" -a \
        -n "$sspill" -a -n "$occ" -a -n "$lds"
    printf '%-18s VGPR=%s SGPR=%s occ=%s scratch=%s spills=%s/%s LDS=%s (oracle only)\n' \
        "$name" "$vgpr" "$sgpr" "$occ" "$scratch" "$vspill" "$sspill" "$lds"
done

bash "$ROOT/runtime/cmake/hipcc_hsaco.sh" "$HIPCC" \
    "$ROCM_ROOT/lib/llvm/bin/clang-offload-bundler" gfx950 \
    "$OUT/pack.elf" plow_mla_materialize_pack_gfx950 64 4 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_materialize_pack_abi_1 \
    "$ROOT/runtime/amd/mla_materialize_pack.hip"

bash "$ROOT/runtime/cmake/hipcc_hsaco.sh" "$HIPCC" \
    "$ROCM_ROOT/lib/llvm/bin/clang-offload-bundler" gfx950 \
    "$OUT/upstream-grid.elf" oracle_mla_materialized_upstream_grid 256 2 \
    -std=c++20 -I"$ROOT/runtime/amd/third_party/aiter_opus" \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_materialized_upstream_grid_oracle_1 \
    "$ROOT/runtime/bench/amd/mla_materialized_prefill/opus_upstream_grid.hip"

"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/mla_materialized_prefill/bench.cpp" \
    -o "$OUT/bench" -lamdhip64
"$ROOT/perf-data/tools/gpulease" -n 1 mla-materialized-prefill \
    "$OUT/bench" "$OUT/kernel.co" "$OUT/opus.elf" "$OUT/pack.elf" \
    "$OUT/upstream-grid.elf" "${SAMPLES:-9}"
