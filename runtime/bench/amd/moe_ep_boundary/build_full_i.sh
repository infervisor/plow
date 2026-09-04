#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/moe_ep_boundary"
OUT=${1:-/tmp/plow-moe-ep-full-i}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
BUNDLER="$TOOLROOT/lib/llvm/bin/clang-offload-bundler"
READELF="$TOOLROOT/lib/llvm/bin/llvm-readelf"
OBJCOPY="$TOOLROOT/lib/llvm/bin/llvm-objcopy"

"$ROOT/runtime/bench/amd/lean_moe_stage1_ref/build_reuse.sh" "$OUT/stage1"
"$ROOT/runtime/cmake/hipcc_hsaco.sh" "$HIPCC" "$BUNDLER" gfx950 \
    "$OUT/stage1/shipping_full_i.elf" plow_moe1_mxfp4_bk256_gfx950 192 2 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_mxfp4_stage1_abi_1 \
    -DPLOW_MOE1_INTER_DIM=3072 -DPLOW_MOE1_EXPERTS=112 \
    "$ROOT/runtime/bench/amd/lean_moe_stage1_ref/native_kernel.hip"
"$HIPCC" -O2 -std=c++17 -DMOE1_I=3072 -DMOE1_E=112 -DMOE1_TOPK=2 \
    "$ROOT/runtime/bench/amd/lean_moe_stage1_ref/reuse_compare.cpp" \
    -o "$OUT/stage1/full_i_compare"

for source in stage2_full_i filter_align combine; do
    "$HIPCC" --genco --offload-arch=gfx950 -O3 -std=c++17 \
        "$HERE/$source.hip" -o "$OUT/$source.bundle"
    "$BUNDLER" --type=o --unbundle --input="$OUT/$source.bundle" \
        --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$OUT/$source.elf"
done

PLOW_MOE2_EP_FULL_I=1 python3 \
    "$ROOT/runtime/bench/amd/lean_moe_stage2_ref/native_manifest.py" \
    "$OUT/stage2_full_i.elf" "$OUT/stage2_full_i.json" "$READELF" "$OBJCOPY"

for object in stage2_full_i filter_align combine; do
    notes=$($READELF --notes "$OUT/$object.elf")
    grep -q '.wavefront_size: 64' <<<"$notes"
    grep -q '.private_segment_fixed_size: 0' <<<"$notes"
    grep -q '.vgpr_spill_count: 0' <<<"$notes"
    grep -q '.sgpr_spill_count: 0' <<<"$notes"
done
echo "built $OUT"
