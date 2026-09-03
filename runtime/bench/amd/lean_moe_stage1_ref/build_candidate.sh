#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/lean_moe_stage1_ref"
OUT=${1:-/tmp/plow-moe1-candidate}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")

HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
BUNDLER="$TOOLROOT/lib/llvm/bin/clang-offload-bundler"
READELF="$TOOLROOT/lib/llvm/bin/llvm-readelf"
OBJDUMP="$TOOLROOT/lib/llvm/bin/llvm-objdump"
BUILD="$ROOT/runtime/cmake/hipcc_hsaco.sh"

bash "$BUILD" "$HIPCC" "$BUNDLER" gfx950 "$OUT/candidate.elf" \
    plow_moe1_mxfp4_bm64_bn128_bk256_xcd8_wgm4_gfx950 168 3 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_candidate_abi_1 \
    "$HERE/candidate_kernel.hip"
bash "$BUILD" "$HIPCC" "$BUNDLER" gfx950 "$OUT/shipping.elf" \
    plow_moe1_mxfp4_bk256_gfx950 192 2 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_mxfp4_stage1_abi_1 \
    "$HERE/native_kernel.hip"

for variant in candidate shipping; do
    "$READELF" -n "$OUT/$variant.elf" > "$OUT/$variant.notes"
    "$READELF" -sW "$OUT/$variant.elf" > "$OUT/$variant.symbols"
    "$OBJDUMP" -d --mcpu=gfx950 "$OUT/$variant.elf" > "$OUT/$variant.isa"
done
for marker in \
    plow_moe1_candidate_abi_1 \
    plow_moe1_candidate_bm64_bn128_bk256_1 \
    plow_moe1_candidate_wave64_wg256_1 \
    plow_moe1_candidate_xcd8_wgm4_1 \
    plow_moe1_candidate_generic_a4w4_1 \
    plow_moe1_candidate_situ_pair_1 \
    plow_moe1_candidate_sorted_mxfp4_scale_1 \
    plow_moe1_candidate_dynamic_lds_52224; do
    grep -qE "OBJECT .* ${marker}$" "$OUT/candidate.symbols" || {
        echo "missing candidate contract marker $marker" >&2
        exit 1
    }
done
python3 "$HERE/check_candidate.py" \
    "$OUT/candidate.notes" "$OUT/candidate.isa" \
    "$OUT/shipping.notes" "$OUT/shipping.isa"
"$HIPCC" -O2 -std=c++17 "$HERE/compare.cpp" -o "$OUT/compare"
