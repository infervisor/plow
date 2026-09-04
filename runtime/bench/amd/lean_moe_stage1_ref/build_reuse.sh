#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/lean_moe_stage1_ref"
OUT=${1:-/tmp/plow-moe1-reuse}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
BUILD="$ROOT/runtime/cmake/hipcc_hsaco.sh"

bash "$BUILD" "$HIPCC" "$TOOLROOT/lib/llvm/bin/clang-offload-bundler" gfx950 \
    "$OUT/reuse.elf" plow_moe1_a4_reuse_16x16x128_gfx950 192 2 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_a4_reuse_abi_1 \
    "$HERE/reuse_kernel.hip"
python3 "$HERE/check_reuse.py" "$TOOLROOT/lib/llvm/bin/llvm-readelf" "$OUT/reuse.elf"
bash "$BUILD" "$HIPCC" "$TOOLROOT/lib/llvm/bin/clang-offload-bundler" gfx950 \
    "$OUT/shipping.elf" plow_moe1_mxfp4_bk256_gfx950 192 2 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_mxfp4_stage1_abi_1 \
    "$HERE/native_kernel.hip"
"$HIPCC" -O2 -std=c++17 "$HERE/reuse_compare.cpp" -o "$OUT/reuse_compare"
