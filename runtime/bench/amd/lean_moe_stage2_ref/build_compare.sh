#!/usr/bin/env bash
# Build the shipping stage-2 object, one candidate object, and the exact comparator.
#   build_compare.sh OUT [candidate hipcc defines...]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/lean_moe_stage2_ref"
OUT=${1:-/tmp/plow-moe2-compare}
shift || true
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
BUILD="$ROOT/runtime/cmake/hipcc_hsaco.sh"
BUNDLER="$TOOLROOT/lib/llvm/bin/clang-offload-bundler"
MAXREG=${MOE2_MAXREG:-128}
SYMBOL=${MOE2_SYMBOL:-plow_moe2_mxfp4_16x16x128_gfx950}

bash "$BUILD" "$HIPCC" "$BUNDLER" gfx950 \
    "$OUT/shipping.elf" plow_moe2_mxfp4_16x16x128_gfx950 100 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe2_mxfp4_stage2_abi_3 \
    "$HERE/native_kernel.hip"
bash "$BUILD" "$HIPCC" "$BUNDLER" gfx950 \
    "$OUT/candidate.elf" "$SYMBOL" "$MAXREG" 2 \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_moe2_mxfp4_stage2_abi_3 "$@" \
    "$HERE/native_kernel.hip"
"$HIPCC" -O2 -std=c++17 "$HERE/stage2_compare.cpp" -o "$OUT/stage2_compare"
