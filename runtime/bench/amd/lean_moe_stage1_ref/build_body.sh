#!/usr/bin/env bash
# Build the shipping A4-reuse object, one candidate body variant, and the exact comparator.
#   build_body.sh OUT [candidate hipcc defines, default -DPLOW_MOE1_BODY=1]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/lean_moe_stage1_ref"
OUT=${1:-/tmp/plow-moe1-body}
shift || true
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
BUILD="$ROOT/runtime/cmake/hipcc_hsaco.sh"
BUNDLER="$TOOLROOT/lib/llvm/bin/clang-offload-bundler"
MAXREG=${MOE1_MAXREG:-192}
if [ "$#" -eq 0 ]; then set -- -DPLOW_MOE1_BODY=1; fi

bash "$BUILD" "$HIPCC" "$BUNDLER" gfx950 \
    "$OUT/shipping.elf" plow_moe1_a4_reuse_16x16x128_gfx950 192 2 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_a4_reuse_abi_1 \
    "$HERE/reuse_kernel.hip"
bash "$BUILD" "$HIPCC" "$BUNDLER" gfx950 \
    "$OUT/candidate.elf" plow_moe1_a4_reuse_16x16x128_gfx950 "$MAXREG" 2 \
    -I "$ROOT/runtime/amd" -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 \
    -DPLOW_NO_SGPR_SPILL=1 -DPLOW_REQUIRED_MARKER=plow_moe1_a4_reuse_abi_1 "$@" \
    "$HERE/reuse_kernel.hip"
"$HIPCC" -O2 -std=c++17 "$HERE/body_compare.cpp" -o "$OUT/body_compare"
