#!/usr/bin/env bash
# Configure the gfx950 object build against a config header that requests the lean MoE body
# variants and print the generated stage-1/stage-2 object commands (no compile).
#   cmake_body_check.sh OUT
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/moe-body-cmk}
mkdir -p "$OUT"
printf '#pragma once\n#define PLOW_PACKET_HASH 0x0ull\n#define PLOW_OBJECT_MOE_STAGE1_BODY 1\n#define PLOW_OBJECT_MOE_STAGE2_BODY 1\n' \
    > "$OUT/plow_config.h"
cmake -S "$ROOT/runtime" -B "$OUT/build" -DPLOW_GFX950_HSACO=ON -DPLOW_HSACO_ARCH=gfx950 \
    -DPLOW_HSACO_HIPCC="${PLOW_HIPCC:-hipcc}" -DPLOW_HSACO_BUNDLER="${PLOW_BUNDLER:-clang-offload-bundler}" \
    -DPLOW_HSACO_CONFIG="$OUT/plow_config.h" -DPLOW_HSACO_DIR="$OUT/hsaco" > "$OUT/config.log" 2>&1
grep -rhoE "plow_moe1_a4_reuse_16x16x128_gfx950 [0-9]+ 2 [^\"]*reuse_kernel.hip|plow_moe2_mxfp4_16x16x128_gfx950 100 2 [^\"]*native_kernel.hip" \
    "$OUT/build" 2>/dev/null | sort -u
