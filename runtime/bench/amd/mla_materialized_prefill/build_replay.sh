#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:-/tmp/plow-mla-boundary-replay}
HIPCC=${HIPCC:-$(command -v hipcc)}
mkdir -p "$OUT"
ROCM_ROOT=$(dirname "$(dirname "$(readlink -f "$HIPCC")")")

"$HIPCC" --offload-arch=gfx950 -O3 -w \
    -DPLOW_BUCKET_FLASH=1 -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 \
    --genco "$ROOT/runtime/bench/amd/mla_materialized_prefill/kernel.hip" \
    -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common" -o "$OUT/kernel.co"

bash "$ROOT/runtime/cmake/hipcc_hsaco.sh" "$HIPCC" \
    "$ROCM_ROOT/lib/llvm/bin/clang-offload-bundler" gfx950 \
    "$OUT/opus.elf" plow_mla_materialized_hd192_v128_gfx950 256 2 \
    -std=c++20 -I"$ROOT/runtime/amd/third_party/aiter_opus" \
    -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
    -DPLOW_REQUIRED_MARKER=plow_mla_materialized_opus_abi_1 \
    "$ROOT/runtime/amd/mla_materialized_opus.hip"
"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/mla_materialized_prefill/replay.cpp" \
    -o "$OUT/replay" -lamdhip64
"$HIPCC" -O2 -w "$ROOT/runtime/bench/amd/mla_materialized_prefill/replay_absorbed.cpp" \
    -o "$OUT/replay-absorbed" -lamdhip64
printf '%s\n' "$OUT/replay" "$OUT/replay-absorbed" "$OUT/opus.elf" "$OUT/kernel.co"
