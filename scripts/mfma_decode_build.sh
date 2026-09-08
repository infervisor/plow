#!/usr/bin/env bash
# Build the MFMA batched-decode primitive bench. Run inside `nix develop /app/plow`.
#   nix develop /app/plow --command scripts/mfma_decode_build.sh <outdir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build-mfma}"
mkdir -p "$OUT"
cd "$ROOT"

HIPCC="${PLOW_HIPCC:-hipcc}"
echo "hipcc: $HIPCC"
"$HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 --genco \
  -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=4 -DPLOW_WG_WAVES=8 -DGV_MFMA4=1 \
  -DGM_BM=192 -DGM_BN=256 -DGM_BK=64 -DGM_DBUF=1 \
  -Iruntime/amd -Iruntime/common runtime/bench/amd/gemma31_mfma_decode_bench.hip \
  -o "$OUT/mf31.co"

c++ -O2 -std=c++17 runtime/bench/amd/gemma31_mfma_decode_bench.cpp \
  -I"$ROCM_PATH/include" -D__HIP_PLATFORM_AMD__ \
  -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o "$OUT/mf31"
echo "built $OUT/mf31.co $OUT/mf31"
