#!/usr/bin/env bash
# PX-3 helper: compile the seg_gemm interp objects and the standalone occ bench.
# Runs with a clean env (nix CPATH conflicts with CUDA math headers).
set -euo pipefail
W=/root/plow/.claude/worktrees/agent-ab9a15a8bb551c881
SRC="$W/runtime/nvidia/interp_sm120.cu"
NVCC=/usr/local/cuda/bin/nvcc
INC="-I $W/runtime/common -I $W/runtime/nvidia"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
unset CPATH LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true

case "${1:-}" in
  ptxas128)
    exec "$NVCC" -arch=sm_120a -O3 $INC \
      -DPLOW_NV_GEMMA=1 -DPLOW_NV_PREFILL=1 -DPLOW_NV_FA_GF=2 \
      -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_SEG_GEMM=1 \
      -Xptxas -v -cubin -o /tmp/pfgemm_bn128.cubin "$SRC" ;;
  ptxas64)
    exec "$NVCC" -arch=sm_120a -O3 $INC \
      -DPLOW_NV_GEMMA=1 -DPLOW_NV_PREFILL=1 -DPLOW_NV_FA_GF=2 \
      -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_SEG_GEMM=1 -DPLOW_NV_SEG_GEMM_BN64=1 \
      -Xptxas -v -cubin -o /tmp/pfgemm_bn64.cubin "$SRC" ;;
  bench)
    BN="${2:?bn}"
    exec "$NVCC" -arch=sm_120a -O3 $INC -DPGM_BN="$BN" \
      -o "/tmp/gemm_occ_bench_bn$BN" "$W/perf-data/gemm_occ_bench.cu" ;;
  *) echo "usage: px3_build.sh {ptxas128|ptxas64|bench <BN>}"; exit 1 ;;
esac
