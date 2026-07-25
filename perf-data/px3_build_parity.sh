#!/usr/bin/env bash
set -euo pipefail
W=/root/plow/.claude/worktrees/agent-ab9a15a8bb551c881
NVCC=/usr/local/cuda/bin/nvcc
INC="-I $W/runtime/common -I $W/runtime/nvidia"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
unset CPATH LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true
BN="${1:?bn}"
exec "$NVCC" -arch=sm_120a -O3 $INC -DPGM_BN="$BN" -o "/tmp/gemm_parity_bn$BN" "$W/perf-data/gemm_parity.cu"
