#!/usr/bin/env bash
set -euo pipefail
W=/root/plow/.claude/worktrees/agent-ab9a15a8bb551c881
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
unset CPATH LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true
# default _pf prefill object (must be unaffected by the PGM_BN guard / BN64 flag)
exec /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 -I "$W/runtime/common" -I "$W/runtime/nvidia" \
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_PREFILL=1 -DPLOW_NV_FA_GF=2 \
  -Xptxas -v -cubin -o /tmp/pf_default.cubin "$W/runtime/nvidia/interp_sm120.cu"
