#!/usr/bin/env bash
set -euo pipefail
W=/root/plow/.claude/worktrees/agent-ab9a15a8bb551c881
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
unset CPATH LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true
exec /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 -o /tmp/occ_probe "$W/perf-data/occ_probe.cu" -lcuda
