#!/usr/bin/env bash
# Lease 4 GPUs (GLM is TP4) and run the shared-gate/up A/B. Blocks until 4 are free rather than
# contending — a contended run silently invalidates every number (knob-contract §5).
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-14400}"
exec "$WT/perf-data/harness/gpulease" -n 4 glu-split sg render -c \
  "cd '$WT' && unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES && nix develop -c bash scripts/glm52_glusplit_run.sh"
