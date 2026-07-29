#!/usr/bin/env bash
# One GPU, one lease, the three arms of scripts/walk_b16_ab.sh. See that file's header.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env GPU_LEASE_TIMEOUT=7200 "$WT/perf-data/harness/gpulease" -n 1 walk-b16 \
  sg render -c "bash $WT/scripts/walk_b16_ab.sh run"
