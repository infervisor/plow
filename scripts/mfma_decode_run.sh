#!/usr/bin/env bash
# Run the MFMA batched-decode primitive bench on one leased card.
#   scripts/mfma_decode_run.sh <builddir> [iters] [mm-filter]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="${1:-$ROOT/build-mfma}"
IT="${2:-41}"
MM="${3:-}"
GPU_LEASE_TIMEOUT=21600 "$ROOT/perf-data/tools/gpulease" -n 1 gemma31-mfma-decode \
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  "$B/mf31" "$B/mf31.co" "$IT" $MM
