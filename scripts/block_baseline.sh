#!/usr/bin/env bash
# scripts/block_baseline.sh — PyTorch single-block baseline, serialized on the GPU.
#
# Wraps scripts/block_baseline.py in `gpulease` (advisory flock) so concurrent
# agents on the one card never contend a timing run. This is the vLLM/PyTorch
# BASELINE side of the block harness; the plow side is `block_run <asset> bench`
# (crates/plowrt/examples/block_run.rs). Both emit the same sweep.json schema.
#
# Usage:
#   ./scripts/block_baseline.sh <descriptor.json> [-- <extra block_baseline.py args>]
#
# Examples:
#   ./scripts/block_baseline.sh crates/plowc/examples/transformer_block_gemma4_12b.json \
#       -- --layers 48 --vllm-tpot-ms 19.78 --out /dev/shm/block-baseline/gemma12b.json
#   ./scripts/block_baseline.sh crates/plowc/examples/moe_gemma4_26b_a4b.json \
#       -- --batch 1,4 --ctx 1024,4096 --out /dev/shm/block-baseline/gemma26b-moe.json
#
# Env:
#   PLOW_PY        python with a CUDA torch (default: python3)
#   GPULEASE       path to the gpulease script (default: perf-data/tools/gpulease)
#   GPU_LEASE_*    forwarded to gpulease (GPU_LEASE_TIMEOUT, ...)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESC="${1:?usage: block_baseline.sh <descriptor.json> [-- extra args]}"; shift
[ "${1:-}" = "--" ] && shift || true

PY="${PLOW_PY:-python3}"
GPULEASE="${GPULEASE:-$REPO/perf-data/tools/gpulease}"
LABEL="blkbase-$(basename "$DESC" .json)"

# gpulease writes its lock/log under /workspace/gpu; make sure it exists.
mkdir -p /workspace/gpu 2>/dev/null || true

if [ -x "$GPULEASE" ]; then
  exec "$GPULEASE" "$LABEL" "$PY" "$REPO/scripts/block_baseline.py" "$DESC" "$@"
else
  echo "WARN: gpulease not found/executable at $GPULEASE — running WITHOUT GPU lease" >&2
  exec "$PY" "$REPO/scripts/block_baseline.py" "$DESC" "$@"
fi
