#!/usr/bin/env bash
# scripts/block_layer_bench.sh — vLLM-native single decoder-layer baseline, GPU-serialized.
#
# Wraps scripts/block_layer_bench.py in `gpulease` (advisory flock) so concurrent
# agents on one card never contend a timing run. Requires a vLLM venv.
#
# Usage:
#   PLOW_PY=/workspace/venvs/vllm-blk/bin/python \
#     ./scripts/block_layer_bench.sh perf-data/block-configs/gemma4-31b.json -- --batch 1,4 --ctx 1024
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:?usage: block_layer_bench.sh <block-config.json> [-- extra args]}"; shift
[ "${1:-}" = "--" ] && shift || true
PY="${PLOW_PY:-/workspace/venvs/vllm-blk/bin/python}"
GPULEASE="${GPULEASE:-$REPO/perf-data/harness/gpulease}"
mkdir -p /workspace/gpu 2>/dev/null || true
if [ -x "$GPULEASE" ]; then
  exec "$GPULEASE" "blklayer-$(basename "$CFG" .json)" "$PY" "$REPO/scripts/block_layer_bench.py" "$CFG" "$@"
else
  echo "WARN: gpulease missing at $GPULEASE — running WITHOUT GPU lease" >&2
  exec "$PY" "$REPO/scripts/block_layer_bench.py" "$CFG" "$@"
fi
