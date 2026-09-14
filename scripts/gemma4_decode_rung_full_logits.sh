#!/usr/bin/env bash
# One-process Gemma-4 decode-rung full-logit sweep. The wrapper takes one GPU
# lease, then loads the widest-rung reference once and the candidate once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:?usage: $0 TEST_BINARY CANDIDATE_ASSETS REFERENCE_ASSETS PACKET_SHA256 /tmp/OUTPUT.jsonl}"
CANDIDATE="${2:?candidate assets required}"
REFERENCE="${3:?reference assets required}"
PACKET_SHA256="${4:?candidate packet SHA256 required}"
OUTPUT="${5:?output path required}"
[[ $# == 5 ]] || { echo "usage: $0 TEST_BINARY CANDIDATE_ASSETS REFERENCE_ASSETS PACKET_SHA256 /tmp/OUTPUT.jsonl" >&2; exit 2; }
[[ "$OUTPUT" == /tmp/* ]] || { echo "output must be under /tmp" >&2; exit 2; }

if [[ -z "${PLOW_GEMMA4_FULL_LOGITS_LEASED:-}" ]]; then
  LEASE="${PLOW_GPULEASE_BIN:-$ROOT/perf-data/tools/gpulease}"
  [[ -x "$LEASE" ]] || { echo "gpulease missing at $LEASE" >&2; exit 2; }
  exec "$LEASE" -n 1 gemma4-decode-full-logits \
    env PLOW_GEMMA4_FULL_LOGITS_LEASED=1 "$0" "$@"
fi

export PLOW_MULTISTEP=0 PLOW_PF_BATCH=0 PLOW_VMM_PREFIX=0
export TEST_DECODE_RUNG_GPU=1
export TEST_DECODE_RUNG_ASSETS="$CANDIDATE"
export TEST_DECODE_RUNG_BASELINE="$REFERENCE"
export TEST_RUNG_PACKET_SHA256="$PACKET_SHA256"
export TEST_DECODE_RUNG_LOGITS_OUT="$OUTPUT"
exec "$BIN" gpu_isolated_decode_rungs_match_widest_logits \
  --ignored --nocapture --test-threads=1
