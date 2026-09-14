#!/usr/bin/env bash
# Compare exact-rung Gemma-4 prefill role packets against an exact-rung control.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:?usage: $0 TEST_BINARY CANDIDATE_ASSETS CONTROL_ASSETS PACKET_SHA256 ROLE_RUNGS /tmp/OUTPUT.jsonl}"
CANDIDATE="${2:?candidate assets required}"
CONTROL="${3:?control assets required}"
PACKET_SHA256="${4:?candidate packet SHA256 required}"
ROLE_RUNGS="${5:?comma-separated role rungs required}"
OUTPUT="${6:?output path required}"
[[ $# == 6 ]] || { echo "usage: $0 TEST_BINARY CANDIDATE_ASSETS CONTROL_ASSETS PACKET_SHA256 ROLE_RUNGS /tmp/OUTPUT.jsonl" >&2; exit 2; }
[[ "$OUTPUT" == /tmp/* ]] || { echo "output must be under /tmp" >&2; exit 2; }

if [[ -z "${PLOW_GEMMA4_FULL_LOGITS_LEASED:-}" ]]; then
  LEASE="${PLOW_GPULEASE_BIN:-$ROOT/perf-data/tools/gpulease}"
  [[ -x "$LEASE" ]] || { echo "gpulease missing at $LEASE" >&2; exit 2; }
  exec "$LEASE" -n 1 gemma4-prefill-role-full-logits \
    env PLOW_GEMMA4_FULL_LOGITS_LEASED=1 "$0" "$@"
fi

export PLOW_MULTISTEP=0 PLOW_PF_BATCH=0 PLOW_VMM_PREFIX=0
export TEST_PREFILL_RUNG_GPU=1
export TEST_PREFILL_RUNG_ASSETS="$CANDIDATE"
export TEST_PREFILL_RUNG_BASELINE="$CONTROL"
export TEST_RUNG_PACKET_SHA256="$PACKET_SHA256"
export TEST_PREFILL_ROLE_RUNGS="$ROLE_RUNGS"
export TEST_PREFILL_RUNG_LOGITS_OUT="$OUTPUT"
exec "$BIN" gpu_prefill_roles_match_control_logits \
  --ignored --nocapture --test-threads=1
