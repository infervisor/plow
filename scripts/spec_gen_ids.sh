#!/usr/bin/env bash
# Generate a real K3 token-id stream, for the speculative-decoding acceptance study.
#
#   ./scripts/spec_gen_ids.sh [assets] [steps]
#
# `perf-data/archive/k3/k3-speculative-decoding.md` §1 measures n-gram acceptance on REAL tokens rather than
# on words, because word granularity flatters prompt-lookup (a matched word is 1-2 BPE tokens).
# This is what produces those tokens; pipe the bracketed id list into `scripts/spec_accept_sim.py`.
#
# Speculative decoding is OUT OF SCOPE for the K3 campaign — that document records the decision and
# the numbers behind it. This script exists so the negative result stays reproducible.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:-/home/lava/models/k3_b1}"
STEPS="${2:-400}"
CKPT="${CKPT:-/home/lava/models/k3_farm}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
REPORT="${SPEC_GEN_REPORT:-/tmp/plow-spec-gen-$USER.json}"
NIX="${PLOW_NIX_BIN:-nix}"
LEASE="${PLOW_GPULEASE_BIN:-$WT/perf-data/tools/gpulease}"

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "steps must be positive" >&2; exit 2; }
PY='import json,sys; print(json.load(open(sys.argv[1]))["parity"]["output_token_ids"][0])'
printf -v PLOW_CMD \
  'exec %q --rt-checkpoint %q bench --assets %q --prompt-ids %q --concurrency 1 --requests 1 --warmup-requests 0 --output-len %q --parity-report >%q' \
  "$BIN" "$CKPT" "$ASSETS" '1008,10484,318,15383,387' "$STEPS" "$REPORT"
printf -v RUN \
  '%q develop %q --command bash -c %q && python3 -c %q %q' \
  "$NIX" "$WT" "$PLOW_CMD" "$PY" "$REPORT"

exec "$LEASE" -n 8 specgen sg render -c \
  "$RUN"
