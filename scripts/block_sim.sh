#!/usr/bin/env bash
# scripts/block_sim.sh — M0 single-block sim loop (CPU only, no GPU).
#
# Compiles ONE representative transformer block, then walks it through the
# plowrt CPU simulator and prints the resulting activation summary (packets
# fired, makespan, per-op-family counts). This is Route S of
# plans/block-asset-harness.md: block descriptor -> plowc -> plowrt simulate.
#
# Usage:
#   ./scripts/block_sim.sh <hf-dir-or-config> [bucket]
#
#   <hf-dir-or-config>  A directory holding config.json + *.safetensors -> the
#                       `plowc --hf-dir` route (needs a real checkpoint), OR a
#                       plow-native NetConfig .json file -> the `plowc --net`
#                       route (weight-free, runs anywhere).
#   [bucket]            `<phase>:<batch>:<seq>` (default `prefill:1:128`).
#
# Examples:
#   ./scripts/block_sim.sh crates/plowc/examples/transformer_block_gemma4_12b.json
#   ./scripts/block_sim.sh /path/to/gemma-hf-dir decode:1:128
#
# Requires: nix develop (cargo), plowc + plowrt built from this tree.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: block_sim.sh <hf-dir-or-config> [bucket]}"
BUCKET="${2:-prefill:1:128}"
OUT="${PLOW_BLOCK_OUT:-/tmp/block_sim_out}"

# bucket = <phase>:<batch>:<seq>
IFS=':' read -r PHASE BATCH SEQ <<<"$BUCKET"
: "${PHASE:?bucket must be <phase>:<batch>:<seq>}" \
  "${BATCH:?}" "${SEQ:?}"

# Directory -> HF checkpoint route; file -> plow-native NetConfig route.
if [ -d "$SRC" ]; then
  SRC_FLAG=(--hf-dir "$SRC")
else
  SRC_FLAG=(--net "$SRC")
fi

cd "$REPO"
rm -rf "$OUT"

echo "== compile ($PHASE b$BATCH s$SEQ) =="
nix develop -c cargo run --release -q -p plowc --bin plowc -- \
  "${SRC_FLAG[@]}" --out "$OUT" \
  --batch "$BATCH" --seq "$SEQ" --phase "$PHASE"

echo
echo "== simulate (CPU) =="
nix develop -c cargo run --release -q -p plowrt --bin plowrt -- \
  simulate --assets "$OUT" --bucket "$BUCKET"
