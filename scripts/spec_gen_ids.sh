#!/usr/bin/env bash
# Generate a real K3 token-id stream, for the speculative-decoding acceptance study.
#
#   ./scripts/spec_gen_ids.sh [assets] [steps]
#
# `perf-data/k3-speculative-decoding.md` §1 measures n-gram acceptance on REAL tokens rather than
# on words, because word granularity flatters prompt-lookup (a matched word is 1-2 BPE tokens).
# This is what produces those tokens; pipe the bracketed id list into `scripts/spec_accept_sim.py`.
#
# Speculative decoding is OUT OF SCOPE for the K3 campaign — that document records the decision and
# the numbers behind it. This script exists so the negative result stays reproducible.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:-/home/lava/models/k3_b1}"
STEPS="${2:-400}"
CKPT="${CKPT:-/home/lava/models/k3_farm}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"

exec "$WT/perf-data/tools/gpulease" -n 8 specgen sg render -c \
  "nix develop $WT --command $BIN amd-bench \
   --blob $ASSETS/model.pkt --hsaco $ASSETS/hsaco --checkpoint $CKPT \
   --tp 8 --steps $STEPS --ctx 512 --prompt '1008,10484,318,15383,387'"
