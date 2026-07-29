#!/usr/bin/env bash
# Slice-level locality census (`PLOW_PLACE_REPORT=1`) + Lean checkpoint D/oracle on one model.
#
# Answers the question a locality-aware placement pass has to answer FIRST: how much of the
# program's slice-level producer->consumer dataflow could same-domain placement capture?
# See `plans/l2-placement-generic.md` §7.
#
#   $1  out.pkt (default /tmp/placediff/census.pkt)   $2..  extra plowc args
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/placediff/census.pkt}"; shift || true
CKPT="${PLOW_CKPT:-$WT/../../../build-amd/g31b-bf16/checkpoint}"
mkdir -p "$(dirname "$OUT")"

export PLOW_PLACE_REPORT=1
export PLOW_L2_PLACE="${PLOW_L2_PLACE:-1}"
export PLOW_VERIFY_BIN="${PLOW_VERIFY_BIN:-$WT/lean-plow/.lake/build/bin/plow_verify}"

cd "$WT"
exec ./target/release/plowc --hf-dir "$CKPT" --emit devblob --arch gfx950 --gpu mi355x \
    --n-cu 256 --max-ctx "${PLOW_CTX:-1024}" --out "$OUT" "$@"
