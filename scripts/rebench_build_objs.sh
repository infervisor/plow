#!/usr/bin/env bash
# Re-benchmark campaign: build the gfx950 objects with BOTH prefill arms.
# Runs OUTSIDE nix, OUTSIDE gpulease (knob-contract §0 / §0a).
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/home/lava/plow/build-amd/rebench-objs}"
export PLOW_MLA_PREFILL=1
export PLOW_MOE_PREFILL=1
exec bash "$WT/scripts/build_gfx950.sh" "$OUT"
