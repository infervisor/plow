#!/usr/bin/env bash
# PLOW_CHAIN_BYPASS ceiling arms for GLM-5.2 TP4 decode.
#
# Each arm splices ONE candidate producer out of the dependency chain: its consumers
# wait on its predecessors instead. The op STILL RUNS on the same workgroups, so packet
# count / workgroup-packet count / memory traffic are identical and only chain depth and
# gate structure change. That is the STRICT UPPER BOUND on any partial-completion or
# tile-granular signalling scheme on that edge (the limit of "consumer never waits").
#
# The control (no knob) MUST stay md5-identical to the SHIPPING blob e818c91b..., which
# is what proves the knob is inert when unset.
set -euo pipefail
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
REPO="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT=/home/lava/models/glm52_bypass
W=/home/lava/models/GLM-5.2-plow
PC="$REPO/target/release/plowc"
cd "$REPO"
emit () {
  local tag="$1"; shift
  unset PLOW_CHAIN_BYPASS PLOW_XR_CUS GLM_SPINE_CUS PLOW_GLM_FUSE_B1 || true
  for kv in "$@"; do export "$kv"; done
  GLM_FULL=1 PLOW_FP8=1 "$PC" --hf-dir "$W" --emit devblob \
      --max-ctx 65536 --n-cu 256 --num-gpus 4 --no-rope-gen --out "$OUT/$tag.pkt" 2>"$OUT/$tag.emit.log"
  grep -h "CHAIN-BYPASS" "$OUT/$tag.emit.log" || echo "   (no bypass)"
  ls -l --time-style=+%H:%M:%S "$OUT/$tag.pkt" | awk '{print "   ",$NF,$5"B",$6}'
}
#           op ids: 1 RmsNorm  4 Residual  24 XReduce  43 MoeCombine  56 MoeRouterTopk
#                   3 HeadNormRope  57 MlaMergeFold  19 GemvGlu
emit ctl
emit b_resid   PLOW_CHAIN_BYPASS=4
emit b_rms     PLOW_CHAIN_BYPASS=1
emit b_spine   PLOW_CHAIN_BYPASS=1,4
emit b_spine3  PLOW_CHAIN_BYPASS=1,4,56
emit b_comb    PLOW_CHAIN_BYPASS=43
emit b_xr      PLOW_CHAIN_BYPASS=24
emit b_rope    PLOW_CHAIN_BYPASS=3
emit b_fold    PLOW_CHAIN_BYPASS=57
md5sum "$OUT"/ctl.pkt /home/lava/models/glm52_tp/glm52_tp4_64k.pkt
