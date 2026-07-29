#!/usr/bin/env bash
# glm52_xgate_phase2.sh — the COST side of the cross-GPU tile/watermark proposal, plus the
# positive control that decides whether the NOWAIT/NOSIG ceilings mean anything.
#
#   (a) DECODE marginal-acquire sweep (C harness), controls interleaved at 1/3/5.
#       -DPLOW_XR_ACQ_N=k takes k system-scope acquires per gate where the protocol takes 1.
#       A C-chunk watermark pays (C-1) of these per gate to win at most what PLOW_XR_NOWAIT
#       deletes, so this is the number the whole design turns on. NUMERICALLY CORRECT — an
#       extra acquire fence cannot change a value — so every arm must reproduce the control's
#       24 ids, which is the instrument's own self-check.
#   (b) PREFILL: base / shuffle / acq2 / base / acq4 / base through `plowrt serve`.
#       `shuffle` (-DPLOW_XR_SHUFFLE=1) is the POSITIVE CONTROL: equally numerically wrong,
#       protocol 100% intact. Without it the NOWAIT/NOSIG prefill deltas cannot be told apart
#       from GLM's data-dependent MoE routing doing less work on garbage activations.
#   (c) the §5 coherence gate on the CONTROL object set.
#
# Companion: perf-data/glm52_xgate_prefill_ab.sh (phase 1), plans/knob-contract.md §7e-XGATE.
W=/home/lava/models/GLM-5.2-plow
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BLOB=/home/lava/models/glm52_xr/base.pkt   # md5 e818c91b = the shipping glm52_tp4_64k.pkt
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd "$D"

echo "===== (a) marginal-acquire sweep, decode, controls interleaved at 1/3/5 ====="
run () { echo "########## $1"
  PLOW_INTERP="$D/obj/$2.elf" "$D/glm52_decode" "$BLOB" "$W" --tp 4 --sweep 1024 --steps 65 --gen 24 2>&1
  echo "########## end $1 rc=$?"; }
run base  d_base
run acq2  d_acq2
run base  d_base
run acq4  d_acq4
run base  d_base

echo "===== (b) PREFILL: marginal-acquire + nosig, all on ONE audit-off control ====="
bash "$D/phase2_prefill.sh"

echo "===== (c) coherence gate on the CONTROL objects (knob-contract §5) ====="
READY=1200 bash "$WT/scripts/rebench_glm_coherence.sh" "$D/assets_base" 8139 glm-5.2 2>&1 | tail -25

echo ALLDONE2
touch "$D/phase2.done"
