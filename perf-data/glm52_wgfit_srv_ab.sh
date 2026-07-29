#!/usr/bin/env bash
# §0-BENCH-legal end-to-end A/B: `vllm bench serve` -> plowrt TP4 GLM-5.2, both arms.
# ctl = PLOW_GLM_WGFIT=0 emit, fit = default. SAME hsaco (ctxsweep-objs), same recipe,
# only the packets' `blocks` fields differ. Coherence BEFORE timing on each arm.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
D=/home/lava/models/glm52_wgfit_srv
OUT=/home/lava/models/glm52_wgfit/srv_raw.txt
PORT=8177
: > "$OUT"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd "$WT" || exit 1
for arm in ctl fit ctl; do
  echo "########## SRV ARM $arm  $(date -Is)" | tee -a "$OUT"
  echo "===== coherence $arm" | tee -a "$OUT"
  bash "$WT/scripts/rebench_glm_coherence.sh" "$D/$arm" "$PORT" glm-5.2 2>&1 | tail -25 | tee -a "$OUT"
  echo "===== bench $arm" | tee -a "$OUT"
  IN_LENS="4096" CONCS="1" NPROMPT=8 OUTLEN=128 \
    bash "$WT/scripts/bench_plowrt_serve.sh" "$D/$arm" "$PORT" glm-5.2 zai-org/GLM-5.2-FP8 2>&1 \
    | tee -a "$OUT"
done
echo "########## SRV DONE $(date -Is)" | tee -a "$OUT"
touch /home/lava/models/glm52_wgfit/srv.done
