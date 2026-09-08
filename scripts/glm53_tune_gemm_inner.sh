#!/usr/bin/env bash
# INSIDE-nix half of the gfx942 GEMM tile campaign for GLM-5.3. Measures the objects this
# campaign SHIPS (build-glm53/hsaco, built with the flake's ROCm 7.14), because the tuning store
# is keyed on defines + toolchain + preprocessed-source digest: a campaign measured against a
# differently-built object is stale the moment it lands.
#
# GLM_FULL=1 and PLOW_MLA_PREFILL are both load-bearing for the DEMAND emit this drives:
# without GLM_FULL the --num-gpus 4 emit panics (tp==1 assert on the bring-up path), and
# without a prefill ladder the emit declares no dense GEMM at all and the command correctly
# refuses an empty campaign.
set -euo pipefail
WT="${1:?repo}"; OBJ="${2:?objdir}"; CKPT="${3:?ckpt}"; OUT="${4:?samples.jsonl}"; TP="${5:-4}"
export LD_LIBRARY_PATH="${ROCM_PATH:?}/lib:${LD_LIBRARY_PATH:-}"
export GLM_FULL=1 PLOW_FP8=1 PLOW_MLA_PREFILL=full
export GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1
cd "$WT"
exec "$WT/target-glm53/release/plowc" --hf-dir "$CKPT" --gpu MI300X --arch gfx942 \
  --max-ctx "${TUNE_CTX:-10240}" --n-cu 304 --num-gpus "$TP" \
  tune gemm --gpu MI300X --root . --obj "$OBJ" --samples "$OUT" \
  --campaign glm53-gfx942-mi300x-tile
