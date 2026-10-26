#!/usr/bin/env bash
# p0_gate.sh <model>...: FP32-reference gate (llm_fp32_ref) on the n_cu 90 packets, plowrt pinned to the
# 90 inference cores; reuses the campaign's ref.json / vLLM capture (same 2K prompt set).
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
G=/tmp/g4c/gate
export PYREF=/tmp/g4c/bin/vllm-py
mkdir -p /tmp/g4c/l2r/gatebin/release && cp /tmp/g4c/l2r/bin/plowrt /tmp/g4c/l2r/gatebin/release/plowrt
for M in "$@"; do
  case $M in
  gemma-4-E2B-it) slug=gemma-4-e2b; ENV=GEMMA4_E2B ;;
  gemma-4-E4B-it) slug=gemma-4-e4b; ENV=GEMMA4_E4B ;;
  esac
  export ${ENV}_CHECKPOINT=/tmp/models/google/$M ${ENV}_FP32_REF=$G/$M.ref.json ${ENV}_FP32_VLLM=$G/$M.vllm.json
  O=/tmp/g4c/l2r/results/p0_gate/$M; rm -rf $O; mkdir -p $(dirname $O)
  R=$WT/recipes/infervisor/$slug/xeon6-amx-bf16.toml
  (cd $WT && python3.11 scripts/campaign/campaign.py gate "$R" --assets /tmp/g4c/l2r/pk90/$M --out $O --only llm_fp32_ref --plowrt /tmp/g4c/l2r/bin/plowrt --dry-run > $O.dryrun.log 2>&1)
  PLOWRT_DIR=/tmp/g4c/l2r/gatebin taskset -c 2-31,34-63,66-95 bash $O/run.sh > $O.run.log 2>&1
  (cd $WT && python3.11 scripts/campaign/campaign.py gate "$R" --assets /tmp/g4c/l2r/pk90/$M --out $O --only llm_fp32_ref --score-only 2>&1 | tail -3)
done
