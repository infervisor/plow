#!/usr/bin/env bash
# final_comb.sh <hf-model>: strict rerun of the plow arm with the PLOW_CPU_COMBINE binary and the recipe's [serve.env]
# (now incl. PLOW_CPU_COMBINE=16): plow grid, FP32 gate capture + score for the same packet with this binary, then
# campaign.py report vs the recorded vLLM 0.30 arm (and the llama.cpp arm). Outputs /tmp/g4c/final/<M>/*.comb.
# Host state as final.sh: numa_balancing=0, qos0 holder, THP madvise, no pseudo-lock held.
set -u
M=$1
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/final/$M
G=/tmp/g4c/gate
HF=/tmp/models/google/$M
ASSETS=/tmp/g4c/build/$M/assets
T=/tmp/g4c/l2r/combtgt  # build_rt.sh: plowrt --features cpu from HEAD
case $M in
gemma-4-E2B-it) slug=gemma-4-e2b; ENV=GEMMA4_E2B ;;
gemma-4-E4B-it) slug=gemma-4-e4b; ENV=GEMMA4_E4B ;;
esac
RECIPE=$WT/recipes/infervisor/$slug/xeon6-amx-bf16.toml
export PYREF=/tmp/g4c/bin/vllm-py
export ${ENV}_CHECKPOINT=$HF ${ENV}_FP32_REF=$G/$M.ref.json ${ENV}_FP32_VLLM=$G/$M.vllm.json
export CARGO_TARGET_DIR=$T
SHA=$(cd $WT && git rev-parse --short=12 HEAD)
log() { echo "[$(date -u +%H:%M:%S)] $M: $*"; }
rm -rf "$R/plow.comb" "$R/gate.comb" "$R/report-vllm.comb" "$R/report-llamacpp.comb"
senv=$(python3.11 -c 'import sys,tomllib; print(" ".join(f"{k}={v}" for k,v in tomllib.load(open(sys.argv[1],"rb")).get("serve",{}).get("env",{}).items()))' "$RECIPE")
log "serve env: $senv"
env $senv ASSETS=$ASSETS PLOWRT=$T/release/plowrt PLOWRT_GIT_SHA=$SHA /tmp/g4c/grid.sh plow "$M" "$R/plow.comb" > "$R/plow.comb.log" 2>&1
grep -E "ttft|rror" "$R/plow.comb.log"
(cd $WT && python3.11 scripts/campaign/campaign.py gate "$RECIPE" --assets "$ASSETS" --out "$R/gate.comb" \
    --only llm_fp32_ref --dry-run > "$R/gate.comb-dryrun.log" 2>&1)
PLOWRT_DIR=$T bash "$R/gate.comb/run.sh" > "$R/gate.comb-run.log" 2>&1
tail -2 "$R/gate.comb/llm_fp32_ref/plow.log"
(cd $WT && python3.11 scripts/campaign/campaign.py gate "$RECIPE" --assets "$ASSETS" --out "$R/gate.comb" \
    --only llm_fp32_ref --score-only 2>&1 | tail -4)
for b in vllm llamacpp; do
    (cd $WT && python3.11 scripts/campaign/campaign.py report --baseline "$R/$b" --infervisor "$R/plow.comb" \
        --gate "$R/gate.comb/gates.json" --out "$R/report-$b.comb" > "$R/report-$b.comb.log" 2>&1)
    echo "report-$b rc=$? $(tail -1 "$R/report-$b.comb.log")"
done
log done
