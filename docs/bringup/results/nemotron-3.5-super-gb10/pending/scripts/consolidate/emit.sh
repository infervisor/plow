#!/usr/bin/env bash
# emit.sh <nemotron|qwen>: CPU-only campaign build (--no-probe) into /home/ifv/plow-scratch/consolidate/<model>.
cd /home/ifv/plow/.claude/worktrees/agent-aabd7b853cbd494b2 || exit 1
export CARGO_TARGET_DIR=/home/ifv/plow-target-consolidate
export CUDA_VISIBLE_DEVICES=
export NP_RUNTIME=bwrap
export PLOW_VERIFY_BIN=/home/ifv/.claude/jobs/72c86dac/tmp/plow-gb10-build/lean-plow/.lake/build/bin/plow_verify
export NEMOTRON35_CHECKPOINT=/home/ifv/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
export QWEN38_CHECKPOINT=/home/ifv/models/Qwen/Qwen3.8-27B
case "$1" in
  nemotron) RECIPE=scripts/campaign/recipes/nemotron-3.5-super.gb10.nvfp4.toml ;;
  qwen) RECIPE=scripts/campaign/recipes/qwen38-27b.gb10.fp8.toml ;;
  *) echo "usage: emit.sh nemotron|qwen"; exit 2 ;;
esac
OUT=/home/ifv/plow-scratch/consolidate/$1
mkdir -p /home/ifv/plow-scratch/consolidate
rm -rf "$OUT"
python3 scripts/campaign/campaign.py build "$RECIPE" --out "$OUT" --no-probe
echo "rc=$?"
