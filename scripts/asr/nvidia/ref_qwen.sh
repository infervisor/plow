#!/usr/bin/env bash
# Qwen3-ASR vLLM baseline. Run under gpulease.
set -u
VENV=${VENV:-/root/asr-work/venv}
export CUDA_HOME=${CUDA_HOME:-/root/asr-work/cuda_home}
export PATH=$CUDA_HOME/bin:$VENV/bin:$PATH
MODEL=${MODEL:-/root/plow/models/Qwen3-ASR-1.7B}
OUT=${OUT:-/root/asr-work/results}
HERE=$(dirname "$0")
"$VENV/bin/python" "$HERE/ref_bench.py" qwen --model "$MODEL" --out "$OUT/qwen_vllm.json" \
  --profile "$OUT/qwen_vllm_prof" "$@" > "$OUT/qwen_vllm.log" 2>&1
echo "qwen rc=$?"
