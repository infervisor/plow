#!/usr/bin/env bash
# One lease, three concurrency points, vLLM Veena baseline (greedy, same prompts).
set -u
export HF_HOME=/root/tts-work/hf VLLM_USE_FLASHINFER_SAMPLER=0
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
PY=/root/tts-work/venv-vllm/bin/python
for c in 1 8 32; do
  n=$(( c < 8 ? 8 : c * 2 ))
  $PY scripts/tts/veena_ref.py --engine vllm --conc $c --n $n --greedy --gpu-mem 0.5 \
     --out /root/tts-work/results/veena 2>&1 | grep -v "Warning\|INFO\|it/s" | tail -n 14
done
