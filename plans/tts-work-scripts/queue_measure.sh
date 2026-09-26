#!/usr/bin/env bash
# Queued GPU measurements; each waits for its lease (4 h timeout).
export GPU_LEASE_TIMEOUT=14400 VLLM_USE_FLASHINFER_SAMPLER=0
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
bash /root/tts-work/run_probe.sh vllm-rungs /root/tts-work/venv-vllm/bin/python scripts/tts/vllm_rungs.py \
  --model /root/tts-work/hf/hub/models--maya-research--Veena/snapshots/8b770f9e69e6b35ef320d4cd70a99a4ab6dd022f \
  > /root/tts-work/results/vllm_rungs.log 2>&1
bash /root/tts-work/run_probe.sh veena-trace /root/tts-work/step_bench /root/tts-work/assets/veena-trace 1 512 64 \
  > /root/tts-work/results/veena_trace_b1.log 2>&1
bash /root/tts-work/run_probe.sh veena-trace16 /root/tts-work/step_bench /root/tts-work/assets/veena-trace 16 512 64 \
  > /root/tts-work/results/veena_trace_b16.log 2>&1
echo QUEUE_DONE >> /root/tts-work/results/veena_trace_b16.log
