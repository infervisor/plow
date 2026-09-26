#!/usr/bin/env bash
export GPU_LEASE_TIMEOUT=14400 HF_HOME=/root/tts-work/hf
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
bash /root/tts-work/run_probe.sh t3-ref /root/tts-work/venv-ref/bin/python scripts/tts/t3_ref.py /root/tts-work/results/t3_ref.json \
  > /root/tts-work/results/t3_ref.log 2>&1
echo DONE >> /root/tts-work/results/t3_ref.log
