#!/usr/bin/env bash
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
for sc in 2.0 1.2 0.8; do
  /root/tts-work/venv-ref/bin/python scripts/tts/sample_kernel_bench.py /root/tts-work/smp_ctrl.cubin \
    /root/tts-work/smp_fast.cubin --iters 30 --scale $sc 2>&1 | grep -E "B=  1|support|TVD" | sed "s/^/scale=$sc /"
done
