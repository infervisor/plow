#!/usr/bin/env bash
# One lease: speech bench with PLOW_MULTISTEP=8 vs 4 vs 8 on veena-v5m16.
WT=/root/plow/.claude/worktrees/tts-veena-chatterbox
R=/root/tts-work/results/ab-ms
for k in 8 4 8b; do
  PLOW_MULTISTEP=${k%b} PLOWRT_BIN=/root/tts-work/plowrt-tts6 $WT/scripts/tts/plow_speech_probe.sh \
    /root/tts-work/assets/veena-v5m16 $R/k$k "--conc 1 --n 8 --stream" "--conc 8 --n 32 --stream" "--conc 32 --n 64 --stream" 2>&1 \
    | grep '"tag"' | sed "s/^/K$k /"
done
