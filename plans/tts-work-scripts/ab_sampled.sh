#!/usr/bin/env bash
# A/B (one lease): per-token sampled (tts1) vs sampled multi-step (tts2), control rerun (tts1b).
WT=/root/plow/.claude/worktrees/tts-veena-chatterbox
R=/root/tts-work/results/ab-sampled
for arm in tts1 tts2 tts1b; do
  PLOWRT_BIN=/root/tts-work/plowrt-${arm%b} $WT/scripts/tts/plow_speech_probe.sh /root/tts-work/assets/veena-tts $R/$arm \
    "--conc 1 --n 8 --stream --wav" "--conc 8 --n 32 --stream --wav" 2>&1 | grep '"tag"' | sed "s/^/$arm /"
done
