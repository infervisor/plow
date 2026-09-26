#!/usr/bin/env bash
# One lease: same plowrt (tts3), control vs fast sampler cubin, control rerun.
WT=/root/plow/.claude/worktrees/tts-veena-chatterbox
R=/root/tts-work/results/ab-sampler-e2e
for arm in ctrl fast ctrl2; do
  PLOW_NV_CUBIN_SAMPLE=/root/tts-work/smp_${arm%2}.cubin PLOWRT_BIN=/root/tts-work/plowrt-tts3 \
    $WT/scripts/tts/plow_speech_probe.sh /root/tts-work/assets/veena-tts $R/$arm \
    "--conc 1 --n 8 --stream --wav" "--conc 8 --n 32 --stream --wav" "--conc 32 --n 64 --stream" 2>&1 \
    | grep '"tag"' | sed "s/^/$arm /"
done
