#!/usr/bin/env bash
for a in veena-v4 veena-v5m16 veena-v4; do
  CONCS="1 8 16 32" bash /root/tts-work/rung_sweep.sh /root/tts-work/plowrt-tts6 /root/tts-work/assets/$a $a
done
