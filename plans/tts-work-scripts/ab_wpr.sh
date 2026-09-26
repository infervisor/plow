#!/usr/bin/env bash
RT=/root/tts-work/plowrt-tts6
for a in veena-ctl veena-gf3 veena-ctl; do
  CONCS="1 4 8 16" bash /root/tts-work/rung_sweep.sh $RT /root/tts-work/assets/$a $a
done
