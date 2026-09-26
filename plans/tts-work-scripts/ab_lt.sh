#!/usr/bin/env bash
# One lease: interp decode (veena-tts) vs cuBLASLt decode (veena-lt), then interp again.
RT=/root/tts-work/plowrt-tts4
for a in veena-tts veena-lt veena-tts; do
  CONCS="1 4 8 16" bash /root/tts-work/rung_sweep.sh $RT /root/tts-work/assets/$a $a
done
