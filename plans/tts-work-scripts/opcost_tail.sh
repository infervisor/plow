#!/usr/bin/env bash
# opcost_tail.sh <assets> <slots> — cost of the decode tail (final norm, lm head, argmax).
A=$1; S=$2
for cap in ${CAPS:-2 13 309 310 311 312 313}; do
  m=$(PLOW_DEBUG_MAX_INST=$cap /root/tts-work/step_bench $A $S 512 48 2>/dev/null | grep -o "mean_ms=[0-9.]*" | cut -d= -f2)
  echo "slots=$S cap=$cap mean_ms=$m"
done
