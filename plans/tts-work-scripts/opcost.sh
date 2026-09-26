#!/usr/bin/env bash
# opcost.sh <assets> <slots> — marginal wall time per op of layer 14 via instruction caps.
A=$1; S=$2
L0=$((2 + 11 * 14))
names=(start qkv hnr_q hnr_k hnr_v flash merge o_proj addnorm1 glu down addnorm2)
prev=""
for k in 0 1 2 3 4 5 6 7 8 9 10 11; do
  cap=$((L0 + k))
  m=$(PLOW_DEBUG_MAX_INST=$cap /root/tts-work/step_bench $A $S 512 48 2>/dev/null | grep -o "mean_ms=[0-9.]*" | cut -d= -f2)
  if [ -n "$prev" ]; then
    echo "slots=$S op=${names[$k]} cap=$cap mean_ms=$m delta_us=$(echo "($m - $prev) * 1000" | bc -l | cut -c1-7)"
  else
    echo "slots=$S layer14-start cap=$cap mean_ms=$m"
  fi
  prev=$m
done
full=$(/root/tts-work/step_bench $A $S 512 48 2>/dev/null | grep -o "mean_ms=[0-9.]*" | cut -d= -f2)
echo "slots=$S full_step_ms=$full"
