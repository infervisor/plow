#!/usr/bin/env bash
# ab_steps.sh <slots> <asset...> — raw engine step (step_bench, ctx 512, 64 steps) per asset, ABA order.
S=$1; shift
for a in "$@"; do
  m=$(/root/tts-work/step_bench /root/tts-work/assets/$a $S 512 64 2>/dev/null | grep -o "mean_ms=[0-9.]*")
  echo "slots=$S $a $m"
done
