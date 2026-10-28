#!/usr/bin/env bash
# run.sh [extra stage-plan args]: E2B / E4B stage plans at batch 1 and 16, ctx 2048 and 16384
cd /tmp/g4c/l2r/plan
for m in E2B E4B; do
  for b in 1 16; do
    for c in 2048 16384; do
      echo "== $m b$b ctx $c"
      /tmp/plow-target-l2r/release/plowc --hf-dir /tmp/models/google/gemma-4-$m-it stage-plan --batch $b --ctx $c \
        --out $m.b$b.c$c.json "$@" 2>/dev/null > $m.b$b.c$c.txt
      head -4 $m.b$b.c$c.txt; tail -3 $m.b$b.c$c.txt
    done
  done
done
