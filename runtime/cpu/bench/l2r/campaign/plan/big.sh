#!/usr/bin/env bash
# big.sh: stage plans for 12B / 26B-A4B / 31B at B=1 ctx 2048 (rc and stderr shown)
cd /tmp/g4c/l2r/plan
for m in 12B 26B-A4B 31B; do
  echo "== $m"
  /tmp/plow-target-l2r/release/plowc --hf-dir /tmp/models/google/gemma-4-$m-it stage-plan --batch 1 --ctx 2048 \
    --out $m.b1.c2048.json > $m.b1.c2048.txt 2> $m.err
  echo "rc=$?"
  head -3 $m.b1.c2048.txt; tail -1 $m.b1.c2048.txt; tail -3 $m.err
done
ls /home/ec2-user/plow/plans/ 2>/dev/null | grep -iE "l2r|xeon|groq|sram|l2"
