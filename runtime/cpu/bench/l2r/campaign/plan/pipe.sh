#!/usr/bin/env bash
# pipe.sh: --split pipe plans (whole layers or row splits on one socket each) for E2B / E4B, 12 / 14 ways, B 1 / 16
cd /tmp/g4c/l2r/plan
mkdir -p pipe
for m in E2B E4B; do
  for kib in 1344 1568; do
    for b in 1 4 8 16; do
      t=pipe/$m.k$kib.b$b
      /tmp/plow-target-l2r/release/plowc --hf-dir /tmp/models/google/gemma-4-$m-it stage-plan --batch $b --ctx 2048 \
        --l2-weight-kib $kib --split pipe --out $t.json > $t.txt 2> $t.err || { echo "$t FAILED"; continue; }
      echo "$m ${kib}KiB b$b: $(tail -1 $t.txt)"
    done
  done
done
