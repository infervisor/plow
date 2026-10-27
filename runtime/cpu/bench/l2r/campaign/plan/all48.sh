#!/usr/bin/env bash
# all48.sh: as all.sh for B=4 / 8
cd /tmp/g4c/l2r/plan
mkdir -p all
for m in E2B E4B 12B 26B-A4B 31B; do
  for kib in 1344 1568; do
    for b in 4 8; do
      t=all/$m.k$kib.b$b
      /tmp/plow-target-l2r/release/plowc --hf-dir /tmp/models/google/gemma-4-$m-it stage-plan --batch $b --ctx 2048 \
        --l2-weight-kib $kib --out $t.json "$@" > $t.txt 2> $t.err || { echo "$t FAILED: $(grep -o 'error.*' $t.err | tail -1)"; continue; }
      echo "$m ${kib}KiB b$b: $(tail -1 $t.txt)"
    done
  done
done
