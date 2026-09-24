#!/usr/bin/env bash
# Sweep the grouped-MoE repro over routing distributions and prefill bucket widths.
# dist 3 additionally runs the real router (op 73) over NaN-filled padding rows,
# which is what a 39-token prompt in a 128-row bucket actually presents.
set -uo pipefail
B=/home/lava/.claude/jobs/ef9d0e7f/tmp/moerepro
for T in 128 512 1024 4096; do
  for d in 0 1 2 3; do
    out=$("$B" "$T" 132 "$d" 2>&1)
    if echo "$out" | grep -q "ALL OK"; then
      line=$(echo "$out" | grep "align: total_tiles")
      echo "T=$T dist=$d  OK   $line"
    else
      echo "T=$T dist=$d  *** FAIL ***"
      echo "$out" | tail -5 | sed 's/^/      /'
    fi
  done
done
