#!/usr/bin/env bash
# p2_split.sh: E4B L2/L3 split. Resident budget per worker (KiB, 0 = plain LRU), no all-gather, L2 lines in + step time.
for ref in e4b.L0.c2048 e4b.L5.c2048; do
  for kib in 0 768 1024 1280 1536; do
    L2R_ENV="L2R_RESIDENT_KIB=$kib" bash /tmp/g4c/l2r/p2_ctr.sh /tmp/g4c/l2r/ref/$ref 10000 split.$ref.k$kib
    python3 /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r/p2_gate.py /tmp/g4c/l2r/results/p2_ctr/split.$ref.k$kib.json | head -1
  done
done
