#!/usr/bin/env bash
# p1_gap.sh: is the C/D weight-phase slowdown KV traffic still in flight? plain 1.25 MiB AMX, 10 s, gap 0/1/3/10 µs.
O=/tmp/g4c/l2r/results/p1_gap; mkdir -p $O
for sc in C D; do
  for g in 0 1000 3000 10000; do
    L2R_GAP_NS=$g L2R_LOCK=0 L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock 1310720 10 $sc > $O/$sc.gap$g.jsonl 2> $O/$sc.gap$g.err
    echo "$sc gap=$g $(grep -o '"per_core_mean":[0-9.]*,"min_gbs":[0-9.]*' $O/$sc.gap$g.jsonl) $(grep -o '"worst_p99_us":[0-9.]*' $O/$sc.gap$g.jsonl)"
  done
done
