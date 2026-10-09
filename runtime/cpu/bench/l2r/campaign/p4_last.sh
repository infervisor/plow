#!/usr/bin/env bash
# p4_last.sh: attention alone vs in the layer (path C), and a KV footprint sweep (alternating copies) around L3.
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/p4_attn
for ref in e2b.L4.c32768 e2b.L4.c131072 e4b.L5.c32768 e4b.L5.c131072; do
  for r in 1 2 3; do
    bash $S/p4_run.sh $O $ref.C.layer.r$r $R/$ref A 1 L2R_KV_TILE_KIB=256 L2R_KV_PFD=8 L2R_KV_PFH=t0
    bash $S/p4_run.sh $O $ref.C.attnonly.r$r $R/$ref A 1 L2R_KV_TILE_KIB=256 L2R_KV_PFD=8 L2R_KV_PFH=t0 L2R_ATTN_ONLY=1
  done
done
python3 $S/p4_sum.py $O > $O/summary.md
O=/tmp/g4c/l2r/results/p4_foot
for c in 1 2 3 4 6 8 12; do
  for r in 1 2 3; do
    bash $S/p4_run.sh $O e4b.L5.c32768.k$c.B.r$r $R/e4b.L5.c32768 B 1 L2R_KV_TILE_KIB=256 L2R_KV_COPIES=$c
    bash $S/p4_run.sh $O e4b.L5.c32768.k$c.C.r$r $R/e4b.L5.c32768 B 1 L2R_KV_TILE_KIB=256 L2R_KV_COPIES=$c L2R_KV_PFD=8 L2R_KV_PFH=t0
  done
done
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
echo P4_LAST_DONE
