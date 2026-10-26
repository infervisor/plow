#!/usr/bin/env bash
# p4_tile2.sh: fill-in of the selection: tile 256 with T0/T2 prefetch on path A, and prefetch on path B (2 reps).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p4_tile
R=/tmp/g4c/l2r/ref
for ref in e2b.L4.c32768 e4b.L5.c32768; do
  for h in t0 t2; do
    for pfd in 2 8; do
      for r in 1 2; do
        bash $S/p4_run.sh $O $ref.A.t256.pf$h$pfd.r$r $R/$ref A 1 L2R_KV_TILE_KIB=256 L2R_KV_PFD=$pfd L2R_KV_PFH=$h
        bash $S/p4_run.sh $O $ref.B.t256.pf$h$pfd.r$r $R/$ref B 1 L2R_KV_TILE_KIB=256 L2R_KV_PFD=$pfd L2R_KV_PFH=$h
        bash $S/p4_run.sh $O $ref.B.t0.pf$h$pfd.r$r $R/$ref B 1 L2R_KV_TILE_KIB=0 L2R_KV_PFD=$pfd L2R_KV_PFH=$h
      done
    done
  done
done
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
echo P4_TILE2_DONE
