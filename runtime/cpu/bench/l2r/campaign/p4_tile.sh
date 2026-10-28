#!/usr/bin/env bash
# p4_tile.sh: KV tile size x path A/B on the 32K full-attention layers (2 reps), then in-loop prefetch at tile 0/64.
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p4_tile
R=/tmp/g4c/l2r/ref
for ref in e2b.L4.c32768 e4b.L5.c32768; do
  for path in B A; do
    for t in 0 16 64 256 1024; do
      for r in 1 2; do bash $S/p4_run.sh $O $ref.$path.t$t.r$r $R/$ref $path 1 L2R_KV_TILE_KIB=$t; done
    done
  done
done
for ref in e2b.L4.c32768 e4b.L5.c32768; do
  for t in 0 64; do
    for h in t0 t2 nta; do
      for pfd in 2 8 32; do
        for r in 1 2; do bash $S/p4_run.sh $O $ref.A.t$t.pf$h$pfd.r$r $R/$ref A 1 L2R_KV_TILE_KIB=$t L2R_KV_PFD=$pfd L2R_KV_PFH=$h; done
      done
    done
  done
done
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
echo P4_TILE_DONE
