#!/usr/bin/env bash
# p4_chain.sh: early staging test, the P4 matrix, 12/14-way lock, AMX GEMV under KV (selected: tile 256, T0 pfd 8).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
cp /tmp/g4c/l2r/l2r_layer.p4b /tmp/g4c/l2r/l2r_layer
O=/tmp/g4c/l2r/results/p4_early
for ref in e2b.L4.c32768 e4b.L5.c32768; do
  for e in 256 1024 4096; do
    for h in t1 t2; do
      for r in 1 2; do
        bash $S/p4_run.sh $O $ref.A.t256.early$h$e.r$r $R/$ref A 1 L2R_KV_TILE_KIB=256 L2R_KV_EARLY_KIB=$e L2R_KV_PFH=$h
      done
    done
  done
done
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "early gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
O=/tmp/g4c/l2r/results/p4_amx
for ref in e2b.L4.c2048 e2b.L4.c32768 e2b.L4.c131072; do
  for g in avx amx; do
    for r in 1 2 3; do
      bash $S/p4_run.sh $O $ref.A.$g.r$r $R/$ref A 1 L2R_KV_TILE_KIB=256 L2R_KV_PFD=8 L2R_KV_PFH=t0 L2R_GEMV=$g
    done
  done
done
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "amx gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
bash /tmp/g4c/l2r/p4_matrix.sh 256 L2R_KV_PFD=8 L2R_KV_PFH=t0
bash /tmp/g4c/l2r/p4_lock.sh 256 L2R_KV_PFD=8 L2R_KV_PFH=t0
echo P4_CHAIN_DONE
