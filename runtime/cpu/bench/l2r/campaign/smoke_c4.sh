#!/usr/bin/env bash
# smoke_c4.sh: balanced-AMX binary numerics on the 6 housekeeping cores (B 1 / 4, AMX / AVX, E2B L4 2K, rows o0-o3)
cd /tmp/g4c/l2r
R=ref/e2b.L4.c2048
rows=$R; for i in 1 2 3; do rows=$rows,$R.o$i; done
for g in amx avx; do
  for b in 1 4; do
    L2R_CPUS=0,1,32,33,64,65 L2R_GEMV=$g L2R_BCAST=repnt L2R_BATCH=$b L2R_ROWS=$rows ./l2r_layer.c4 $R 20 \
      > results/smoke_b/c4.$g.b$b.json 2> results/smoke_b/c4.$g.b$b.err
    python3 /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r/p2_gate.py results/smoke_b/c4.$g.b$b.json | head -1 | cut -c1-70
  done
done
head -3 results/smoke_b/c4.amx.b4.err
