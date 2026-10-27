#!/usr/bin/env bash
# smoke_m.sh: l2r_layer.m numerics on the 6 housekeeping cores: 12B full layers (no TP) and TP=4 slices r0 / r3, AVX / AMX
cd /tmp/g4c/l2r
mkdir -p results/smoke_m
G=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r/p2_gate.py
for d in 12b.L0.c2048 12b.L5.c2048 12b.L0.c2048.tp4r0 12b.L0.c2048.tp4r3 12b.L5.c2048.tp4r0 12b.L5.c2048.tp4r3; do
  for g in avx amx; do
    t=$d.$g
    L2R_CPUS=0,1,32,33,64,65 L2R_GEMV=$g L2R_BCAST=repnt ./l2r_layer.m ref/$d 10 > results/smoke_m/$t.json 2> results/smoke_m/$t.err
    echo "$t rc=$? $(python3 $G results/smoke_m/$t.json 2>&1 | head -1 | cut -c1-150) $(head -c 200 results/smoke_m/$t.err)"
  done
done
