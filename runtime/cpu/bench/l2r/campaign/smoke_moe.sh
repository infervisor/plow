#!/usr/bin/env bash
# smoke_moe.sh: 26B-A4B L0 head socket and expert groups g5 (4 picks) / g2 (1 pick), AVX / AMX, 6 housekeeping cores
cd /tmp/g4c/l2r
mkdir -p results/smoke_moe
G=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r/p2_gate.py
for d in 26b.L0.c2048.head 26b.L0.c2048.ex13g5 26b.L0.c2048.ex13g2; do
  for g in avx amx; do
    t=$d.$g
    L2R_CPUS=0,1,32,33,64,65 L2R_GEMV=$g L2R_BCAST=repnt ./l2r_layer.m2 ref/$d 10 > results/smoke_moe/$t.json 2> results/smoke_moe/$t.err
    echo "$t rc=$? $(python3 $G results/smoke_moe/$t.json 2>&1 | head -2 | tr '\n' ' ' | cut -c1-330) $(head -c 200 results/smoke_moe/$t.err)"
  done
done
