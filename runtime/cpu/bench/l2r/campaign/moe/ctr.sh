#!/usr/bin/env bash
# ctr.sh: DRAM CAS + L2 lines-in for the G=3 / G=2 expert groups, AVX, B=16 and B=1, rotating routing
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/m_moe23ctr; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m3
rows() { local s=$R/$1$2; for i in $(seq 1 $(( $3 - 1 ))); do s=$s,$R/$1.o$i$2; done; echo $s; }
for c in "3 1" "2 0"; do
  set -- $c
  for b in 1 16; do
    bash $S/p4_ctr.sh $O g$1.b$b $R/26bg$1.L0.c2048.ex$1g$2 B $b L2R_GEMV=avx L2R_EXPERTS_ROTATE=37 L2R_ROWS=$(rows 26bg$1.L0.c2048 .ex$1g$2 $b) | tail -1
  done
done
