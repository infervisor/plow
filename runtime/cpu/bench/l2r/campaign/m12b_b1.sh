#!/usr/bin/env bash
# m12b_b1.sh: 12B TP=4 slice stages at B=1 on the 90 workers, AVX / AMX, 2 reps (path B, warm KV)
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/m12b
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m
for d in 12b.L0.c2048.tp4r0 12b.L5.c2048.tp4r0 12b.L0.c2048.tp4r3 12b.L5.c2048.tp4r3; do
  for g in avx amx; do
    for rep in 1 2; do
      tag=$d.$g.b1.r$rep
      bash $S/p4_run.sh $O $tag $R/$d B 1 L2R_GEMV=$g > /dev/null
      python3 $S/p2_gate.py $O/$tag.json 2>&1 | head -1 | sed "s/^/$tag /" | cut -c1-200
    done
  done
done
echo M12B_B1_DONE
