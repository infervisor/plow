#!/usr/bin/env bash
# p2_final.sh: final P2 sweep (vectorised softmax) + 128K timing rows (KV tiled 8x from the 16K dumps; numerics unchecked).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p2v2
$S/p2_sweep.sh /tmp/g4c/l2r/ref $O 5000
mkdir -p $O/c128k
for ref in e2b.L4.c16384 e4b.L5.c16384; do
  for nb in 0 1; do
    for rep in 1 2 3; do
      L2R_CTX_REPEAT=8 L2R_NOBCAST=$nb /tmp/g4c/l2r/l2r_layer /tmp/g4c/l2r/ref/$ref 2000 > $O/c128k/${ref%.c16384}.c131072.avx.nb$nb.r$rep.json 2> $O/c128k/${ref%.c16384}.c131072.avx.nb$nb.r$rep.err
    done
  done
done
echo P2_FINAL_DONE
