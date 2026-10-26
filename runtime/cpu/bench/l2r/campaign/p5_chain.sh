#!/usr/bin/env bash
# p5_chain.sh: sequential timed work after build_rt.sh: strict E2B / E4B reruns with COMBINE, then the batch-row
# reference dumps, then the batched-stage sweep. One at a time so no run shares the cores with another.
set -u
L=/tmp/g4c/l2r
for M in gemma-4-E2B-it gemma-4-E4B-it; do bash $L/final_comb.sh $M; done
bash $L/gen_refs_p5.sh
bash $L/p5_batch.sh
bash $L/p5_stab.sh
bash $L/p5_srvctr.sh
echo P5_CHAIN_DONE
