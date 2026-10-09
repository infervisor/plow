#!/usr/bin/env bash
# gen_refs_m2.sh: 12B TP=4 socket slices of L0 (sliding) and L5 (full, k_eq_v) at 2K
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
for L in 0 5; do
  s=$(date +%s)
  /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-12B-it $L 2048 $R/12b.L$L.c2048 8192 0 4 > $R/12b.L$L.c2048.tp4.log 2>&1
  echo "12b L$L tp4 rc=$? $(( $(date +%s) - s ))s $(tail -1 $R/12b.L$L.c2048.tp4.log | cut -c1-220)"
done
ls -d $R/12b.*
echo REFS_M2_DONE
