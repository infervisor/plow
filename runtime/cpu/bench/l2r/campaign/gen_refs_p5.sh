#!/usr/bin/env bash
# gen_refs_p5.sh: batch rows 1..15 for the P5 batched stage: same layer and context as the row-0 dump, text offset
# r * 997 tokens (BOS kept), so every row has its own hidden state, KV cache and FP32 reference.
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
for spec in "E2B 0 2048" "E2B 4 2048" "E2B 4 16384" "E4B 5 2048" "E2B 0 16384"; do
  set -- $spec
  m=$(echo $1 | tr A-Z a-z)
  for r in $(seq 0 15); do
    d=$R/$m.L$2.c$3; [ $r -gt 0 ] && d=$d.o$r
    [ -f $d/meta.json ] && continue
    s=$(date +%s)
    /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-$1-it $2 $3 $d 8192 $((r * 997)) > $d.log 2>&1
    echo "$spec o$r rc=$? $(( $(date +%s) - s ))s $(tail -1 $d.log | cut -c1-160)"
  done
done
echo REFS_P5_DONE
