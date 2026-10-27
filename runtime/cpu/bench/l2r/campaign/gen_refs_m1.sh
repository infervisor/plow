#!/usr/bin/env bash
# gen_refs_m1.sh: ref_layer.py regression (E2B L4 2K vs the P2-P5 dump) + 12B TP=4 slices of L0 (sliding) and L5 (full)
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
s=$(date +%s)
/tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-E2B-it 4 2048 /tmp/g4c/l2r/refchk/e2b.L4.c2048 > /tmp/g4c/l2r/refchk/e2b.log 2>&1
echo "e2b regression rc=$? $(( $(date +%s) - s ))s $(tail -1 /tmp/g4c/l2r/refchk/e2b.log | cut -c1-200)"
for f in ref.out ref.o ref.down ref.attn w.self_attn.q_proj.weight kcache x_in per_layer_input; do
  for e in f32 bf16; do
    [ -f $R/e2b.L4.c2048/$f.$e ] && { cmp -s $R/e2b.L4.c2048/$f.$e /tmp/g4c/l2r/refchk/e2b.L4.c2048/$f.$e && echo "  $f same" || echo "  $f DIFF"; }
  done
done
for L in 0 5; do
  s=$(date +%s)
  /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-12B-it $L 2048 $R/12b.L$L.c2048 8192 0 4 > $R/12b.L$L.c2048.log 2>&1
  echo "12b L$L tp4 rc=$? $(( $(date +%s) - s ))s $(tail -1 $R/12b.L$L.c2048.log | cut -c1-220)"
done
echo REFS_M1_DONE
