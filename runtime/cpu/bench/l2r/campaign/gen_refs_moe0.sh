#!/usr/bin/env bash
# gen_refs_moe0.sh: 26B-A4B L0 (sliding) at 2K: head socket + 13 expert groups
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
s=$(date +%s)
/tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-26B-A4B-it 0 2048 $R/26b.L0.c2048 8192 0 moe:13 > $R/26b.L0.c2048.log 2>&1
echo "26b L0 moe:13 rc=$? $(( $(date +%s) - s ))s $(tail -1 $R/26b.L0.c2048.log | cut -c1-300)"
