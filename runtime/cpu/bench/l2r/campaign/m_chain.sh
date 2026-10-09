#!/usr/bin/env bash
# m_chain.sh: after gen_refs_m3 -> 26B dumps (L0 / L5 at 2K, rows o1..o15; L5 16K row 0) -> multi-model timing sweep
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
until grep -q REFS_M3_DONE /tmp/g4c/l2r/gen_refs_m3.log; do sleep 20; done
R=/tmp/g4c/l2r/ref
one() { # layer ctx row
  local d=$R/26b.L$1.c$2; [ $3 -gt 0 ] && d=$d.o$3
  [ -f $d.head/meta.json ] && return
  local s=$(date +%s)
  /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-26B-A4B-it $1 $2 $d 8192 $(($3 * 997)) moe:13 > $d.log 2>&1
  echo "26b L$1 c$2 o$3 rc=$? $(( $(date +%s) - s ))s $(tail -1 $d.log | cut -c1-200)"
}
for L in 0 5; do for r in $(seq 0 15); do one $L 2048 $r; done; done
one 5 16384 0
echo REFS_M4_DONE
bash /tmp/g4c/l2r/m_sweep.sh
