#!/usr/bin/env bash
# gen_refs_m3.sh: batch rows and long context for the TP slices.
#   12B TP=4: L0 / L5 at 2K rows o1..o15 (rank 0); L5 at 16K rows o0..o15 (o0 every rank, others rank 0)
#   31B TP=8: L0 / L5 at 2K o0 every rank, o1..o15 rank 0; L5 at 16K o0..o15 (o0 every rank)
# Row r: text offset r * 997 tokens (BOS kept), as P5.
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
one() { # model tag layer ctx row tp
  local d=$R/$2.L$3.c$4; [ $5 -gt 0 ] && d=$d.o$5
  [ -f $d.tp${6%%:*}r0/meta.json ] && return
  local s=$(date +%s)
  /tmp/g4c/bin/vllm-py ref_layer.py /tmp/models/google/gemma-4-$1-it $3 $4 $d 8192 $(($5 * 997)) $6 > $d.log 2>&1
  echo "$2 L$3 c$4 o$5 tp $6 rc=$? $(( $(date +%s) - s ))s $(tail -1 $d.log | cut -c1-170)"
}
for L in 0 5; do for r in $(seq 1 15); do one 12B 12b $L 2048 $r 4:0; done; done
one 12B 12b 5 16384 0 4
for r in $(seq 1 15); do one 12B 12b 5 16384 $r 4:0; done
for L in 0 5; do one 31B 31b $L 2048 0 8; for r in $(seq 1 15); do one 31B 31b $L 2048 $r 8:0; done; done
one 31B 31b 5 16384 0 8
for r in $(seq 1 15); do one 31B 31b 5 16384 $r 8:0; done
echo REFS_M3_DONE
