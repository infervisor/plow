#!/usr/bin/env bash
# p3_final.sh: sync/broadcast modes on three stages x 3 reps, plus one long fid run (stale epochs / tail stalls).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p3v2; mkdir -p $O
for ref in e2b.L0.c2048 e2b.L4.c16384 e4b.L0.c2048; do
  for mode in direct.diss rep.diss repnt.diss repnt.hier fid.diss nobcast.diss; do
    bc=${mode%.*} br=${mode#*.}
    for rep in 1 2 3; do
      if [ $bc = nobcast ]; then envs="L2R_NOBCAST=1"; else envs="L2R_BCAST=$bc"; fi
      env $envs L2R_BARRIER=$br /tmp/g4c/l2r/l2r_layer /tmp/g4c/l2r/ref/$ref 5000 > $O/$ref.$mode.r$rep.json 2> $O/$ref.$mode.r$rep.err
    done
  done
done
L2R_BCAST=fid /tmp/g4c/l2r/l2r_layer /tmp/g4c/l2r/ref/e2b.L0.c2048 200000 > $O/long.e2b.L0.fid.json 2> $O/long.err
L2R_BCAST=repnt /tmp/g4c/l2r/l2r_layer /tmp/g4c/l2r/ref/e2b.L0.c2048 200000 > $O/long.e2b.L0.repnt.json 2>> $O/long.err
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "gate rc=$?"
echo P3_DONE
