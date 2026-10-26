#!/usr/bin/env bash
# p3_run.sh <tag> <ref> <steps> [ENV=VAL ...]: one l2r_layer run, gate line + per-phase compute/prologue/barrier.
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p3; mkdir -p $O
tag=$1 ref=$2 n=$3; shift 3
env "$@" /tmp/g4c/l2r/l2r_layer $ref $n > $O/$tag.json 2> $O/$tag.err
echo "$tag: $(python3 $S/p2_gate.py $O/$tag.json | head -1 | cut -d' ' -f1,5-9)"
python3 -c "
import json; s=json.load(open('$O/$tag.json')); print('   ', ' '.join('%d:%.1f/%.1f/b%.1f' % (i,p['compute_max'],p['prologue_mean'],p['barrier_mean']) for i,p in enumerate(s['phase_us'])))"
