#!/usr/bin/env bash
# p3_modes.sh <ref> [steps]: broadcast x barrier modes on one stage dump (AVX), gate + step time + phase breakdown.
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p3; mkdir -p $O
ref=$1 n=${2:-5000} t=$(basename $1)
for bc in direct rep repcld; do
  for br in diss hier; do
    L2R_BCAST=$bc L2R_BARRIER=$br /tmp/g4c/l2r/l2r_layer $ref $n > $O/$t.$bc.$br.json 2> $O/$t.$bc.$br.err
    echo "$bc $br: $(python3 $S/p2_gate.py $O/$t.$bc.$br.json | head -1 | cut -d' ' -f1,5-9) | $(python3 -c "
import json; s=json.load(open('$O/$t.$bc.$br.json')); print(' '.join('%.1f/%.1f' % (p['compute_max'], p['barrier_mean']) for p in s['phase_us']))")"
  done
done
