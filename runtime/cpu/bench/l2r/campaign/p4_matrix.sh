#!/usr/bin/env bash
# p4_matrix.sh <tile_kib> <path-C env...>: context x concurrency x path A/B/C, 3 reps, full and sliding layers.
# Path C = path A plus the selected prefetch settings (passed as ENV=VAL args).
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
O=/tmp/g4c/l2r/results/p4_matrix
R=/tmp/g4c/l2r/ref
T=$1; shift
CENV=("$@")
kvmib() { python3 -c "
import json; m = json.load(open('$R/$1/meta.json'))
print(round(2 * m['kv_heads'] * (m['cache_len'] + 1) * m['head_dim'] * 2 * $2 / 2**20))"; }
cell() { # ref seqs
  local ref=$1 s=$2 kv; kv=$(kvmib $ref $s)
  for r in 1 2 3; do
    bash $S/p4_run.sh $O $ref.s$s.B.r$r $R/$ref B $s L2R_KV_TILE_KIB=$T
    if [ $kv -lt 1536 ]; then
      bash $S/p4_run.sh $O $ref.s$s.A.r$r $R/$ref A $s L2R_KV_TILE_KIB=$T
      bash $S/p4_run.sh $O $ref.s$s.C.r$r $R/$ref A $s L2R_KV_TILE_KIB=$T "${CENV[@]}"
    else
      bash $S/p4_run.sh $O $ref.s$s.C.r$r $R/$ref B $s L2R_KV_TILE_KIB=$T "${CENV[@]}"
    fi
  done
}
for m in e2b.L4 e4b.L5; do
  for c in 2048 8192 16384 32768 65536 131072; do
    for s in 1 4 16; do cell $m.c$c $s; done
  done
done
for ref in e2b.L0.c2048 e2b.L0.c131072 e4b.L0.c2048 e4b.L0.c131072; do
  for s in 1 4 16; do cell $ref $s; done
done
python3 $S/p2_gate.py $O/*.json > $O/gate.txt; echo "gate rc=$?"
python3 $S/p4_sum.py $O > $O/summary.md
echo P4_MATRIX_DONE
