#!/usr/bin/env bash
# m_sweep.sh: multi-model stage sweep on the 90 workers (l2r_layer.m2), 2 reps, warm KV (path B), repnt:
#   12B TP=4 rank 0 and 31B TP=8 rank 0: L0 (sliding) / L5 (full) at 2K, L5 at 16K (P4 KV policy)
#   26B-A4B: L0 / L5 head socket and the expert group with the most picks of row 0
#   B 1 / 4 / 8 / 16 with AMX, B 1 / 16 with AVX; rows = distinct sequences (o1..o15)
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/m_sweep; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.m2
rows() { # <base> <suffix> <n>: row dirs base[.o<r>]suffix
  local s=$R/$1$2
  for i in $(seq 1 $(( $3 - 1 ))); do s=$s,$R/$1.o$i$2; done
  echo $s
}
maxg() { python3 -c "
import json, glob
b = max(glob.glob('$R/$1.ex13g*/meta.json'), key=lambda f: json.load(open(f))['pairs'])
print(b.split('/')[-2].split('.')[-1])"; }
run() { # <base> <suffix> <tag-stage> <kvenv>
  local base=$1 suf=$2 st=$3 kv=$4
  for g in amx avx; do
    for b in 1 4 8 16; do
      [ $g = avx ] && [ $b != 1 ] && [ $b != 16 ] && continue
      [ ! -d $R/$base.o$(( b > 1 ? b - 1 : 0 ))$suf ] && [ $b -gt 1 ] && continue
      for rep in 1 2; do
        tag=$st.$g.b$b.r$rep
        bash $S/p4_run.sh $O $tag $R/$base$suf B $b L2R_GEMV=$g L2R_ROWS=$(rows $base $suf $b) $kv > /dev/null
        python3 $S/p2_gate.py $O/$tag.json 2>&1 | head -1 | sed "s/^/$tag /" | cut -c1-170
      done
    done
  done
}
KV16="L2R_KV_TILE_KIB=256 L2R_KV_PFD=8"
run 12b.L0.c2048 .tp4r0 12b.L0.c2048.tp4r0 ""
run 12b.L5.c2048 .tp4r0 12b.L5.c2048.tp4r0 ""
run 12b.L5.c16384 .tp4r0 12b.L5.c16384.tp4r0 "$KV16"
run 31b.L0.c2048 .tp8r0 31b.L0.c2048.tp8r0 ""
run 31b.L5.c2048 .tp8r0 31b.L5.c2048.tp8r0 ""
run 31b.L5.c16384 .tp8r0 31b.L5.c16384.tp8r0 "$KV16"
for L in 0 5; do
  run 26b.L$L.c2048 .head 26b.L$L.c2048.head ""
  g=$(maxg 26b.L$L.c2048); echo "26b L$L expert group of record: $g"
  run 26b.L$L.c2048 .$g 26b.L$L.c2048.$g ""
done
run 26b.L5.c16384 .head 26b.L5.c16384.head "$KV16"
echo M_SWEEP_DONE
