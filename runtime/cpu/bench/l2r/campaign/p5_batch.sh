#!/usr/bin/env bash
# p5_batch.sh: batched stage sweep. B 1/2/4/8/16 x AMX/AVX x {E2B L0, L4} at 2K / 16K + E4B L5 2K, path B (warm KV),
# distinct per-row dumps (L2R_ROWS), P3 sync (repnt + dissemination), P4 KV policy at 16K (256 KiB tiles, T0 PFD 8),
# 2 reps, gate every run. Then B=16 AMX residency: plain vs L2 lock (12 ways, 1344 KiB) under counters.
set -u
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/p5_batch; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.b
rows() { local d=$R/$1 r=$1; for i in $(seq 1 15); do d=$d,$R/$r.o$i; done; echo $d; }
kvp() { case $1 in *c16384*) echo "L2R_KV_TILE_KIB=256 L2R_KV_PFD=8";; esac; }
for ref in e2b.L0.c2048 e2b.L4.c2048 e2b.L4.c16384 e2b.L0.c16384 e4b.L5.c2048; do
  for g in amx avx; do
    for b in 1 2 4 8 16; do
      for rep in 1 2; do
        tag=$ref.$g.b$b.r$rep
        bash $S/p4_run.sh $O $tag $R/$ref B $b L2R_GEMV=$g L2R_ROWS=$(rows $ref) $(kvp $ref) > /dev/null
        python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
      done
    done
  done
done
for ref in e2b.L0.c2048 e2b.L4.c16384; do
  for bc in rep repcld repnt; do
    for rep in 1 2; do
      tag=$ref.amx.b16.$bc.r$rep
      bash $S/p4_run.sh $O $tag $R/$ref B 16 L2R_GEMV=amx L2R_BCAST=$bc L2R_ROWS=$(rows $ref) $(kvp $ref) > /dev/null
      python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
    done
  done
done
echo P5_SWEEP_DONE
for ref in e2b.L0.c2048 e2b.L4.c2048 e2b.L4.c16384 e2b.L0.c16384; do
  for arm in plain lock; do
    ex=""; [ $arm = lock ] && ex="L2R_RESIDENT_KIB=1344 L2R_LOCK=1"
    for b in 1 16; do
      tag=$ref.amx.b$b.$arm
      bash $S/p4_ctr.sh $O $tag $R/$ref B $b L2R_GEMV=amx L2R_ROWS=$(rows $ref) $(kvp $ref) $ex | tail -1
      python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
      grep -o '"lock":{[^}]*}' $O/$tag.json
    done
  done
done
echo P5_BATCH_DONE
