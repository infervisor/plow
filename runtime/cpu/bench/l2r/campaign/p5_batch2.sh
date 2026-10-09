#!/usr/bin/env bash
# p5_batch2.sh: balanced AMX partition (4-row units, 16-row tiles + AVX tail; l2r_layer.c4) rerun of the AMX batch sweep
# and the 12-way residency matrix: E2B L0 2K / L4 2K / L4 16K / E4B L5 2K, B 1/2/4/8/16, 2 reps; B=16 rep vs repnt;
# residency E2B L0 / L4 2K / 16K, B 1 / 16, plain vs lock 12 under counters. Expects pseudo_lock_sram at 12 ways.
set -u
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/p5_batch2; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.c4
grep -q "1376256 B/core" /sys/class/misc/pseudo_lock/caps || { echo "driver not at 12 ways"; exit 1; }
rows() { local d=$R/$1 r=$1; for i in $(seq 1 15); do d=$d,$R/$r.o$i; done; echo $d; }
kvp() { case $1 in *c16384*) echo "L2R_KV_TILE_KIB=256 L2R_KV_PFD=8";; esac; }
for ref in e2b.L0.c2048 e2b.L4.c2048 e2b.L4.c16384 e4b.L5.c2048; do
  for b in 1 2 4 8 16; do
    for rep in 1 2; do
      tag=$ref.amx.b$b.r$rep
      bash $S/p4_run.sh $O $tag $R/$ref B $b L2R_GEMV=amx L2R_ROWS=$(rows $ref) $(kvp $ref) > /dev/null
      python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
    done
  done
  for rep in 1 2; do
    tag=$ref.amx.b16.rep.r$rep
    bash $S/p4_run.sh $O $tag $R/$ref B 16 L2R_GEMV=amx L2R_BCAST=rep L2R_ROWS=$(rows $ref) $(kvp $ref) > /dev/null
    python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
  done
done
echo P5_SWEEP2_DONE
for ref in e2b.L0.c2048 e2b.L4.c2048 e2b.L4.c16384; do
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
echo P5_BATCH2_DONE
