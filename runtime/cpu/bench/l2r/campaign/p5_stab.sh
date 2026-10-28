#!/usr/bin/env bash
# p5_stab.sh: (1) final-table E4B stage rows: E4B L5 at 2K / 16K / 128K, batch 1, AVX GEMV, L2 lock 12 ways
#   (1344 KiB resident, FFN rest streamed), P3 sync, P4 KV policy at >= 16K, under DRAM CAS + L2 lines-in counters,
#   2 reps, gated. (2) stability: 30 consecutive E2B L4 16K batch-16 AMX lock-12 runs of 20000 steps, each gated and
#   counted, turbostat (frequency, package / DRAM power) sampled every 10 s across all of them.
set -u
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/p5_stab; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.b
KVP="L2R_KV_TILE_KIB=256 L2R_KV_PFD=8"
for c in 2048 16384 131072; do
  ex=""; [ $c -gt 2048 ] && ex=$KVP
  for rep in 1 2; do
    tag=e4b.L5.c$c.avx.lock12.r$rep
    bash $S/p4_ctr.sh $O $tag $R/e4b.L5.c$c B 1 L2R_GEMV=avx L2R_RESIDENT_KIB=1344 L2R_LOCK=1 $ex | tail -1
    python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
    grep -o '"lock":{[^}]*}' $O/$tag.json
  done
done
echo P5_E4B_DONE
rows=$R/e2b.L4.c16384; for i in $(seq 1 15); do rows=$rows,$R/e2b.L4.c16384.o$i; done
sudo turbostat --quiet --interval 10 --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgWatt,RAMWatt -o $O/stab.turbostat.txt &
TS=$!
t0=$(date +%s)
for i in $(seq -w 1 30); do
  tag=stab.e2b.L4.c16384.amx.b16.lock12.$i
  P4_STEPS=20000 bash $S/p4_ctr.sh $O $tag $R/e2b.L4.c16384 B 16 L2R_GEMV=amx L2R_ROWS=$rows $KVP L2R_RESIDENT_KIB=1344 L2R_LOCK=1 | tail -1
  python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag t=$(( $(date +%s) - t0 ))s /"
  grep -o '"lock":{[^}]*}' $O/$tag.json
done
sudo kill $TS
echo P5_STAB_DONE
