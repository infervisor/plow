#!/usr/bin/env bash
# p5_stab2.sh: E2B L4 (full attention, 1,409,024 B per core with the AMX packing) exceeds the 12-way lock budget
# (1,376,256 B). Reload pseudo_lock_sram with l2_ways=14 (1,605,632 B per core) while no region is live, then
# (1) E2B L4 residency plain vs lock 14 at 2K / 16K, batch 1 and 16, AMX, under counters; (2) 30 consecutive
# E2B L4 16K batch-16 AMX lock-14 runs of 20000 steps, each gated and counted, turbostat every 10 s; restore 12 ways.
set -u
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
KO=$S/../../driver/pseudo_lock_sram.ko
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/p5_stab2; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.b
KVP="L2R_KV_TILE_KIB=256 L2R_KV_PFD=8"
reload() {
  if [ -n "$(grep -v '^#' /sys/class/misc/pseudo_lock/regions 2>/dev/null | grep -v '^$')" ]; then echo "live regions; not reloading"; exit 1; fi
  sudo rmmod pseudo_lock_sram && sudo insmod "$KO" "$@" && sudo chgrp "$(id -g)" /dev/pseudo_lock && sudo chmod 0660 /dev/pseudo_lock
  cat /sys/class/misc/pseudo_lock/caps
}
rows() { local d=$R/$1 r=$1; for i in $(seq 1 15); do d=$d,$R/$r.o$i; done; echo $d; }
reload l2_ways=14 l3_ways=0
for ref in e2b.L4.c2048 e2b.L4.c16384; do
  kv=""; [ $ref = e2b.L4.c16384 ] && kv=$KVP
  for arm in plain lock14; do
    ex=""; [ $arm = lock14 ] && ex="L2R_RESIDENT_KIB=1568 L2R_LOCK=1"
    for b in 1 16; do
      tag=$ref.amx.b$b.$arm
      bash $S/p4_ctr.sh $O $tag $R/$ref B $b L2R_GEMV=amx L2R_ROWS=$(rows $ref) $kv $ex | tail -1
      python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag /"
      grep -o '"lock":{[^}]*}' $O/$tag.json
    done
  done
done
echo P5_RES14_DONE
sudo turbostat --quiet --interval 10 --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgWatt,RAMWatt -o $O/stab.turbostat.txt &
TS=$!
t0=$(date +%s)
for i in $(seq -w 1 30); do
  tag=stab.e2b.L4.c16384.amx.b16.lock14.$i
  P4_STEPS=20000 bash $S/p4_ctr.sh $O $tag $R/e2b.L4.c16384 B 16 L2R_GEMV=amx L2R_ROWS=$(rows e2b.L4.c16384) $KVP L2R_RESIDENT_KIB=1568 L2R_LOCK=1 | tail -1
  python3 $S/p2_gate.py $O/$tag.json | head -1 | sed "s/^/$tag t=$(( $(date +%s) - t0 ))s /"
  grep -o '"lock":{[^}]*}' $O/$tag.json
done
sudo kill $TS
echo P5_STAB_DONE
reload l2_ways=12 l3_ways=0
echo P5_STAB2_DONE
