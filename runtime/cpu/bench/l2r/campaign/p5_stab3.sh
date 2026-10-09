#!/usr/bin/env bash
# p5_stab3.sh: continuous stability, >= 30 min: 80 back-to-back E2B L4 16K batch-16 AMX lock-14 runs of 20000 steps
# (~26 s each), each gated (16 rows) and counted (DRAM CAS, L2 lines in), lock held fraction before / after each run,
# turbostat (busy MHz, package / DRAM power) every 10 s across all of them. Expects pseudo_lock_sram at l2_ways=14;
# restores l2_ways=12 at the end.
set -u
S=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r
KO=$S/../../driver/pseudo_lock_sram.ko
R=/tmp/g4c/l2r/ref
O=/tmp/g4c/l2r/results/p5_stab3; rm -rf $O; mkdir -p $O
export L2R_BIN=/tmp/g4c/l2r/l2r_layer.b
grep -q "1605632 B/core" /sys/class/misc/pseudo_lock/caps || { echo "driver not at 14 ways"; exit 1; }
rows=$R/e2b.L4.c16384; for i in $(seq 1 15); do rows=$rows,$R/e2b.L4.c16384.o$i; done
sudo turbostat --quiet --interval 10 --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgWatt,RAMWatt -o $O/turbostat.txt &
TS=$!
t0=$(date +%s)
for i in $(seq -w 1 80); do
  tag=stab.$i
  P4_STEPS=20000 bash $S/p4_ctr.sh $O $tag $R/e2b.L4.c16384 B 16 L2R_GEMV=amx L2R_ROWS=$rows L2R_KV_TILE_KIB=256 L2R_KV_PFD=8 \
    L2R_RESIDENT_KIB=1568 L2R_LOCK=1 > /dev/null 2>&1
  echo "$tag t=$(( $(date +%s) - t0 ))s $(python3 $S/p2_gate.py $O/$tag.json | head -1) $(grep -o '"lock":{[^}]*}' $O/$tag.json)"
done
sudo pkill -f "turbostat --quiet --interval 10" ; wait $TS 2>/dev/null
echo P5_STAB3_RUNS_DONE
if [ -z "$(grep -v '^#' /sys/class/misc/pseudo_lock/regions | grep -v '^$')" ]; then
  sudo rmmod pseudo_lock_sram && sudo insmod "$KO" l2_ways=12 l3_ways=0 && sudo chgrp "$(id -g)" /dev/pseudo_lock && sudo chmod 0660 /dev/pseudo_lock
fi
cat /sys/class/misc/pseudo_lock/caps
echo P5_STAB3_DONE
