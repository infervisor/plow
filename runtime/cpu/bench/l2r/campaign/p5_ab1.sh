#!/usr/bin/env bash
# p5_ab1.sh: existing plowrt flags on real E2B serving, c1 ISL 1000 OSL 128, REPS 2, T4 order.
#   ctl = P0 config (WEIGHT_AFFINE=1); huge = + PLOW_CPU_HUGE_PAGES=1; hugena = HUGE_PAGES=1 without affine;
#   sram = + PLOW_CPU_SRAM=1 with the driver at l2_ways=12 l3_ways=8 (restored after).
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p5_ab1; mkdir -p $R
M=gemma-4-E2B-it
A=/tmp/g4c/l2r/rtx90-2k/$M
RT=/tmp/g4c/l2r/bin/plowrt-iso-smt
KO=$WT/runtime/cpu/driver/pseudo_lock_sram.ko
cd $WT
SHA=$(git rev-parse --short=12 HEAD)
run() {
  local tag=$1; shift
  env "$@" ASSETS=$A PLOWRT=$RT PLOWRT_GIT_SHA=$SHA CONCS=1 ISL=1000 OSL=128 REPS=2 \
    /tmp/g4c/grid.sh plow $M $R/$tag > $R/$tag.log 2>&1
  echo "$tag rc=$? $(grep -h 'tpot p50' $R/$tag.log | awk '{printf "%s ", $7}')"
}
reload() {
  sudo rmmod pseudo_lock_sram && sudo insmod "$KO" "$@" && sudo chgrp "$(id -g)" /dev/pseudo_lock && sudo chmod 0660 /dev/pseudo_lock
  cat /sys/class/misc/pseudo_lock/caps
}
run ctl PLOW_CPU_WEIGHT_AFFINE=1
run huge PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_HUGE_PAGES=1
run ctl2 PLOW_CPU_WEIGHT_AFFINE=1
run huge2 PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_HUGE_PAGES=1
run hugena PLOW_CPU_HUGE_PAGES=1
run hugena2 PLOW_CPU_HUGE_PAGES=1
reload l2_ways=12 l3_ways=8 > $R/caps.sram.txt
run sram PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_SRAM=1
run ctl3 PLOW_CPU_WEIGHT_AFFINE=1
run sram2 PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_SRAM=1
reload l2_ways=12 l3_ways=0 l3_io_ways=0 > $R/caps.restored.txt
echo P5_AB1_DONE
