#!/usr/bin/env bash
# p5_ab2.sh: L2-only pseudo-lock (driver l2_ways=12, l3_ways=0: worker scratch heads locked in L2, nothing locked in
# L3, L3 left as a normal cache) vs control, E2B c1 ISL 1000 OSL 128, REPS 2, T4 order.
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p5_ab2; mkdir -p $R
M=gemma-4-E2B-it
A=/tmp/g4c/l2r/rtx90-2k/$M
RT=/tmp/g4c/l2r/bin/plowrt-iso-smt
cd $WT
SHA=$(git rev-parse --short=12 HEAD)
cat /sys/class/misc/pseudo_lock/caps > $R/caps.txt
run() {
  local tag=$1; shift
  env "$@" ASSETS=$A PLOWRT=$RT PLOWRT_GIT_SHA=$SHA CONCS=1 ISL=1000 OSL=128 REPS=2 \
    /tmp/g4c/grid.sh plow $M $R/$tag > $R/$tag.log 2>&1
  echo "$tag rc=$? $(grep -h 'tpot p50' $R/$tag.log | awk '{printf "%s ", $7}') $(grep -ho 'locked_mib[^ ]* [^ ]*held[^ ]*\|locked_kib[^ ]*' $R/$tag/server.log 2>/dev/null | tr '\n' ' ')"
}
run ctl PLOW_CPU_WEIGHT_AFFINE=1
run l2lock PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_SRAM=1
run ctl2 PLOW_CPU_WEIGHT_AFFINE=1
run l2lock2 PLOW_CPU_WEIGHT_AFFINE=1 PLOW_CPU_SRAM=1
echo P5_AB2_DONE
