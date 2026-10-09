#!/usr/bin/env bash
# p5_grid.sh: PLOW_CPU_COMBINE=16 vs 0 on the P0 serving grid: E2B/E4B, ISL 1900 / 15900, OSL 128, c1/4/16, REPS 2,
# pk90 packets (n_cu 90, max ctx 16K), same binary both arms, arm order ctl then treat per cell.
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p5_grid; mkdir -p $R
cd $WT
SHA=$(git rev-parse --short=12 HEAD)+comb
for M in gemma-4-E2B-it gemma-4-E4B-it; do
  for isl in 1900 15900; do
    for g in 0 16; do
      out=$R/$M.isl$isl.g$g
      PLOW_CPU_COMBINE=$g PLOW_CPU_WEIGHT_AFFINE=1 PLOW_SESSION_SLACK=32 ASSETS=/tmp/g4c/l2r/pk90/$M PLOWRT=/tmp/g4c/l2r/bin/plowrt-comb-iso \
        PLOWRT_GIT_SHA=$SHA CONCS="1 4 16" ISL=$isl OSL=128 REPS=2 /tmp/g4c/grid.sh plow $M $out > $out.log 2>&1
      echo "$M isl $isl g$g rc=$?"
      grep -h "tpot p50" $out.log | sed 's/^/   /'
    done
  done
done
echo P5_GRID_DONE
