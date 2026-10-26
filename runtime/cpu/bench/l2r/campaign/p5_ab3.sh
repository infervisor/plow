#!/usr/bin/env bash
# p5_ab3.sh: PLOW_CPU_COMBINE serving A/B (new binary for every arm), E2B and E4B c1 ISL 1000 OSL 128 REPS 2, T4 order.
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p5_ab3; mkdir -p $R
cat > /tmp/g4c/l2r/bin/plowrt-comb-iso <<'EOF'
#!/usr/bin/env bash
exec taskset -c 2-31,34-63,66-95,98-127,130-159,162-191 env PLOW_CPU_THREADS=90 /tmp/g4c/l2r/bin/plowrt.comb "$@"
EOF
chmod +x /tmp/g4c/l2r/bin/plowrt-comb-iso
cd $WT
SHA=$(git rev-parse --short=12 HEAD)+comb
run() {
  local M=$1 tag=$2; shift 2
  env "$@" PLOW_CPU_WEIGHT_AFFINE=1 ASSETS=$( [ $M = gemma-4-E4B-it ] && echo /tmp/g4c/l2r/pk90/$M || echo /tmp/g4c/l2r/rtx90-2k/$M ) PLOWRT=/tmp/g4c/l2r/bin/plowrt-comb-iso PLOWRT_GIT_SHA=$SHA \
    CONCS=1 ISL=1000 OSL=128 REPS=2 /tmp/g4c/grid.sh plow $M $R/$M.$tag > $R/$M.$tag.log 2>&1
  echo "$M $tag rc=$? $(grep -h 'tpot p50' $R/$M.$tag.log | awk '{printf "%s ", $7}')"
}
for M in gemma-4-E2B-it gemma-4-E4B-it; do
  
  run $M ctl PLOW_CPU_COMBINE=0
  run $M g16 PLOW_CPU_COMBINE=16
  run $M ctl2 PLOW_CPU_COMBINE=0
  run $M g16b PLOW_CPU_COMBINE=16
  run $M g8 PLOW_CPU_COMBINE=8
  run $M g30 PLOW_CPU_COMBINE=30
done
echo P5_AB3_DONE
