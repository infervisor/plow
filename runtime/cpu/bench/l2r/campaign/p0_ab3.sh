#!/usr/bin/env bash
# p0_ab3.sh: current plowc with the production recipe target (--gpu rtx6000pro --arch sm_120a), E2B 2K c1.
#   g: n_cu 96, all cores (compare d = 4fea6d8c3932 from plowc 878dac7b)
#   h: n_cu 90, isolated
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p0_ab; mkdir -p $R
M=gemma-4-E2B-it
emit() {
  local d=$1 ncu=$2
  [ -s $d/model.pkt ] && return 0
  rm -rf $d; mkdir -p $d
  PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 PLOW_MAX_CHUNK=2048 /tmp/g4c/l2r/bin/plowc --hf-dir /tmp/models/google/$M \
    --gpu rtx6000pro --arch sm_120a --n-cu $ncu --emit devblob --max-ctx 2048 --out $d > $d.emit.log 2>&1 || { echo "emit $d failed"; exit 1; }
  ln -sfn /tmp/models/google/$M $d/checkpoint
  cp /tmp/models/google/$M/tokenizer.json $d/
}
emit /tmp/g4c/l2r/rtx96-2k/$M 96
emit /tmp/g4c/l2r/rtx90-2k/$M 90
cd $WT
SHA=$(git rev-parse --short=12 HEAD)
run() {
  local tag=$1 assets=$2 rt=$3
  PLOW_CPU_WEIGHT_AFFINE=1 ASSETS=$assets PLOWRT=$rt PLOWRT_GIT_SHA=$SHA CONCS=1 ISL=1000 OSL=128 REPS=2 \
    /tmp/g4c/grid.sh plow $M $R/$M.$tag > $R/$M.$tag.log 2>&1
  echo "$tag rc=$?"
}
run g /tmp/g4c/l2r/rtx96-2k/$M /tmp/g4c/l2r/bin/plowrt
run h /tmp/g4c/l2r/rtx90-2k/$M /tmp/g4c/l2r/bin/plowrt-iso
echo AB3_DONE
