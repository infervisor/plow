#!/usr/bin/env bash
# pollvar.sh: build cpu_profile with the wait-loop poll mask at 255 and 15 (temporary source edit, restored after),
# then profile E2B decode with each and with the stock 63.
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
F=$WT/crates/plowrt/src/exec/cpu/interp.rs
E=/tmp/plow-target-l2r/release/examples/cpu_profile
cd $WT
cp $E /tmp/g4c/l2r/cpu_profile.m63
for m in 255 15; do
  sed -i "s/if n & 63 == 0 {/if n \& $m == 0 {/" $F
  CARGO_TARGET_DIR=/tmp/plow-target-l2r nix develop -c cargo build --release -p plowrt --no-default-features --features cpu --example cpu_profile > /tmp/g4c/l2r/pollvar.build.$m.log 2>&1
  cp $E /tmp/g4c/l2r/cpu_profile.m$m
  sed -i "s/if n & $m == 0 {/if n \& 63 == 0 {/" $F
done
git diff --stat -- $F
for m in 63 255 15 63 255 15; do
  taskset -c 2-31,34-63,66-95,98-127,130-159,162-191 env PLOW_CPU_WEIGHT_AFFINE=1 PROF_DUMP=/tmp/g4c/l2r/results/p5_poll.m$m.csv \
    /tmp/g4c/l2r/cpu_profile.m$m /tmp/g4c/l2r/rtx90-2k/gemma-4-E2B-it/model.pkt /tmp/models/google/gemma-4-E2B-it --threads 90 --prompt-tokens 1000 2>&1 | grep -E "decode step" | sed "s/^/m$m /"
  python3 /tmp/g4c/l2r/prof_edge.py /tmp/g4c/l2r/results/p5_poll.m$m.csv.decode_step 104 105 | sed -n 2p
done
