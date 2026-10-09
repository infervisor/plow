#!/usr/bin/env bash
# build_rt.sh: plowrt (CPU engine) from the worktree HEAD into /tmp/g4c/l2r/combtgt, housekeeping cores only
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
git diff --quiet HEAD -- crates/plowrt runtime/cpu/dev || { echo "plowrt sources dirty"; exit 1; }
export CARGO_TARGET_DIR=/tmp/g4c/l2r/combtgt
nice -n 19 taskset -c 0,1,32,33,64,65 nix develop --command cargo build --release -p plowrt --no-default-features --features cpu 2>&1 | grep -E "^error|Finished"
sha256sum $CARGO_TARGET_DIR/release/plowrt
