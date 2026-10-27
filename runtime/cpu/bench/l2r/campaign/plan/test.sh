#!/usr/bin/env bash
# test.sh: plowc stage_plan unit tests (release, housekeeping cores) and the release plowc build
cd /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
export CARGO_TARGET_DIR=/tmp/plow-target-l2r
nice -n 19 taskset -c 0,1,32,33,64,65 nix develop --command cargo test --release -p plowc --lib stage_plan 2>&1 | grep -E "^test |test result|^error"
nice -n 19 taskset -c 0,1,32,33,64,65 nix develop --command cargo build --release -p plowc 2>&1 | grep -E "^error|Finished"
