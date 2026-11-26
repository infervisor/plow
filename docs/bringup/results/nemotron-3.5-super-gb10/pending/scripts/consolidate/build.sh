#!/usr/bin/env bash
# build.sh: plowc + plowrt (cuda) release builds in the consolidated worktree.
cd /home/ifv/plow/.claude/worktrees/agent-aabd7b853cbd494b2 || exit 1
export CARGO_TARGET_DIR=/home/ifv/plow-target-consolidate
NP_RUNTIME=bwrap nix develop --command bash -c '
set -e
echo "ROCM_PATH=${ROCM_PATH:-unset} PLOW_NIX_GPU=${PLOW_NIX_GPU:-unset}"
cargo build --release -p plowc
cargo build --release -p plowrt --features cuda --bin plowrt --example step_bench --example block_run --example packet_run
echo BUILD_OK'
