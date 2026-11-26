#!/usr/bin/env bash
# muxtest.sh (gpuq, CPU): plowrt mux lib tests on the shared tree (HEAD + uncommitted edits).
cd /home/ifv/plow/.claude/worktrees/gb10-sm121a || exit 1
export CARGO_TARGET_DIR=/home/ifv/plow-target-merge
NP_RUNTIME=bwrap nix develop --command cargo test -q -p plowrt --features cuda,hsa --lib ${FILTER:-serve::mux} 2>&1 \
  | grep -E "^test .*FAILED|^test result|panicked|error(\[|:)" | tail -n 30
