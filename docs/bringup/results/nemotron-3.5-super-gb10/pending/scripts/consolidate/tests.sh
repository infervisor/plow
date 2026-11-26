#!/usr/bin/env bash
# tests.sh: cargo + python test suite for the consolidated worktree; one summary line per suite.
cd /home/ifv/plow/.claude/worktrees/agent-aabd7b853cbd494b2 || exit 1
export CARGO_TARGET_DIR=/home/ifv/plow-target-consolidate
NP_RUNTIME=bwrap nix develop --command bash -c '
run() { local name=$1; shift; echo "=== $name"; "$@" > "/home/ifv/.claude/jobs/72c86dac/tmp/consolidate/t_$name.log" 2>&1; local rc=$?; grep -E "^test result:|^Ran |^OK|^FAILED|panicked|^error" "/home/ifv/.claude/jobs/72c86dac/tmp/consolidate/t_$name.log" | tail -12; echo "rc=$rc"; }
run devgen cargo test -p devgen --lib
run packet cargo test -p packet --lib
run plowasset cargo test -p plow-asset
run plowrt cargo test -p plowrt --features cuda,hsa --lib
run devgen_knob cargo test -p devgen --lib knob
run plowrt_knob cargo test -p plowrt --features cuda,hsa --lib knob
run costkc cargo test -p costmodel -p kernelcaps
run py_op_roof python3 scripts/bench/test_op_roof.py
run py_block_roofline python3 scripts/campaign/test_block_roofline.py
run py_certify bash -c "cd scripts/campaign && python3 test_certify_kernel.py"
run py_gen_search python3 scripts/gen_kernels/test_gen_search.py
for t in scripts/tests/test_*.py; do run "py_$(basename $t .py)" python3 "$t"; done
'
