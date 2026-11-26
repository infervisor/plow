#!/usr/bin/env bash
# Full CPU suites on a clean export of the pushed commit (no working-tree edits). Run inside gpuq so it
# never overlaps another agent's timing lease. Args: <commit>
C=$1
D=/home/ifv/plow-scratch/consolidate/src-$C
OUT=/home/ifv/plow-scratch/consolidate/suite-$C.log
rm -rf "$D"; mkdir -p "$D"
git -C /home/ifv/plow/.claude/worktrees/gb10-sm121a archive "$C" | tar -x -C "$D"
cd "$D" || exit 1
export CARGO_TARGET_DIR=/home/ifv/plow-target-merge NP_RUNTIME=bwrap
{
  for spec in "-p hwspec" "-p costmodel" "-p packet" "-p plow-asset" "-p devgen --lib" "-p plowrt --features cuda,hsa --lib"; do
    echo "== cargo test $spec"
    # shellcheck disable=SC2086
    nix develop --command cargo test -q $spec 2>&1 | grep -E "^test result|FAILED|panicked|error(\[|:)" | tail -n 20
    echo "rc=${PIPESTATUS[0]}"
  done
  for t in scripts/bench/test_op_roof.py scripts/campaign/test_block_roofline.py; do
    echo "== $t"
    nix develop --command python3 "$t" 2>&1 | tail -n 3
    echo "rc=${PIPESTATUS[0]}"
  done
  echo SUITE-DONE
} > "$OUT" 2>&1
