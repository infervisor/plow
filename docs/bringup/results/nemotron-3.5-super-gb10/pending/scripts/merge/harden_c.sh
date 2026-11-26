#!/usr/bin/env bash
# harden_c.sh : CPU validation of the merge-harden patch on a HEAD+patch snapshot worktree.
# Runs in the flake's dev env captured by `nix print-dev-env` (devenv.sh): userns is restricted on
# this boot and nix develop nested under the proot fallback cannot stat /nix/store.
# Summary to stdout, full cargo output to harden_c.full.log.
M=/home/ifv/.claude/jobs/72c86dac/tmp/merge
W=/home/ifv/plow-scratch/merge/hard
L=$M/harden_c.full.log
: > "$L"
source $M/devenv.sh > /dev/null 2>&1
[ -n "${NIX_BUILD_TOP:-}" ] && rmdir "$NIX_BUILD_TOP" 2>/dev/null
unset TMPDIR TMP TEMP TEMPDIR NIX_BUILD_TOP
export CARGO_TARGET_DIR=/home/ifv/plow-target-harden CUDA_VISIBLE_DEVICES=
export PLOW_VERIFY_BIN=$W/lean-plow/.lake/build/bin/plow_verify
cd "$W" || exit 1
date -u +%T
run() {
  echo "== cargo $*"
  cargo "$@" > /tmp/harden-suite.log 2>&1
  echo "rc=$?"
  cat /tmp/harden-suite.log >> "$L"
  grep -E "^test result|panicked|FAILED|^error(\[|:)" /tmp/harden-suite.log | sort | uniq -c | sed -n 1,40p
}
py() { echo "== python $*"; python3 "$@" 2>&1 | tail -3; }
run test -p plowrt --features cuda,hsa --lib serve::mux::tests::cold_burst
run test -p plowrt --features cuda,hsa --lib knob
run test -p devgen --lib knob
run test -p packet
run test -p plow-asset
run test -p devgen --lib
run test -p plowrt --features cuda,hsa --lib
run test -p hwspec -p costmodel -p kernelcaps -p tunedb -p nn-graph -p rewrite
run test -p plowc
echo "== cargo clippy plowrt"
cargo clippy -p plowrt --features cuda,hsa --lib --tests > /tmp/harden-clippy.log 2>&1
echo "rc=$?"
cat /tmp/harden-clippy.log >> "$L"
grep -E "^(warning|error)" /tmp/harden-clippy.log | sort | uniq -c | sort -rn | sed -n 1,30p
grep -E -A3 "^(warning|error)" /tmp/harden-clippy.log | grep -E "^ *--> crates/plowrt/src/(serve/mux|serve/chat|config|knob_spec|exec/gpu|exec/gpu/kv_blocks)\.rs" | sed -n 1,40p
py scripts/bench/test_op_roof.py
py scripts/campaign/test_block_roofline.py
py scripts/campaign/test_certify_kernel.py
py scripts/gen_kernels/test_gen_search.py
echo "== check_recipe"
python3 scripts/check_recipe.py recipes/infervisor/*/*.toml scripts/campaign/recipes/nemotron-3.5-super.gb10.nvfp4.toml scripts/campaign/recipes/qwen38-27b.gb10.fp8.toml > /tmp/harden-recipe.log 2>&1
echo "check_recipe rc=$?"
grep -v "^  note: plow_git" /tmp/harden-recipe.log | grep -v ": OK$" | tail -40
date -u +%T
