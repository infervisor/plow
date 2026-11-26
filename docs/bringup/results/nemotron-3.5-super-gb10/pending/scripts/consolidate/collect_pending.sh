#!/usr/bin/env bash
# collect_pending.sh: gather every agent's in-flight (ungated) work as patches into /home/ifv/plow-scratch/pending.
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
P=/home/ifv/plow-scratch/pending
T=/home/ifv/.claude/jobs/72c86dac/tmp
rm -rf "$P"; mkdir -p "$P"
cd "$W" || exit 1
git diff HEAD -- crates/plowrt docs/flags-reference.md runtime/nvidia/op_gemv_fp8_tc.cuh scripts/campaign/recipes > "$P/merge-hardening.patch"
git diff HEAD -- scripts/gen_kernels/bench_lib_attn.py > "$P/libattn-bench.patch"
# dev trees: diff against the HEAD they were cut from, including untracked new files
for pair in "mamba-chunk-out-v1:/home/ifv/plow-mamba-v1" "mamba-chunk-out-dev:/home/ifv/plow-mamba" "qwen-fp8tc-row-invariant:/home/ifv/plow-tcinv"; do
  name=${pair%%:*}; d=${pair#*:}
  if git -C "$d" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$d" add -N . >/dev/null 2>&1
    git -C "$d" diff HEAD > "$P/$name.patch"
    echo "$name base $(git -C "$d" rev-parse --short HEAD)"
  else
    echo "$name: $d is not a git tree"
  fi
done
# fp4 candidate: its own patch generator
if [ -x "$T/fp4loop/candpatch.sh" ] || [ -f "$T/fp4loop/candpatch.sh" ]; then
  C=/home/ifv/plow-fp4loop/cand3 OUT="$P/fp4-tma-split-kernel.patch" bash "$T/fp4loop/candpatch.sh" > /dev/null 2>&1 || echo "candpatch failed"
fi
cp /home/ifv/plow-scratch/hd256/hd256-dev-full.patch "$P/hd256-decode-rgm-dev.patch" 2>/dev/null
wc -l "$P"/*.patch
