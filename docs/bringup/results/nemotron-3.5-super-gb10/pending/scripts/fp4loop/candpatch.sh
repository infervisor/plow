#!/usr/bin/env bash
# candpatch.sh : the TMA-port candidate (/home/ifv/plow-fp4loop/cand) vs the shared worktree's HEAD, this agent's
# files only -> fp4loop-tma-cand.patch (apply in the worktree with patch -p1)
set -uo pipefail
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
C=${C:-/home/ifv/plow-fp4loop/cand}
OUT=${OUT:-/home/ifv/.claude/jobs/72c86dac/tmp/fp4loop/fp4loop-tma-cand.patch}
T=$(mktemp -d)
: > "$OUT"
for f in crates/devgen/src/components/nvfp4_moe.rs crates/devgen/src/knob_spec.rs crates/devgen/src/nemotron_h.rs \
  crates/packet/src/dev.rs crates/packet/src/devbuild.rs crates/packet/src/rope.rs crates/packet/src/slots.rs \
  crates/packet/src/typed_abi.rs crates/plow-asset/src/live_kv.rs crates/plow-asset/src/logical_effects.rs \
  crates/plow-asset/src/splitk.rs crates/plowrt/src/device/cuda.rs crates/plowrt/src/exec/gpu.rs \
  runtime/nvidia/interp_sm120.cu runtime/nvidia/op_moe_nvfp4_pf.cuh; do
  mkdir -p "$T/a/$(dirname $f)"
  git -C "$W" show "HEAD:$f" > "$T/a/$f"
  (cd "$T" && diff -u "a/$f" "$C/$f" | sed "1,2s#^+++ $C/#+++ b/#") >> "$OUT"
done
rm -rf "$T"
grep -c "^@@" "$OUT"; wc -l "$OUT"
cd "$W" && patch -p1 --dry-run < "$OUT" | grep -v "^checking" || true
