#!/usr/bin/env bash
# AITER-rate collective schedule TP8 gate: the candidate bundle's prefill objects are built with
# -DPLOW_XR_SCHED=aiter and its packet is emitted with PLOW_XR_CUS=48 (the 16-byte schedule's
# workgroup count; only the collective packets' CU sets change), then one audited candidate run
# and three alternating 8192->256 folds against the control bundle built from the same source with
# neither. Both arms must produce fnv1a64:71a28c1449921c95.
set -uo pipefail
export GPU_LEASE_TIMEOUT=14400
src=${PLOW_BUNDLE_SRC:-/home/lava/plow/.claude/worktrees/agent-ae3f4f443b6abf8a2}
out=/tmp/k3-xr-sched-gate; ckpt=/tmp/k3-farm.dvzmZN; S=$src/docs/k3-mi355x-20260904/scripts
mkdir -p "$out"
export PLOW_BUNDLE_SRC=$src
rm -rf /tmp/k3-xrsched /tmp/k3-xrsched-ctl
"$S/showdown_bundle.sh" /tmp/k3-xrsched-ctl > "$out/ctl-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/ctl-bundle.log"
PLOW_BUNDLE_CMAKE_EXTRA="-DPLOW_XR_SCHED=aiter" "$S/showdown_bundle.sh" /tmp/k3-xrsched PLOW_XR_CUS=48 > "$out/cand-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/cand-bundle.log"
grep -c "PLOW_XR_SCHED_AITER=1" /tmp/k3-xrsched/cmake/CMakeFiles/gfx950_hsaco.dir/build.make
cp -f /tmp/k3-xrsched-ctl/bin/plowrt /tmp/k3-xrsched/bin/plowrt
cd "$src"
bench() { local B=$1 tag=$2 audit=$3
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "xrsched-$tag" nix develop -c env RUST_LOG=info \
    /tmp/k3-xrsched-ctl/bin/plowrt --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" $audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len 256 >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log")"
}
bench /tmp/k3-xrsched xs-audit ""
bench /tmp/k3-xrsched xs-1 --amd-tp-no-audit
bench /tmp/k3-xrsched-ctl ctl-1 --amd-tp-no-audit
bench /tmp/k3-xrsched-ctl ctl-2 --amd-tp-no-audit
bench /tmp/k3-xrsched xs-2 --amd-tp-no-audit
bench /tmp/k3-xrsched xs-3 --amd-tp-no-audit
bench /tmp/k3-xrsched-ctl ctl-3 --amd-tp-no-audit
echo "[$(date +%T)] XR_SCHED_GATE_DONE"
