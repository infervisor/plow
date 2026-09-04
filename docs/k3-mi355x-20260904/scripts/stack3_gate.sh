#!/usr/bin/env bash
# Stack-3 candidate: stack-2 + PLOW_MOE_ALIGN_PAR=1 + GemmWide c8 tile at 8192x1536x7168.
# Both levers are exact, so the candidate's 8192->256 checksum must equal stack-2's served arm
# (fnv1a64:71a28c1449921c95). Three alternating 8192->1 folds, then one 256 pair.
set -uo pipefail
export GPU_LEASE_TIMEOUT=14400
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
rm -rf /tmp/k3-stack3
"$S/showdown_bundle.sh" /tmp/k3-stack3 PLOW_MOE_ALIGN_PAR=1 PLOW_GEMM_WIDE_C8_SHAPE=8192x1536x7168 > "$out/stack3-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/stack3-bundle.log"
grep -c "want gfx950" /tmp/k3-stack3/emit.log
cd "$repo"
bench() { local B=$1 tag=$2 outlen=$3
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "stack3-$tag" nix develop -c env RUST_LOG=info \
    "$B/bin/plowrt" --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len "$outlen" >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log")"
}
bench /tmp/k3-stack3 s3-1 1
bench /tmp/k3-stack2 s2-1 1
bench /tmp/k3-stack2 s2-2 1
bench /tmp/k3-stack3 s3-2 1
bench /tmp/k3-stack3 s3-3 1
bench /tmp/k3-stack2 s2-3 1
bench /tmp/k3-stack3 s3-256 256
bench /tmp/k3-stack2 s2-256 256
echo "[$(date +%T)] STACK3_GATE_DONE"
