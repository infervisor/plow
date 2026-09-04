#!/usr/bin/env bash
# Stack-2 gate: exact-arm 8192->256 checksum must equal the control's (fnv1a64:b7682a38c151ac99),
# then a TTFT/TPOT reading of the served (f32-mix) bundle; on pass, launch the served showdown.
set -uo pipefail
export GPU_LEASE_TIMEOUT=14400
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
until grep -q STACK2_PREP_DONE "$out/stack2-prep.log"; do sleep 30; done
cd "$repo"
bench() { local B=$1 tag=$2 outlen=$3
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "stack2-$tag" nix develop -c env RUST_LOG=info \
    "$B/bin/plowrt" --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len "$outlen" >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log")"
}
bench /tmp/k3-stack2-exact ex256-1 256
bench /tmp/k3-stack2       sv256-1 256
bench /tmp/k3-stack2-exact ex256-2 256
ck1=$(grep -o -E '"output_checksum": "[^"]+"' "$out/bench-ex256-1.log" | head -1 | cut -d'"' -f4)
ck2=$(grep -o -E '"output_checksum": "[^"]+"' "$out/bench-ex256-2.log" | head -1 | cut -d'"' -f4)
echo "[$(date +%T)] exact-arm checksums: $ck1 $ck2 (control fnv1a64:b7682a38c151ac99)"
if [ "$ck1" = "fnv1a64:b7682a38c151ac99" ] && [ "$ck2" = "fnv1a64:b7682a38c151ac99" ]; then
  echo "[$(date +%T)] EXACT_GATE_PASS"
  rm -rf /tmp/k3-showdown-c1-stack2-20260904
  "$S/run_showdown.sh" /tmp/k3-stack2 k3-showdown-c1-stack2-20260904 > "$out/showdown-stack2.log" 2>&1
  echo "[$(date +%T)] showdown rc=$?"
else
  echo "[$(date +%T)] EXACT_GATE_FAIL"
fi
echo "[$(date +%T)] STACK2_GATE_DONE"
