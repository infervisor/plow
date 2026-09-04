#!/usr/bin/env bash
# Exactness probe for the register-state KDA carry route: ragged tails (300, 8400, 9000 tokens)
# and a concurrency-2 slot probe, stack-3 (regstate on, fixed runtime) vs the opted-out control.
set -uo pipefail
export GPU_LEASE_TIMEOUT=7200
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
cd "$repo"
bench() { local B=$1 tag=$2 inlen=$3 conc=$4 reqs=$5
  perf-data/tools/gpulease -n 8 "regprobe-$tag" nix develop -c env RUST_LOG=info \
    "$B/bin/plowrt" --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len "$inlen" --seed 20260904 \
    --concurrency "$conc" --requests "$reqs" --warmup-requests 1 --output-len 32 >"$out/bench-$tag.log" 2>&1
  rc=$?; ck=$(grep -o -E '"output_checksum": "[^"]+"' "$out/bench-$tag.log" | head -1 | cut -d'"' -f4)
  echo "[$(date +%T)] rc=$rc $tag in=$inlen c=$conc: ck=$ck $(grep -o -E 'Error: .{0,120}' "$out/bench-$tag.log" | head -1)"
}
for L in 300 8400 9000; do
  bench /tmp/k3-stack3       reg-$L  $L 1 2
  bench /tmp/k3-stack3-noreg ctl-$L  $L 1 2
done
bench /tmp/k3-stack3       reg-c2 8192 2 4
bench /tmp/k3-stack3-noreg ctl-c2 8192 2 4
for t in 300 8400 9000 c2; do
  a=$(grep -o -E '"output_checksum": "[^"]+"' "$out/bench-reg-$t.log" | head -1 | cut -d'"' -f4)
  b=$(grep -o -E '"output_checksum": "[^"]+"' "$out/bench-ctl-$t.log" | head -1 | cut -d'"' -f4)
  [ -n "$a" ] && [ "$a" = "$b" ] && echo "MATCH $t $a" || echo "MISMATCH $t reg=$a ctl=$b"
done
echo "[$(date +%T)] REGPROBE_DONE"
