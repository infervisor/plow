#!/usr/bin/env bash
# L1 router top-k gate: bundle from the merged head (flag-free), then alternating 8192->256 pairs
# vs the stack-3 bundle with the same runtime. Checksum must stay fnv1a64:71a28c1449921c95.
set -uo pipefail
export GPU_LEASE_TIMEOUT=7200
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
until grep -q SEAMS_GATE_DONE "$out/seams-gate.log"; do sleep 15; done
until grep -q 'l1 build rc=' "$out/build-l1.log"; do sleep 15; done
rm -rf /tmp/k3-l1
"$S/showdown_bundle.sh" /tmp/k3-l1 > "$out/l1-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/l1-bundle.log"
echo "packet: l1=$(sha256sum /tmp/k3-l1/assets/model.pkt | cut -c1-16) stack3=$(sha256sum /tmp/k3-stack3/assets/model.pkt | cut -c1-16)"
cp -f /tmp/k3-l1/bin/plowrt /tmp/k3-stack3/bin/plowrt
cd "$repo"
bench() { local B=$1 tag=$2
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "l1-$tag" nix develop -c env RUST_LOG=info \
    /tmp/k3-l1/bin/plowrt --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len 256 >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log")"
}
bench /tmp/k3-l1     l1-1
bench /tmp/k3-stack3 c-1
bench /tmp/k3-stack3 c-2
bench /tmp/k3-l1     l1-2
echo "[$(date +%T)] L1_GATE_DONE"
