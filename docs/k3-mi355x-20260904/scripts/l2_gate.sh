#!/usr/bin/env bash
# L2 gate: bundle with PLOW_GQ_ORDER=asap-seg from the merged worktree, then alternating
# 8192->256 pairs vs /tmp/k3-l1 (served config) with the same runtime; checksum must stay
# fnv1a64:71a28c1449921c95.
set -uo pipefail
export GPU_LEASE_TIMEOUT=7200
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
rm -rf /tmp/k3-l2
"$S/showdown_bundle.sh" /tmp/k3-l2 PLOW_GQ_ORDER=asap-seg > "$out/l2-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/l2-bundle.log"
echo "packet: l2=$(sha256sum /tmp/k3-l2/assets/model.pkt | cut -c1-16) l1=$(sha256sum /tmp/k3-l1/assets/model.pkt | cut -c1-16)"
cd "$repo"
bench() { local B=$1 tag=$2
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "l2-$tag" nix develop -c env RUST_LOG=info \
    /tmp/k3-l2/bin/plowrt --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len 256 >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log")"
}
bench /tmp/k3-l2 l2-1
bench /tmp/k3-l1 c-1
bench /tmp/k3-l2 l2-2
bench /tmp/k3-l1 c-2
echo "[$(date +%T)] L2_GATE_DONE"
