#!/usr/bin/env bash
# Screen: exact KDA key-factor objects ON (packet unchanged), vs the final stack bundle, same runtime.
set -uo pipefail
export GPU_LEASE_TIMEOUT=7200
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
rm -rf /tmp/k3-kf
"$S/showdown_bundle_kf.sh" /tmp/k3-kf > "$out/kf-bundle.log" 2>&1
grep -E "^lean:|^objects:|BUNDLE_DONE" "$out/kf-bundle.log"
echo "packet: kf=$(sha256sum /tmp/k3-kf/assets/model.pkt | cut -c1-16) l2=$(sha256sum /tmp/k3-l2/assets/model.pkt | cut -c1-16)"
ls /tmp/k3-kf/hsaco | grep -i 'key_factor' | tr '\n' ' '; echo
cd "$repo"
bench() { local B=$1 tag=$2
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "kf-$tag" nix develop -c env RUST_LOG=info \
    /tmp/k3-kf/bin/plowrt --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len 256 >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log") warn=$(grep -c 'interpreter fallback' "$out/bench-$tag.log")"
}
bench /tmp/k3-kf kf-1
bench /tmp/k3-l2 c-1
bench /tmp/k3-kf kf-2
echo "[$(date +%T)] KF_GATE_DONE"
