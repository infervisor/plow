#!/usr/bin/env bash
# Long-context probe: re-emit the default stack at --max-ctx 32768, then 16384->1024 (3 requests)
# and 8192->1024 on the same bundle for a like-for-like reading; plus 300->32 exactness vs /tmp/k3-l2.
set -uo pipefail
export GPU_LEASE_TIMEOUT=7200
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
sed 's/--max-ctx 16384/--max-ctx 32768/' "$S/showdown_bundle.sh" > "$S/showdown_bundle_32k.sh"; chmod +x "$S/showdown_bundle_32k.sh"
rm -rf /tmp/k3-32k
"$S/showdown_bundle_32k.sh" /tmp/k3-32k > "$out/32k-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/32k-bundle.log"
python3 -c "import json;d=json.load(open('/tmp/k3-32k/assets/build.json'));print('buckets', sorted(set(p.get('t',0) for p in d['programs']))[:12])" 2>/dev/null
cd "$repo"
bench() { local B=$1 tag=$2 inlen=$3 outlen=$4 reqs=$5
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "lc-$tag" nix develop -c env RUST_LOG=info \
    "$B/bin/plowrt" --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len "$inlen" --seed 20260904 \
    --concurrency 1 --requests "$reqs" --warmup-requests 1 --output-len "$outlen" >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag in=$inlen out=$outlen: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log") $(grep -o -E 'Error: .{0,120}' "$out/bench-$tag.log" | head -1)"
}
bench /tmp/k3-32k lc-16k 16384 1024 3
bench /tmp/k3-32k lc-8k 8192 1024 3
bench /tmp/k3-32k lc-300 300 32 2
bench /tmp/k3-l2 lc-300-ctl 300 32 2
echo "[$(date +%T)] LONGCTX_DONE"
