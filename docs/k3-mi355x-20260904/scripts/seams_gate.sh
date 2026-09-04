#!/usr/bin/env bash
# Sequence-parallel seams TP8 gate: bundle with PLOW_SEQ_PAR_SEAMS=1, one audited run, then three
# alternating 8192->256 folds vs the served-config bundle using the SAME runtime binary.
# Both arms must produce fnv1a64:71a28c1449921c95.
set -uo pipefail
export GPU_LEASE_TIMEOUT=7200
repo=/home/lava/plow; out=/tmp/k3-xr-phase-gate; ckpt=/tmp/k3-farm.dvzmZN; S=/home/lava/.claude/jobs/3cf96348/tmp
until grep -q 'fmt rc=' "$out/build-seams.log"; do sleep 20; done
grep -E 'rc=' "$out/build-seams.log" | tr '\n' ' '; echo
rm -rf /tmp/k3-seqpar
"$S/showdown_bundle.sh" /tmp/k3-seqpar PLOW_SEQ_PAR_SEAMS=1 > "$out/seqpar-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:|BUNDLE_DONE" "$out/seqpar-bundle.log"
grep -oE 'XReduceTwoShot[^,]{0,30}|XReduceScatter[^,]{0,30}|XAllGather[^,]{0,30}' /tmp/k3-seqpar/emit.log | sort | uniq -c | head -6
grep -c 'plow_seq_par_seams_arm_1' /tmp/k3-seqpar/build.log
cp -f /tmp/k3-seqpar/bin/plowrt /tmp/k3-stack3/bin/plowrt
cd "$repo"
bench() { local B=$1 tag=$2 audit=$3
  echo "[$(date +%T)] start $tag"
  perf-data/tools/gpulease -n 8 "seqpar-$tag" nix develop -c env RUST_LOG=info \
    /tmp/k3-seqpar/bin/plowrt --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" $audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len 256 >"$out/bench-$tag.log" 2>&1
  rc=$?; echo "[$(date +%T)] rc=$rc $tag: $(python3 "$S/bench_fields.py" "$out/bench-$tag.log")"
}
bench /tmp/k3-seqpar sp-audit ""
bench /tmp/k3-seqpar sp-1 --amd-tp-no-audit
bench /tmp/k3-stack3 s3-1 --amd-tp-no-audit
bench /tmp/k3-stack3 s3-2 --amd-tp-no-audit
bench /tmp/k3-seqpar sp-2 --amd-tp-no-audit
bench /tmp/k3-seqpar sp-3 --amd-tp-no-audit
bench /tmp/k3-stack3 s3-3 --amd-tp-no-audit
echo "[$(date +%T)] SEAMS_GATE_DONE"
