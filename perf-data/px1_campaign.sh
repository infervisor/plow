#!/usr/bin/env bash
# px1_campaign.sh — PX-1 stage-1 A/B: serialized (off) vs cross-request
# batched (on) prefill, same binary, same assets, same harness/profile as the
# B2 concurrency campaign. Each config's whole server+sweep runs under the
# shared GPU bench lock.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${ASSETS:-/root/gpu-assets-px1}"
VUS="${VUS:-1 4 8 16 32}"

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS="$ASSETS" TAG=px1-off CAMPAIGN=PX1-s1 \
      VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1-bench-off.log 2>&1
echo "OFF rc=$?" | tee /tmp/px1-bench-off.rc

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS="$ASSETS" TAG=px1-on CAMPAIGN=PX1-s1 PLOW_PF_BATCH=1 \
      VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1-bench-on.log 2>&1
echo "ON rc=$?" | tee /tmp/px1-bench-on.rc
