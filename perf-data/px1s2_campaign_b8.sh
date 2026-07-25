#!/usr/bin/env bash
# px1s2_campaign_b8.sh — PX-1 stage-2 three-arm sweep on the 12B ctx8k B=8
# blob (VRAM fallback: a foreign long-ctx plowrt held ~33 GB, and the B=16
# blob's 66.6 GiB plan does not co-fit on the 96 GB card). Same binary, same
# harness/profile as px1s2_campaign.sh; only the blob (B=8) differs. The three
# arms are mutually comparable; comparisons against stage-1's committed B=16
# numbers carry the blob caveat.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VUS="${VUS:-1 4 8 16 32}"
PORT="${PORT:-8093}"

# Contention sampler: the box is shared — a foreign long-ctx plowrt (pid
# 850160, port 8091) serves traffic OUTSIDE the bench lock, and its ~100 ms
# 132k-ctx token kernels time-slice against our cooperative decode (measured:
# VU1 ITL flaps 22 ms clean -> exactly 100.0 ms while it serves). A sibling
# agent's microbenches do the same. Per-second `nvidia-smi pmon` rows record
# WHICH pid burned SM in every second; windows where a pid other than our
# server shows sm% > 0 are poisoned and the affected VU points rerun.
( nvidia-smi pmon -d 1 -s u -o T > /tmp/px1s2-pmon.log 2>&1 ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS=/root/gpu-assets-px1s2b8 TAG=px1s2b8-off CAMPAIGN=PX1-s2-b8 \
      PORT="$PORT" VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1s2b8-bench-off.log 2>&1
echo "OFF rc=$?" | tee /tmp/px1s2b8-bench-off.rc

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS=/root/gpu-assets-px1s1b8 TAG=px1s2b8-s1 CAMPAIGN=PX1-s2-b8 \
      PORT="$PORT" PLOW_PF_BATCH=1 VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1s2b8-bench-s1.log 2>&1
echo "S1 rc=$?" | tee /tmp/px1s2b8-bench-s1.rc

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS=/root/gpu-assets-px1s2b8 TAG=px1s2b8-varlen CAMPAIGN=PX1-s2-b8 \
      PORT="$PORT" PLOW_PF_BATCH=1 VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1s2b8-bench-varlen.log 2>&1
echo "VARLEN rc=$?" | tee /tmp/px1s2b8-bench-varlen.rc
