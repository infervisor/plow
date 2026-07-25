#!/usr/bin/env bash
# px1s2_campaign.sh — PX-1 stage-2 three-arm sweep: serialized prefill (off)
# vs stage-1 batched GEMM + per-request-serial attention (s1) vs stage-2
# batched GEMM + block-diagonal varlen flash (varlen). Same binary (stage-2
# branch, Rust identical to stage-1), same 12B ctx8k B=16 blob, same
# harness/profile as the B2 campaign. Only the prefill cubin + PLOW_PF_BATCH
# differ per arm:
#   px1s2-off    ASSETS=/root/gpu-assets-px1s2  PLOW_PF_BATCH unset (legacy path)
#   px1s2-s1     ASSETS=/root/gpu-assets-px1    PLOW_PF_BATCH=1     (stage-1 pf cubin)
#   px1s2-varlen ASSETS=/root/gpu-assets-px1s2  PLOW_PF_BATCH=1     (stage-2 pf cubin)
# Each arm's whole server+sweep runs under the shared GPU bench lock (per-run
# hold, released between arms). PORT=8093: 8091 is a foreign long-running
# plowrt — do not touch it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VUS="${VUS:-1 4 8 16 32}"
PORT="${PORT:-8093}"

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS=/root/gpu-assets-px1s2 TAG=px1s2-off CAMPAIGN=PX1-s2 \
      PORT="$PORT" VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1s2-bench-off.log 2>&1
echo "OFF rc=$?" | tee /tmp/px1s2-bench-off.rc

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS=/root/gpu-assets-px1 TAG=px1s2-s1 CAMPAIGN=PX1-s2 \
      PORT="$PORT" PLOW_PF_BATCH=1 VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1s2-bench-s1.log 2>&1
echo "S1 rc=$?" | tee /tmp/px1s2-bench-s1.rc

flock -w 14400 /tmp/plow-gpu-bench.lock \
  env ENGINE=plow ASSETS=/root/gpu-assets-px1s2 TAG=px1s2-varlen CAMPAIGN=PX1-s2 \
      PORT="$PORT" PLOW_PF_BATCH=1 VUS="$VUS" DO_SWEEP=0 bash "$ROOT/perf-data/bench_b2_ib.sh" \
  > /tmp/px1s2-bench-varlen.log 2>&1
echo "VARLEN rc=$?" | tee /tmp/px1s2-bench-varlen.rc
