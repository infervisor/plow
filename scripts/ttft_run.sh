#!/usr/bin/env bash
# §TTFT: the §0-BENCH-legal GLM-5.2 TP4 run with `PLOW_TTFT_LOG=1`, so the
# server log carries one breakdown table per request.
#
# Runs INSIDE `sg render -c` (this login session predates the render gid) and
# unsets HIP_/CUDA_VISIBLE_DEVICES, which COMPOSE with the ROCR ids gpulease
# exports and otherwise leave a HIP binary with "no ROCm-capable device"
# (plans/knob-contract.md §0a).
set -u
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PLOW_TTFT_LOG=1
export NPROMPT="${NPROMPT:-32}"
export BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:---num-warmups 4}"
export LOG="${LOG:-/tmp/ttft_server.log}"
exec bash "$WT/scripts/bench_plowrt_serve.sh" \
  "${ASSETS:-/home/lava/models/glm52_ttft}" "${PORT:-8123}" glm-5.2 zai-org/GLM-5.2-FP8
