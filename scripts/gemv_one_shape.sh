#!/usr/bin/env bash
# Sweep ONE (N,K) and ingest it — for a shape the census turned up after a campaign ran.
#   $1 N  $2 K  $3 label  [$4 objdir]
set -eu
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:?N}"; K="${2:?K}"; LABEL="${3:?label}"
OBJ="${4:-/home/lava/plow/build-amd/gemvsweep-objs}"
JSONL="/tmp/gemv_one_${LABEL}.jsonl"
rm -f "$JSONL"
env GPU_LEASE_TIMEOUT=3600 "$WT/perf-data/harness/gpulease" -n 1 "gemv-$LABEL" sg render -c \
  "cd $OBJ && unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES && PLOW_GEMV_JSONL=$JSONL ./gemv_row_sweep $N $K $LABEL"
cd "$WT"
# TUNE_GPU DECIDES THE CELL. `tunedb-gemv` used to hardcode gfx950's, so a sweep taken here
# published into MI350X's cell and the compiler never looked at it. Default is this box.
nix develop -c cargo run --release -p tunedb --bin tunedb-gemv -- \
    ingest --db tuning --gpu "${TUNE_GPU:-MI300X}" --samples "$JSONL" \
    --campaign gemv-row-inventory
