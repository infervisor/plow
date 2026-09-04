#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 build-dir [gpu0 ... gpu7]" >&2
    exit 2
fi

BUILD_DIR="$(cd "$1" && pwd)"
shift
BENCH="./tp_allreduce_prefill_bench"
ROWS="${TP_ROWS:-8192}"
FULL_HIDDEN="${TP_FULL_HIDDEN:-7168}"
HALF_HIDDEN="${TP_HALF_HIDDEN:-3584}"
NWG_SWEEP="${TP_NWG_SWEEP:-64 80 96 128 160 192 224 256}"

if [ ! -x "$BUILD_DIR/$BENCH" ]; then
    echo "benchmark not found: $BUILD_DIR/$BENCH" >&2
    exit 2
fi

if [ "${TP_CHECK_CONFIG:-0}" = 1 ]; then
    BENCH_ARGS=(--check-config)
elif [ "$#" -eq 8 ]; then
    BENCH_ARGS=("$@")
else
    echo "expected exactly 8 GPU ids" >&2
    exit 2
fi

cd "$BUILD_DIR"
for nwg in $NWG_SWEEP; do
    TP_ROWS="$ROWS" TP_HIDDEN="$HALF_HIDDEN" TP_NWG="$nwg" TP_GATHER=0 \
        "$BENCH" "${BENCH_ARGS[@]}"
    TP_ROWS="$ROWS" TP_HIDDEN="$FULL_HIDDEN" TP_NWG="$nwg" TP_GATHER=0 \
        "$BENCH" "${BENCH_ARGS[@]}"
    TP_ROWS="$ROWS" TP_HIDDEN="$FULL_HIDDEN" TP_NWG="$nwg" TP_GATHER=1 \
        "$BENCH" "${BENCH_ARGS[@]}"
done
