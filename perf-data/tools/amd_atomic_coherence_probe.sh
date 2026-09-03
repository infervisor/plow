#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC="$ROOT/perf-data/tools/amd_atomic_coherence_probe.hip"
ARCH=${PLOW_HIP_ARCH:-gfx950}
HIPCC=${PLOW_HIPCC:-hipcc}
OUT_DIR=${PLOW_ATOMIC_PROBE_OUT:-/tmp/plow-atomic-coherence}
BIN="$OUT_DIR/amd-atomic-coherence-$ARCH"

TOKENS=${1:-64}
HIDDEN=${2:-128}
# UINT_MAX asks the probe to use one workgroup per CU.
BLOCKS=${3:-4294967295}
COLD_REPS=${4:-3}
HOT_REPS=${5:-6}

if [[ ${1:-} == --help || $# -gt 5 ]]; then
    echo "usage: $0 [tokens [hidden [blocks [cold_reps [hot_reps]]]]]"
    echo "env: PLOW_HIP_ARCH, PLOW_HIPCC, PLOW_ATOMIC_PROBE_OUT, PLOW_COMPILE_ONLY"
    exit 0
fi

mkdir -p "$OUT_DIR"
"$HIPCC" --offload-arch="$ARCH" -O3 -std=c++17 "$SRC" -o "$BIN"
echo "BUILD PASS arch=$ARCH binary=$BIN"

if [[ ${PLOW_COMPILE_ONLY:-0} == 1 ]]; then
    exit 0
fi

exec "$ROOT/perf-data/tools/gpulease" -n 1 "atomic-coherence-$ARCH" \
    "$BIN" "$TOKENS" "$HIDDEN" "$BLOCKS" "$COLD_REPS" "$HOT_REPS"
