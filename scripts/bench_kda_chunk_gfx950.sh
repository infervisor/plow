#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HIPCC=${PLOW_HIPCC:-hipcc}
ROCM=${ROCM_PATH:-/opt/rocm}
OUT=${TMPDIR:-/tmp}/plow-kda-chunk-bench
WAVES=${KDA_CHUNK_WAVES:-8}
case "$WAVES" in
  4|8) ;;
  *) echo "KDA_CHUNK_WAVES must be 4 or 8" >&2; exit 2 ;;
esac
THREADS=$((WAVES * 64))

"$HIPCC" --offload-arch=gfx950 -O3 -w -DKDA3_DEVICE -DPLOW_WG_WAVES="$WAVES" --genco \
  "$ROOT/runtime/tests/kda_step_cdna3_test.hip" -o "$OUT.co" \
  -I"$ROOT/runtime/amd" -I"$ROOT/runtime/common"
"$HIPCC" -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -DKDA_BENCH_THREADS="$THREADS" -I"$ROCM/include" \
  "$ROOT/runtime/tests/kda_step_cdna3_test.hip" -o "$OUT" \
  -L"$ROCM/lib" -lamdhip64

exec "$ROOT/perf-data/tools/gpulease" -n 1 kda-chunk-gfx950 \
  env KDA3_BENCH=1 "$OUT" "$OUT.co"
